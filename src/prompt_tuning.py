import copy
import logging
import os
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

import sys

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

from src.uncertsam2.data import ColorJitter
from src.loss_fns import sam2_loss
from src.metrics import IoU, safe_logit
from src.uncertsam2.modeling.sam.prompt_encoder import PromptEncoder
from src.prompt_utils import prepare_prompt_inputs, preturb_bounding_box
from src.uncertainsam2 import UncertainSAM2
from src.uncertsam2.config import build_uncertain_sam2_kwargs
from src.utils import get_sam, prepare_modules_for_finetuning, setup_device
from src.vos_dataset import (ConcatUncertSAMDataset, RandomUniformSampler,
                             UncertSAMDataset, VOSDataset,
                             uncertsam_collate_fn)

hydra.core.global_hydra.GlobalHydra.instance().clear()
log = logging.getLogger(__name__)
device = setup_device()

class PromptTuningTrainer(pl.LightningModule):
    """LightningModule for prompt tuning refinement over SAM 2 outputs."""

    def __init__(
        self,
        uncertainsam_model: UncertainSAM2,
        preturb_prompts: bool = True,
        sample_from_gt_probability: float = 0.5,
        base_lr: float = 1e-4,
        weight_decay: float = 0.01,
        seed: int = 42,
        num_points_to_sample: int = 8,
        multistep_training: bool = True,
        sparse_prompts: bool = True,
        ones_baseline: bool = False,
    ) -> None:
        super().__init__()
        
        # Save hyperparameters to disk
        self.save_hyperparameters(ignore=['uncertainsam_model'])
        
        # Store base model
        self.uncertain_model = uncertainsam_model
        self.refinement_network = None 
        
        # Whether to calculate baselines (done during testing)
        self.baselines = False 
        
        # whether to use uncertainty maps as input, or dumb baseline (ones/noise)
        self.ones_baseline = ones_baseline
        if self.ones_baseline:
            log.info('Using ones baseline !!!')
        
        self.sparse_prompts = sparse_prompts
        
        # Create copy of inference model for a clean separation of the base model
        # potentially containing laplace hooks etc and the base model with default 
        # prediction for which we optimise the prompt. 
        self.inference_model = copy.deepcopy(uncertainsam_model)
        self.inference_model.la = None 
        self.inference_model.uncertainty_method = 'default'
        self.inference_model.hook = None 
        
        # Generator for stochasticity in prompt perturbation
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.loss_fn = sam2_loss
        
        # Freeze entire UncertainSAM backbone
        for param in self.uncertain_model.parameters():
            param.requires_grad = False 
        for param in self.inference_model.parameters():
            param.requires_grad = False 
            
        pretrained_prompt_encoder_state = self.inference_model.model.sam_prompt_encoder.state_dict()
            
        # Duplicate prompt encoder, with uncertainty embedding network. 
        updated_sam_prompt_encoder = PromptEncoder(
            embed_dim=self.inference_model.model.sam_prompt_embed_dim,
            image_embedding_size=(
                self.inference_model.model.sam_image_embedding_size,
                self.inference_model.model.sam_image_embedding_size,
            ),
            input_image_size=(self.inference_model.model.image_size, self.inference_model.model.image_size),
            mask_in_chans=16,
            use_uncertainty_prompt=True,
            uncertainty_channel_size=1,
        )
        updated_sam_prompt_encoder = updated_sam_prompt_encoder.to(self.inference_model.device)
        
        # Use the weights of the pretrained network everywhere, except in the embedding network 
        updated_sam_prompt_encoder_state = updated_sam_prompt_encoder.state_dict()
        updated_state_dict = {k: v for k, v in pretrained_prompt_encoder_state.items() if k in updated_sam_prompt_encoder_state}
        
        updated_sam_prompt_encoder.load_state_dict(updated_state_dict, strict=False)
        
        # Use the updated "uncertain-aware" prompt encoder in the inference model
        self.inference_model.model.sam_prompt_encoder = updated_sam_prompt_encoder
        
        # Freeze the prompt encoder except the uncertain embedder
        for param in self.inference_model.model.sam_prompt_encoder.parameters():
            param.requires_grad = False 
        for param in self.inference_model.model.sam_prompt_encoder.uncertainty_downscaling.parameters():
            param.requires_grad = True
        for param in self.inference_model.model.sam_prompt_encoder.uncertainty_fusion.parameters():
            param.requires_grad = True 
        
    def forward(self, inp: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run two-stage forward: uncertainty model, then refinement/inference model.

        Returns (final_logits_high_res, umap, std_map, ious, aggregated_scores).
        """
        # Load input, targets and bounding box
        inp = inp.to(device)
        targets = inp.masks
        target_bbox = inp.bboxes[0]
        
        # Store metrics of the potentially multiple forward passes 
        metrics = []
        ### Initial forward doesn't require grad
        with torch.no_grad():
            # Sample num_points, and prepare prompts for forward
            point_coords, point_labels, box_coords, box_labels = prepare_prompt_inputs(
                    bbox_prompts=target_bbox,
                    masks=targets,
                    num_points_to_sample=self.hparams.num_points_to_sample,
                    device=device,
                    generator=self.generator,
                    sampling_strategy='uniform',
                    sample_from_gt_probability=self.hparams.sample_from_gt_probability
                )
            
            if self.hparams.preturb_prompts:
                # Preturb bounding box, not point prompts. These are already randomly sampled
                box_coords = box_coords.reshape(-1, 4)
                box_coords = preturb_bounding_box(box_coords, device=device, generator=self.generator)
                box_coords = box_coords.reshape(-1, 2, 2)
            
            # forward using uncertainty model
            out, umap, std_map, ious, all_prompts, backbone_out = self.uncertain_model(
                inp,
                point_coords, point_labels, box_coords, box_labels,
                mask_inputs=None,
                multistep_inference=self.hparams.multistep_training,
                return_backbone_out=True
            )
            # Calculate initial IoU's
            metrics.append(self._calculate_metrics(out, targets, IoU))
        
        all_prompts = [inp[0] for inp in all_prompts]
        # Void prompt list for the forward passes that do not use sparse prompts
        void_prompt_inputs = [None for _ in all_prompts]
        if self.baselines:
            with torch.no_grad():
                # Calculate baseline using same prompts, but logits as mask inputs
                b1_out, _, _, _, _ = self.inference_model(
                    inp,
                    mask_inputs=out,
                    point_inputs=all_prompts,
                    backbone_out=backbone_out,
                    multistep_inference=self.hparams.multistep_training,
                    return_backbone_out=False,
                    default_prediction=True
                )
                metrics.append(self._calculate_metrics(b1_out, targets.float(), IoU))

                gt_mask_input = targets.squeeze(0)  # num_objects, H, W
                gt_mask_input = gt_mask_input.float().expand(out.shape[0], *gt_mask_input.shape)

                gt_mask_input = safe_logit(gt_mask_input)
                
                # Calculate baseline using same prompts, but gt mask as input
                b1_out, _, _, _, _ = self.inference_model(
                    inp,
                    mask_inputs=gt_mask_input,
                    point_inputs=all_prompts,
                    backbone_out=backbone_out,
                    multistep_inference=self.hparams.multistep_training,
                    return_backbone_out=False,
                    default_prediction=True
                )
                metrics.append(self._calculate_metrics(b1_out, targets.float(), IoU))
                
                # Calculate baseline using no sparse prompts, only logits
                b1_out, _, _, _, _ = self.inference_model(
                    inp,
                    mask_inputs=out,
                    point_inputs=void_prompt_inputs,
                    backbone_out=backbone_out,
                    multistep_inference=self.hparams.multistep_training,
                    return_backbone_out=False,
                    default_prediction=True
                )
                metrics.append(self._calculate_metrics(b1_out, targets.float(), IoU))
                
                # Calculate baseline using no sparse prompts, only gt masks
                b1_out, _, _,  _, _ = self.inference_model(
                    inp,
                    mask_inputs=gt_mask_input,
                    point_inputs=void_prompt_inputs,
                    backbone_out=backbone_out,
                    multistep_inference=self.hparams.multistep_training,
                    return_backbone_out=False,
                    default_prediction=True
                )
                metrics.append(self._calculate_metrics(b1_out, targets.float(), IoU))
                
                
        if self.ones_baseline:
            umap = torch.ones_like(umap, device=umap.device, dtype=umap.dtype)
        
        # Second forward pass with default prediction 
        final_out, _, std_map, final_ious, _ = self.inference_model(
            inp,
            mask_inputs = out,
            uncertainty_inputs = umap, 
            point_inputs=all_prompts if self.sparse_prompts else void_prompt_inputs,
            backbone_out=backbone_out,
            multistep_inference=self.hparams.multistep_training,
            return_backbone_out=False,
            default_prediction=True
        )
        
        metrics.append(self._calculate_metrics(final_out, targets, IoU))
        
        results = torch.cat(metrics, dim=1)
        results = torch.mean(results, dim=0)
        
        high_res_out = F.interpolate(final_out, size=(1024, 1024), mode='bilinear', align_corners=False)
        
        return high_res_out, umap, std_map, final_ious, results
    
    def _calculate_metrics(self, out: torch.Tensor, targets: torch.Tensor, metric_func: Callable[[torch.Tensor, torch.Tensor], torch.Tensor]) -> torch.Tensor:
        results = []
        
        high_res_out = F.interpolate(out, size=(1024, 1024), mode='bilinear', align_corners=False)
        # Loop over steps
        for i in range(high_res_out.shape[0]):
            pred = high_res_out[i:i+1] if self.hparams.multistep_training else high_res_out
            preds = (pred > 0.0).float()
            targets = targets.float()
            
            # Take average result over num_objects to get image level metric
            result = torch.mean(metric_func(preds, targets), dim=1, keepdim=True)
            
            results.append(result)
        
        return torch.vstack(results)
    

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        # Forward pass 
        out, umap, std_map, ious, mean_scores = self(batch)
        
        targets = batch.masks.squeeze(0)  # 1, num_obj, 1024, 1024 -> num_obj, 1, 1024, 1024
        batch_size = len(batch.img_batch)

        # Loss
        loss = self.loss_fn(out, targets.float(), ious)['core']

        self.log(f"train_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        
        self.log("lr", self.lr_schedulers().get_last_lr()[0], on_step=True, logger=True)
        self.log("mIoU_improvement", (mean_scores[-1] - mean_scores[0]).item(), on_step=True, logger=True)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, umap, std_map, ious, mean_scores = self(batch)
        out = out.detach()
        umap = umap.detach()
        ious = ious.detach()
        
        targets = batch.masks.squeeze(0)   # [1(frame, always 1), num_objects, 1024, 1024] -> [num_objects, 1, 1024, 1024], like the output of the models
        batch_size = len(batch.img_batch)

        loss = self.loss_fn(out, targets.float(), ious)
        core_loss = loss['core']
        
        self.log(f"val_loss", core_loss, prog_bar=True, logger=True, batch_size=batch_size)

        self.val_losses.append(core_loss)
        self.val_scores.append(mean_scores)
        
        return core_loss

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        out, umap, std_map, ious, mean_scores = self(batch)
        targets = batch.masks.squeeze(0)
        batch_size = len(batch.img_batch)

        loss = self.loss_fn(out, targets.float(), ious)['core']

        self.log(f"test_loss", loss, prog_bar=True, logger=True, batch_size=batch_size)
        
        self.test_losses.append(loss)
        self.test_scores.append(mean_scores)

        return loss

    def configure_optimizers(self):
        # Use AdamW for all trainable parameters
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.base_lr,
            weight_decay=self.hparams.weight_decay,
        )
        
        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * (1_000 / 90_000))      # Same ratio as SAM training 
        cooldown_steps = int(total_steps * (5_000 / 90_000))    # Same ratio as SAM training 
        cooldown_start = total_steps - cooldown_steps
        
        def custom_schedule(step):
            # Constant learning rate schedule with warmup and cooldown
            if step < warmup_steps:
                return step / warmup_steps
            elif step >= cooldown_start:
                return max(0.0, (total_steps - step) / cooldown_steps)
            else:
                return 1.0
        
        scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, custom_schedule)
        
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
                "frequency": 1
            }
        }

    def on_train_epoch_start(self):
        # Control stochasticity, but ensure different sampling behaviour per epoch.
        log.info(f"Starting Epoch {self.current_epoch}...")
        

    def on_train_epoch_end(self):
        log.info(f"Finished Epoch {self.current_epoch}")
        
        
    def on_validation_epoch_start(self):
        log.info(f"Starting validation...")
        self.val_losses = []
        self.val_scores = []

    def on_validation_epoch_end(self):
        if self.val_losses == []:
            return 
        log.info(f"Validation Epoch {self.current_epoch} ended.")
        avg_validation_loss = torch.mean(torch.stack(self.val_losses))
        
        log.info(f"Average loss : {avg_validation_loss}")
        self.log('avg_val_loss', avg_validation_loss.item(), logger=True)
        
        avg_scores = torch.mean(torch.stack(self.val_scores), dim=0)
        log.info('Validation average scores : ')
        log.info(-avg_scores[:-1] + avg_scores[-1])
        
        
        
    def on_test_epoch_start(self):
        log.info(f"Starting test...")
        self.test_losses = []
        self.test_scores = []

    def on_test_epoch_end(self):
        log.info(f"Testing completed for epoch {self.current_epoch}")
        log.info('Average test loss : ')
        log.info(torch.mean(torch.stack(self.test_losses)))
        
        avg_scores = torch.mean(torch.stack(self.test_scores), dim=0)
        log.info('Test average scores : ')
        log.info(-avg_scores[:-1] + avg_scores[-1])
        
def get_sam_datasets(dataset_dir: Path | str, split: str = 'train') -> ConcatUncertSAMDataset:
    dataset = []
    target_datasets = ["Fill in the desired training datasets here"]
    
    for target in target_datasets:
        dataset.append(
            UncertSAMDataset(
                        dataset_path=dataset_dir,
                        dataset_name=target,
                        dataset_split=split, 
                        )
                     )
    dataset = ConcatUncertSAMDataset(dataset)
    
    return dataset
        
def get_dataloaders(target_dataset, cfg, tta_transform):
    train_dataset = UncertSAMDataset(
                    dataset_path=cfg.data_dir,
                    dataset_name=target_dataset,
                    dataset_split='train', 
                    )
    val_dataset = UncertSAMDataset(
                    dataset_path=cfg.data_dir,
                    dataset_name=target_dataset,
                    dataset_split='val', 
                    )
    test_dataset = UncertSAMDataset(
                    dataset_path=cfg.data_dir,
                    dataset_name=target_dataset,
                    dataset_split='test', 
                    )
    
    sampler = RandomUniformSampler(
        num_frames=1, # images, so 1 frame per image
        max_num_objects=cfg.experiment.max_num_objects, # controls how many objects per image to sample 
    )
    
    # Instantiate SAM transforms 
    train_transforms = instantiate(cfg.experiment.train_transforms)
    eval_transforms = instantiate(cfg.experiment.eval_transforms)
    
    train_vos = VOSDataset(
        transforms=train_transforms, 
        training=True, 
        video_dataset=train_dataset, 
        sampler=sampler,
        multiplier=1,
        tta_transform = tta_transform,
        tta_samples=cfg.experiment.tta_samples
    )
    
    val_vos = VOSDataset(
        transforms=eval_transforms, 
        training=True, 
        video_dataset=val_dataset, 
        sampler=sampler,
        multiplier=1,
        tta_transform = tta_transform,
        tta_samples=cfg.experiment.tta_samples
    )
    
    test_vos = VOSDataset(
        transforms=eval_transforms, 
        training=True, 
        video_dataset=test_dataset, 
        sampler=sampler,
        multiplier=1,  
        tta_transform = tta_transform,
        tta_samples=cfg.experiment.tta_samples
    )
    
    generator = torch.Generator().manual_seed(cfg.experiment.seed)
    custom_collate = partial(uncertsam_collate_fn, dict_key="train", laplace=False, generator=generator, out_resolution=128, tta=(tta_transform is not None))
    train_loader = DataLoader(
        train_vos, batch_size=1,  shuffle=True, generator=generator,# persistent_workers=False,
        collate_fn=custom_collate, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )
    
    custom_collate = partial(uncertsam_collate_fn, dict_key="val", laplace=False, generator=generator, out_resolution=128, tta=(tta_transform is not None))
    val_loader = DataLoader(
        val_vos, batch_size=1,  shuffle=False, generator=generator, #persistent_workers=False,
        collate_fn=custom_collate, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )
    
        
    custom_collate = partial(uncertsam_collate_fn, dict_key="test", laplace=False, generator=generator, out_resolution=128, tta=(tta_transform is not None))
    test_loader = DataLoader(
        test_vos, batch_size=1,  shuffle=False, generator=generator, #persistent_workers=False,
        collate_fn=custom_collate, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )
    
    return train_loader, val_loader, test_loader


def load_laplace_artifacts(cfg: Any) -> Tuple[Optional[dict], Optional[str]]:
    """Load Laplace checkpoint (if provided) and return auxiliary metadata."""
    laplace_path = cfg.experiment.laplace_checkpoint
    if laplace_path is None:
        return None, None

    laplace_checkpoint: dict = torch.load(laplace_path, map_location=device)
    laplace_target_module = 'sam_mask_decoder.output_hypernetworks_mlps.0.layers.2'
    log.info(laplace_checkpoint.keys())
    log.info(laplace_checkpoint['mean'].shape)
    log.info(laplace_target_module)
    for key, value in laplace_checkpoint.items():
        log.info(key)
        log.info(value)

    return laplace_checkpoint, laplace_target_module


def load_variance_head_weights(cfg: Any, checkpoint_model: torch.nn.Module) -> torch.nn.Module:
    """Attach variance-network weights onto the provided SAM checkpoint if available."""
    variance_ckpt_path = cfg.experiment.variance_network_checkpoint
    if variance_ckpt_path is None:
        return checkpoint_model

    targets = ['model.sam_mask_decoder.']
    variance_network_ckpt = torch.load(variance_ckpt_path, map_location=device, weights_only=False)
    state_dict = variance_network_ckpt['state_dict']

    uncertainty_statedict = OrderedDict()
    for key, value in state_dict.items():
        for target in targets:
            if key.startswith(target):
                log.info(key)
                uncertainty_statedict[key.replace(target, "")] = value

    checkpoint_model.sam_mask_decoder.load_state_dict(uncertainty_statedict, strict=False)
    return checkpoint_model


def resolve_tta_transform(cfg: Any) -> Optional[Callable]:
    """Map the configuration flag to the optional TTA transformation callable."""
    if cfg.experiment.uncertainty_method != 'TTA':
        return None

    match cfg.experiment.tta_transform:
        case 'hue':
            return ColorJitter(
                consistent_transform=True,
                brightness=0.0,
                contrast=0.0,
                saturation=0.0,
                hue=0.5,
            )
        case _:
            return None


def build_prompt_trainer(cfg: Any, log_dir: str) -> Trainer:
    """Create the PyTorch Lightning trainer used for prompt tuning."""
    checkpoint_callback = ModelCheckpoint(
        monitor='avg_val_loss',
        save_top_k=1,
        mode='min',
        filename='best-checkpoint',
        verbose=True,
    )

    return Trainer(
        default_root_dir=log_dir,
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        precision="32-true",
        accelerator="auto",
        devices=1,
        max_epochs=cfg.experiment.max_epochs,
        log_every_n_steps=1,
        val_check_interval=0.5,
        gradient_clip_val=cfg.experiment.grad_clip_norm,
    )


@hydra.main(version_base=None, config_path="../configs", config_name="local_config")
def main(cfg):
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(log_dir)
    
    # Load pretrained SAM model
    checkpoint_model = get_sam(cfg.experiment.sam_model, cfg.checkpoint_mapping, device=device, train_mode=False, multimask_mode=False)

    laplace_checkpoint, laplace_target_module = load_laplace_artifacts(cfg)
    checkpoint_model = load_variance_head_weights(cfg, checkpoint_model)
        
    if laplace_target_module is not None:
        # Select subset of weights to approximate posterior weight distributions for using Laplace
        checkpoint_model = prepare_modules_for_finetuning(checkpoint_model, [laplace_target_module])
    
    tta_transform = resolve_tta_transform(cfg)

    uncertain_kwargs = build_uncertain_sam2_kwargs(
        sam_base_model=checkpoint_model,
        experiment_cfg=cfg.experiment,
        device=device,
        laplace_checkpoint=laplace_checkpoint,
        laplace_target_module=laplace_target_module,
        overrides={
            "prompt_refinement_method": None,
            "refine_with_sparse_prompts": None,
            "ones_baseline": None,
        },
    )
    model = UncertainSAM2(**uncertain_kwargs)

    # Setup target dataset & dataloaders
    target_dataset = ''
    train_loader, val_loader, test_loader = get_dataloaders(target_dataset, cfg, tta_transform)
    

    finetune_model = PromptTuningTrainer(
        model, 
        base_lr=cfg.experiment.base_lr, 
        seed=cfg.experiment.seed,
        preturb_prompts=cfg.experiment.preturb_prompts,
        sample_from_gt_probability=cfg.experiment.sample_from_gt_probability,
        weight_decay=cfg.experiment.weight_decay,
        ones_baseline=cfg.experiment.ones_baseline,
        num_points_to_sample=cfg.experiment.num_points_to_sample,
        multistep_training=cfg.experiment.multistep_training,
        sparse_prompts=cfg.experiment.sparse_prompts,
        )
        
    trainer = build_prompt_trainer(cfg, log_dir)
    
    trainer.fit(finetune_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    
    torch.save(finetune_model.inference_model.model.sam_prompt_encoder.state_dict(), os.path.join(log_dir, "prompt_encoder.pth"))
    
    best_model_path = trainer.checkpoint_callback.best_model_path
    best_model = PromptTuningTrainer.load_from_checkpoint(
            best_model_path,
            uncertainsam_model=model,
            base_lr=cfg.experiment.base_lr, 
            seed=cfg.experiment.seed,
            preturb_prompts=cfg.experiment.preturb_prompts,
            sample_from_gt_probability=cfg.experiment.sample_from_gt_probability,
            weight_decay=cfg.experiment.weight_decay,
            ones_baseline=cfg.experiment.ones_baseline,
            num_points_to_sample=cfg.experiment.num_points_to_sample,
            multistep_training=cfg.experiment.multistep_training,
            sparse_prompts=cfg.experiment.sparse_prompts,
        )
    
    torch.save(best_model.inference_model.model.sam_prompt_encoder.state_dict(), os.path.join(log_dir, "prompt_encoder_best.pth"))
    
    
if __name__ == '__main__':
    main()
