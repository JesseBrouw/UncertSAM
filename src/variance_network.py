import logging
from collections import OrderedDict
from functools import partial
from pathlib import Path
from typing import Any, Dict, List, Tuple

if __package__ is None or __package__ == "":
    # Allow running `python src/variance_network.py` directly.
    import sys

    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import pytorch_lightning as pl
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from pytorch_lightning.callbacks import ModelCheckpoint, TQDMProgressBar
from pytorch_lightning.trainer import Trainer
from torch.utils.data import DataLoader

from src.loss_fns import variance_network_loss
from src.metrics import pearson_corr
from src.prompt_utils import prepare_prompt_inputs, preturb_bounding_box
from src.utils import get_sam, setup_device
from src.vos_dataset import (ConcatUncertSAMDataset, RandomUniformSampler,
                             UncertSAMDataset, VOSDataset,
                             uncertsam_collate_fn)

hydra.core.global_hydra.GlobalHydra.instance().clear()
log = logging.getLogger(__name__)
device = setup_device()
    
class VarianceNetworkTrainerSAM2(pl.LightningModule):
    """LightningModule to train SAM 2 variance (logvar) head with frozen backbone."""

    VARIANCE_HEAD_PREFIXES = (
        "model.sam_mask_decoder.output_upscaling_uncertainty",
        "model.sam_mask_decoder.uncertainty_hypernetworks_mlps",
    )

    def __init__(
        self,
        sam_base_model: nn.Module,
        preturb_prompts: bool = True,
        bbox_probability: float = 0.5,
        base_lr: float = 1e-4,
        weight_decay: float = 0.1,
        seed: int = 42,
        multistep_training: bool = True,
    ) -> None:
        super().__init__()
        
        self.save_hyperparameters(ignore=['sam_base_model'])
        self.model = sam_base_model
        
        # Generator for stochasticity in prompt perturbation
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.loss_fn = variance_network_loss

        self.model.multimask_output_in_sam = False
        
    def forward(self, inp: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass yielding logits, log-variance, and IoU per step.

        Expects a VOSDataset-like batch object with attributes: ``masks``,
        ``bboxes``, ``flat_img_batch``, ``num_frames``, and ``flat_obj_to_img_idx``.
        Returns tensors shaped as multi-step stacks if ``multistep_training``.
        """
        inp = inp.to(device)
        targets = inp.masks
        backbone_out: Dict[str, Any] = self.model.forward_image(inp.flat_img_batch)  # {"backbone_fpn": ..., "vision_pos_enc": ...}
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self.model._prepare_backbone_features(backbone_out)
        
        # Images scenario: T is always 1; use frame index 0.
        frame_idx = 0
        num_frames = inp.num_frames
        img_ids = inp.flat_obj_to_img_idx[frame_idx]
        
        # Select the correct vision features
        current_vision_feats = [x[:, img_ids] for x in vision_feats] 
        current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds] 
        
        bboxes = inp.bboxes[frame_idx]                  # Tensor of [T, O, 4]
            
        point_coords, point_labels, box_coords, box_labels = prepare_prompt_inputs(
            bbox_prompts=bboxes, 
            masks=targets, 
            device=device,
            generator=self.generator,
            sample_from_gt_probability=0.5,             # Probability first step uses bbox
            num_points_to_sample=8,                     # Number of steps
            sampling_strategy='uniform'                 # Could extend to uncertainty-aware sampling
        )
        
        if self.hparams.multistep_training:
            point_input = self._construct_multistep_prompt(point_coords, point_labels, box_coords, box_labels)
            
            # Get list of predictive_averages / low_res_logit predictions and umaps per step
            return self._multistep_forward(
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                num_frames=num_frames,
                point_input=point_input,
                mask_inputs=None
            )
        else:
            point_input = self._construct_singlestep_prompt(point_coords, point_labels, box_coords, box_labels)
            
            return self._single_step_forward(
                current_vision_feats=current_vision_feats,
                current_vision_pos_embeds=current_vision_pos_embeds,
                feat_sizes=feat_sizes,
                num_frames=num_frames,
                point_input=point_input,
                mask_inputs=None
            )

    def _step(self, point_inputs: Dict[str, torch.Tensor], **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single SAM forward pass given prepared point inputs."""
        _, sam_outputs, _, _ = self.model._track_step(
            0,
            True,
            kwargs['current_vision_feats'],
            kwargs['current_vision_pos_embeds'],
            kwargs['feat_sizes'],
            point_inputs,
            None,
            {},             # output_dict (for tracking memory) 
            kwargs['num_frames'],
            False,          # track_in_reverse
            None,           # prev_sam_mask_logits  
        )
        (
            ious,
            high_res_masks,
            high_res_logvar_maps,
        ) = sam_outputs[4], sam_outputs[6], sam_outputs[8]
        
        return high_res_masks, high_res_logvar_maps, ious
    
    def _construct_multistep_prompt(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_coords: torch.Tensor,
        box_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build multi-step prompt sequence; optionally perturb bbox and prepend it."""
        # Use bounding box as first step prompt with certain probability
        if torch.rand(1, device=self.device, generator=self.generator) <= self.hparams.bbox_probability:
            if self.hparams.preturb_prompts:
                # Preturb bounding box, not point prompts. These are already randomly sampled
                box_coords = box_coords.reshape(-1, 4)
                box_coords = preturb_bounding_box(box_coords, device=device, generator=self.generator)
                box_coords = box_coords.reshape(-1, 2, 2)
            
            prepared_input = {
                'point_coords' : torch.cat((box_coords, point_coords[:, 1:]), dim=1),  # [num_objects, num_steps, 2]
                'point_labels' : torch.cat((box_labels, point_labels[:, 1:]), dim=1),  # [num_objects, num_steps]
            }
        else:
            prepared_input = {
                'point_coords' : point_coords,  # [num_objects, num_steps, 2]
                'point_labels' : point_labels,  # [num_objects, num_steps]
            }
        
        return prepared_input
    
    def _construct_singlestep_prompt(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_coords: torch.Tensor,
        box_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Build single-step prompt input (either bbox or points)."""
        if torch.rand(1, device=self.device, generator=self.generator) <= self.hparams.bbox_probability:
            if self.hparams.preturb_prompts:
                # Preturb bounding box, not point prompts. These are already randomly sampled
                box_coords = box_coords.reshape(-1, 4)
                box_coords = preturb_bounding_box(box_coords, device=device, generator=self.generator)
                box_coords = box_coords.reshape(-1, 2, 2)
                
            prepared_input = {
                'point_coords' : box_coords,  # [num_objects, num_steps, 2]
                'point_labels' : box_labels,  # [num_objects, num_steps]
            }
        else:
            prepared_input = {
                'point_coords' : point_coords,  # [num_objects, num_steps, 2]
                'point_labels' : point_labels,  # [num_objects, num_steps]
            }
        
        return prepared_input
    
    def _multistep_forward(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iteratively add prompts and collect per-step predictions and logvars."""
        all_low_res_masks, all_umaps, all_ious = [], [], []
        multistep_prompt = kwargs['point_input']
        
        start = 2 if torch.any(multistep_prompt['point_labels'][:, 0]==2) else 1
        
        n_steps = multistep_prompt['point_coords'].shape[1]  # Extract the number of steps from point prompt tensor

        step_inputs = {
                'point_coords' : multistep_prompt['point_coords'][:, :start],  # Step 1, first point [num_objects, 1, 2]
                'point_labels' : multistep_prompt['point_labels'][:, :start],  # Step 1, first point [num_objects, 1]
            }
        
        low_res_masks, umap, ious = self._step(
            step_inputs,
            current_vision_feats=kwargs['current_vision_feats'],
            current_vision_pos_embeds=kwargs['current_vision_pos_embeds'],
            feat_sizes=kwargs['feat_sizes'],
            num_frames=kwargs['num_frames'],
        )        
        all_low_res_masks.append(low_res_masks.squeeze(1))
        all_umaps.append(umap.squeeze(1))
        all_ious.append(ious.squeeze(1))
        
        # Predict n_steps, where in each step a new sampled point is simulated and added to the already existing input. 
        for i in range(start, n_steps):
            old_coords = step_inputs['point_coords']
            old_labels = step_inputs['point_labels']
            
            step_inputs = {
                'point_coords' : torch.cat((old_coords, multistep_prompt['point_coords'][:, i:i+1]), dim=1),  # Step n, [num_objects, n, 2]
                'point_labels' : torch.cat((old_labels, multistep_prompt['point_labels'][:, i:i+1]), dim=1)   # Step n, first label [num_objects, n, 1]
            }
            
            idx = i - 1 if start == 2 else i
            low_res_masks, umap, ious = self._step(
                step_inputs,
                current_vision_feats=kwargs['current_vision_feats'],
                current_vision_pos_embeds=kwargs['current_vision_pos_embeds'],
                feat_sizes=kwargs['feat_sizes'],
                num_frames=kwargs['num_frames']
            )        
            all_low_res_masks.append(low_res_masks.squeeze(1))
            all_umaps.append(umap.squeeze(1))
            all_ious.append(ious.squeeze(1))
        
        # Return [num_steps, num_objects, 256, 256] tensors containing predicted averages and uncertainty maps per step.   
        return torch.stack(all_low_res_masks), torch.stack(all_umaps), torch.stack(all_ious)
        
    def _single_step_forward(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step forward used when not training with multi-step prompts."""
        point_inputs = kwargs['point_input']
        
        low_res_masks, umap, ious = self._step(
            point_inputs,
            kwargs['mask_inputs'][0:1].permute(1, 0, 2, 3) if kwargs['mask_inputs'] is not None else None,
            current_vision_feats=kwargs['current_vision_feats'],
            current_vision_pos_embeds=kwargs['current_vision_pos_embeds'],
            feat_sizes=kwargs['feat_sizes'],
            num_frames=kwargs['num_frames']
        )
        return low_res_masks.squeeze(1).unsqueeze(0), umap.squeeze(1).unsqueeze(0), ious.squeeze(1).unsqueeze(0)
    
    def _prepare_backbone_features(
        self, backbone_out: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], List[torch.Tensor], List[torch.Tensor], List[Tuple[int, int]]]:
        """Prepare and flatten visual features. [Unused helper for parity]"""
        backbone_out = backbone_out.copy()
        assert len(backbone_out["backbone_fpn"]) == len(backbone_out["vision_pos_enc"])
        num_feature_levels = getattr(self.model, 'num_feature_levels', 1)
        assert len(backbone_out["backbone_fpn"]) >= num_feature_levels

        feature_maps = backbone_out["backbone_fpn"][-num_feature_levels:]
        vision_pos_embeds = backbone_out["vision_pos_enc"][-num_feature_levels:]

        feat_sizes = [(x.shape[-2], x.shape[-1]) for x in vision_pos_embeds]
        # flatten NxCxHxW to HWxNxC
        vision_feats = [x.flatten(2).permute(2, 0, 1) for x in feature_maps]
        vision_pos_embeds = [x.flatten(2).permute(2, 0, 1) for x in vision_pos_embeds]

        return backbone_out, vision_feats, vision_pos_embeds, feat_sizes

    def _filter_variance_state_dict(self, state_dict: Dict[str, torch.Tensor]) -> "OrderedDict[str, torch.Tensor]":
        """Return only the variance head parameters from a (possibly full) state dict."""
        return OrderedDict(
            (key, value)
            for key, value in state_dict.items()
            if key.startswith(self.VARIANCE_HEAD_PREFIXES)
        )

    def state_dict(self, *args: Any, **kwargs: Any) -> "OrderedDict[str, torch.Tensor]":
        """Persist only the variance head weights to keep checkpoints lightweight."""
        full_state = super().state_dict(*args, **kwargs)
        return self._filter_variance_state_dict(full_state)

    def load_state_dict(self, state_dict: Dict[str, torch.Tensor], strict: bool = True):
        """Restore only the variance head weights while keeping the SAM backbone frozen."""
        filtered_state = self._filter_variance_state_dict(state_dict)
        current_state = super().state_dict()
        if not filtered_state:
            log.warning("No variance head weights found in provided state_dict; keeping defaults.")
            return super().load_state_dict(current_state, strict=strict)

        current_state.update(filtered_state)
        return super().load_state_dict(current_state, strict=strict)

    def load_variance_head_from_checkpoint(self, checkpoint_path: str) -> None:
        """Attach a trained variance head from ``checkpoint_path`` to the current SAM model."""
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        state_dict = checkpoint["state_dict"] if "state_dict" in checkpoint else checkpoint
        incompatible = self.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            log.warning(f"Missing variance head keys when loading checkpoint: {incompatible.missing_keys}")
        if incompatible.unexpected_keys:
            log.warning(f"Unexpected keys when loading checkpoint: {incompatible.unexpected_keys}")

    def training_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Compute heteroscedastic loss on training batch and log it."""
        # Forward pass
        out, logvar, iou = self(batch)
        targets = batch.masks.squeeze(0)  # 1, num_obj, 1024, 1024 -> num_obj, 1, 1024, 1024
        batch_size = len(batch.img_batch)

        # Loss
        loss = self.loss_fn(out, targets.float(), logvar)

        self.log(f"train_loss", loss.item(), prog_bar=True, logger=True, batch_size=batch_size)

        return loss
    
    @torch.no_grad()
    def validation_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Evaluate and log validation loss and Pearson correlation."""
        out, logvar, iou = self(batch)
        out = out.detach()
        logvar = logvar.detach()
        iou = iou.detach()
        
        targets = batch.masks.squeeze(0)   # [1(frame, always 1), num_objects, 1024, 1024] -> [num_objects, 1, 1024, 1024], like the output of the models
        batch_size = len(batch.img_batch)

        loss = self.loss_fn(out, targets.float(), logvar)
        
        # Calculate pearson between error and logvar for each step
        pearsons = [pearson_corr(logvar[i].view(-1), torch.abs(targets.float() - out[i].sigmoid()).view(-1)).detach() for i in range(logvar.shape[0])]
        # report average over steps
        pearson = torch.stack(pearsons).mean().detach().cpu()
        
        self.log(f"val_loss", loss.item(), prog_bar=True, logger=True, batch_size=batch_size)
        self.log(f"val_pearson", pearson.item(), prog_bar=True, logger=True, batch_size=batch_size)

        self.val_losses.append(loss.item())
        self.val_pearson.append(pearson.item())
        
        return loss

    @torch.no_grad()
    def test_step(self, batch: Any, batch_idx: int) -> torch.Tensor:
        """Evaluate and log test loss."""
        out, logvar, iou = self(batch)
        targets = batch.masks.squeeze(0)
        batch_size = len(batch.img_batch)

        loss = self.loss_fn(out, targets.float(), logvar)

        self.log(f"test_loss", loss.item(), prog_bar=True, logger=True, batch_size=batch_size)

        return loss

    def configure_optimizers(self) -> Dict[str, Any]:
        """Set up AdamW with no weight decay for LN and bias, and a custom LR schedule."""
        # According to fine-tuning advice, no weight decay for LN and Bias terms
        no_weight_decay = []
        weight_decay = []
    
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            # According to SAM 2 training config, no layernorm weight decay or bias weight decay
            if "bias" in name or "LayerNorm" in name:
                no_weight_decay.append(param)
            else:
                weight_decay.append(param)

        optimizer = torch.optim.AdamW([
            {
                'params': no_weight_decay,                  
                'lr': self.hparams.base_lr, 
                'weight_decay': 0.0,
            },
            {
                'params': weight_decay,
                'lr': self.hparams.base_lr,
                'weight_decay': self.hparams.weight_decay,
            }
        ])

        total_steps = self.trainer.estimated_stepping_batches
        warmup_steps = int(total_steps * (1_000 / 90_000))  # Same ratio as SAM training 
        cooldown_steps = int(total_steps * (5_000 / 90_000))    # Same ratio as SAM training 
        cooldown_start = total_steps - cooldown_steps

        def custom_schedule(step):
            # Constant learning rate schedule with warmup and cooldown
            if step < warmup_steps:
                return step / warmup_steps
            elif step >= cooldown_start:
                return (total_steps - step) / cooldown_steps
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

    def on_train_epoch_start(self) -> None:
        """Epoch-start hook to trace progress and control stochasticity if needed."""
        log.info(f"Starting Epoch {self.current_epoch}...")

    def on_train_epoch_end(self) -> None:
        log.info(f"Finished Epoch {self.current_epoch}")
        
    def on_validation_epoch_start(self) -> None:
        log.info(f"Starting validation...")
        self.val_losses = []
        self.val_pearson = []

    def on_validation_epoch_end(self) -> None:
        if self.val_losses == []:
            return 
        log.info(f"Validation Epoch {self.current_epoch} ended.")
        avg_validation_loss = torch.mean(torch.tensor(self.val_losses))
        log.info(f"Average loss : {avg_validation_loss}")
        
        avg_validation_corr = torch.mean(torch.tensor(self.val_pearson))
        log.info(f"Average pearson : {avg_validation_corr}")

    def on_test_epoch_end(self) -> None:
        log.info(f"Testing completed for epoch {self.current_epoch}")

            
def setup_logvar_head(
    model: pl.LightningModule, initialize_heads: bool = True
) -> pl.LightningModule:
    """Freeze backbone and optionally (re)initialize the variance head for training."""
    if hasattr(model, 'sam2_train_model'): 
        model = model.sam2_train_model

    # Instantiate new logvar head and set to trainable.
    for name, module in model.named_modules():
        if 'uncertainty_hypernetworks_mlps' in name:
            log.info(f'Setting up {name}...')
            for param in module.parameters():
                param.requires_grad = True 
                
            if initialize_heads:
                for m in module.modules():
                    if isinstance(m, nn.Linear):
                        nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                        if m.bias is not None:
                            nn.init.constant_(m.bias, 0)

        elif "output_upscaling_uncertainty" in name:
            log.info(f'Setting up {name}...')
            for param in module.parameters():
                param.requires_grad = True 
        
        else:
            for param in module.parameters():
                param.requires_grad = False
            
    for name, module in model.named_modules():
        if sum(p.numel() for p in module.parameters() if p.requires_grad) > 0:
            log.info(f'Module [{name}] has trainable parameters.')
            
    return model 

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


def _build_vos_datasets(
    cfg: Any,
    sampler: RandomUniformSampler,
    train_transforms: Any,
    eval_transforms: Any,
) -> Tuple[VOSDataset, VOSDataset, VOSDataset]:
    """Instantiate train/val/test datasets with the shared sampler + transforms."""
    train_vos = VOSDataset(
        transforms=train_transforms,
        training=True,
        video_dataset=get_sam_datasets(cfg.data_dir, split='train'),
        sampler=sampler,
        multiplier=1,
    )

    val_vos = VOSDataset(
        transforms=eval_transforms,
        training=True,
        video_dataset=get_sam_datasets(cfg.data_dir, split='val'),
        sampler=sampler,
        multiplier=1,
    )

    test_vos = VOSDataset(
        transforms=eval_transforms,
        training=True,
        video_dataset=get_sam_datasets(cfg.data_dir, split='test'),
        sampler=sampler,
        multiplier=1,
    )

    return train_vos, val_vos, test_vos


def _build_dataloader(dataset: VOSDataset, dict_key: str, cfg: Any, shuffle: bool) -> DataLoader:
    """Shared helper to create our Lightning dataloaders."""
    custom_collate = partial(uncertsam_collate_fn, dict_key=dict_key, laplace=False)
    return DataLoader(
        dataset,
        batch_size=1,
        shuffle=shuffle,
        collate_fn=custom_collate,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory,
    )


def build_dataloaders(cfg: Any) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train/val/test dataloaders used during training/evaluation."""
    sampler = RandomUniformSampler(
        num_frames=1,
        max_num_objects=cfg.experiment.max_num_objects,
    )
    train_transforms = instantiate(cfg.experiment.train_transforms)
    eval_transforms = instantiate(cfg.experiment.eval_transforms)

    train_vos, val_vos, test_vos = _build_vos_datasets(
        cfg=cfg,
        sampler=sampler,
        train_transforms=train_transforms,
        eval_transforms=eval_transforms,
    )

    train_loader = _build_dataloader(train_vos, "train", cfg, shuffle=True)
    val_loader = _build_dataloader(val_vos, "val", cfg, shuffle=False)
    test_loader = _build_dataloader(test_vos, "test", cfg, shuffle=False)

    return train_loader, val_loader, test_loader

@hydra.main(version_base=None, config_path="../configs", config_name="local_config")
def main(cfg: Any) -> None:
    """Train the variance head with heteroscedastic loss and evaluate on validation/test."""
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(log_dir)
    
    # make sure only the first prediction head is used. 
    # Turn multimask mode off to make sure the first head is constantly used 
    checkpoint_model = get_sam(cfg.experiment.sam_model, cfg.checkpoint_mapping, device=device, train_mode=False, multimask_mode=False)
    
    train_loader, val_loader, test_loader = build_dataloaders(cfg)
    
    finetune_model = VarianceNetworkTrainerSAM2(
        checkpoint_model, 
        base_lr=cfg.experiment.base_lr, 
        seed=cfg.experiment.seed, 
        weight_decay=cfg.experiment.weight_decay,
        multistep_training=cfg.experiment.multistep_training,
        preturb_prompts=cfg.experiment.preturb_prompts,
    )

    initialize_heads = cfg.experiment.resume_from_checkpoint_path is None
    finetune_model = setup_logvar_head(finetune_model, initialize_heads=initialize_heads)

    if cfg.experiment.resume_from_checkpoint_path is not None:
        finetune_model.load_variance_head_from_checkpoint(cfg.experiment.resume_from_checkpoint_path)
        log.info('Attached variance head from checkpoint to pretrained SAM backbone.')
        
    for param in finetune_model.model.sam_mask_decoder.uncertainty_hypernetworks_mlps.parameters():
        assert param.requires_grad
        
    for param in finetune_model.model.sam_mask_decoder.output_upscaling_uncertainty.parameters():
        assert param.requires_grad
    
        
    # Save best checkpoint based on validation loss 
    checkpoint_callback = ModelCheckpoint(
        monitor='val_loss',          
        save_top_k=1,               
        mode='min',                 
        filename='best-checkpoint', 
        verbose=True
    )

    trainer = Trainer(
        default_root_dir=log_dir, # Hydra sets cwd to the logging directory. 
        callbacks=[checkpoint_callback, TQDMProgressBar(refresh_rate=1)],
        precision="bf16-mixed",
        accelerator="auto",
        devices=1,
        max_epochs=2,
        log_every_n_steps=1000,
        val_check_interval=1.0,
        gradient_clip_val=cfg.experiment.grad_clip_norm
    )
    
    trainer.fit(finetune_model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    trainer.test(finetune_model, dataloaders=test_loader)
    
    
if __name__ == '__main__':
    main()
