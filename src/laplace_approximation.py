import logging
import os
import time
from functools import partial
from pathlib import Path
from typing import Any, Dict
import sys

if __package__ is None or __package__ == "":
    PROJECT_ROOT = Path(__file__).resolve().parents[1]
    if str(PROJECT_ROOT) not in sys.path:
        sys.path.insert(0, str(PROJECT_ROOT))

import hydra
import torch
import torch.nn as nn
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from hydra.utils import instantiate
from laplace import Laplace
from torch.utils.data import DataLoader

from src.prompt_utils import prepare_prompt_inputs, preturb_bounding_box
from src.utils import get_sam, prepare_modules_for_finetuning, setup_device
from src.vos_dataset import (ConcatUncertSAMDataset, RandomUniformSampler,
                             UncertSAMDataset, VOSDataset,
                             uncertsam_collate_fn)

hydra.core.global_hydra.GlobalHydra.instance().clear()
log = logging.getLogger(__name__)
device = setup_device()
    
class LaplaceSAM2(nn.Module):
    """Laplace approximation wrapper for a SAM2-like model (post-hoc).

    Fits a diagonal-Hessian Laplace approximation to a target layer of the
    mask prediction head and uses it to derive uncertainty estimates.
    """

    def __init__(
        self,
        sam_base_model: nn.Module,
        preturb_prompts: bool = True,
        bbox_probability: float = 0.5,
        seed: int = 42,
        num_points_to_sample: int = 8,
        sample_from_gt_probability: float = 0.5,
        out_resolution: int = 128,
    ) -> None:
        super().__init__()
        self.model = sam_base_model
        
        self.bbox_probability = bbox_probability
        self.preturb_prompts = preturb_prompts
        self.generator = torch.Generator(device=device).manual_seed(seed)
        
        self.num_points_to_sample = num_points_to_sample
        self.sample_from_gt_probability = sample_from_gt_probability

        self.out_resolution = out_resolution
        self.model.eval()
            
        
    def forward(self, inp: Any) -> torch.Tensor:
        """Produce downsampled logits for Laplace fitting.

        Expects a VOSDataset-like batch with attributes: ``masks``, ``bboxes``,
        ``flat_img_batch``, ``num_frames``, and ``flat_obj_to_img_idx``.
        Returns a tensor of shape ``[1, N]`` suitable for Laplace.
        """
        inp = inp.to(device)
        targets = inp.masks
        target_bbox = inp.bboxes[0]
        
        backbone_out: Dict[str, Any] = self.model.forward_image(inp.flat_img_batch)
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self.model._prepare_backbone_features(backbone_out)
        
        frame_idx = 0   # Images scenario: frame index is always 0
        num_frames = inp.num_frames
        img_ids = inp.flat_obj_to_img_idx[frame_idx]
        
        # Select the correct vision features
        current_vision_feats = [x[:, img_ids] for x in vision_feats] 
        current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds] 
        
        point_coords, point_labels, box_coords, box_labels = prepare_prompt_inputs(
                bbox_prompts=target_bbox,
                masks=targets,
                num_points_to_sample=self.num_points_to_sample,
                device=device,
                generator=self.generator,
                sampling_strategy='uniform',
                sample_from_gt_probability=self.sample_from_gt_probability
            )
            
        point_inputs = self._construct_multistep_prompt(point_coords, point_labels, box_coords, box_labels)
            
        sam_outputs = self.model._track_step(
            0,
            True,   
            current_vision_feats,
            current_vision_pos_embeds,
            feat_sizes,
            point_inputs,
            None,
            {},           
            num_frames,
            False,         
            None,           
        )[1]
        low_res_masks = sam_outputs[5]
        
        # Resize low-resolution SAM logits to the configured Laplace resolution
        out = F.interpolate(
            low_res_masks.float(),
            size=(self.out_resolution, self.out_resolution),
            mode='bilinear',
        ).squeeze(0)
        
        return out.reshape(1, -1)       # Laplace expects [B, N] output for pixel wise cross_entropy
    
    
    def _construct_multistep_prompt(
        self,
        point_coords: torch.Tensor,
        point_labels: torch.Tensor,
        box_coords: torch.Tensor,
        box_labels: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Create a reduced multi-step prompt sequence for Laplace fitting.

        Optionally perturbs the bounding box, then randomly slices the sequence
        to simulate multi-step prompting with fewer steps.
        """
        if self.preturb_prompts:
            # Preturb bounding box, not point prompts. These are already randomly sampled
            box_coords = box_coords.reshape(-1, 4)
            box_coords = preturb_bounding_box(box_coords, device=device, generator=self.generator)
            box_coords = box_coords.reshape(-1, 2, 2)
        
        # With probability ``bbox_probability``, use bbox as the first step; otherwise use only sampled points
        if torch.rand(1, device=device, generator=self.generator) <= self.bbox_probability:
            prepared_input = {
                'point_coords' : torch.cat((box_coords, point_coords[:, 1:]), dim=1),  # [num_objects, num_steps, 2]
                'point_labels' : torch.cat((box_labels, point_labels[:, 1:]), dim=1),  # [num_objects, num_steps]
            }
        else:
            prepared_input = {
                'point_coords' : point_coords,  # [num_objects, num_steps, 2]
                'point_labels' : point_labels,  # [num_objects, num_steps]
            }
            
        # Because the current Laplace implementation can't deal with the dimensionality of multistep prompts,
        # we sample a random subset of the multistep prompts to simulate a similar procedure but fewer total steps.
        
        # Since prepared_input can be of size [num_obj, num_points, ...] or [num_obj, num_points+1, ...] (when bbox included),
        # sample an integer in [0, num_points_to_sample) and slice accordingly.
        subset_index = torch.randint(0, self.num_points_to_sample, (1,), device=device, generator=self.generator).item()
        if subset_index > 0:
            prepared_input = {
                'point_coords' : prepared_input['point_coords'][:, :-subset_index],  # [num_objects, sampled_number, 2]
                'point_labels' : prepared_input['point_labels'][:, :-subset_index],  # [num_objects, sampled_number]
            }
        
        return prepared_input

    
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


@hydra.main(version_base=None, config_path="../configs", config_name="local_config")
def main(cfg: Any) -> None:
    """Entry point for fitting a Laplace approximation on SAM2 outputs."""
    log_dir = HydraConfig.get().runtime.output_dir
    log.info(log_dir)
    
    # make sure only the first prediction head is used. 
    # Turn multimask mode off to make sure the first head is constantly used. Necessary to know for sure the outputs stem from the layer we approximate the weight distributions for 
    checkpoint_model = get_sam(cfg.experiment.sam_model, cfg.checkpoint_mapping, device=device, train_mode=False, multimask_mode=False)
    
    train_dataset = get_sam_datasets(cfg.data_dir, split='train')
    
    # Instantiate sampler
    sampler = RandomUniformSampler(
        num_frames=1,           # images, so 1 frame per image
        max_num_objects=1,      # controls how many objects per image to sample, because of memory constraints use 1 for Laplace
    )

    # Use same transforms as SAM2 during training 
    transforms = instantiate(cfg.experiment.train_transforms)
    train_vos = VOSDataset(
        transforms=transforms, 
        training=True, 
        video_dataset=train_dataset, 
        sampler=sampler,
        multiplier=3   # Loop over dataset three times
        )
    
    # Set resolution to downsample output to. 
    laplace_resolution = 128
    custom_collate = partial(uncertsam_collate_fn, dict_key="train", laplace=True, out_resolution=laplace_resolution)
    train_loader = DataLoader(
        train_vos, batch_size=1,  shuffle=True,
        collate_fn=custom_collate, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory
        )

    # Select subset of weights to approximate posterior weight distributions for using Laplace
    model = prepare_modules_for_finetuning(checkpoint_model, ['sam_mask_decoder.output_hypernetworks_mlps.0.layers.2'])
    
    # Instantiate model that is used by Laplace for forward passes.
    approximation_model = LaplaceSAM2(
        model, 
        out_resolution=laplace_resolution, 
        preturb_prompts=True,             
        bbox_probability=0.5,             
        num_points_to_sample=8,           
        sample_from_gt_probability=0.5,   
        seed=cfg.experiment.seed,
        ).to(device)
    
    log.info('Start laplace approximation on SAM2...')
    start = time.time()
    
    la = Laplace(
        approximation_model,
        likelihood="classification",
        # note that this means all weights with requires_grad = True -> last layer
        subset_of_weights="all",
        hessian_structure="diag"
    )
    
    la.fit(train_loader)
    
    log.info(f'Finished fitting laplace on SA-1B subset. \nTook {time.time() - start} seconds!')
    log.info('Saving state dicts...')
    # Save Laplace approximation object (this includes the posterior precision, etc.)
    torch.save(la.state_dict(), os.path.join(log_dir, 'laplace_state_dict.pth'))
    torch.save(approximation_model.state_dict(), os.path.join(log_dir, 'model_weights_state_dict.pth'))
    
    log.info('Successfully saved states! \n\nProcess Finished.')
    

if __name__ == '__main__':
    main()
