import logging
from typing import Any, Dict, List, Optional, Tuple, Union

import torch
from torch import nn

from src.metrics import entropy, safe_logit
from src.uncertsam2.modeling.sam.mask_decoder import MaskDecoder
from src.uncertsam2.modeling.sam.prompt_encoder import PromptEncoder
from src.uncertsam2.modeling.sam.transformer import TwoWayTransformer
from src.uncertainty import (
    LaplaceConfig,
    LaplaceMixin,
    PromptPerturbationConfig,
    PromptPerturbationMixin,
    RefinementConfig,
    RefinementMixin,
)

log = logging.getLogger(__name__)

class UncertainSAM2(LaplaceMixin, PromptPerturbationMixin, RefinementMixin, nn.Module):
    def __init__(
        self,
        sam_base_model: nn.Module,
        uncertainty_method: str,
        device: torch.device,
        preturb_prompts: bool = True,
        bbox_probability: Optional[float] = None,
        seed: int = 42,
        target_module_name: Optional[str] = None,
        laplace_checkpoint: Optional[Dict[str, Any]] = None,
        laplace_samples: Optional[int] = None,
        prompt_perturbations: Optional[int] = None,
        prompt_refinement_method: Optional[str] = None,
        refine_with_sparse_prompts: Optional[bool] = None,
        ones_baseline: Optional[bool] = None,
        uncertain_prompt_encoder_path: Optional[str] = None,
        uncertain_mask_decoder_path: Optional[str] = None,
        laplace_config: Optional[Union[LaplaceConfig, Dict[str, Any]]] = None,
        prompt_config: Optional[Union[PromptPerturbationConfig, Dict[str, Any]]] = None,
        refinement_config: Optional[Union[RefinementConfig, Dict[str, Any]]] = None,
    ) -> None:
        nn.Module.__init__(self)
        PromptPerturbationMixin.__init__(self, prompt_config)
        RefinementMixin.__init__(self, refinement_config)
        LaplaceMixin.__init__(self, laplace_config)

        self.device = device
        self.model = sam_base_model
        self.preturb_prompts = preturb_prompts
        self.generator = torch.Generator(device=device).manual_seed(seed)
        self.uncertainty_method = uncertainty_method

        if bbox_probability is not None:
            self.bbox_probability = bbox_probability
        if prompt_perturbations is not None:
            self.prompt_perturbations = prompt_perturbations
        if laplace_samples is not None:
            self.laplace_samples = laplace_samples
        if prompt_refinement_method is not None:
            self.prompt_refinement_method = prompt_refinement_method
        if refine_with_sparse_prompts is not None:
            self.refine_with_sparse_prompts = refine_with_sparse_prompts
        if ones_baseline is not None:
            self.ones_baseline = ones_baseline

        self.configure_laplace(laplace_checkpoint, target_module_name)

        if uncertain_prompt_encoder_path is not None:
            self._setup_uncertain_prompt_encoder(uncertain_prompt_encoder_path)
            
        if uncertain_mask_decoder_path is not None:
            self._setup_uncertain_mask_decoder(uncertain_mask_decoder_path)
        
        for param in self.parameters():
            param.requires_grad = False

        self.model.use_mask_input_as_output_without_sam = False  # when mask is passed as prompt, actually use it as additional prompt.
        self.model.multimask_output_in_sam = False
        
    def _setup_uncertain_prompt_encoder(self, path: str, unc_channel_size: int=1) -> None:
        """Load and attach an uncertainty-aware prompt encoder from ``path``."""
        try:
            uncertain_prompt_encoder = PromptEncoder(
                embed_dim=self.model.sam_prompt_embed_dim,
                image_embedding_size=(
                    self.model.sam_image_embedding_size,
                    self.model.sam_image_embedding_size,
                ),
                input_image_size=(self.model.image_size, self.model.image_size),
                mask_in_chans=16,
                use_uncertainty_prompt=True,
                uncertainty_channel_size=unc_channel_size
            )
            state_dict = torch.load(path, map_location=self.device)
            missing_keys, unexpected_keys = uncertain_prompt_encoder.load_state_dict(state_dict, strict=True)

            log.info(f"Loaded prompt encoder from {path}")
            log.info(f"Missing keys: {missing_keys}")
            log.info(f"Unexpected keys: {unexpected_keys}")

            self.model.sam_prompt_encoder = uncertain_prompt_encoder

        except Exception as e:
            log.error(f"Failed to load prompt encoder from {path}. Exception: {e}")
            
    def _setup_uncertain_mask_decoder(self, path: str) -> None:
        """Load and attach a mask decoder with variance head from ``path``."""
        try:
            uncertain_mask_decoder = MaskDecoder(
                num_multimask_outputs=3,
                transformer=TwoWayTransformer(
                    depth=2,
                    embedding_dim=self.model.sam_prompt_embed_dim,
                    mlp_dim=2048,
                    num_heads=8,
                ),
                transformer_dim=self.model.sam_prompt_embed_dim,
                iou_head_depth=3,
                iou_head_hidden_dim=256,
                use_high_res_features=self.model.use_high_res_features_in_sam,
                iou_prediction_use_sigmoid=self.model.iou_prediction_use_sigmoid,
                pred_obj_scores=self.model.pred_obj_scores,
                pred_obj_scores_mlp=self.model.pred_obj_scores_mlp,
                use_multimask_token_for_obj_ptr=self.model.use_multimask_token_for_obj_ptr,
                **(self.model.sam_mask_decoder_extra_args or {}),
            )
            
            state_dict = torch.load(path, map_location=self.device)
            missing_keys, unexpected_keys = uncertain_mask_decoder.load_state_dict(state_dict, strict=True)

            log.info(f"Loaded mask decoder from {path}")
            log.info(f"Missing keys: {missing_keys}")
            log.info(f"Unexpected keys: {unexpected_keys}")

            self.model.uncertain_mask_decoder = uncertain_mask_decoder
            # Only use it for refinement pass 
            self.model.use_uncertain_mask_decoder = False

        except Exception as e:
            log.error(f"Failed to load prompt encoder from {path}. Exception: {e}")


    def forward(
        self,
        inp: Any,
        point_coords: Optional[torch.Tensor] = None,
        point_labels: Optional[torch.Tensor] = None,
        box_coords: Optional[torch.Tensor] = None,
        box_labels: Optional[torch.Tensor] = None,
        point_inputs: Optional[List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]] = None,
        mask_inputs: Optional[torch.Tensor] = None,
        uncertainty_inputs: Optional[torch.Tensor] = None,
        backbone_out: Optional[Dict[str, Any]] = None,
        multistep_inference: bool = False,
        return_backbone_out: bool = False,
        default_prediction: bool = False,
    ) -> Tuple[Any, ...]:
        """Forward that supports prompt perturbation, multi-step prompting, and refinement.

        Returns a tuple (logits, umap, std_map, ious, prompts[, backbone_out]).
        """
        assert point_coords is not None or box_coords is not None or point_inputs is not None, 'Either input points, input bounding boxes or prepared prompts must be provided!'
        
        inp = inp.to(self.device)
        targets = inp.masks
        
        if backbone_out is None :
            backbone_out:dict = self.model.forward_image(inp.flat_img_batch)  
            
        (
            _,
            vision_feats,
            vision_pos_embeds,
            feat_sizes,
        ) = self.model._prepare_backbone_features(backbone_out)
            
        # We act on images, so T is always one, otherwise we would have to wrap the following logic in a loop for frames. 
        frame_idx = 0   # We predict images
        num_frames = inp.num_frames
        img_ids = inp.flat_obj_to_img_idx[frame_idx]
        
        # Select the correct vision features
        current_vision_feats = [x[:, img_ids] for x in vision_feats] 
        current_vision_pos_embeds = [x[:, img_ids] for x in vision_pos_embeds] 
            
        if point_inputs is not None:
            prepared_point_inputs = point_inputs
        else:
            prepared_point_inputs = self._build_point_inputs(
                None,
                point_coords,
                point_labels,
                box_coords,
                box_labels,
                targets,
                enable_perturbation=self.preturb_prompts and self.uncertainty_method == "prompt_perturbation",
            )

        if len(prepared_point_inputs) == 0:
            raise ValueError(
                "No prompt inputs were provided; expected either explicit `point_inputs` "
                "or coordinates/boxes to construct prompts."
            )
        
        all_low_res_masks = [] 
        all_umaps = []
        all_std_maps = []
        all_ious = []
        
        all_prompts = []
        
        for point_input in prepared_point_inputs: 
            if multistep_inference:
                # If prepared_prompt is not passed to the forward call, construct prompt
                if isinstance(point_input, tuple):
                    # If no bbox/point, pass tuple of None values
                    point_input = self._construct_multistep_prompt(*point_input)
                
                all_prompts.append((point_input, mask_inputs))
                
                # Get list of predictive_averages / low_res_logit predictions and umaps per step
                step_low_res_masks, step_umaps, step_std_maps, step_ious = self._multistep_forward(
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    num_frames=num_frames,
                    point_input=point_input,
                    mask_inputs=mask_inputs,
                    default_prediction=default_prediction,
                    uncertainty_inputs=uncertainty_inputs
                )
            else:
                # If prepared_prompt is not passed to the forward call, construct prompt
                if isinstance(point_input, tuple):
                    point_input = self._construct_singlestep_prompt(*point_input)
                
                all_prompts.append((point_input, mask_inputs))
                
                step_low_res_masks, step_umaps, step_std_maps, step_ious = self._single_step_forward(
                    current_vision_feats=current_vision_feats,
                    current_vision_pos_embeds=current_vision_pos_embeds,
                    feat_sizes=feat_sizes,
                    num_frames=num_frames,
                    point_input=point_input,
                    mask_inputs=mask_inputs,
                    default_prediction=default_prediction,
                    uncertainty_inputs=uncertainty_inputs
                )
            
            all_low_res_masks.append(step_low_res_masks)
            all_umaps.append(step_umaps)
            all_std_maps.append(step_std_maps)
            all_ious.append(step_ious)
        
        # Prompt perturbation scenario
        if len(all_low_res_masks) > 1:
            # Reduce list to single tensor. Note that list only contains more than one element when prompt perturbation is used.
            probs = torch.stack(all_low_res_masks).sigmoid()
            pred_avg = probs.mean(0)
            
            std_map = probs.std(0)
            
            umap = entropy(pred_avg)
            
            out = safe_logit(pred_avg)
            ious = torch.stack(all_ious).mean(0)
        else:
            out, umap, std_map, ious = all_low_res_masks[0], all_umaps[0], all_std_maps[0], all_ious[0]
            
            
        out, umap, std_map, ious, all_prompts = self._iterative_refinement(
            inp,
            out,
            umap,
            std_map,
            ious,
            all_prompts,
            backbone_out,
            targets,
            multistep_inference=multistep_inference,
        )
        self.model.use_uncertain_mask_decoder = False

        self._logit_steps = []
        self._umap_steps = []

        if return_backbone_out:
            return out, umap, std_map, ious, all_prompts, backbone_out
        return out, umap, std_map, ious, all_prompts
