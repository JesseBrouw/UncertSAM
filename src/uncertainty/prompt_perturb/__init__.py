from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from src.metrics import entropy
from src.prompt_utils import preturb_prompts
from .config import PromptPerturbationConfig


class PromptPerturbationMixin:
    """Handles prompt sampling, perturbations, and SAM forward steps."""
    def __init__(
        self,
        prompt_config: Optional[Union[PromptPerturbationConfig, Dict[str, Any]]] = None,
    ) -> None:
        config = PromptPerturbationConfig.load(prompt_config)
        self.prompt_perturbations: int = int(config.prompt_perturbations)
        self.bbox_probability: float = float(config.bbox_probability)

    def _build_point_inputs(
        self,
        existing_inputs: Optional[List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]]],
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        box_coords: Optional[torch.Tensor],
        box_labels: Optional[torch.Tensor],
        targets: torch.Tensor,
        enable_perturbation: bool,
    ) -> List[Any]:
        """Return the list of prompts to iterate over during inference."""
        if existing_inputs is not None:
            return existing_inputs

        if point_coords is None and box_coords is None:
            return []

        prompts: List[Any] = [(point_coords, point_labels, box_coords, box_labels)]
        if enable_perturbation:
            for _ in range(max(0, self.prompt_perturbations - 1)):
                prompts.append(
                    preturb_prompts(
                        point_coords,
                        point_labels,
                        box_coords,
                        box_labels,
                        targets,
                        device=self.device,  
                        generator=self.generator, 
                    )
                )
        return prompts

    def _construct_multistep_prompt(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        box_coords: Optional[torch.Tensor],
        box_labels: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build multi-step prompt: optionally prepend bbox before point clicks."""
        if torch.rand(1, device=self.device, generator=self.generator) <= self.bbox_probability:  
            return (
                {
                    "point_coords": torch.cat((box_coords, point_coords[:, 1:]), dim=1),
                    "point_labels": torch.cat((box_labels, point_labels[:, 1:]), dim=1),
                }
                if box_coords is not None and point_coords is not None
                else None
            )

        return (
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
            if point_coords is not None
            else None
        )

    def _construct_singlestep_prompt(
        self,
        point_coords: Optional[torch.Tensor],
        point_labels: Optional[torch.Tensor],
        box_coords: Optional[torch.Tensor],
        box_labels: Optional[torch.Tensor],
    ) -> Optional[Dict[str, torch.Tensor]]:
        """Build single-step prompt using either bbox or a set of points."""
        if torch.rand(1, device=self.device, generator=self.generator) <= self.bbox_probability:  
            return (
                {
                    "point_coords": box_coords,
                    "point_labels": box_labels,
                }
                if box_coords is not None
                else None
            )
        return (
            {
                "point_coords": point_coords,
                "point_labels": point_labels,
            }
            if point_coords is not None
            else None
        )

    def _step(
        self,
        point_inputs: Optional[Dict[str, torch.Tensor]],
        mask_inputs: Optional[torch.Tensor],
        **kwargs: Any,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single SAM forward step with prepared point/mask inputs."""
        _, sam_outputs, _, _ = self.model._track_step( 
            0,
            True,
            kwargs["current_vision_feats"],
            kwargs["current_vision_pos_embeds"],
            kwargs["feat_sizes"],
            point_inputs,
            mask_inputs,
            {},
            kwargs["num_frames"],
            False,
            None,
            uncertainty_inputs=kwargs["uncertainty_inputs"],
        )

        (
            ious,
            low_res_masks,
            low_res_logvar_maps,
        ) = sam_outputs[4], sam_outputs[5], sam_outputs[7]

        if kwargs["default_prediction"]:
            return (
                low_res_masks,
                entropy(low_res_masks.sigmoid()),
                torch.zeros_like(low_res_masks, dtype=low_res_masks.dtype, device=low_res_masks.device),
                ious,
            )

        if self.uncertainty_method == "laplace":
            low_res_masks, umap, mi_map, std_map = self._predict_with_sampled_weights()
            if getattr(self, "uncertainty_measure", None) == "mutual_information":
                umap = mi_map
        elif self.uncertainty_method == "variance_network":
            low_res_masks, umap, std_map = (
                low_res_masks,
                low_res_logvar_maps,
                torch.zeros_like(low_res_masks, dtype=low_res_masks.dtype, device=low_res_masks.device),
            )
        else:
            low_res_masks, umap, std_map = (
                low_res_masks,
                entropy(low_res_masks.sigmoid()),
                torch.zeros_like(low_res_masks, dtype=low_res_masks.dtype, device=low_res_masks.device),
            )

        return low_res_masks, umap, std_map, ious

    def _multistep_forward(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Iteratively add prompts and collect per-step logits/umaps/std/IoUs."""
        all_low_res_masks, all_umaps, all_std_maps, all_ious = [], [], [], []
        multistep_prompt = kwargs["point_input"]

        if kwargs["uncertainty_inputs"] is not None:
            uncertain_input = kwargs["uncertainty_inputs"]
            uncertain_input = uncertain_input.unsqueeze(2) if uncertain_input.ndim == 4 else uncertain_input
        else:
            uncertain_input = None

        if multistep_prompt is None:
            for i in range(kwargs["mask_inputs"].shape[0]):
                low_res_masks, umap, std_map, ious = self._step(
                    None,
                    kwargs["mask_inputs"][i : i + 1].permute(1, 0, 2, 3) if kwargs["mask_inputs"] is not None else None,
                    current_vision_feats=kwargs["current_vision_feats"],
                    current_vision_pos_embeds=kwargs["current_vision_pos_embeds"],
                    feat_sizes=kwargs["feat_sizes"],
                    num_frames=kwargs["num_frames"],
                    default_prediction=kwargs["default_prediction"],
                    uncertainty_inputs=uncertain_input[0] if uncertain_input is not None else None,
                )

                all_low_res_masks.append(low_res_masks.squeeze(1))
                all_umaps.append(umap.squeeze(1))
                all_ious.append(ious.squeeze(1))
                all_std_maps.append(std_map.squeeze(1))

            return (
                torch.stack(all_low_res_masks),
                torch.stack(all_umaps),
                torch.stack(all_std_maps),
                torch.stack(all_ious),
            )

        start = 2 if torch.any(multistep_prompt["point_labels"][:, 0] == 2) else 1
        n_steps = multistep_prompt["point_coords"].shape[1]

        step_inputs = {
            "point_coords": multistep_prompt["point_coords"][:, :start],
            "point_labels": multistep_prompt["point_labels"][:, :start],
        }

        low_res_masks, umap, std_map, ious = self._step(
            step_inputs,
            kwargs["mask_inputs"][0:1].permute(1, 0, 2, 3) if kwargs["mask_inputs"] is not None else None,
            current_vision_feats=kwargs["current_vision_feats"],
            current_vision_pos_embeds=kwargs["current_vision_pos_embeds"],
            feat_sizes=kwargs["feat_sizes"],
            num_frames=kwargs["num_frames"],
            default_prediction=kwargs["default_prediction"],
            uncertainty_inputs=uncertain_input[0] if uncertain_input is not None else None,
        )
        all_low_res_masks.append(low_res_masks.squeeze(1))
        all_umaps.append(umap.squeeze(1))
        all_ious.append(ious.squeeze(1))
        all_std_maps.append(std_map.squeeze(1))

        for i in range(start, n_steps):
            old_coords = step_inputs["point_coords"]
            old_labels = step_inputs["point_labels"]

            step_inputs = {
                "point_coords": torch.cat((old_coords, multistep_prompt["point_coords"][:, i : i + 1]), dim=1),
                "point_labels": torch.cat((old_labels, multistep_prompt["point_labels"][:, i : i + 1]), dim=1),
            }

            idx = i - 1 if start == 2 else i
            low_res_masks, umap, std_map, ious = self._step(
                step_inputs,
                kwargs["mask_inputs"][idx : idx + 1].permute(1, 0, 2, 3) if kwargs["mask_inputs"] is not None else None,
                current_vision_feats=kwargs["current_vision_feats"],
                current_vision_pos_embeds=kwargs["current_vision_pos_embeds"],
                feat_sizes=kwargs["feat_sizes"],
                num_frames=kwargs["num_frames"],
                default_prediction=kwargs["default_prediction"],
                uncertainty_inputs=uncertain_input[idx] if uncertain_input is not None else None,
            )
            all_low_res_masks.append(low_res_masks.squeeze(1))
            all_umaps.append(umap.squeeze(1))
            all_ious.append(ious.squeeze(1))
            all_std_maps.append(std_map.squeeze(1))

        return (
            torch.stack(all_low_res_masks),
            torch.stack(all_umaps),
            torch.stack(all_std_maps),
            torch.stack(all_ious),
        )

    def _single_step_forward(self, **kwargs: Any) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Single-step prediction path used when not doing multi-step prompting."""
        point_inputs = kwargs["point_input"]

        low_res_masks, umap, std_map, ious = self._step(
            point_inputs,
            kwargs["mask_inputs"][0:1].permute(1, 0, 2, 3) if kwargs["mask_inputs"] is not None else None,
            current_vision_feats=kwargs["current_vision_feats"],
            current_vision_pos_embeds=kwargs["current_vision_pos_embeds"],
            feat_sizes=kwargs["feat_sizes"],
            num_frames=kwargs["num_frames"],
            default_prediction=kwargs["default_prediction"],
            uncertainty_inputs=kwargs["uncertainty_inputs"],
        )
        return (
            low_res_masks.squeeze(1).unsqueeze(0),
            umap.squeeze(1).unsqueeze(0),
            std_map.squeeze(1).unsqueeze(0),
            ious.squeeze(1).unsqueeze(0),
        )


__all__ = ["PromptPerturbationMixin", "PromptPerturbationConfig"]

