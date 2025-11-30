from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from src.metrics import IoU, safe_logit
from .config import RefinementConfig

class RefinementMixin:
    def __init__(
        self,
        refinement_config: Optional[Union[RefinementConfig, Dict[str, Any]]] = None,
    ) -> None:
        config = RefinementConfig.load(refinement_config)
        self.prompt_refinement_method: Optional[str] = config.method
        self.max_refinement_iterations: int = int(config.max_iterations)
        self.refine_with_sparse_prompts: bool = bool(config.refine_with_sparse_prompts)
        self.ones_baseline: bool = bool(config.ones_baseline)
        self._logit_steps: List[torch.Tensor] = []
        self._umap_steps: List[torch.Tensor] = []

    def _iterative_refinement(
        self,
        inp: Any,
        logits: torch.Tensor,
        umap: torch.Tensor,
        std_map: torch.Tensor,
        ious: torch.Tensor,
        all_prompts: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]],
        backbone_out: Dict[str, Any],
        targets: torch.Tensor,
        multistep_inference: bool,
        max_iterations: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Any]:
        """Optionally refine predictions using selected method and return updated outputs."""
        method = self.prompt_refinement_method
        if method is None:
            return logits, umap, std_map, ious, all_prompts

        max_iters = max_iterations or self.max_refinement_iterations

        if len(self._logit_steps) > max(0, max_iters - 1):
            return logits, umap, std_map, ious, all_prompts

        if len(self._logit_steps) > 0:
            mask_prev = (self._logit_steps[-1] > 0.0).float()
            mask_curr = (logits > 0.0).float()
            if torch.mean(IoU(mask_prev, mask_curr)) > 0.99:
                return logits, umap, std_map, ious, all_prompts

        self._logit_steps.append(logits)
        self._umap_steps.append(umap)

        if method == "dense_refinement":
            self.model.use_uncertain_mask_decoder = True
            return self._dense_refinement(
                inp,
                logits,
                umap,
                std_map,
                all_prompts,
                backbone_out,
                multistep_inference,
                sparse_prompts=self.refine_with_sparse_prompts,
            )
        if method == "gt_baseline":
            return self._gt_baseline(
                inp,
                logits,
                umap,
                std_map,
                all_prompts,
                backbone_out,
                targets,
                multistep_inference,
                sparse_prompts=self.refine_with_sparse_prompts,
            )
        return logits, umap, std_map, ious, all_prompts

    def _dense_refinement(
        self,
        inp: Any,
        logits: torch.Tensor,
        umap: torch.Tensor,
        std_map: torch.Tensor,
        all_prompts: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]],
        backbone_out: Dict[str, Any],
        multistep_inference: bool,
        sparse_prompts: bool = True,
    ):
        point_inputs, _ = [inp[0] for inp in all_prompts], [inp[1] for inp in all_prompts]

        void_prompt_inputs = [None for _ in point_inputs]

        if self.ones_baseline:
            mask_inp = torch.ones_like(umap, dtype=umap.dtype, device=umap.device)
        else:
            mask_inp = umap

        return self(
            inp,
            mask_inputs=logits,
            uncertainty_inputs=mask_inp.permute(1, 0, 2, 3),
            point_inputs=point_inputs if sparse_prompts else void_prompt_inputs,
            backbone_out=backbone_out,
            multistep_inference=multistep_inference,
            return_backbone_out=False,
            default_prediction=True,
        )

    def _gt_baseline(
        self,
        inp: Any,
        logits: torch.Tensor,
        umap: torch.Tensor,
        std_map: torch.Tensor,
        all_prompts: List[Tuple[Optional[Dict[str, torch.Tensor]], Optional[torch.Tensor]]],
        backbone_out: Dict[str, Any],
        targets: torch.Tensor,
        multistep_inference: bool,
        sparse_prompts: bool = True,
    ):
        gt_mask_input = targets.squeeze(0)
        gt_mask_input = gt_mask_input.float().expand(logits.shape[0], *gt_mask_input.shape)
        gt_mask_input = safe_logit(gt_mask_input)

        point_inputs, _ = [inp[0] for inp in all_prompts], [inp[1] for inp in all_prompts]
        void_prompt_inputs = [None for _ in point_inputs]

        return self(
            inp,
            mask_inputs=gt_mask_input,
            point_inputs=point_inputs if sparse_prompts else void_prompt_inputs,
            backbone_out=backbone_out,
            multistep_inference=multistep_inference,
            return_backbone_out=False,
            default_prediction=True,
        )


__all__ = ["RefinementMixin", "RefinementConfig"]
