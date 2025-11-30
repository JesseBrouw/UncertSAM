from __future__ import annotations

from typing import Any, Dict, Optional

import torch
from torch import nn


def _safe_get(cfg: Any, key: str, default: Any = None) -> Any:
    if hasattr(cfg, key):
        return getattr(cfg, key)
    try:
        return cfg.get(key, default)
    except Exception:
        return default


def build_uncertain_sam2_kwargs(
    *,
    sam_base_model: nn.Module,
    experiment_cfg: Any,
    device: torch.device,
    laplace_checkpoint: Optional[Dict[str, Any]] = None,
    laplace_target_module: Optional[str] = None,
    overrides: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    
    uncertainty_method = _safe_get(experiment_cfg, "uncertainty_method")
    preturb_prompts = _safe_get(experiment_cfg, "preturb_prompts", True)
    if uncertainty_method == "prompt_perturbation":
        preturb_prompts = True

    kwargs: Dict[str, Any] = {
        "sam_base_model": sam_base_model,
        "uncertainty_method": uncertainty_method,
        "device": device,
        "preturb_prompts": preturb_prompts,
        "bbox_probability": _safe_get(experiment_cfg, "bbox_probability"),
        "seed": _safe_get(experiment_cfg, "seed", 42),
        "target_module_name": laplace_target_module,
        "laplace_checkpoint": laplace_checkpoint,
        "laplace_samples": _safe_get(experiment_cfg, "laplace_samples"),
        "prompt_perturbations": _safe_get(experiment_cfg, "prompt_perturbations"),
        "prompt_refinement_method": _safe_get(experiment_cfg, "prompt_refinement_method"),
        "refine_with_sparse_prompts": _safe_get(experiment_cfg, "refine_with_sparse_prompts"),
        "ones_baseline": _safe_get(experiment_cfg, "ones_baseline"),
        "uncertain_prompt_encoder_path": _safe_get(
            experiment_cfg, "uncertain_prompt_encoder_path"
        ),
        "uncertain_mask_decoder_path": _safe_get(
            experiment_cfg, "uncertain_mask_decoder_path"
        ),
        "laplace_config": _safe_get(experiment_cfg, "laplace_config"),
        "prompt_config": _safe_get(experiment_cfg, "prompt_config"),
        "refinement_config": _safe_get(experiment_cfg, "refinement_config"),
    }

    if overrides:
        kwargs.update(overrides)

    return kwargs


__all__ = ["build_uncertain_sam2_kwargs"]