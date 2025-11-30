from __future__ import annotations

import logging
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Optional, Union

import torch
from laplace import Laplace

from src.metrics import entropy, safe_logit
from .config import LaplaceConfig

log = logging.getLogger(__name__)


class LaplaceMixin:
    def __init__(
        self,
        laplace_config: Optional[Union[LaplaceConfig, Dict[str, Any]]] = None,
    ) -> None:
        config = LaplaceConfig.load(laplace_config)
        self.laplace_samples: int = int(config.laplace_samples)
        self.laplace_checkpoint_path: Optional[str] = config.checkpoint_path
        self.default_target_module_name: Optional[str] = config.target_module_name

        self.la: Optional[Laplace] = None
        self.hook = None
        self.captured_input = None
        self.upscaled_embeddings_state = None
        self.laplace_checkpoint_dict: Optional[Dict[str, Any]] = None
        self.laplace_approximation_mode: bool = True
        self.target_layer = None

    def configure_laplace(
        self,
        laplace_checkpoint: Optional[Union[str, Dict[str, Any]]],
        target_module_name: Optional[str],
    ) -> None:
        """Load checkpoint + target layer metadata if both are available."""
        checkpoint: Optional[Dict[str, Any]] = None
        if isinstance(laplace_checkpoint, dict):
            checkpoint = laplace_checkpoint
        elif laplace_checkpoint is not None:
            ckpt_path = Path(str(laplace_checkpoint))
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location="cpu")
            else:
                raise FileNotFoundError(f"Laplace checkpoint path {ckpt_path} was not found.")
        elif self.laplace_checkpoint_path:
            ckpt_path = Path(self.laplace_checkpoint_path)
            if ckpt_path.exists():
                checkpoint = torch.load(ckpt_path, map_location="cpu")

        layer_name = target_module_name or self.default_target_module_name

        self.target_layer = None
        if checkpoint is not None and layer_name is not None:
            try:
                self.target_layer = self.model.get_submodule(layer_name)  # type: ignore[attr-defined]
            except Exception as exc:  # noqa: BLE001
                log.error("Failed to resolve Laplace target layer '%s': %s", layer_name, exc)
                self.target_layer = None
            else:
                self.laplace_checkpoint_dict = checkpoint
                self._create_laplace_object(checkpoint)
                self._activate_hook()
                self.laplace_approximation_mode = False

    def _create_laplace_object(self, laplace_dict: Dict[str, Any]) -> None:
        """Recreate a Laplace object."""
        self.la = Laplace(
            self.model,
            likelihood=laplace_dict["likelihood"],
            subset_of_weights="all",
            hessian_structure="diag",
        )
        for key, value in laplace_dict.items():
            setattr(self.la, key, value)

    def _populate_module(self, module: torch.nn.Module, flattened_weight_samples: torch.Tensor) -> OrderedDict:
        output_state_dict: OrderedDict[str, torch.Tensor] = OrderedDict()
        i = 0
        for name, param in module.named_parameters():
            numel = param.numel()
            output_state_dict[name] = flattened_weight_samples[i : i + numel].view_as(param)
            i += numel
        return output_state_dict

    def _activate_hook(self) -> None:
        self.captured_input = None
        self.upscaled_embeddings_state = None
        if self.target_layer is not None and self.hook is None:
            self.hook = self.target_layer.register_forward_pre_hook(self._laplace_hook_fn)

    def _deactivate_hook(self) -> None:
        if self.hook is not None:
            self.hook.remove()
            self.hook = None

    def _laplace_hook_fn(self, _module: torch.nn.Module, input: tuple) -> None:
        torch.cuda.synchronize()
        captured = input[0].detach().clone().contiguous().cpu()
        if torch.isnan(captured).any():
            log.warning("Detected NaNs in Laplace hook input.")
        self.captured_input = captured

        upscaled = getattr(self.model.sam_mask_decoder, "upscaled_embedding", None) 
        if upscaled is not None:
            upscaled = upscaled.detach().clone().contiguous().cpu()
            if torch.isnan(upscaled).any():
                log.warning("Detected NaNs in cached upscaled embeddings.")
            self.upscaled_embeddings_state = upscaled

    def _predict_with_sampled_weights(
        self,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample decoder weights and aggregate predictions + uncertainties."""
        if self.la is None or self.target_layer is None:
            raise RuntimeError("Laplace object or target layer is not initialized.")

        if self.captured_input is not None:
            self._deactivate_hook()

        outputs = []
        weight_samples = self.la.sample(n_samples=self.laplace_samples)

        dev = self.target_layer.weight.device
        if self.upscaled_embeddings_state is None:
            raise RuntimeError("Laplace expected cached upscaled embeddings but found None.")

        for weights in weight_samples:
            param_dict = self._populate_module(self.target_layer, weights)
            param_dict = {k: v.to(dev) for k, v in param_dict.items()}
            output = torch.func.functional_call(
                self.target_layer,
                param_dict,
                self.captured_input.to(dev),
            )
            outputs.append(output)

        output_stack = torch.stack(outputs, dim=1)

        upscaled_embedding = self.upscaled_embeddings_state.to(dev)
        b, c, h, w = upscaled_embedding.shape
        masks = (output_stack @ upscaled_embedding.view(b, c, h * w)).view(b, -1, h, w)

        self._activate_hook()

        probs = masks.sigmoid()
        avg_pred = torch.mean(probs, dim=1, keepdim=True)
        std_map = torch.std(probs, dim=1, keepdim=True)
        avg_pred_entropy = entropy(avg_pred)
        mutual_information = avg_pred_entropy - torch.mean(entropy(probs), dim=1, keepdim=True)

        return safe_logit(avg_pred), avg_pred_entropy, mutual_information, std_map


__all__ = ["LaplaceMixin"]
