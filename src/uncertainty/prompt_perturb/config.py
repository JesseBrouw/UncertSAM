from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

@dataclass
class PromptPerturbationConfig:
    """Configuration for prompt perturbation sampling."""

    prompt_perturbations: int = 10
    bbox_probability: float = 0.5

    @classmethod
    def load(
        cls,
        config: Optional[Union["PromptPerturbationConfig", Dict[str, Any]]] = None,
    ) -> "PromptPerturbationConfig":
        if isinstance(config, cls):
            return config

        defaults = cls()
        data = {field.name: getattr(defaults, field.name) for field in fields(cls)}

        if isinstance(config, dict):
            data.update({k: v for k, v in config.items() if v is not None})

        return cls(**data)


__all__ = ["PromptPerturbationConfig"]
