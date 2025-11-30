from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

@dataclass
class LaplaceConfig:
    """Container for Laplace approximation hyper-parameters."""

    laplace_samples: int = 1000
    checkpoint_path: Optional[str] = None
    target_module_name: Optional[str] = None

    @classmethod
    def load(
        cls,
        config: Optional[Union["LaplaceConfig", Dict[str, Any]]] = None,
    ) -> "LaplaceConfig":
        if isinstance(config, cls):
            return config

        defaults = cls()
        data = {field.name: getattr(defaults, field.name) for field in fields(cls)}

        if isinstance(config, dict):
            data.update({k: v for k, v in config.items() if v is not None})

        return cls(**data)


__all__ = ["LaplaceConfig"]
