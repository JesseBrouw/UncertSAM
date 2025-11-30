from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Dict, Optional, Union

@dataclass
class RefinementConfig:
    
    method: Optional[str] = None
    max_iterations: int = 1
    refine_with_sparse_prompts: bool = True
    ones_baseline: bool = False

    @classmethod
    def load(
        cls,
        config: Optional[Union["RefinementConfig", Dict[str, Any]]] = None,
    ) -> "RefinementConfig":
        if isinstance(config, cls):
            return config

        defaults = cls()
        field_names = {field.name for field in fields(cls)}
        data = {name: getattr(defaults, name) for name in field_names}

        if isinstance(config, dict):
            data.update({k: v for k, v in config.items() if v is not None and k in field_names})

        return cls(**data)


__all__ = ["RefinementConfig"]
