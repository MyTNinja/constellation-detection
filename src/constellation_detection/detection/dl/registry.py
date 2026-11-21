"""Architecture registry ensuring Colab-trained and local CPU models match."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from constellation_detection.detection.dl.models import StarCenterUNetMini

__all__ = ["MODEL_REGISTRY", "build_model"]

MODEL_REGISTRY: dict[str, Callable[..., Any]] = {
    "star_center_unet_mini": StarCenterUNetMini,
}


def build_model(model_name: str, model_kwargs: dict[str, Any] | None = None) -> Any:
    """Instantiate a registered Phase 2+ PyTorch star detector."""
    key = str(model_name).strip()
    if key not in MODEL_REGISTRY:
        raise KeyError(
            f"Unknown model {key!r}; registered models: {sorted(MODEL_REGISTRY)}"
        )
    return MODEL_REGISTRY[key](**(model_kwargs or {}))
