"""Typed training configuration matching the Task 07 reference setup."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

__all__ = ["TrainingConfig"]


@dataclass(frozen=True)
class TrainingConfig:
    """Reproducible hyperparameter specification for PyTorch star-center detection."""

    experiment_id: str = "p2_stardet_starcenterunetmini_ds01_s42"
    dataset_version: str = "ds01-v1.0"
    model_name: str = "star_center_unet_mini"
    in_channels: int = 1
    base_channels: int = 16
    crop_size: int = 256
    sigma_px: float = 1.5
    batch_size: int = 16
    epochs: int = 60
    early_stopping_patience: int = 15
    lr: float = 1e-3
    eta_min: float = 1e-5
    weight_decay: float = 1e-4
    warmup_epochs: int = 5
    grad_clip_norm: float = 1.0
    focal_alpha: float = 2.0
    focal_beta: float = 4.0
    lambda_offset: float = 1.0
    amp_cuda_only: bool = True
    seed: int = 42
    deterministic: bool = True
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        """Serialize configuration to a plain JSON-compatible dictionary."""
        return asdict(self)

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> TrainingConfig:
        """Instantiate ``TrainingConfig`` from a dictionary."""
        return cls(**raw)
