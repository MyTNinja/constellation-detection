"""Phase 2+ PyTorch training configuration, losses, and seeded training engine."""

from __future__ import annotations

from constellation_detection.training.config import TrainingConfig
from constellation_detection.training.engine import (
    build_optimizer_and_scheduler,
    evaluate_loss,
    run_training,
    train_one_epoch,
)
from constellation_detection.training.losses import StarCenterLoss

__all__ = [
    "StarCenterLoss",
    "TrainingConfig",
    "build_optimizer_and_scheduler",
    "evaluate_loss",
    "run_training",
    "train_one_epoch",
]
