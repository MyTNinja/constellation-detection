"""Seeded PyTorch training loop, CUDA AMP handling, and artifact persistence."""

from __future__ import annotations

import csv
import io
import time
from typing import Any

import torch
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from constellation_detection.artifacts import RunArtifactManager
from constellation_detection.runtime import DeviceInfo, seed_everything
from constellation_detection.training.config import TrainingConfig
from constellation_detection.training.losses import StarCenterLoss

__all__ = [
    "build_optimizer_and_scheduler",
    "evaluate_loss",
    "run_training",
    "train_one_epoch",
]


def build_optimizer_and_scheduler(
    model: torch.nn.Module,
    cfg: TrainingConfig,
) -> tuple[AdamW, Any]:
    """Create ``AdamW`` optimizer and warmup + cosine decay LR schedule."""
    optimizer = AdamW(
        model.parameters(),
        lr=float(cfg.lr),
        weight_decay=float(cfg.weight_decay),
    )
    total_epochs = max(1, int(cfg.epochs))
    warmup = min(max(0, int(cfg.warmup_epochs)), max(0, total_epochs - 1))
    if warmup > 0:
        sched_warm = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup,
        )
        sched_cos = CosineAnnealingLR(
            optimizer,
            T_max=max(1, total_epochs - warmup),
            eta_min=float(cfg.eta_min),
        )
        scheduler = SequentialLR(
            optimizer,
            schedulers=[sched_warm, sched_cos],
            milestones=[warmup],
        )
    else:
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=total_epochs,
            eta_min=float(cfg.eta_min),
        )
    return optimizer, scheduler


def train_one_epoch(
    model: torch.nn.Module,
    loader: Any,
    criterion: StarCenterLoss,
    optimizer: AdamW,
    device: torch.device,
    grad_clip_norm: float = 1.0,
    use_amp: bool = False,
    scaler: Any = None,
) -> dict[str, float]:
    """Run one training epoch over ``loader`` and return mean loss metrics."""
    model.train()
    n_batches = 0
    loss_acc = 0.0
    hm_acc = 0.0
    off_acc = 0.0
    amp_enabled = bool(use_amp and device.type == "cuda")

    for batch in loader:
        images = batch["image"].to(device, dtype=torch.float32)
        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
            outputs = model(images)
            loss, stats = criterion(outputs, batch)

        if amp_enabled and scaler is not None:
            scaler.scale(loss).backward()
            if grad_clip_norm > 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

        n_batches += 1
        loss_acc += stats["loss"]
        hm_acc += stats["heatmap_loss"]
        off_acc += stats["offset_loss"]

    denom = float(max(1, n_batches))
    return {
        "loss": loss_acc / denom,
        "heatmap_loss": hm_acc / denom,
        "offset_loss": off_acc / denom,
    }


def evaluate_loss(
    model: torch.nn.Module,
    loader: Any,
    criterion: StarCenterLoss,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate model loss on validation/test loader without gradients."""
    model.eval()
    n_batches = 0
    loss_acc = 0.0
    hm_acc = 0.0
    off_acc = 0.0

    with torch.inference_mode():
        for batch in loader:
            images = batch["image"].to(device, dtype=torch.float32)
            outputs = model(images)
            _, stats = criterion(outputs, batch)
            n_batches += 1
            loss_acc += stats["loss"]
            hm_acc += stats["heatmap_loss"]
            off_acc += stats["offset_loss"]

    denom = float(max(1, n_batches))
    return {
        "loss": loss_acc / denom,
        "heatmap_loss": hm_acc / denom,
        "offset_loss": off_acc / denom,
    }


def run_training(
    model: torch.nn.Module,
    train_loader: Any,
    val_loader: Any,
    cfg: TrainingConfig,
    artifact_manager: RunArtifactManager,
    device_info: DeviceInfo,
    parity_input_shape: tuple[int, ...] = (1, 1, 64, 64),
) -> dict[str, Any]:
    """Execute full seeded training loop with atomic best/last checkpoints."""
    seed_everything(seed=cfg.seed, deterministic=cfg.deterministic)
    device = torch.device(device_info.device_str)
    model = model.to(device)

    criterion = StarCenterLoss(
        alpha=cfg.focal_alpha,
        beta=cfg.focal_beta,
        lambda_offset=cfg.lambda_offset,
    )
    optimizer, scheduler = build_optimizer_and_scheduler(model, cfg)
    use_amp = bool(cfg.amp_cuda_only and device_info.amp_supported)
    scaler = torch.amp.GradScaler("cuda") if use_amp else None

    artifact_manager.write_json_atomic("train_config.json", cfg.to_dict())

    best_val_loss = float("inf")
    best_epoch = 0
    no_improve = 0
    history: list[dict[str, Any]] = []
    best_weights_path = None

    for epoch in range(1, int(cfg.epochs) + 1):
        t0 = time.perf_counter()
        tr = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device=device,
            grad_clip_norm=cfg.grad_clip_norm,
            use_amp=use_amp,
            scaler=scaler,
        )
        va = evaluate_loss(model, val_loader, criterion, device=device)
        lr_now = float(optimizer.param_groups[0]["lr"])
        scheduler.step()
        elapsed = time.perf_counter() - t0

        row = {
            "epoch": epoch,
            "train_loss": round(tr["loss"], 6),
            "val_loss": round(va["loss"], 6),
            "train_hm_loss": round(tr["heatmap_loss"], 6),
            "val_hm_loss": round(va["heatmap_loss"], 6),
            "train_off_loss": round(tr["offset_loss"], 6),
            "val_off_loss": round(va["offset_loss"], 6),
            "lr": lr_now,
            "epoch_sec": round(elapsed, 3),
        }
        history.append(row)

        if va["loss"] < best_val_loss:
            best_val_loss = float(va["loss"])
            best_epoch = epoch
            no_improve = 0
            best_weights_path = artifact_manager.save_weights_artifact(
                model=model,
                model_name=cfg.model_name,
                model_kwargs={
                    "in_channels": cfg.in_channels,
                    "base_channels": cfg.base_channels,
                },
                tag="best",
                extra_metadata={"best_epoch": epoch, "best_val_loss": best_val_loss},
            )
            artifact_manager.write_parity_reference(
                model=model,
                input_shape=parity_input_shape,
                seed=cfg.seed,
            )
        else:
            no_improve += 1

        artifact_manager.save_training_state(
            {
                "epoch": epoch,
                "best_val_loss": best_val_loss,
                "model_state_dict": {
                    k: v.detach().cpu() for k, v in model.state_dict().items()
                },
                "optimizer_state_dict": optimizer.state_dict(),
                "scheduler_state_dict": scheduler.state_dict(),
            },
            tag="last",
        )

        if no_improve >= int(cfg.early_stopping_patience):
            break

    buf = io.StringIO()
    writer = csv.DictWriter(buf, fieldnames=list(history[0].keys()))
    writer.writeheader()
    writer.writerows(history)
    artifact_manager.write_text_atomic("metrics/epoch_metrics.csv", buf.getvalue())

    return {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "epochs_completed": len(history),
        "best_weights_path": str(best_weights_path) if best_weights_path else None,
        "history": history,
    }
