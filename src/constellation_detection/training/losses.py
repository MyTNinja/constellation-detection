"""Masked penalty-reduced focal loss and sub-pixel offset regression loss."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["StarCenterLoss"]


class StarCenterLoss(nn.Module):
    """Combined CenterNet-style heatmap focal loss and masked L1 offset loss."""

    def __init__(
        self,
        alpha: float = 2.0,
        beta: float = 4.0,
        lambda_offset: float = 1.0,
    ) -> None:
        super().__init__()
        self.alpha = float(alpha)
        self.beta = float(beta)
        self.lambda_offset = float(lambda_offset)

    def forward(
        self,
        outputs: dict[str, torch.Tensor],
        targets: dict[str, torch.Tensor],
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute total scalar loss plus detached component breakdown."""
        logits = outputs["heatmap_logits"]
        pred_offset = outputs["offset"]

        gt_hm = targets["heatmap"].to(logits.device, dtype=torch.float32)
        gt_off = targets["offset"].to(logits.device, dtype=torch.float32)
        pos_mask = targets["pos_mask"].to(logits.device, dtype=torch.float32)
        ignore_mask = targets["ignore_mask"].to(logits.device, dtype=torch.float32)

        probs = torch.clamp(torch.sigmoid(logits.float()), 1e-6, 1.0 - 1e-6)
        valid_mask = 1.0 - ignore_mask
        is_pos = pos_mask > 0.5
        is_neg = (~is_pos) & (valid_mask > 0.5)

        pos_count = torch.clamp(pos_mask.sum(), min=1.0)
        pos_loss = -(
            ((1.0 - probs) ** self.alpha) * torch.log(probs) * pos_mask
        ).sum()
        neg_weights = (1.0 - gt_hm) ** self.beta
        neg_loss = -(
            neg_weights * (probs**self.alpha) * torch.log(1.0 - probs) * is_neg.float()
        ).sum()
        hm_loss = (pos_loss + neg_loss) / pos_count

        pos_mask_2ch = pos_mask.expand_as(pred_offset)
        if float(pos_mask.sum().item()) > 0.0:
            off_l1 = F.l1_loss(
                pred_offset.float() * pos_mask_2ch,
                gt_off * pos_mask_2ch,
                reduction="sum",
            )
            off_loss = off_l1 / (2.0 * pos_count)
        else:
            off_loss = torch.zeros((), device=logits.device, dtype=torch.float32)

        total = hm_loss + self.lambda_offset * off_loss
        stats = {
            "loss": float(total.detach().item()),
            "heatmap_loss": float(hm_loss.detach().item()),
            "offset_loss": float(off_loss.detach().item()),
        }
        return total, stats
