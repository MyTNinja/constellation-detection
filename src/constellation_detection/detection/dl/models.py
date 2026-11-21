"""Compact PyTorch architectures for sparse point-source star localization.

Models preserve stride-1 spatial resolution to keep adjacent 2-6 px stars separated
and use ``GroupNorm`` for deterministic, batch-size-invariant execution across
Colab GPUs and local CPU tiled inference.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

__all__ = ["StarCenterUNetMini"]


class _ConvBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, num_groups: int = 4) -> None:
        super().__init__()
        groups = min(num_groups, out_ch)
        while groups > 1 and (out_ch % groups) != 0:
            groups -= 1
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(num_groups=groups, num_channels=out_ch),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class StarCenterUNetMini(nn.Module):
    """3-level stride-1 U-Net predicting star-center heatmaps and sub-pixel offsets."""

    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 16,
        num_groups: int = 4,
        heatmap_bias_init: float = -4.595,
    ) -> None:
        super().__init__()
        c1 = int(base_channels)
        c2 = c1 * 2
        c3 = c1 * 4
        c4 = c1 * 8

        self.enc1 = _ConvBlock(in_channels, c1, num_groups=num_groups)
        self.enc2 = _ConvBlock(c1, c2, num_groups=num_groups)
        self.enc3 = _ConvBlock(c2, c3, num_groups=num_groups)
        self.bottleneck = _ConvBlock(c3, c4, num_groups=num_groups)

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dec3 = _ConvBlock(c4 + c3, c3, num_groups=num_groups)
        self.dec2 = _ConvBlock(c3 + c2, c2, num_groups=num_groups)
        self.dec1 = _ConvBlock(c2 + c1, c1, num_groups=num_groups)

        self.heatmap_head = nn.Conv2d(c1, 1, kernel_size=1)
        self.offset_head = nn.Conv2d(c1, 2, kernel_size=1)

        if self.heatmap_head.bias is not None:
            nn.init.constant_(self.heatmap_head.bias, float(heatmap_bias_init))

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Run forward pass returning ``heatmap_logits`` and ``offset`` ``[-0.5, 0.5]``."""
        s1 = self.enc1(x)
        s2 = self.enc2(self.pool(s1))
        s3 = self.enc3(self.pool(s2))
        b = self.bottleneck(self.pool(s3))

        d3 = F.interpolate(b, size=s3.shape[-2:], mode="bilinear", align_corners=False)
        d3 = self.dec3(torch.cat([d3, s3], dim=1))

        d2 = F.interpolate(d3, size=s2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, s2], dim=1))

        d1 = F.interpolate(d2, size=s1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, s1], dim=1))

        heatmap_logits = self.heatmap_head(d1)
        offset = 0.5 * torch.tanh(self.offset_head(d1))
        return {"heatmap_logits": heatmap_logits, "offset": offset}
