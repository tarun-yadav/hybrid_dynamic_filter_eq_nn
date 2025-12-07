"""Differential invariant feature blocks."""
from __future__ import annotations

import torch
from torch import Tensor, nn


def gradient_features(x: Tensor) -> Tensor:
    """Compute gradient magnitude and orientation features."""
    sobel_x = torch.tensor([[1, 0, -1], [2, 0, -2], [1, 0, -1]], device=x.device, dtype=x.dtype)
    sobel_y = sobel_x.t()
    sobel_x = sobel_x.view(1, 1, 3, 3)
    sobel_y = sobel_y.view(1, 1, 3, 3)
    gx = torch.nn.functional.conv2d(x, sobel_x, padding=1)
    gy = torch.nn.functional.conv2d(x, sobel_y, padding=1)
    mag = torch.sqrt(gx ** 2 + gy ** 2 + 1e-6)
    ang = torch.atan2(gy, gx)
    return torch.cat([mag, ang], dim=1)


class InvariantBlock(nn.Module):
    """Lightweight block that mixes invariant features."""

    def __init__(self, channels: int):
        super().__init__()
        groups = max(1, channels // 6)
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv2d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)
