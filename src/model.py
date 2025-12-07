"""Hybrid dynamic filter equivariant network prototype."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .invariant_features import InvariantBlock, gradient_features
from .moving_frame import MovingFrame


class HybridEquivariantNet(nn.Module):
    """Simple backbone that couples moving frames with invariant filters."""

    def __init__(self, image_size: tuple[int, int], num_classes: int = 10):
        super().__init__()
        self.frame = MovingFrame(image_size=image_size)
        self.low_level = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(8, 16, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.invariant = InvariantBlock(18)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(18, num_classes),
        )

    def forward(self, x: Tensor) -> dict[str, Tensor]:
        warped, homography = self.frame(x)
        features = self.low_level(warped)
        enriched = torch.cat([features, gradient_features(warped)], dim=1)
        encoded = self.invariant(enriched)
        logits = self.head(encoded)
        return {"logits": logits, "homography": homography, "warped": warped}


def build_model(image_size: tuple[int, int], num_classes: int = 10) -> HybridEquivariantNet:
    return HybridEquivariantNet(image_size=image_size, num_classes=num_classes)
