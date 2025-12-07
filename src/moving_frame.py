"""Moving frame estimator for projective equivariance."""
from __future__ import annotations

import torch
from torch import Tensor, nn

from .projective import build_homography, warp_image


def default_canonical_corners(batch: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    """Return unit square corners in clockwise order."""
    corners = torch.tensor(
        [[-1.0, -1.0], [1.0, -1.0], [1.0, 1.0], [-1.0, 1.0]], device=device, dtype=dtype
    )
    return corners.unsqueeze(0).repeat(batch, 1, 1)


class MovingFrame(nn.Module):
    """Predicts projective frames and applies a canonical warp.

    The module predicts four control points in normalized coordinates, computes
    the induced homography that maps them to a canonical square, and warps the
    input accordingly. This implements the moving frame normalization described
    in the paper to achieve equivariance to PGL(3).
    """

    def __init__(self, image_size: tuple[int, int], patch_size: tuple[int, int] | None = None):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size or image_size
        self.predictor = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(4, 16),
            nn.GELU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(4, 32),
            nn.GELU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(64, 8),
        )

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        if x.shape[1] != 1:
            raise ValueError("MovingFrame expects grayscale inputs for stability")
        b, _, h, w = x.shape
        offsets = torch.tanh(self.predictor(x).view(b, 4, 2))
        offsets = torch.where(torch.isfinite(offsets), offsets, torch.zeros_like(offsets))
        canonical = default_canonical_corners(b, x.device, x.dtype)
        coords = canonical + 0.5 * offsets
        homography = build_homography(coords, canonical)
        warped = warp_image(x, homography, self.patch_size)
        return warped, homography
