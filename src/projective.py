"""Utilities for projective geometry operations used in equivariant layers."""
from __future__ import annotations

import torch
from torch import Tensor


def normalize_points(points: Tensor) -> Tensor:
    """Normalize homogeneous 2D points so the last coordinate is 1.

    Args:
        points: ``(..., 3)`` tensor of homogeneous points.

    Returns:
        Tensor with the same shape where the last coordinate is 1.
    """
    if points.size(-1) != 3:
        raise ValueError("points must be homogeneous with shape (..., 3)")
    scale = points[..., 2:].clamp(min=1e-8)
    return points / scale


def build_homography(src: Tensor, dst: Tensor) -> Tensor:
    """Compute homography that maps ``src`` to ``dst``.

    The implementation uses the direct linear transform formulation with four
    corner correspondences and solves the system in closed form to avoid
    numerical instability during training.

    Args:
        src: ``(B, 4, 2)`` source points.
        dst: ``(B, 4, 2)`` target points.

    Returns:
        ``(B, 3, 3)`` homography matrices.
    """
    if src.shape != dst.shape or src.shape[-2:] != (4, 2):
        raise ValueError("src and dst must both be (B, 4, 2)")

    batch = src.shape[0]
    device = src.device

    ones = torch.ones((batch, 4, 1), device=device, dtype=src.dtype)
    zeros = torch.zeros((batch, 4, 3), device=device, dtype=src.dtype)
    src_h = torch.cat([src, ones], dim=-1)
    dst_x = dst[..., 0:1]
    dst_y = dst[..., 1:2]

    # Build A matrix for Ah = 0
    row1 = torch.cat([src_h, zeros, -dst_x * src_h], dim=-1)
    row2 = torch.cat([zeros, src_h, -dst_y * src_h], dim=-1)
    a_mat = torch.cat([row1, row2], dim=1)  # (B, 8, 9)

    # Solve using SVD; last column of V gives homography parameters.
    _, _, vh = torch.linalg.svd(a_mat)
    h = vh[..., -1]  # (B, 9)
    h = h / h[..., -1:].clamp(min=1e-8)
    return h.view(batch, 3, 3)


def warp_image(image: Tensor, homography: Tensor, output_size: tuple[int, int]) -> Tensor:
    """Apply projective warp to image using a differentiable grid sampler."""
    if image.dim() != 4:
        raise ValueError("image must be BCHW")
    b, c, h, w = image.shape
    oh, ow = output_size

    device = image.device
    dtype = image.dtype
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, steps=oh, device=device, dtype=dtype),
        torch.linspace(-1, 1, steps=ow, device=device, dtype=dtype),
        indexing="ij",
    )
    ones = torch.ones_like(grid_x)
    base_grid = torch.stack([grid_x, grid_y, ones], dim=-1)  # (oh, ow, 3)
    base_grid = base_grid.view(1, oh * ow, 3).repeat(b, 1, 1)

    h_inv = torch.linalg.inv(homography)
    warped = base_grid @ h_inv.transpose(1, 2)
    warped = normalize_points(warped)
    x = warped[..., 0].view(b, oh, ow)
    y = warped[..., 1].view(b, oh, ow)

    sample_grid = torch.stack([x, y], dim=-1)
    return torch.nn.functional.grid_sample(
        image,
        sample_grid,
        mode="bilinear",
        align_corners=True,
        padding_mode="border",
    )
