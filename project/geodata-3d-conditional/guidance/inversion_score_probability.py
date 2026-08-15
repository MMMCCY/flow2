"""Minimal 8^3 inversion-score calibration primitives for Stage15-F."""

from __future__ import annotations

import torch


FINE_SHAPE = (64, 64, 64)
COARSE_SHAPE = (8, 8, 8)
REPEAT_FACTOR = (8, 8, 8)


def upsample_inversion_score(q_coarse: torch.Tensor) -> torch.Tensor:
    """Nearest-neighbour expansion from a fixed 8^3 score grid to 64^3."""
    if q_coarse.ndim != 5 or tuple(q_coarse.shape[1:]) != (1, *COARSE_SHAPE):
        raise ValueError("q_coarse must have shape [B,1,8,8,8]")
    if not q_coarse.is_floating_point() or not torch.isfinite(q_coarse).all():
        raise ValueError("q_coarse must be finite floating point")
    if bool(((q_coarse < 0) | (q_coarse > 1)).any()):
        raise ValueError("q_coarse must lie in [0,1]")
    result = q_coarse
    for axis, repeat in zip((-3, -2, -1), REPEAT_FACTOR):
        result = result.repeat_interleave(repeat, dim=axis)
    return result.contiguous()


def coarse_truth_occupancy_8(
    binary_truth: torch.Tensor, subsurface_mask: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return target occupancy, presence, and subsurface support per 8^3 cell."""
    expected = (1, 1, *FINE_SHAPE)
    if binary_truth.shape != subsurface_mask.shape or tuple(binary_truth.shape) != expected:
        raise ValueError("truth and subsurface must both have shape [1,1,64,64,64]")
    target = binary_truth.bool() & subsurface_mask.bool()
    support = subsurface_mask.bool()
    target_blocks = target.reshape(1, 1, 8, 8, 8, 8, 8, 8)
    support_blocks = support.reshape(1, 1, 8, 8, 8, 8, 8, 8)
    target_count = target_blocks.sum(dim=(3, 5, 7))
    support_count = support_blocks.sum(dim=(3, 5, 7))
    occupancy = target_count.float() / support_count.clamp_min(1).float()
    occupancy = torch.where(support_count > 0, occupancy, torch.zeros_like(occupancy))
    return occupancy.contiguous(), (target_count > 0).contiguous(), support_count.contiguous()
