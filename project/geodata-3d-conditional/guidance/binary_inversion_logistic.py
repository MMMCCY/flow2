"""Small binary inversion-feature mapper used by Stage15-G."""

from __future__ import annotations

import torch


FEATURE_NAMES = ("raw_q", "within_case_percentile", "vertical_contrast", "depth")


def coarse_support_count_8(subsurface: torch.Tensor) -> torch.Tensor:
    if tuple(subsurface.shape) != (1, 1, 64, 64, 64):
        raise ValueError("subsurface must have shape [1,1,64,64,64]")
    blocks = subsurface.bool().reshape(1, 1, 8, 8, 8, 8, 8, 8)
    return blocks.sum(dim=(3, 5, 7)).contiguous()


def within_case_percentile(q: torch.Tensor, domain: torch.Tensor) -> torch.Tensor:
    """Tie-aware empirical percentile among supported cells of one case."""
    if q.shape != domain.shape or tuple(q.shape) != (1, 1, 8, 8, 8):
        raise ValueError("q and domain must match [1,1,8,8,8]")
    selected = q[domain.bool()]
    if not selected.numel():
        raise ValueError("case has no supported coarse cells")
    percentile = (selected[:, None] >= selected[None, :]).float().mean(dim=1)
    result = torch.zeros_like(q)
    result[domain.bool()] = percentile
    return result


def binary_inversion_features(q: torch.Tensor, support_count: torch.Tensor) -> torch.Tensor:
    """Return fixed [8,8,8,4] case-relative binary inversion features."""
    if tuple(q.shape) != (1, 1, 8, 8, 8) or support_count.shape != q.shape:
        raise ValueError("q and support_count must match [1,1,8,8,8]")
    domain = support_count > 0
    percentile = within_case_percentile(q, domain)
    contrast = torch.zeros_like(q)
    difference = (q[..., 1:] - q[..., :-1]).abs()
    valid_edge = domain[..., 1:] & domain[..., :-1]
    difference = torch.where(valid_edge, difference, torch.zeros_like(difference))
    contrast[..., 1:] = torch.maximum(contrast[..., 1:], difference)
    contrast[..., :-1] = torch.maximum(contrast[..., :-1], difference)
    depth_axis = torch.linspace(1.0, 0.0, 8, dtype=q.dtype, device=q.device)
    depth = depth_axis.view(1, 1, 1, 1, 8).expand_as(q)
    features = torch.stack((q, percentile, contrast, depth), dim=-1)
    return torch.where(domain.unsqueeze(-1), features, torch.zeros_like(features))[0, 0]


def weighted_mean_std(features: torch.Tensor, weights: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError("features must be [N,4]")
    if weights.ndim != 1 or weights.shape[0] != features.shape[0]:
        raise ValueError("weights must be [N]")
    total = weights.sum()
    if float(total) <= 0:
        raise ValueError("weights must have positive mass")
    mean = (features * weights[:, None]).sum(dim=0) / total
    variance = ((features - mean).square() * weights[:, None]).sum(dim=0) / total
    return mean, variance.sqrt().clamp_min(1e-6)


def apply_linear_probability(
    features: torch.Tensor,
    feature_mean: torch.Tensor,
    feature_std: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    normalized = (features - feature_mean) / feature_std
    return torch.sigmoid(normalized @ weight + bias)
