"""Metrics for generator-free Stage6Q soft/hard transfer localization."""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import torch

from .causality_gradient_audit import response_truth_direction_fraction
from .observation_closure_audit import epsilon_safe_attainment


SOFT_HARD_TRANSFER_VERSION = "phase6q_soft_hard_transfer_v1"


def response_distance_geometry(
    *,
    baseline: torch.Tensor,
    guided: torch.Tensor,
    truth: torch.Tensor,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Classify whether a response is closer to its paired baseline or truth."""
    b = baseline.detach().double().reshape(-1)
    g = guided.detach().double().reshape(-1)
    t = truth.detach().double().reshape(-1)
    distance_baseline = torch.linalg.vector_norm(g - b)
    distance_truth = torch.linalg.vector_norm(g - t)
    baseline_truth = torch.linalg.vector_norm(b - t)
    scale = max(float(baseline_truth), eps)
    tolerance = eps * scale
    if float(distance_baseline) + tolerance < float(distance_truth):
        closer = "baseline"
    elif float(distance_truth) + tolerance < float(distance_baseline):
        closer = "truth"
    else:
        closer = "equidistant"
    fraction, orthogonal = response_truth_direction_fraction(baseline, guided, truth)
    return {
        "distance_to_baseline": float(distance_baseline),
        "distance_to_truth": float(distance_truth),
        "distance_baseline_to_truth": float(baseline_truth),
        "closer_to": closer,
        "truth_direction_fraction": fraction,
        "orthogonal_response_fraction": orthogonal,
    }


def paired_attainment(
    *,
    soft_baseline_loss: float,
    guided_soft_loss: float,
    soft_truth_loss: float,
    hard_baseline_loss: float,
    guided_hard_loss: float,
    hard_truth_loss: float,
) -> dict[str, float]:
    """Compute soft/hard attainment with matching reference denominators."""
    return {
        "soft_attainment": epsilon_safe_attainment(
            soft_baseline_loss, guided_soft_loss, soft_truth_loss
        ),
        "hard_attainment": epsilon_safe_attainment(
            hard_baseline_loss, guided_hard_loss, hard_truth_loss
        ),
    }


def spatial_energy_fractions(
    value: torch.Tensor,
    regions: Mapping[str, torch.Tensor],
    *,
    eps: float = 1e-30,
) -> dict[str, float]:
    """Allocate squared gradient/update energy to diagnostic voxel regions."""
    energy = value.detach().double().square().sum(dim=1, keepdim=True)
    total = float(energy.sum())
    output: dict[str, float] = {}
    for name, mask in regions.items():
        selected = mask.to(device=energy.device, dtype=torch.bool).expand_as(energy)
        output[f"{name}_energy_fraction"] = (
            float(energy[selected].sum()) / max(total, eps)
        )
    return output


def decision_boundary_statistics(
    *,
    probabilities_before: torch.Tensor,
    probabilities_after: torch.Tensor,
    similarities_before: torch.Tensor,
    categories_before: torch.Tensor,
    categories_after: torch.Tensor,
    target_category: int,
    search_mask: torch.Tensor,
    margin_edges: Sequence[float],
) -> dict[str, object]:
    """Aggregate probability motion and hard crossings by similarity margin."""
    if len(margin_edges) != 2 or not 0 <= margin_edges[0] < margin_edges[1]:
        raise ValueError("margin_edges must contain two increasing non-negative values")
    search = search_mask.to(device=probabilities_before.device, dtype=torch.bool)[:, 0]
    top2 = similarities_before.topk(2, dim=1)
    margin = top2.values[:, 0] - top2.values[:, 1]
    entropy = -(
        probabilities_before.clamp_min(1e-12)
        * probabilities_before.clamp_min(1e-12).log()
    ).sum(dim=1)
    target_change = probabilities_after[:, target_category] - probabilities_before[:, target_category]
    changed = (categories_before != categories_after) & search
    bins = {
        "near": search & (margin < float(margin_edges[0])),
        "medium": search
        & (margin >= float(margin_edges[0]))
        & (margin < float(margin_edges[1])),
        "high": search & (margin >= float(margin_edges[1])),
    }
    output: dict[str, object] = {
        "hard_crossing_count": int(changed.sum()),
        "hard_crossing_rate": float(changed.sum() / search.sum().clamp_min(1)),
        "crossing_into_target": int((changed & (categories_after == target_category)).sum()),
        "crossing_out_of_target": int((changed & (categories_before == target_category)).sum()),
        "crossing_into_wrong_class": int(
            (changed & (categories_after != target_category)).sum()
        ),
        "similarity_margin_mean": float(margin[search].mean()),
        "entropy_mean": float(entropy[search].mean()),
        "target_probability_change_mean": float(target_change[search].mean()),
    }
    for name, mask in bins.items():
        count = int(mask.sum())
        output[f"margin_{name}_voxel_count"] = count
        output[f"margin_{name}_target_probability_change_mean"] = (
            float(target_change[mask].mean()) if count else float("nan")
        )
        output[f"margin_{name}_crossing_rate"] = (
            float(changed[mask].float().mean()) if count else float("nan")
        )
    return output


def projection_erasure_fraction(
    *,
    loss_before: float,
    loss_pre_projection: float,
    loss_post_projection: float,
    eps: float = 1e-15,
) -> float:
    improvement = float(loss_before) - float(loss_pre_projection)
    if improvement <= eps:
        return float("nan")
    lost = float(loss_post_projection) - float(loss_pre_projection)
    return lost / improvement
