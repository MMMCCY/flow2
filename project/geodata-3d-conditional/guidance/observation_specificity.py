"""Small-matrix diagnostics for Stage-7 observation specificity.

The helpers in this module deliberately operate on already computed vectors.  They
do not know geological truth and cannot select or modify an inference result.
"""

from __future__ import annotations

from itertools import combinations
import math
from typing import Mapping, Sequence

import torch


OBSERVATION_SPECIFICITY_VERSION = "stage7_observation_specificity_v1"


def finite_cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    """Return a float64 flattened cosine, or NaN for a zero vector."""
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    if float(denominator) <= eps:
        return float("nan")
    return float(torch.dot(a, b) / denominator)


def voxelwise_correlation(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    """Pearson correlation over all channel/voxel entries."""
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    a = a - a.mean()
    b = b - b.mean()
    return finite_cosine(a, b, eps=eps)


def pairwise_geometry(
    vectors: Mapping[str, torch.Tensor],
    *,
    include_correlation: bool = False,
) -> list[dict[str, float | str]]:
    """Return deterministic pairwise cosine, norm ratio and optional correlation."""
    rows: list[dict[str, float | str]] = []
    for left_name, right_name in combinations(vectors, 2):
        left = vectors[left_name]
        right = vectors[right_name]
        left_norm = float(torch.linalg.vector_norm(left.detach().double()))
        right_norm = float(torch.linalg.vector_norm(right.detach().double()))
        row: dict[str, float | str] = {
            "left": left_name,
            "right": right_name,
            "cosine": finite_cosine(left, right),
            "left_norm": left_norm,
            "right_norm": right_norm,
            "norm_ratio_left_over_right": left_norm / right_norm if right_norm > 0 else float("nan"),
        }
        if include_correlation:
            row["voxelwise_correlation"] = voxelwise_correlation(left, right)
        rows.append(row)
    return rows


def masked_pairwise_geometry(
    vectors: Mapping[str, torch.Tensor], mask: torch.Tensor
) -> list[dict[str, float | str]]:
    """Pairwise geometry after broadcasting one spatial mask over channels."""
    selected: dict[str, torch.Tensor] = {}
    for name, value in vectors.items():
        expanded = mask.to(device=value.device, dtype=torch.bool).expand_as(value)
        selected[name] = value[expanded]
    return pairwise_geometry(selected, include_correlation=True)


def sensitivity_spectrum(
    columns: Sequence[torch.Tensor],
    names: Sequence[str],
    *,
    truth_column_indices: Sequence[int],
    relative_rank_tolerance: float = 1e-6,
) -> dict[str, object]:
    """Summarize a tall local sensitivity matrix through its small Gram matrix."""
    if not columns or len(columns) != len(names):
        raise ValueError("columns and names must be non-empty and have equal length")
    flattened = [value.detach().double().reshape(-1).cpu() for value in columns]
    gram = torch.empty((len(flattened), len(flattened)), dtype=torch.float64)
    for i, left in enumerate(flattened):
        for j in range(i, len(flattened)):
            value = torch.dot(left, flattened[j])
            gram[i, j] = value
            gram[j, i] = value
    eigenvalues = torch.linalg.eigvalsh(gram).clamp_min(0).flip(0)
    singular = eigenvalues.sqrt()
    maximum = float(singular[0])
    threshold = maximum * float(relative_rank_tolerance)
    effective = singular[singular > threshold]
    rank = int(effective.numel())
    condition = (
        float(effective[0] / effective[-1]) if rank > 1 and float(effective[-1]) > 0 else float("nan")
    )
    normalized = gram.diag().clamp_min(0).sqrt()
    denominator = normalized[:, None] * normalized[None, :]
    cosine = torch.where(denominator > 0, gram / denominator.clamp_min(1e-30), torch.nan)
    pair_values = cosine[torch.triu(torch.ones_like(cosine, dtype=torch.bool), diagonal=1)]
    finite_pairs = pair_values[torch.isfinite(pair_values)]
    truth_indices = [int(value) for value in truth_column_indices]
    truth_direction = sum(flattened[index] for index in truth_indices)
    column_norms = normalized.tolist()
    median_norm = float(torch.median(normalized[normalized > 0])) if bool((normalized > 0).any()) else 0.0
    return {
        "basis_names": list(names),
        "output_dimension": int(flattened[0].numel()),
        "column_norms": {name: float(value) for name, value in zip(names, column_norms)},
        "singular_values": [float(value) for value in singular],
        "effective_rank": rank,
        "column_count": len(flattened),
        "relative_rank_tolerance": float(relative_rank_tolerance),
        "condition_number_effective": condition,
        "pairwise_column_cosine_min": float(finite_pairs.min()) if finite_pairs.numel() else float("nan"),
        "pairwise_column_cosine_median": float(finite_pairs.median()) if finite_pairs.numel() else float("nan"),
        "pairwise_column_cosine_max": float(finite_pairs.max()) if finite_pairs.numel() else float("nan"),
        "truth_direction_sensitivity_norm": float(torch.linalg.vector_norm(truth_direction)),
        "truth_direction_sensitivity_over_median_column_norm": (
            float(torch.linalg.vector_norm(truth_direction)) / median_norm if median_norm > 0 else float("nan")
        ),
        "truth_basis_pair_cosine": (
            float(cosine[truth_indices[0], truth_indices[1]]) if len(truth_indices) == 2 else float("nan")
        ),
    }


def hidden_target_metrics(
    labels: torch.Tensor,
    *,
    target_label: int,
    truth_hidden_mask: torch.Tensor,
    evaluation_domain: torch.Tensor,
) -> dict[str, float | int]:
    """Retrospective hidden-body metrics restricted to a declared search domain."""
    predicted = (labels == int(target_label)) & evaluation_domain
    truth = truth_hidden_mask & evaluation_domain
    intersection = int((predicted & truth).sum())
    predicted_count = int(predicted.sum())
    truth_count = int(truth.sum())
    union = int((predicted | truth).sum())
    return {
        "hidden_target_iou": intersection / union if union else 1.0,
        "hidden_target_recall": intersection / truth_count if truth_count else 1.0,
        "hidden_target_precision": intersection / predicted_count if predicted_count else 0.0,
        "hidden_target_true_positive_voxels": intersection,
        "hidden_target_predicted_voxels": predicted_count,
        "hidden_target_truth_voxels": truth_count,
    }


def rank_mechanisms(scores: Mapping[str, float]) -> list[dict[str, float | int | str]]:
    """Rank precomputed bounded support scores without silently replacing NaNs."""
    clean = {
        name: (float(value) if math.isfinite(float(value)) else 0.0)
        for name, value in scores.items()
    }
    ordered = sorted(clean.items(), key=lambda item: (-item[1], item[0]))
    return [
        {"rank": rank, "mechanism": name, "support_score": value}
        for rank, (name, value) in enumerate(ordered, start=1)
    ]
