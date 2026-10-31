"""Oracle three-dimensional target-probability construction and loss.

The target built here is intentionally derived from a truth model.  It is an
upper-bound mechanism probe, not a representation of a measured geophysical
observation.
"""

from __future__ import annotations

import hashlib
import math
from typing import Dict, Sequence

import torch
import torch.nn.functional as F

from geology_io_utils import connected_components_3d
from guided_geophysical_sampling import soft_decode_to_probs
from inference_runtime import normalize_single_geology


COMPONENT_MODES = ("all", "largest", "selected")
SPATIAL_GRADIENT_LOSS = "roi_normalized_axis_l1_target_gradient_v1"
SOFT_DIAGNOSTIC_VERSION = "final_target_probability_and_cosine_margin_v1"
SOFT_BOUNDARY_SIMILARITY_THRESHOLDS = (0.01, 0.05)
LEGACY_PROBABILITY_LOSS_MODE = "balanced_soft_bce_soft_dice_v1"
CALIBRATED_PROBABILITY_LOSS_MODE = "calibrated_soft_bce_hard_dice_v2"
COARSE_OCCUPANCY_LOSS_MODE = "binary_coarse_occupancy_bce_v1"
PROBABILITY_LOSS_MODES = (
    LEGACY_PROBABILITY_LOSS_MODE,
    CALIBRATED_PROBABILITY_LOSS_MODE,
    COARSE_OCCUPANCY_LOSS_MODE,
)


def tensor_sha256(value: torch.Tensor) -> str:
    """Return a stable digest of tensor shape, dtype, and contiguous bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _component_summary(component: Dict[str, object], rank: int) -> Dict[str, object]:
    return {
        "rank_by_volume": int(rank),
        "source_component_id": int(component["component_id"]),
        "voxel_count": int(component["voxel_count"]),
        "bbox_min": list(component["bbox_min"]),
        "bbox_max": list(component["bbox_max"]),
        "centroid": list(component["centroid"]),
    }


def build_target_mask(
    truth_model: torch.Tensor,
    target_label: int,
    component_mode: str = "all",
    component_rank: int | None = None,
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Build a deterministic target mask with shape ``[1,1,X,Y,Z]``.

    Connected components use the repository's six-neighbour definition and
    are ranked by decreasing volume, then by their scan-order component ID.
    ``component_rank`` therefore remains stable for immutable truth assets.
    """
    if component_mode not in COMPONENT_MODES:
        raise ValueError(f"component_mode must be one of {COMPONENT_MODES}")
    if component_rank is not None and component_rank < 0:
        raise ValueError("component_rank must be non-negative")

    truth = normalize_single_geology(truth_model, "truth_model").long()
    full_mask = truth == int(target_label)
    target_voxels = int(full_mask.sum().item())
    if target_voxels == 0:
        raise ValueError(f"target label {target_label} is absent from the truth model")

    components = connected_components_3d(full_mask[0, 0])
    ranked = sorted(
        components,
        key=lambda item: (-int(item["voxel_count"]), int(item["component_id"])),
    )
    summaries = [_component_summary(component, rank) for rank, component in enumerate(ranked)]

    selected_summary: Dict[str, object] | None = None
    if component_mode == "all":
        if component_rank is not None:
            raise ValueError("component_rank is only valid with component_mode='selected'")
        selected = full_mask.clone()
    else:
        rank = 0 if component_mode == "largest" else component_rank
        if rank is None:
            raise ValueError("component_mode='selected' requires component_rank")
        if rank >= len(ranked):
            raise ValueError(
                f"component_rank {rank} is out of range for {len(ranked)} components"
            )
        selected = torch.zeros_like(full_mask)
        coords = ranked[rank]["coords"]
        selected[0, 0, coords[:, 0], coords[:, 1], coords[:, 2]] = True
        selected_summary = summaries[rank]

    selected_voxels = int(selected.sum().item())
    metadata: Dict[str, object] = {
        "target_label": int(target_label),
        "component_mode": component_mode,
        "component_rank": component_rank,
        "connectivity": 6,
        "truth_target_voxels": target_voxels,
        "selected_target_voxels": selected_voxels,
        "selected_fraction_of_label": selected_voxels / target_voxels,
        "component_count": len(ranked),
        "selected_component": selected_summary,
        "ranked_components": summaries,
    }
    metadata["target_mask_sha256"] = tensor_sha256(selected)
    return selected, metadata


def _gaussian_kernel_1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    if sigma <= 0:
        raise ValueError("sigma must be positive")
    radius = max(1, int(math.ceil(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / float(sigma)).square())
    return kernel / kernel.sum()


def gaussian_blur_3d(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply separable replicate-padded Gaussian blur to a ``[B,1,X,Y,Z]`` tensor."""
    if volume.ndim != 5 or volume.shape[1] != 1:
        raise ValueError("volume must have shape [B,1,X,Y,Z]")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return volume

    kernel = _gaussian_kernel_1d(float(sigma), volume.device, volume.dtype)
    radius = kernel.numel() // 2
    result = volume
    kernels = (
        kernel.view(1, 1, -1, 1, 1),
        kernel.view(1, 1, 1, -1, 1),
        kernel.view(1, 1, 1, 1, -1),
    )
    paddings = (
        (0, 0, 0, 0, radius, radius),
        (0, 0, radius, radius, 0, 0),
        (radius, radius, 0, 0, 0, 0),
    )
    for conv_kernel, padding in zip(kernels, paddings):
        result = F.conv3d(F.pad(result, padding, mode="replicate"), conv_kernel)
    return result


def build_probability_volume(
    target_mask: torch.Tensor,
    sigmas: Sequence[float] = (0.0, 1.5),
    scale_weights: Sequence[float] | None = None,
) -> tuple[torch.Tensor, Dict[str, object]]:
    """Return a normalized multiscale probability target and metadata."""
    if target_mask.ndim != 5 or target_mask.shape[:2] != (1, 1):
        raise ValueError("target_mask must have shape [1,1,X,Y,Z]")
    if not sigmas:
        raise ValueError("at least one target sigma is required")
    sigma_values = [float(value) for value in sigmas]
    if any(value < 0 for value in sigma_values):
        raise ValueError("target sigmas must be non-negative")

    weights = (
        [1.0] * len(sigma_values)
        if scale_weights is None
        else [float(value) for value in scale_weights]
    )
    if len(weights) != len(sigma_values):
        raise ValueError("scale_weights must match sigmas")
    if any(value < 0 for value in weights) or sum(weights) <= 0:
        raise ValueError("scale_weights must be non-negative with positive sum")
    total_weight = sum(weights)
    weights = [value / total_weight for value in weights]

    binary = target_mask.float()
    probability = torch.zeros_like(binary)
    for sigma, weight in zip(sigma_values, weights):
        probability = probability + weight * gaussian_blur_3d(binary, sigma)
    probability = probability.clamp(0.0, 1.0)
    metadata = {
        "target_sigmas": sigma_values,
        "target_scale_weights": weights,
        "target_probability_min": float(probability.min().item()),
        "target_probability_max": float(probability.max().item()),
        "target_probability_mean": float(probability.mean().item()),
        "target_probability_sha256": tensor_sha256(probability),
    }
    return probability, metadata


def dilate_mask(target_mask: torch.Tensor, radius: int) -> torch.Tensor:
    """Dilate a target mask by a cubic Chebyshev radius."""
    if target_mask.ndim != 5 or target_mask.shape[1] != 1:
        raise ValueError("target_mask must have shape [B,1,X,Y,Z]")
    if radius < 0:
        raise ValueError("radius must be non-negative")
    if radius == 0:
        return target_mask.bool()
    width = 2 * int(radius) + 1
    return F.max_pool3d(
        target_mask.float(),
        kernel_size=width,
        stride=1,
        padding=radius,
    ).bool()


def spatial_gradient_matching_loss(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
    roi_mask: torch.Tensor,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Match target-probability gradients inside the ROI.

    Isolated probability islands introduce gradients where the oracle target
    is locally flat.  Matching axis-aligned first differences therefore
    penalizes this high-frequency fragmentation while retaining gradients at
    the target's own boundaries.  Normalization by target total variation
    keeps the loss scale usable across target volumes of different sizes.
    """
    if predicted_probability.ndim != 5 or predicted_probability.shape[1] != 1:
        raise ValueError("predicted_probability must have shape [B,1,X,Y,Z]")
    target = target_probability.to(
        device=predicted_probability.device,
        dtype=predicted_probability.dtype,
    )
    roi = roi_mask.to(device=predicted_probability.device).bool()
    if target.shape[0] == 1 and predicted_probability.shape[0] > 1:
        target = target.expand(predicted_probability.shape[0], -1, -1, -1, -1)
        roi = roi.expand(predicted_probability.shape[0], -1, -1, -1, -1)
    if target.shape != predicted_probability.shape or roi.shape != target.shape:
        raise ValueError("target_probability and roi_mask must match prediction shape")

    error_sum = predicted_probability.new_zeros(())
    predicted_tv_sum = predicted_probability.new_zeros(())
    target_tv_sum = predicted_probability.new_zeros(())
    edge_count = predicted_probability.new_zeros(())
    for dimension in (2, 3, 4):
        predicted_gradient = torch.diff(predicted_probability, dim=dimension)
        target_gradient = torch.diff(target, dim=dimension)
        left = roi.narrow(dimension, 0, roi.shape[dimension] - 1)
        right = roi.narrow(dimension, 1, roi.shape[dimension] - 1)
        edge_mask = (left & right).to(dtype=predicted_probability.dtype)
        error_sum = error_sum + (
            (predicted_gradient - target_gradient).abs() * edge_mask
        ).sum()
        predicted_tv_sum = predicted_tv_sum + (
            predicted_gradient.abs() * edge_mask
        ).sum()
        target_tv_sum = target_tv_sum + (target_gradient.abs() * edge_mask).sum()
        edge_count = edge_count + edge_mask.sum()

    edge_count = edge_count.clamp_min(1.0)
    gradient_error_mean = error_sum / edge_count
    predicted_gradient_mean = predicted_tv_sum / edge_count
    target_gradient_mean = target_tv_sum / edge_count
    normalized_loss = gradient_error_mean / target_gradient_mean.clamp_min(eps)
    diagnostics = {
        "roi_spatial_gradient_error_mean": gradient_error_mean,
        "roi_predicted_gradient_mean": predicted_gradient_mean,
        "roi_target_gradient_mean": target_gradient_mean,
    }
    return normalized_loss, diagnostics


def probability_target_loss_terms(
    predicted_probability: torch.Tensor,
    target_probability: torch.Tensor,
    target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    loss_mode: str,
    eps: float = 1e-6,
) -> Dict[str, torch.Tensor]:
    """Return auditable BCE/Dice terms for one target-label probability.

    The legacy mode class-balances a continuous soft target.  That operation
    is intentionally retained for exact reproduction, but its optimum is not
    calibrated to the supplied target probability.  The calibrated mode uses
    an unweighted proper BCE for the continuous multiscale target and applies
    Dice only to the exact binary target core.  It therefore does not promote
    a low-probability Gaussian halo merely because the target class is rare.
    """
    if loss_mode not in PROBABILITY_LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {PROBABILITY_LOSS_MODES}")
    if predicted_probability.ndim != 5 or predicted_probability.shape[1] != 1:
        raise ValueError("predicted_probability must have shape [B,1,X,Y,Z]")
    target = target_probability.to(
        device=predicted_probability.device,
        dtype=predicted_probability.dtype,
    )
    core = target_mask.to(device=predicted_probability.device).bool()
    roi = roi_mask.to(device=predicted_probability.device).bool()
    if target.shape[0] == 1 and predicted_probability.shape[0] > 1:
        target = target.expand(predicted_probability.shape[0], -1, -1, -1, -1)
        core = core.expand(predicted_probability.shape[0], -1, -1, -1, -1)
        roi = roi.expand(predicted_probability.shape[0], -1, -1, -1, -1)
    if not (
        target.shape == predicted_probability.shape == core.shape == roi.shape
    ):
        raise ValueError(
            "predicted_probability, target_probability, target_mask, and "
            "roi_mask must have matching shapes"
        )
    if bool(((target < 0) | (target > 1)).any()):
        raise ValueError("target_probability must lie in [0,1]")

    roi_float = roi.to(dtype=predicted_probability.dtype)
    roi_count = roi_float.sum().clamp_min(1.0)
    clipped = predicted_probability.clamp(eps, 1.0 - eps)
    if loss_mode == LEGACY_PROBABILITY_LOSS_MODE:
        positive_mass = (target * roi_float).sum().clamp_min(eps)
        negative_mass = ((1.0 - target) * roi_float).sum().clamp_min(eps)
        positive_scale = roi_count / (2.0 * positive_mass)
        negative_scale = roi_count / (2.0 * negative_mass)
        bce_voxels = -(
            positive_scale * target * torch.log(clipped)
            + negative_scale * (1.0 - target) * torch.log1p(-clipped)
        )
        dice_target = target
    else:
        positive_scale = predicted_probability.new_ones(())
        negative_scale = predicted_probability.new_ones(())
        bce_voxels = -(
            target * torch.log(clipped)
            + (1.0 - target) * torch.log1p(-clipped)
        )
        dice_target = core.to(dtype=predicted_probability.dtype)

    bce = (bce_voxels * roi_float).sum() / roi_count
    predicted_roi = predicted_probability * roi_float
    dice_target_roi = dice_target * roi_float
    intersection = (predicted_roi * dice_target_roi).sum()
    dice_score = (2.0 * intersection + eps) / (
        predicted_roi.square().sum() + dice_target_roi.square().sum() + eps
    )

    core_float = (core & roi).to(dtype=predicted_probability.dtype)
    background_float = ((~core) & roi).to(dtype=predicted_probability.dtype)
    halo_float = ((target > 0) & (~core) & roi).to(
        dtype=predicted_probability.dtype
    )
    core_count = core_float.sum().clamp_min(1.0)
    background_count = background_float.sum().clamp_min(1.0)
    halo_count = halo_float.sum().clamp_min(1.0)
    return {
        "probability_bce": bce,
        "probability_dice_loss": 1.0 - dice_score,
        "probability_dice_score": dice_score,
        "probability_soft_calibration_mae": (
            (predicted_probability - target).abs() * roi_float
        ).sum()
        / roi_count,
        "loss_positive_scale": positive_scale,
        "loss_negative_scale": negative_scale,
        "roi_hard_core_probability_mean": (
            predicted_probability * core_float
        ).sum()
        / core_count,
        "roi_background_probability_mean": (
            predicted_probability * background_float
        ).sum()
        / background_count,
        "roi_soft_halo_probability_mean": (
            predicted_probability * halo_float
        ).sum()
        / halo_count,
        "roi_soft_halo_target_mean": (target * halo_float).sum() / halo_count,
    }


def probability_volume_loss(
    x: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_probability: torch.Tensor,
    roi_mask: torch.Tensor,
    target_label: int,
    tau: float,
    bce_weight: float = 1.0,
    dice_weight: float = 1.0,
    spatial_gradient_weight: float = 0.0,
    loss_mode: str = LEGACY_PROBABILITY_LOSS_MODE,
    target_mask: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Match the target-label soft probability inside a selected 3-D ROI."""
    if bce_weight < 0 or dice_weight < 0 or bce_weight + dice_weight <= 0:
        raise ValueError("loss weights must be non-negative with positive sum")
    if spatial_gradient_weight < 0:
        raise ValueError("spatial_gradient_weight must be non-negative")
    if loss_mode not in PROBABILITY_LOSS_MODES:
        raise ValueError(f"loss_mode must be one of {PROBABILITY_LOSS_MODES}")
    if loss_mode == CALIBRATED_PROBABILITY_LOSS_MODE and target_mask is None:
        raise ValueError("calibrated loss mode requires the exact target_mask")
    if target_label < -1:
        raise ValueError("target_label must be a raw geology label")

    probs = soft_decode_to_probs(x, embedding_weight, tau=tau)
    category_index = int(target_label) + 1
    if category_index >= probs.shape[1]:
        raise ValueError(
            f"target label {target_label} exceeds {probs.shape[1]} categories"
        )
    predicted = probs[:, category_index : category_index + 1]
    target = target_probability.to(device=x.device, dtype=x.dtype)
    roi = roi_mask.to(device=x.device).bool()
    if loss_mode == COARSE_OCCUPANCY_LOSS_MODE:
        if target.ndim != 5 or tuple(target.shape[1:]) != (1, 8, 8, 8):
            raise ValueError("coarse occupancy target must have shape [B,1,8,8,8]")
        if tuple(predicted.shape[2:]) != (64, 64, 64):
            raise ValueError("coarse occupancy loss requires a 64^3 Flow probability")
        if target.shape[0] == 1 and predicted.shape[0] > 1:
            target = target.expand(predicted.shape[0], -1, -1, -1, -1)
            roi = roi.expand(predicted.shape[0], -1, -1, -1, -1)
        if target.shape[0] != predicted.shape[0] or roi.shape != predicted.shape:
            raise ValueError("coarse target batch and fine ROI must match prediction")
        if bool(((target < 0) | (target > 1)).any()):
            raise ValueError("coarse occupancy target must lie in [0,1]")
        roi_float = roi.to(predicted.dtype)
        support = F.avg_pool3d(roi_float, kernel_size=8, stride=8) * (8**3)
        predicted_mass = F.avg_pool3d(
            predicted * roi_float, kernel_size=8, stride=8
        ) * (8**3)
        predicted_coarse = predicted_mass / support.clamp_min(1.0)
        domain = support > 0
        clipped = predicted_coarse.clamp(eps, 1.0 - eps)
        bce_cells = -(target * torch.log(clipped) + (1.0 - target) * torch.log1p(-clipped))
        bce = (bce_cells * support).sum() / support.sum().clamp_min(1.0)
        base_loss = float(bce_weight) * bce
        calibration_mae = ((predicted_coarse - target).abs() * support).sum() / support.sum().clamp_min(1.0)
        positive = (target > target.mean(dim=(2, 3, 4), keepdim=True)) & domain
        negative = (~positive) & domain
        positive_count = positive.sum().clamp_min(1)
        negative_count = negative.sum().clamp_min(1)
        zero = bce.new_zeros(())
        one = bce.new_ones(())
        support_sum = support.sum().clamp_min(1.0)
        coarse_entropy = -(
            clipped * torch.log(clipped) + (1.0 - clipped) * torch.log1p(-clipped)
        )
        diagnostics = {
            "probability_base_loss": base_loss,
            "probability_bce": bce,
            "probability_dice_loss": zero,
            "probability_dice_score": one,
            "probability_spatial_gradient_loss": zero,
            "probability_soft_calibration_mae": calibration_mae,
            "loss_positive_scale": one,
            "loss_negative_scale": one,
            "roi_hard_core_probability_mean": predicted_coarse[positive].sum() / positive_count,
            "roi_background_probability_mean": predicted_coarse[negative].sum() / negative_count,
            "roi_soft_halo_probability_mean": zero,
            "roi_soft_halo_target_mean": zero,
            "roi_spatial_gradient_error_mean": zero,
            "roi_predicted_gradient_mean": zero,
            "roi_target_gradient_mean": zero,
            "roi_target_probability_mean": (predicted_coarse * support).sum() / support_sum,
            "roi_soft_margin_mean": zero,
            "roi_entropy_mean": (coarse_entropy * support).sum() / support_sum,
        }
        return base_loss, diagnostics
    core = (
        target >= 0.5
        if target_mask is None
        else target_mask.to(device=x.device).bool()
    )
    if target.shape[0] == 1 and x.shape[0] > 1:
        target = target.expand(x.shape[0], -1, -1, -1, -1)
        roi = roi.expand(x.shape[0], -1, -1, -1, -1)
        core = core.expand(x.shape[0], -1, -1, -1, -1)
    if not (target.shape == predicted.shape == roi.shape == core.shape):
        raise ValueError(
            "target_probability, target_mask, and roi_mask must match x spatial shape"
        )

    roi_float = roi.to(dtype=x.dtype)
    roi_count = roi_float.sum().clamp_min(1.0)
    probability_terms = probability_target_loss_terms(
        predicted,
        target,
        core,
        roi,
        loss_mode=loss_mode,
        eps=eps,
    )
    bce = probability_terms["probability_bce"]
    dice_loss = probability_terms["probability_dice_loss"]
    dice_score = probability_terms["probability_dice_score"]
    base_loss = float(bce_weight) * bce + float(dice_weight) * dice_loss
    if spatial_gradient_weight > 0:
        spatial_loss, spatial_diagnostics = spatial_gradient_matching_loss(
            predicted,
            target,
            roi,
            eps=eps,
        )
        total = base_loss + float(spatial_gradient_weight) * spatial_loss
    else:
        spatial_loss = predicted.new_zeros(())
        spatial_diagnostics = {
            "roi_spatial_gradient_error_mean": predicted.new_zeros(()),
            "roi_predicted_gradient_mean": predicted.new_zeros(()),
            "roi_target_gradient_mean": predicted.new_zeros(()),
        }
        total = base_loss

    other_probs = torch.cat(
        (probs[:, :category_index], probs[:, category_index + 1 :]),
        dim=1,
    )
    max_other = other_probs.max(dim=1, keepdim=True).values
    margin = predicted - max_other
    entropy = -(probs.clamp_min(eps) * torch.log(probs.clamp_min(eps))).sum(
        dim=1,
        keepdim=True,
    )
    diagnostics = {
        "probability_base_loss": base_loss,
        "probability_bce": bce,
        "probability_dice_loss": dice_loss,
        "probability_dice_score": dice_score,
        "probability_spatial_gradient_loss": spatial_loss,
        "roi_target_probability_mean": (predicted * roi_float).sum() / roi_count,
        "roi_soft_margin_mean": (margin * roi_float).sum() / roi_count,
        "roi_entropy_mean": (entropy * roi_float).sum() / roi_count,
        **{
            key: value
            for key, value in probability_terms.items()
            if key
            not in {
                "probability_bce",
                "probability_dice_loss",
                "probability_dice_score",
            }
        },
        **spatial_diagnostics,
    }
    return total, diagnostics


def compute_target_soft_fields(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_label: int,
    tau: float,
) -> Dict[str, torch.Tensor]:
    """Return final target soft fields and the tau-independent hard margin.

    ``target_similarity_margin`` is target cosine similarity minus the best
    competing cosine similarity. Its sign therefore describes the actual
    nearest-embedding hard decision boundary without depending on ``tau``.
    ``target_probability_margin`` has the same sign but is useful for
    interpreting the temperature-scaled loss.
    """
    if state.ndim != 5:
        raise ValueError("state must have shape [B,E,X,Y,Z]")
    if embedding_weight.ndim != 2 or state.shape[1] != embedding_weight.shape[1]:
        raise ValueError("state and embedding_weight dimensions must match")
    if tau <= 0:
        raise ValueError("tau must be positive")
    category_index = int(target_label) + 1
    if not 0 <= category_index < embedding_weight.shape[0]:
        raise ValueError("target_label is outside the embedding category range")

    embeddings = embedding_weight.to(device=state.device, dtype=state.dtype)
    normalized_state = F.normalize(state, dim=1)
    normalized_embeddings = F.normalize(embeddings, dim=1)
    similarities = torch.einsum(
        "bexyz,ce->bcxyz",
        normalized_state,
        normalized_embeddings,
    )
    probabilities = torch.softmax(similarities / float(tau), dim=1)
    target_similarity = similarities[:, category_index : category_index + 1]
    target_probability = probabilities[:, category_index : category_index + 1]
    other_similarities = torch.cat(
        (
            similarities[:, :category_index],
            similarities[:, category_index + 1 :],
        ),
        dim=1,
    )
    other_probabilities = torch.cat(
        (
            probabilities[:, :category_index],
            probabilities[:, category_index + 1 :],
        ),
        dim=1,
    )
    return {
        "target_probability": target_probability,
        "target_probability_margin": (
            target_probability
            - other_probabilities.max(dim=1, keepdim=True).values
        ),
        "target_similarity_margin": (
            target_similarity
            - other_similarities.max(dim=1, keepdim=True).values
        ),
        "soft_hard_target": (
            similarities.argmax(dim=1, keepdim=True) == category_index
        ),
    }


def _expand_mask_like(mask: torch.Tensor, reference: torch.Tensor, name: str) -> torch.Tensor:
    normalized = normalize_single_geology(mask, name).to(device=reference.device).bool()
    if normalized.shape[0] == 1 and reference.shape[0] > 1:
        normalized = normalized.expand(reference.shape[0], -1, -1, -1, -1)
    if normalized.shape != reference.shape:
        raise ValueError(f"{name} must match the soft-field shape")
    return normalized


def _masked_soft_statistics(
    fields: Dict[str, torch.Tensor],
    mask: torch.Tensor,
) -> Dict[str, object]:
    count = int(mask.sum().item())
    row: Dict[str, object] = {"voxel_count": count}
    field_names = (
        "target_probability",
        "target_probability_margin",
        "target_similarity_margin",
    )
    for name in field_names:
        values = fields[name][mask]
        if count == 0:
            row.update(
                {
                    f"{name}_mean": float("nan"),
                    f"{name}_p10": float("nan"),
                    f"{name}_p50": float("nan"),
                    f"{name}_p90": float("nan"),
                }
            )
            continue
        quantiles = torch.quantile(
            values.float(),
            torch.tensor((0.1, 0.5, 0.9), device=values.device),
        )
        row.update(
            {
                f"{name}_mean": float(values.float().mean().detach().cpu()),
                f"{name}_p10": float(quantiles[0].detach().cpu()),
                f"{name}_p50": float(quantiles[1].detach().cpu()),
                f"{name}_p90": float(quantiles[2].detach().cpu()),
            }
        )

    hard_count = int((fields["soft_hard_target"] & mask).sum().item())
    row["soft_hard_target_count"] = hard_count
    row["soft_hard_target_fraction"] = (
        hard_count / count if count else float("nan")
    )
    similarity_margin = fields["target_similarity_margin"]
    for threshold in SOFT_BOUNDARY_SIMILARITY_THRESHOLDS:
        key = str(threshold).replace(".", "p")
        near_count = int(((similarity_margin.abs() <= threshold) & mask).sum().item())
        row[f"near_boundary_{key}_count"] = near_count
        row[f"near_boundary_{key}_fraction"] = (
            near_count / count if count else float("nan")
        )
    return row


def target_soft_region_stats(
    fields: Dict[str, torch.Tensor],
    truth_model: torch.Tensor,
    selected_target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    target_label: int,
    sample_id: int,
) -> list[Dict[str, object]]:
    """Summarize final soft fields in scientifically distinct spatial regions."""
    required = {
        "target_probability",
        "target_probability_margin",
        "target_similarity_margin",
        "soft_hard_target",
    }
    if set(fields) != required:
        raise ValueError(f"soft fields must contain exactly {sorted(required)}")
    reference = fields["target_probability"]
    if reference.ndim != 5 or reference.shape[1] != 1:
        raise ValueError("soft fields must have shape [B,1,X,Y,Z]")
    if any(value.shape != reference.shape for value in fields.values()):
        raise ValueError("all soft fields must have identical shapes")

    truth = normalize_single_geology(truth_model, "truth_model").to(
        device=reference.device
    ).long()
    if truth.shape[0] == 1 and reference.shape[0] > 1:
        truth = truth.expand(reference.shape[0], -1, -1, -1, -1)
    if truth.shape != reference.shape:
        raise ValueError("truth_model must match the soft-field shape")
    selected = _expand_mask_like(selected_target_mask, reference, "selected_target_mask")
    roi = _expand_mask_like(roi_mask, reference, "roi_mask")
    condition = _expand_mask_like(condition_mask, reference, "condition_mask")
    truth_target = truth == int(target_label)
    regions = {
        "selected_truth_target": selected,
        "selected_truth_target_unconditioned": selected & ~condition,
        "truth_target_unselected": truth_target & ~selected,
        "roi_true_non_target": roi & ~truth_target,
        "outside_roi_true_non_target": ~roi & ~truth_target,
    }
    rows: list[Dict[str, object]] = []
    for region, mask in regions.items():
        row = {
            "sample_id": int(sample_id),
            "region": region,
            "desired_margin_direction": (
                1
                if region in {
                    "selected_truth_target",
                    "selected_truth_target_unconditioned",
                    "truth_target_unselected",
                }
                else -1
            ),
            **_masked_soft_statistics(fields, mask),
        }
        rows.append(row)
    return rows


def paired_target_soft_deltas(
    baseline_fields: Dict[str, torch.Tensor],
    guided_fields: Dict[str, torch.Tensor],
    truth_model: torch.Tensor,
    selected_target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    condition_mask: torch.Tensor,
    baseline_decoded: torch.Tensor,
    guided_decoded: torch.Tensor,
    target_label: int,
    sample_id: int,
) -> list[Dict[str, object]]:
    """Return region-wise paired soft motion and hard-boundary crossings."""
    baseline_rows = target_soft_region_stats(
        baseline_fields,
        truth_model,
        selected_target_mask,
        roi_mask,
        condition_mask,
        target_label,
        sample_id,
    )
    guided_rows = target_soft_region_stats(
        guided_fields,
        truth_model,
        selected_target_mask,
        roi_mask,
        condition_mask,
        target_label,
        sample_id,
    )
    reference = guided_fields["target_probability"]
    truth = normalize_single_geology(truth_model, "truth_model").to(
        device=reference.device
    ).long()
    selected = _expand_mask_like(selected_target_mask, reference, "selected_target_mask")
    roi = _expand_mask_like(roi_mask, reference, "roi_mask")
    condition = _expand_mask_like(condition_mask, reference, "condition_mask")
    truth_target = truth == int(target_label)
    region_masks = {
        "selected_truth_target": selected,
        "selected_truth_target_unconditioned": selected & ~condition,
        "truth_target_unselected": truth_target & ~selected,
        "roi_true_non_target": roi & ~truth_target,
        "outside_roi_true_non_target": ~roi & ~truth_target,
    }
    baseline_hard = _expand_mask_like(
        normalize_single_geology(baseline_decoded, "baseline_decoded")
        == int(target_label),
        reference,
        "baseline_hard_target",
    )
    guided_hard = _expand_mask_like(
        normalize_single_geology(guided_decoded, "guided_decoded")
        == int(target_label),
        reference,
        "guided_hard_target",
    )
    baseline_by_region = {str(row["region"]): row for row in baseline_rows}
    guided_by_region = {str(row["region"]): row for row in guided_rows}
    margin_delta = (
        guided_fields["target_similarity_margin"]
        - baseline_fields["target_similarity_margin"]
    )
    rows: list[Dict[str, object]] = []
    for region, mask in region_masks.items():
        baseline_row = baseline_by_region[region]
        guided_row = guided_by_region[region]
        desired_direction = int(guided_row["desired_margin_direction"])
        count = int(mask.sum().item())
        entered = int(((~baseline_hard) & guided_hard & mask).sum().item())
        exited = int((baseline_hard & (~guided_hard) & mask).sum().item())
        correct_motion = (margin_delta * desired_direction) > 0
        correct_motion_count = int((correct_motion & mask).sum().item())
        rows.append(
            {
                "sample_id": int(sample_id),
                "region": region,
                "desired_margin_direction": desired_direction,
                "voxel_count": count,
                "delta_target_probability_mean": float(
                    guided_row["target_probability_mean"]
                )
                - float(baseline_row["target_probability_mean"]),
                "delta_target_probability_margin_mean": float(
                    guided_row["target_probability_margin_mean"]
                )
                - float(baseline_row["target_probability_margin_mean"]),
                "delta_target_similarity_margin_mean": float(
                    guided_row["target_similarity_margin_mean"]
                )
                - float(baseline_row["target_similarity_margin_mean"]),
                "margin_moved_correct_direction_count": correct_motion_count,
                "margin_moved_correct_direction_fraction": (
                    correct_motion_count / count if count else float("nan")
                ),
                "hard_target_entered_count": entered,
                "hard_target_exited_count": exited,
                "desired_hard_crossing_count": (
                    entered if desired_direction > 0 else exited
                ),
                "undesired_hard_crossing_count": (
                    exited if desired_direction > 0 else entered
                ),
            }
        )
    return rows
