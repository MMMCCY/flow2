"""Model-based post-stack acoustic inversion utilities for Phase 5a.

The inversion is deliberately separate from flow sampling.  It estimates a
regular-time log-impedance correction around each fixed geological prior,
maps that correction back to depth with the prior slowness, and preserves all
surface/borehole properties exactly.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F

from guidance.seismic import (
    ConvolutionalSeismic,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
)


PHASE5A_INVERSION_VERSION = 1
PHASE5A_CONFIG_SCHEMA = "phase5a_model_based_log_impedance_config_v1"
PHASE5A_INVERSION_MODE = "linearized_poststack_log_impedance_tikhonov_v1"


@dataclass(frozen=True)
class ModelBasedInversionConfig:
    """Frozen dimensionless regularization for one Phase-5a operating point."""

    config_id: str
    prior_relative_weight: float
    vertical_smoothness_relative_weight: float


def parse_inversion_config(
    config: Mapping[str, object],
) -> ModelBasedInversionConfig:
    """Validate the frozen Phase-5a configuration without resolving data."""
    if config.get("schema") != PHASE5A_CONFIG_SCHEMA:
        raise ValueError(f"inversion schema must be {PHASE5A_CONFIG_SCHEMA!r}")
    if config.get("inversion_mode") != PHASE5A_INVERSION_MODE:
        raise ValueError(f"inversion_mode must be {PHASE5A_INVERSION_MODE!r}")
    required = {
        "regularization_scale": "mean_diagonal_gtg",
        "time_difference": "forward_first_difference_last_row_zero",
        "wavelet_boundary": "zero_padding_same_length_no_wraparound",
        "time_depth_mapping": "fixed_prior_slowness_cell_center_linear_interpolation_v1",
        "slowness_update": "none_keep_prior_v1",
        "subsurface_air_policy": (
            "codebook_rock_closest_to_median_log_impedance_v1"
        ),
        "impedance_bounds": "non_air_codebook_minmax",
        "condition_policy": (
            "surface_and_boreholes_exact_before_and_after_inversion_v1"
        ),
        "posterior_statistics": "fixed12_population_mean_std_v1",
        "truth_tuned": False,
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(f"{field} must be {expected!r}")
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("inversion config requires a non-empty id")
    prior = float(config.get("prior_relative_weight", float("nan")))
    smooth = float(
        config.get("vertical_smoothness_relative_weight", float("nan"))
    )
    if not math.isfinite(prior) or prior <= 0:
        raise ValueError("prior_relative_weight must be finite and positive")
    if not math.isfinite(smooth) or smooth < 0:
        raise ValueError(
            "vertical_smoothness_relative_weight must be finite and non-negative"
        )
    return ModelBasedInversionConfig(config_id, prior, smooth)


def forward_difference_matrix(
    num_samples: int,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Return ``D`` with ``(D m)[i] = m[i+1]-m[i]`` and a zero last row."""
    count = int(num_samples)
    if count <= 1:
        raise ValueError("num_samples must be greater than one")
    matrix = torch.zeros((count, count), device=device, dtype=dtype)
    indices = torch.arange(count - 1, device=device)
    matrix[indices, indices] = -1.0
    matrix[indices, indices + 1] = 1.0
    return matrix


def same_length_convolution_matrix(
    wavelet: torch.Tensor,
    num_samples: int,
) -> torch.Tensor:
    """Return the exact PyTorch zero-padded same-length 1-D operator matrix."""
    if wavelet.ndim != 1 or wavelet.numel() % 2 != 1:
        raise ValueError("wavelet must be a one-dimensional odd-length tensor")
    if not wavelet.is_floating_point() or not torch.isfinite(wavelet).all():
        raise ValueError("wavelet must be finite floating point")
    count = int(num_samples)
    if count <= 1:
        raise ValueError("num_samples must be greater than one")
    basis = torch.eye(count, device=wavelet.device, dtype=wavelet.dtype)
    responses = F.conv1d(
        basis[:, None, :],
        wavelet[None, None, :],
        padding=wavelet.numel() // 2,
    )[:, 0, :]
    return responses.transpose(0, 1).contiguous()


def linearized_log_impedance_operator(
    wavelet: torch.Tensor,
    num_samples: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ``G = W (0.5 D)`` and its first-difference matrix."""
    difference = forward_difference_matrix(
        num_samples, device=wavelet.device, dtype=wavelet.dtype
    )
    convolution = same_length_convolution_matrix(wavelet, num_samples)
    return convolution @ (0.5 * difference), difference


def solve_log_impedance_correction(
    residual: torch.Tensor,
    wavelet: torch.Tensor,
    config: ModelBasedInversionConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Solve the frozen Tikhonov system for every lateral trace at once."""
    if residual.ndim != 5 or residual.shape[1] != 1:
        raise ValueError("residual must have shape [B,1,X,Y,T]")
    if not residual.is_floating_point() or not torch.isfinite(residual).all():
        raise ValueError("residual must be finite floating point")
    count = residual.shape[-1]
    wavelet = wavelet.to(device=residual.device, dtype=residual.dtype)
    operator, difference = linearized_log_impedance_operator(wavelet, count)
    normal = operator.transpose(0, 1) @ operator
    scale = normal.diagonal().mean()
    if not torch.isfinite(scale) or float(scale) <= 0:
        raise ValueError("linearized operator has an invalid normal-matrix scale")
    prior_lambda = config.prior_relative_weight * scale
    smooth_lambda = config.vertical_smoothness_relative_weight * scale
    system = normal + prior_lambda * torch.eye(
        count, device=residual.device, dtype=residual.dtype
    )
    if config.vertical_smoothness_relative_weight:
        system = system + smooth_lambda * (difference.transpose(0, 1) @ difference)
    cholesky = torch.linalg.cholesky(system)
    flat = residual.reshape(-1, count).transpose(0, 1)
    right_hand_side = operator.transpose(0, 1) @ flat
    solved = torch.cholesky_solve(right_hand_side, cholesky)
    correction = solved.transpose(0, 1).reshape_as(residual).contiguous()
    if not torch.isfinite(correction).all():
        raise FloatingPointError("log-impedance inversion produced non-finite values")
    diagnostics = {
        "normal_diagonal_mean": float(scale.detach().cpu()),
        "prior_lambda": float(prior_lambda.detach().cpu()),
        "smoothness_lambda": float(smooth_lambda.detach().cpu()),
        "correction_abs_max_time": float(correction.abs().max().detach().cpu()),
        "correction_rms_time": float(correction.square().mean().sqrt().detach().cpu()),
    }
    return correction, diagnostics


def cell_center_twt_ms(
    slowness: torch.Tensor,
    subsurface_mask: torch.Tensor,
    cell_size_z_m: float,
) -> torch.Tensor:
    """Return prior-derived top-down cell-centre TWT in milliseconds."""
    if slowness.ndim != 5 or slowness.shape[1] != 1:
        raise ValueError("slowness must have shape [B,1,X,Y,Z]")
    if not slowness.is_floating_point() or not torch.isfinite(slowness).all():
        raise ValueError("slowness must be finite floating point")
    if bool((slowness <= 0).any()):
        raise ValueError("slowness must be positive")
    if subsurface_mask.ndim != 5 or subsurface_mask.shape[1] != 1:
        raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape[2:] != slowness.shape[2:]:
        raise ValueError("subsurface_mask spatial shape must match slowness")
    if subsurface_mask.shape[0] not in (1, slowness.shape[0]):
        raise ValueError("subsurface_mask batch must be one or match slowness")
    dz = float(cell_size_z_m)
    if not math.isfinite(dz) or dz <= 0:
        raise ValueError("cell_size_z_m must be finite and positive")
    mask = subsurface_mask.to(device=slowness.device, dtype=torch.bool)
    mask = mask.expand_as(slowness).flip(-1)
    top_down = slowness.flip(-1)
    layer_ms = 2.0 * dz * 1000.0 * top_down * mask.to(top_down.dtype)
    return (layer_ms.cumsum(dim=-1) - 0.5 * layer_ms).contiguous()


def sample_time_correction_to_depth(
    correction_time: torch.Tensor,
    cell_center_times_ms: torch.Tensor,
    subsurface_mask: torch.Tensor,
    sample_interval_ms: float,
) -> torch.Tensor:
    """Linearly sample a regular-time correction at cell centres in depth."""
    if correction_time.ndim != 5 or correction_time.shape[1] != 1:
        raise ValueError("correction_time must have shape [B,1,X,Y,T]")
    if cell_center_times_ms.ndim != 5 or cell_center_times_ms.shape[1] != 1:
        raise ValueError("cell_center_times_ms must have shape [B,1,X,Y,Z]")
    if correction_time.shape[:4] != cell_center_times_ms.shape[:4]:
        raise ValueError("time correction and cell-centre grids must match laterally")
    if subsurface_mask.shape != cell_center_times_ms.shape:
        if not (
            subsurface_mask.ndim == 5
            and subsurface_mask.shape[0] == 1
            and subsurface_mask.shape[1:] == cell_center_times_ms.shape[1:]
        ):
            raise ValueError("subsurface_mask must match cell-centre grid")
    interval = float(sample_interval_ms)
    if not math.isfinite(interval) or interval <= 0:
        raise ValueError("sample_interval_ms must be finite and positive")
    count = correction_time.shape[-1]
    position = cell_center_times_ms / interval
    left = torch.floor(position).long().clamp(0, count - 1)
    right = (left + 1).clamp(0, count - 1)
    fraction = (position - torch.floor(position)).clamp(0.0, 1.0)
    left_value = torch.gather(correction_time, -1, left)
    right_value = torch.gather(correction_time, -1, right)
    sampled_top_down = left_value * (1.0 - fraction) + right_value * fraction
    mask = subsurface_mask.to(device=sampled_top_down.device, dtype=torch.bool)
    mask = mask.expand_as(sampled_top_down).flip(-1)
    sampled_top_down = torch.where(mask, sampled_top_down, torch.zeros_like(sampled_top_down))
    return sampled_top_down.flip(-1).contiguous()


def neutral_rock_category(property_table: torch.Tensor) -> int:
    """Choose the rock category closest to the codebook median log-impedance."""
    if property_table.ndim != 2 or property_table.shape[0] != 2:
        raise ValueError("property_table must have shape [2,C]")
    if property_table.shape[1] <= 2:
        raise ValueError("property_table must contain air and at least two rocks")
    rock_log_impedance = property_table[0, 1:].log()
    if not torch.isfinite(rock_log_impedance).all():
        raise ValueError("rock impedance must be finite and positive")
    median = rock_log_impedance.median()
    return int((rock_log_impedance - median).abs().argmin().item()) + 1


def labels_to_clean_prior_acoustic(
    labels: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Map labels to acoustics after deterministic underground-air cleanup."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape != labels.shape:
        if not (
            subsurface_mask.ndim == 5
            and subsurface_mask.shape[0] == 1
            and subsurface_mask.shape[1:] == labels.shape[1:]
        ):
            raise ValueError("subsurface_mask must match labels")
    cleaned = labels.long().clone()
    rock = subsurface_mask.to(device=labels.device, dtype=torch.bool).expand_as(labels)
    invalid = rock & (cleaned == -1)
    category = neutral_rock_category(property_table)
    cleaned[invalid] = category - 1
    acoustic = hard_labels_to_acoustic(cleaned, property_table)
    return acoustic, {
        "underground_air_voxels_replaced": int(invalid.sum().item()),
        "neutral_category": category,
        "neutral_raw_label": category - 1,
    }


def build_exact_condition_acoustic(
    boreholes: torch.Tensor,
    subsurface_mask: torch.Tensor,
    property_table: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build exact surface/borehole acoustics without reading unconstrained truth."""
    if boreholes.ndim != 5 or boreholes.shape[1] != 1:
        raise ValueError("boreholes must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape != boreholes.shape:
        raise ValueError("subsurface_mask must match boreholes")
    condition_mask = (~subsurface_mask.bool()) | (boreholes != -1)
    target = hard_labels_to_acoustic(boreholes.long(), property_table)
    return target, condition_mask


def invert_acoustic_member(
    prior_acoustic: torch.Tensor,
    *,
    observed_seismic: torch.Tensor,
    subsurface_mask: torch.Tensor,
    condition_target: torch.Tensor,
    condition_mask: torch.Tensor,
    property_table: torch.Tensor,
    forward_operator: ConvolutionalSeismic,
    config: ModelBasedInversionConfig,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, float]]:
    """Invert one member and return prior/updated fields plus diagnostics."""
    if prior_acoustic.ndim != 5 or prior_acoustic.shape[1] != 2:
        raise ValueError("prior_acoustic must have shape [B,2,X,Y,Z]")
    prior_exact = overwrite_exact_condition_acoustic(
        prior_acoustic, condition_target, condition_mask
    ).to(dtype=observed_seismic.dtype)
    mask = subsurface_mask.to(device=prior_exact.device, dtype=torch.bool)
    with torch.no_grad():
        prior_field = forward_operator(
            prior_exact[:, 0:1], prior_exact[:, 1:2], mask
        )
        residual = observed_seismic.to(prior_field) - prior_field
        correction_time, diagnostics = solve_log_impedance_correction(
            residual,
            forward_operator.wavelet(prior_field.device, prior_field.dtype),
            config,
        )
        center_times = cell_center_twt_ms(
            prior_exact[:, 1:2], mask, forward_operator.cell_size_m[2]
        )
        correction_depth = sample_time_correction_to_depth(
            correction_time,
            center_times,
            mask,
            forward_operator.sample_interval_ms,
        )
        rock_impedance = property_table[0, 1:].to(prior_exact)
        minimum = rock_impedance.min()
        maximum = rock_impedance.max()
        updated_impedance = torch.exp(prior_exact[:, 0:1].log() + correction_depth)
        updated_impedance = updated_impedance.clamp(minimum, maximum)
        updated_impedance = torch.where(
            mask.expand_as(updated_impedance),
            updated_impedance,
            prior_exact[:, 0:1],
        )
        updated = torch.cat((updated_impedance, prior_exact[:, 1:2]), dim=1)
        updated = overwrite_exact_condition_acoustic(
            updated, condition_target.to(updated), condition_mask
        )
        updated_field = forward_operator(
            updated[:, 0:1], updated[:, 1:2], mask
        )
    diagnostics.update(
        {
            "correction_abs_max_depth": float(
                correction_depth.abs().max().detach().cpu()
            ),
            "correction_rms_depth_subsurface": float(
                correction_depth[mask].square().mean().sqrt().detach().cpu()
            ),
        }
    )
    return prior_exact, updated, torch.stack((prior_field, updated_field), dim=0), diagnostics


def posterior_statistics(members: torch.Tensor) -> dict[str, torch.Tensor]:
    """Return geometric-impedance and arithmetic-slowness posterior moments."""
    if members.ndim != 5 or members.shape[1] != 2:
        raise ValueError("members must have shape [N,2,X,Y,Z]")
    if members.shape[0] < 2:
        raise ValueError("posterior requires at least two members")
    if not members.is_floating_point() or not torch.isfinite(members).all():
        raise ValueError("posterior members must be finite floating point")
    if bool((members <= 0).any()):
        raise ValueError("posterior acoustic properties must be positive")
    log_impedance = members[:, 0:1].log()
    slowness = members[:, 1:2]
    log_mean = log_impedance.mean(dim=0, keepdim=True)
    slowness_mean = slowness.mean(dim=0, keepdim=True)
    return {
        "log_impedance_mean": log_mean,
        "log_impedance_std": log_impedance.std(dim=0, unbiased=False, keepdim=True),
        "slowness_mean": slowness_mean,
        "slowness_std": slowness.std(dim=0, unbiased=False, keepdim=True),
        "acoustic_mean": torch.cat((log_mean.exp(), slowness_mean), dim=1),
    }


def nearest_codebook_labels(
    acoustic: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
) -> torch.Tensor:
    """Project acoustics to the nearest normalized rock code for diagnostics."""
    if acoustic.ndim != 5 or acoustic.shape[1] != 2:
        raise ValueError("acoustic must have shape [B,2,X,Y,Z]")
    if property_table.ndim != 2 or property_table.shape[0] != 2:
        raise ValueError("property_table must have shape [2,C]")
    values = torch.stack((acoustic[:, 0].log(), acoustic[:, 1].log()), dim=1)
    rock = torch.stack(
        (property_table[0, 1:].log(), property_table[1, 1:].log()), dim=0
    ).to(values)
    ranges = (rock.max(dim=1).values - rock.min(dim=1).values).clamp_min(
        torch.finfo(values.dtype).eps
    )
    normalized_values = values / ranges[None, :, None, None, None]
    normalized_rock = rock / ranges[:, None]
    distance = (
        normalized_values[:, :, None]
        - normalized_rock[None, :, :, None, None, None]
    ).square().sum(dim=1)
    raw_labels = distance.argmin(dim=1, keepdim=True).long()
    mask = subsurface_mask.to(device=raw_labels.device, dtype=torch.bool)
    mask = mask.expand_as(raw_labels)
    return torch.where(mask, raw_labels, torch.full_like(raw_labels, -1))
