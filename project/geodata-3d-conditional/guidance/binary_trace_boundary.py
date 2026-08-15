"""Full-trace binary impedance inversion utilities for Stage15-H."""

from __future__ import annotations

import math

import torch

from guidance.binary_seismic_inversion import (
    BinaryAcousticProperties,
    binary_occupancy_to_acoustic,
)
from guidance.seismic_inversion import (
    ModelBasedInversionConfig,
    cell_center_twt_ms,
    sample_time_correction_to_depth,
    solve_log_impedance_correction,
)


def impedance_to_binary_score(
    impedance: torch.Tensor,
    subsurface_mask: torch.Tensor,
    properties: BinaryAcousticProperties,
) -> torch.Tensor:
    """Map log impedance between the known binary endpoints to ``[0,1]``."""
    if impedance.ndim != 5 or impedance.shape[1] != 1:
        raise ValueError("impedance must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape != impedance.shape:
        raise ValueError("subsurface_mask must match impedance")
    if not torch.isfinite(impedance).all() or bool((impedance <= 0).any()):
        raise ValueError("impedance must be finite and positive")
    lower = math.log(properties.background_impedance)
    upper = math.log(properties.target_impedance)
    score = (impedance.log() - lower) / (upper - lower)
    score = score.clamp(0.0, 1.0)
    return torch.where(subsurface_mask.bool(), score, torch.zeros_like(score)).contiguous()


def vertical_boundary_strength(score: torch.Tensor) -> torch.Tensor:
    """Return a cell-centred magnitude of adjacent vertical score changes."""
    if score.ndim != 5 or score.shape[1] != 1:
        raise ValueError("score must have shape [B,1,X,Y,Z]")
    difference = torch.diff(score, dim=-1).abs()
    boundary = torch.zeros_like(score)
    boundary[..., :-1] = torch.maximum(boundary[..., :-1], difference)
    boundary[..., 1:] = torch.maximum(boundary[..., 1:], difference)
    return boundary.contiguous()


def refine_binary_trace_volume(
    observed_seismic: torch.Tensor,
    subsurface_mask: torch.Tensor,
    operator,
    properties: BinaryAcousticProperties,
    inversion_config: ModelBasedInversionConfig,
    refinement_passes: int,
    fixed_occupancy: torch.Tensor | None = None,
    fixed_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, list[dict[str, float]]]:
    """Invert all complete traces jointly without lateral mixing or truth access."""
    if refinement_passes <= 0:
        raise ValueError("refinement_passes must be positive")
    if observed_seismic.ndim != 5 or observed_seismic.shape[1] != 1:
        raise ValueError("observed_seismic must have shape [B,1,X,Y,T]")
    if subsurface_mask.ndim != 5 or subsurface_mask.shape[1] != 1:
        raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
    device, dtype = observed_seismic.device, observed_seismic.dtype
    support = subsurface_mask.to(device=device).bool()
    score = torch.zeros(support.shape, device=device, dtype=dtype)
    if fixed_occupancy is None:
        fixed_occupancy = torch.zeros_like(score)
    else:
        fixed_occupancy = fixed_occupancy.to(device=device, dtype=dtype)
    if fixed_mask is None:
        fixed_mask = torch.zeros_like(support)
    else:
        fixed_mask = fixed_mask.to(device=device).bool()
    if fixed_occupancy.shape != score.shape or fixed_mask.shape != score.shape:
        raise ValueError("fixed occupancy and mask must match the subsurface volume")
    score = torch.where(fixed_mask, fixed_occupancy, score)
    impedance, slowness = binary_occupancy_to_acoustic(score, support, properties)
    trace: list[dict[str, float]] = []
    for pass_index in range(refinement_passes):
        predicted = operator(impedance, slowness, support)
        residual = observed_seismic - predicted
        correction_time, diagnostics = solve_log_impedance_correction(
            residual,
            operator.wavelet(device, dtype),
            inversion_config,
        )
        center_times = cell_center_twt_ms(slowness, support, operator.cell_size_m[2])
        correction_depth = sample_time_correction_to_depth(
            correction_time,
            center_times,
            support,
            operator.sample_interval_ms,
        )
        updated_impedance = torch.exp(impedance.log() + correction_depth).clamp(
            properties.background_impedance,
            properties.target_impedance,
        )
        score = impedance_to_binary_score(updated_impedance, support, properties)
        score = torch.where(fixed_mask, fixed_occupancy, score)
        impedance, slowness = binary_occupancy_to_acoustic(score, support, properties)
        after = operator(impedance, slowness, support)
        trace.append(
            {
                "pass": float(pass_index),
                "seismic_rmse_before": float(residual.square().mean().sqrt().cpu()),
                "seismic_rmse_after": float((after - observed_seismic).square().mean().sqrt().cpu()),
                "score_mean_subsurface": float(score[support].mean().cpu()),
                "score_max": float(score.max().cpu()),
                **diagnostics,
            }
        )
    predicted = operator(impedance, slowness, support)
    return score.cpu(), torch.cat((impedance, slowness), dim=1).cpu(), predicted.cpu(), trace
