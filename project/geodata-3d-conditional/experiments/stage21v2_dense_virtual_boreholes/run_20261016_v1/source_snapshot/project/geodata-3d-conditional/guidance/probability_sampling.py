"""Strictly paired fixed-Euler sampling with oracle 3-D probability guidance."""

from __future__ import annotations

import math
from typing import Dict, List

import torch

from guided_geophysical_sampling import (
    clip_gradient_by_norm,
    guidance_weight as legacy_guidance_weight,
)

from .probability_volume import (
    LEGACY_PROBABILITY_LOSS_MODE,
    PROBABILITY_LOSS_MODES,
    probability_volume_loss,
)


TEMPERATURE_SCHEDULES = ("constant", "linear", "cosine")
GUIDANCE_SCHEDULES = (
    "late_quadratic",
    "quadratic",
    "constant_after_start",
    "windowed_sine",
)
LEGACY_GUIDANCE_SCALING_MODE = "unit_norm_relative_v1"
REFERENCE_GUIDANCE_SCALING_MODE = "reference_norm_relative_v2"
GUIDANCE_SCALING_MODES = (
    LEGACY_GUIDANCE_SCALING_MODE,
    REFERENCE_GUIDANCE_SCALING_MODE,
)
GUIDANCE_GRADIENT_REFERENCE_POLICY = (
    "first_nonzero_active_used_gradient_norm_per_sample_v1"
)


def temperature_at_time(
    t: float,
    start: float,
    end: float,
    schedule: str,
) -> float:
    """Return a positive soft-decoding temperature for ``t in [0,1]``."""
    if start <= 0 or end <= 0:
        raise ValueError("temperature endpoints must be positive")
    if not 0.0 <= float(t) <= 1.0:
        raise ValueError("t must be in [0,1]")
    if float(t) == 0.0:
        return float(start)
    if float(t) == 1.0:
        return float(end)
    if schedule == "constant":
        if start != end:
            raise ValueError("constant temperature schedule requires equal endpoints")
        return float(start)
    if schedule == "linear":
        fraction = float(t)
    elif schedule == "cosine":
        fraction = 0.5 - 0.5 * math.cos(math.pi * float(t))
    else:
        raise ValueError(f"temperature schedule must be one of {TEMPERATURE_SCHEDULES}")
    return float(start) + fraction * (float(end) - float(start))


def probability_guidance_weight(t: float, schedule: str, start: float) -> float:
    """Return the Phase-1 guidance envelope at continuous time ``t``.

    Legacy schedules delegate to the historical implementation.  The new
    windowed sine schedule is zero before ``start``, peaks halfway through the
    active interval, and returns to zero at the endpoint so guidance cannot
    keep increasing after the target residual has already fallen.
    """
    t_value = float(t)
    if not 0.0 <= t_value <= 1.0:
        raise ValueError("t must be in [0,1]")
    if not 0.0 <= float(start) < 1.0:
        raise ValueError("start must satisfy 0 <= start < 1")
    if schedule == "windowed_sine":
        if t_value <= float(start) or t_value >= 1.0:
            return 0.0
        active_fraction = (t_value - float(start)) / (1.0 - float(start))
        return math.sin(math.pi * active_fraction)
    if schedule not in GUIDANCE_SCHEDULES:
        raise ValueError(f"guidance schedule must be one of {GUIDANCE_SCHEDULES}")
    return float(legacy_guidance_weight(t_value, schedule, start))


def _batch_l2(value: torch.Tensor) -> float:
    return float(value.detach().flatten(1).norm(dim=1).mean().cpu())


def _project_conditions(
    state: torch.Tensor,
    embedded_truth: torch.Tensor,
    condition_mask: torch.Tensor,
) -> tuple[torch.Tensor, float]:
    values = embedded_truth.to(device=state.device, dtype=state.dtype)
    mask = condition_mask.to(device=state.device).bool()
    if values.shape[0] == 1 and state.shape[0] > 1:
        values = values.expand(state.shape[0], -1, -1, -1, -1)
        mask = mask.expand(state.shape[0], -1, -1, -1, -1)
    expanded_mask = mask.expand(-1, state.shape[1], -1, -1, -1)
    projected = torch.where(expanded_mask, values, state)
    change_norm = _batch_l2(projected - state)
    return projected, change_norm


def _decode(model, state: torch.Tensor) -> torch.Tensor:
    """Decode to raw labels with shape ``[B,1,X,Y,Z]``."""
    return (model.decode(state) - 1).unsqueeze(1)


def _condition_violations(
    model,
    state: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
) -> int:
    decoded = _decode(model, state)
    truth = truth_model.to(device=decoded.device).long()
    mask = condition_mask.to(device=decoded.device).bool()
    if truth.shape[0] == 1 and decoded.shape[0] > 1:
        truth = truth.expand(decoded.shape[0], -1, -1, -1, -1)
        mask = mask.expand(decoded.shape[0], -1, -1, -1, -1)
    return int(((decoded != truth) & mask).sum().item())


def _relative_guidance_velocity(
    gradient: torch.Tensor,
    prior_velocity: torch.Tensor,
    requested_ratio: float,
    max_ratio: float,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, float]]:
    """Scale the loss gradient to a capped fraction of prior velocity norm."""
    if requested_ratio < 0 or max_ratio < 0:
        raise ValueError("guidance ratios must be non-negative")
    gradient_norm = gradient.flatten(1).norm(dim=1)
    prior_norm = prior_velocity.flatten(1).norm(dim=1)
    used_ratio = min(float(requested_ratio), float(max_ratio))
    shape = (-1, *([1] * (gradient.ndim - 1)))
    valid = gradient_norm > eps
    unit = gradient / (gradient_norm + eps).view(shape)
    velocity = used_ratio * prior_norm.view(shape) * unit
    velocity = torch.where(valid.view(shape), velocity, torch.zeros_like(velocity))
    diagnostics = {
        "requested_guidance_ratio": float(requested_ratio),
        "used_guidance_ratio": used_ratio,
        "uncapped_guidance_ratio": float(requested_ratio),
        "guidance_cap_fraction": float(float(requested_ratio) > float(max_ratio)),
        "guidance_gradient_reference_norm": float("nan"),
        "guidance_gradient_reference_ratio": float("nan"),
        "guidance_velocity_norm": _batch_l2(velocity),
        "effective_guidance_ratio": float(
            (velocity.detach().flatten(1).norm(dim=1) / (prior_norm + eps))
            .mean()
            .cpu()
        ),
    }
    return velocity, diagnostics


def build_probability_guidance_velocity(
    gradient: torch.Tensor,
    prior_velocity: torch.Tensor,
    requested_ratio: float,
    max_ratio: float,
    scaling_mode: str,
    reference_gradient_norm: torch.Tensor | None = None,
    eps: float = 1e-8,
) -> tuple[torch.Tensor, Dict[str, float], torch.Tensor | None]:
    """Scale a probability-loss gradient without discarding convergence.

    ``unit_norm_relative_v1`` exactly preserves the protocol-v1/v3 behavior.
    ``reference_norm_relative_v2`` records the first non-zero active gradient
    norm for each sample and subsequently scales relative to that fixed norm.
    A shrinking gradient therefore produces a shrinking velocity.  The final
    velocity remains capped relative to the paired prior velocity.
    """
    if scaling_mode not in GUIDANCE_SCALING_MODES:
        raise ValueError(f"scaling_mode must be one of {GUIDANCE_SCALING_MODES}")
    if requested_ratio < 0 or max_ratio < 0:
        raise ValueError("guidance ratios must be non-negative")
    if gradient.shape != prior_velocity.shape:
        raise ValueError("gradient and prior_velocity must have matching shapes")
    if scaling_mode == LEGACY_GUIDANCE_SCALING_MODE:
        velocity, diagnostics = _relative_guidance_velocity(
            gradient,
            prior_velocity,
            requested_ratio=requested_ratio,
            max_ratio=max_ratio,
            eps=eps,
        )
        return velocity, diagnostics, reference_gradient_norm

    gradient_norm = gradient.detach().flatten(1).norm(dim=1)
    prior_norm = prior_velocity.detach().flatten(1).norm(dim=1)
    if reference_gradient_norm is None:
        reference = gradient_norm.clone()
    else:
        reference = reference_gradient_norm.to(
            device=gradient_norm.device,
            dtype=gradient_norm.dtype,
        ).clone()
        if reference.shape != gradient_norm.shape:
            raise ValueError("reference_gradient_norm must have one value per sample")
        reference = torch.where(
            (reference <= eps) & (gradient_norm > eps),
            gradient_norm,
            reference,
        )

    valid = (gradient_norm > eps) & (reference > eps)
    reference_ratio = torch.where(
        valid,
        gradient_norm / reference.clamp_min(eps),
        torch.zeros_like(gradient_norm),
    )
    uncapped_ratio = float(requested_ratio) * reference_ratio
    used_ratio = uncapped_ratio.clamp(max=float(max_ratio))
    shape = (-1, *([1] * (gradient.ndim - 1)))
    unit = gradient / (gradient_norm + eps).view(shape)
    velocity = used_ratio.view(shape) * prior_norm.view(shape) * unit
    velocity = torch.where(valid.view(shape), velocity, torch.zeros_like(velocity))
    cap_hits = valid & (uncapped_ratio > float(max_ratio) + eps)
    diagnostics = {
        "requested_guidance_ratio": float(requested_ratio),
        "used_guidance_ratio": float(used_ratio.mean().cpu()),
        "uncapped_guidance_ratio": float(uncapped_ratio.mean().cpu()),
        "guidance_cap_fraction": float(cap_hits.float().mean().cpu()),
        "guidance_gradient_reference_norm": float(reference.mean().cpu()),
        "guidance_gradient_reference_ratio": float(reference_ratio.mean().cpu()),
        "guidance_velocity_norm": _batch_l2(velocity),
        "effective_guidance_ratio": float(
            (velocity.detach().flatten(1).norm(dim=1) / (prior_norm + eps))
            .mean()
            .cpu()
        ),
    }
    return velocity, diagnostics, reference.detach()


def fixed_euler_probability_sample(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_truth: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
    target_probability: torch.Tensor,
    target_mask: torch.Tensor,
    roi_mask: torch.Tensor,
    target_label: int,
    n_steps: int,
    alpha: float,
    max_guidance_ratio: float,
    tau_start: float,
    tau_end: float,
    tau_schedule: str,
    guidance_start: float,
    guidance_schedule: str,
    grad_clip_norm: float,
    bce_weight: float,
    dice_weight: float,
    spatial_gradient_weight: float = 0.0,
    probability_loss_mode: str = LEGACY_PROBABILITY_LOSS_MODE,
    guidance_scaling_mode: str = LEGACY_GUIDANCE_SCALING_MODE,
    sample_id: int = 0,
) -> tuple[torch.Tensor, List[Dict[str, object]]]:
    """Integrate one or more samples with strict hard-condition projection.

    The same function implements the paired baseline.  When ``alpha == 0`` it
    takes an explicit no-gradient branch, so baseline equivalence does not rely
    on multiplying a computed guidance tensor by floating-point zero.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if alpha < 0:
        raise ValueError("alpha must be non-negative")
    if max_guidance_ratio < 0:
        raise ValueError("max_guidance_ratio must be non-negative")
    if grad_clip_norm < 0:
        raise ValueError("grad_clip_norm must be non-negative")
    if spatial_gradient_weight < 0:
        raise ValueError("spatial_gradient_weight must be non-negative")
    if probability_loss_mode not in PROBABILITY_LOSS_MODES:
        raise ValueError(
            f"probability_loss_mode must be one of {PROBABILITY_LOSS_MODES}"
        )
    if guidance_scaling_mode not in GUIDANCE_SCALING_MODES:
        raise ValueError(
            f"guidance_scaling_mode must be one of {GUIDANCE_SCALING_MODES}"
        )
    if initial_state.ndim != 5:
        raise ValueError("initial_state must have shape [B,E,X,Y,Z]")
    if conditioning.shape[1:] != initial_state.shape[1:]:
        raise ValueError("conditioning and initial_state must match")

    state, initial_projection_norm = _project_conditions(
        initial_state.detach(),
        embedded_truth,
        condition_mask,
    )
    conditioning_device = conditioning.to(
        device=state.device,
        dtype=state.dtype,
    ).expand(state.shape[0], -1, -1, -1, -1)
    embedding_weight = model.embedding.weight
    dt = 1.0 / int(n_steps)
    trace: List[Dict[str, object]] = []
    target_device = target_mask.to(device=state.device).bool()
    roi_device = roi_mask.to(device=state.device).bool()
    reference_gradient_norm: torch.Tensor | None = None

    for step in range(n_steps):
        t_value = (step + 0.5) / n_steps
        time = torch.full(
            (state.shape[0],),
            t_value,
            device=state.device,
            dtype=state.dtype,
        )
        weight = probability_guidance_weight(
            t_value,
            guidance_schedule,
            guidance_start,
        )
        tau = temperature_at_time(t_value, tau_start, tau_end, tau_schedule)
        decoded_before = _decode(model, state)

        differentiable_state = state.detach().requires_grad_(alpha > 0 and weight > 0)
        with torch.no_grad():
            prior_velocity = model.net(differentiable_state, conditioning_device, time)

        if alpha > 0 and weight > 0:
            probability_loss, loss_diagnostics = probability_volume_loss(
                differentiable_state,
                embedding_weight,
                target_probability,
                roi_device,
                target_label=target_label,
                tau=tau,
                bce_weight=bce_weight,
                dice_weight=dice_weight,
                spatial_gradient_weight=spatial_gradient_weight,
                loss_mode=probability_loss_mode,
                target_mask=target_device,
            )
            raw_gradient = torch.autograd.grad(probability_loss, differentiable_state)[0]
            used_gradient = clip_gradient_by_norm(raw_gradient, grad_clip_norm)
            (
                guidance_velocity,
                guidance_diagnostics,
                reference_gradient_norm,
            ) = build_probability_guidance_velocity(
                used_gradient,
                prior_velocity,
                requested_ratio=float(alpha) * weight,
                max_ratio=max_guidance_ratio,
                scaling_mode=guidance_scaling_mode,
                reference_gradient_norm=reference_gradient_norm,
            )
        else:
            with torch.no_grad():
                probability_loss, loss_diagnostics = probability_volume_loss(
                    differentiable_state,
                    embedding_weight,
                    target_probability,
                    roi_device,
                    target_label=target_label,
                    tau=tau,
                    bce_weight=bce_weight,
                    dice_weight=dice_weight,
                    spatial_gradient_weight=spatial_gradient_weight,
                    loss_mode=probability_loss_mode,
                    target_mask=target_device,
                )
            raw_gradient = torch.zeros_like(state)
            used_gradient = torch.zeros_like(state)
            guidance_velocity = torch.zeros_like(prior_velocity)
            guidance_diagnostics = {
                "requested_guidance_ratio": 0.0,
                "used_guidance_ratio": 0.0,
                "uncapped_guidance_ratio": 0.0,
                "guidance_cap_fraction": 0.0,
                "guidance_gradient_reference_norm": 0.0,
                "guidance_gradient_reference_ratio": 0.0,
                "guidance_velocity_norm": 0.0,
                "effective_guidance_ratio": 0.0,
            }

        guided_velocity = prior_velocity - guidance_velocity
        candidate = state.detach() + dt * guided_velocity.detach()
        pre_projection_violations = _condition_violations(
            model,
            candidate,
            truth_model,
            condition_mask,
        )
        next_state, projection_norm = _project_conditions(
            candidate,
            embedded_truth,
            condition_mask,
        )
        post_projection_violations = _condition_violations(
            model,
            next_state,
            truth_model,
            condition_mask,
        )
        if post_projection_violations != 0:
            raise RuntimeError(
                "hard-condition projection failed: "
                f"{post_projection_violations} decoded violations"
            )

        decoded_after = _decode(model, next_state)
        changed = decoded_after != decoded_before
        target_before = decoded_before == int(target_label)
        target_after = decoded_after == int(target_label)
        roi = roi_device.expand(decoded_after.shape[0], -1, -1, -1, -1)
        selected = target_device.expand(decoded_after.shape[0], -1, -1, -1, -1)
        trace.append(
            {
                "sample_id": int(sample_id),
                "step": int(step),
                "t": t_value,
                "w_t": weight,
                "tau": tau,
                "probability_loss": float(probability_loss.detach().cpu()),
                "probability_base_loss": float(
                    loss_diagnostics["probability_base_loss"].detach().cpu()
                ),
                "probability_bce": float(
                    loss_diagnostics["probability_bce"].detach().cpu()
                ),
                "probability_dice_loss": float(
                    loss_diagnostics["probability_dice_loss"].detach().cpu()
                ),
                "probability_dice_score": float(
                    loss_diagnostics["probability_dice_score"].detach().cpu()
                ),
                "probability_soft_calibration_mae": float(
                    loss_diagnostics["probability_soft_calibration_mae"]
                    .detach()
                    .cpu()
                ),
                "loss_positive_scale": float(
                    loss_diagnostics["loss_positive_scale"].detach().cpu()
                ),
                "loss_negative_scale": float(
                    loss_diagnostics["loss_negative_scale"].detach().cpu()
                ),
                "roi_hard_core_probability_mean": float(
                    loss_diagnostics["roi_hard_core_probability_mean"]
                    .detach()
                    .cpu()
                ),
                "roi_background_probability_mean": float(
                    loss_diagnostics["roi_background_probability_mean"]
                    .detach()
                    .cpu()
                ),
                "roi_soft_halo_probability_mean": float(
                    loss_diagnostics["roi_soft_halo_probability_mean"]
                    .detach()
                    .cpu()
                ),
                "roi_soft_halo_target_mean": float(
                    loss_diagnostics["roi_soft_halo_target_mean"].detach().cpu()
                ),
                "probability_spatial_gradient_loss": float(
                    loss_diagnostics["probability_spatial_gradient_loss"]
                    .detach()
                    .cpu()
                ),
                "roi_spatial_gradient_error_mean": float(
                    loss_diagnostics["roi_spatial_gradient_error_mean"]
                    .detach()
                    .cpu()
                ),
                "roi_predicted_gradient_mean": float(
                    loss_diagnostics["roi_predicted_gradient_mean"].detach().cpu()
                ),
                "roi_target_gradient_mean": float(
                    loss_diagnostics["roi_target_gradient_mean"].detach().cpu()
                ),
                "roi_target_probability_mean": float(
                    loss_diagnostics["roi_target_probability_mean"].detach().cpu()
                ),
                "roi_soft_margin_mean": float(
                    loss_diagnostics["roi_soft_margin_mean"].detach().cpu()
                ),
                "roi_entropy_mean": float(
                    loss_diagnostics["roi_entropy_mean"].detach().cpu()
                ),
                "raw_grad_norm": _batch_l2(raw_gradient),
                "used_grad_norm": _batch_l2(used_gradient),
                "prior_velocity_norm": _batch_l2(prior_velocity),
                "guidance_velocity_norm": guidance_diagnostics[
                    "guidance_velocity_norm"
                ],
                "guided_velocity_norm": _batch_l2(guided_velocity),
                "requested_guidance_ratio": guidance_diagnostics[
                    "requested_guidance_ratio"
                ],
                "used_guidance_ratio": guidance_diagnostics["used_guidance_ratio"],
                "uncapped_guidance_ratio": guidance_diagnostics[
                    "uncapped_guidance_ratio"
                ],
                "guidance_cap_fraction": guidance_diagnostics[
                    "guidance_cap_fraction"
                ],
                "guidance_gradient_reference_norm": guidance_diagnostics[
                    "guidance_gradient_reference_norm"
                ],
                "guidance_gradient_reference_ratio": guidance_diagnostics[
                    "guidance_gradient_reference_ratio"
                ],
                "effective_guidance_ratio": guidance_diagnostics[
                    "effective_guidance_ratio"
                ],
                "continuous_step_norm": _batch_l2(next_state - state),
                "condition_projection_norm": projection_norm,
                "initial_condition_projection_norm": (
                    initial_projection_norm if step == 0 else 0.0
                ),
                "pre_projection_condition_violations": pre_projection_violations,
                "post_projection_condition_violations": post_projection_violations,
                "hard_change_count": int(changed.sum().item()),
                "hard_change_fraction": float(changed.float().mean().item()),
                "hard_change_inside_roi": int((changed & roi).sum().item()),
                "hard_change_outside_roi": int((changed & ~roi).sum().item()),
                "target_hard_voxels_before": int(target_before.sum().item()),
                "target_hard_voxels_after": int(target_after.sum().item()),
                "selected_target_recovered_after": int(
                    (target_after & selected).sum().item()
                ),
                "alpha": float(alpha),
                "spatial_gradient_weight": float(spatial_gradient_weight),
                "probability_loss_mode": probability_loss_mode,
                "guidance_scaling_mode": guidance_scaling_mode,
            }
        )
        state = next_state.detach()

    return state, trace
