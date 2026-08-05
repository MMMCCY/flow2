"""Strict fixed-Euler sampling for ideal 3-D property-volume guidance."""

from __future__ import annotations

from typing import Callable, Dict, List, Mapping, Sequence

import torch

from guided_geophysical_sampling import clip_gradient_by_norm

from .probability_sampling import (
    GUIDANCE_GRADIENT_REFERENCE_POLICY,
    GUIDANCE_SCALING_MODES,
    REFERENCE_GUIDANCE_SCALING_MODE,
    build_probability_guidance_velocity,
    probability_guidance_weight,
    temperature_at_time,
)
from .property_volume import PROPERTY_LOSS_MODE, property_volume_loss


PROPERTY_SAMPLER_VERSION = "projected_fixed_euler_property_v1"


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
    return projected, _batch_l2(projected - state)


def _decode(model, state: torch.Tensor) -> torch.Tensor:
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


def _loss_trace_values(diagnostics: Dict[str, torch.Tensor]) -> Dict[str, float]:
    values: Dict[str, float] = {}
    for name, value in diagnostics.items():
        if not isinstance(value, torch.Tensor) or value.numel() != 1:
            continue
        values[name] = float(value.detach().cpu())
    return values


def fixed_euler_property_sample(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_truth: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
    target_properties: torch.Tensor,
    property_table: torch.Tensor,
    confidence: torch.Tensor,
    property_sigmas: Sequence[float],
    property_scale_weights: Sequence[float],
    property_channel_weights: torch.Tensor,
    n_steps: int,
    alpha: float,
    max_guidance_ratio: float,
    tau_start: float,
    tau_end: float,
    tau_schedule: str,
    guidance_start: float,
    guidance_schedule: str,
    grad_clip_norm: float,
    guidance_scaling_mode: str = REFERENCE_GUIDANCE_SCALING_MODE,
    sample_id: int = 0,
    loss_function: Callable[..., tuple[torch.Tensor, Dict[str, torch.Tensor]]] = property_volume_loss,
    loss_extra_kwargs: Mapping[str, object] | None = None,
    loss_mode: str = PROPERTY_LOSS_MODE,
) -> tuple[torch.Tensor, List[Dict[str, object]]]:
    """Integrate the Phase-2 property loss with hard-condition projection.

    ``alpha == 0`` always takes an explicit no-gradient branch. The loss is
    still evaluated for diagnostics, but no gradient or guidance velocity is
    constructed and the trajectory is the paired projected fixed-Euler path.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if alpha < 0 or max_guidance_ratio < 0 or grad_clip_norm < 0:
        raise ValueError("alpha, guidance cap, and gradient clip must be non-negative")
    if guidance_scaling_mode not in GUIDANCE_SCALING_MODES:
        raise ValueError(f"guidance_scaling_mode must be one of {GUIDANCE_SCALING_MODES}")
    if initial_state.ndim != 5:
        raise ValueError("initial_state must have shape [B,E,X,Y,Z]")
    if conditioning.shape[1:] != initial_state.shape[1:]:
        raise ValueError("conditioning and initial_state must match")
    if target_properties.ndim != 5:
        raise ValueError("target_properties must have shape [1,P,X,Y,Z]")
    if property_table.ndim != 2:
        raise ValueError("property_table must have shape [P,C]")
    if confidence.ndim != 5 or confidence.shape[1] != 1:
        raise ValueError("confidence must have shape [1,1,X,Y,Z]")

    state, initial_projection_norm = _project_conditions(
        initial_state.detach(),
        embedded_truth,
        condition_mask,
    )
    conditioning_device = conditioning.to(
        device=state.device,
        dtype=state.dtype,
    ).expand(state.shape[0], -1, -1, -1, -1)
    target_device = target_properties.to(device=state.device, dtype=state.dtype)
    table_device = property_table.to(device=state.device, dtype=state.dtype)
    confidence_device = confidence.to(device=state.device, dtype=state.dtype)
    channel_weights_device = property_channel_weights.to(
        device=state.device,
        dtype=state.dtype,
    )
    embedding_weight = model.embedding.weight
    guidance_region = confidence_device[:, :1] > 0
    dt = 1.0 / n_steps
    trace: List[Dict[str, object]] = []
    reference_gradient_norm: torch.Tensor | None = None
    extra_loss_arguments = dict(loss_extra_kwargs or {})

    for step in range(n_steps):
        t_value = (step + 0.5) / n_steps
        time = torch.full(
            (state.shape[0],),
            t_value,
            device=state.device,
            dtype=state.dtype,
        )
        guidance_weight = probability_guidance_weight(
            t_value,
            guidance_schedule,
            guidance_start,
        )
        tau = temperature_at_time(t_value, tau_start, tau_end, tau_schedule)
        decoded_before = _decode(model, state)
        guidance_active = alpha > 0 and guidance_weight > 0
        differentiable_state = state.detach().requires_grad_(guidance_active)
        with torch.no_grad():
            prior_velocity = model.net(differentiable_state, conditioning_device, time)

        if guidance_active:
            loss, loss_diagnostics = loss_function(
                differentiable_state,
                embedding_weight,
                target_device,
                table_device,
                confidence_device,
                tau=tau,
                sigmas=property_sigmas,
                scale_weights=property_scale_weights,
                channel_weights=channel_weights_device,
                **extra_loss_arguments,
            )
            raw_gradient = torch.autograd.grad(loss, differentiable_state)[0]
            used_gradient = clip_gradient_by_norm(raw_gradient, grad_clip_norm)
            (
                guidance_velocity,
                guidance_diagnostics,
                reference_gradient_norm,
            ) = build_probability_guidance_velocity(
                used_gradient,
                prior_velocity,
                requested_ratio=float(alpha) * guidance_weight,
                max_ratio=max_guidance_ratio,
                scaling_mode=guidance_scaling_mode,
                reference_gradient_norm=reference_gradient_norm,
            )
        else:
            with torch.no_grad():
                loss, loss_diagnostics = loss_function(
                    differentiable_state,
                    embedding_weight,
                    target_device,
                    table_device,
                    confidence_device,
                    tau=tau,
                    sigmas=property_sigmas,
                    scale_weights=property_scale_weights,
                    channel_weights=channel_weights_device,
                    **extra_loss_arguments,
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
        if post_projection_violations:
            raise RuntimeError(
                "hard-condition projection failed: "
                f"{post_projection_violations} decoded violations"
            )

        decoded_after = _decode(model, next_state)
        changed = decoded_after != decoded_before
        region = guidance_region.expand(decoded_after.shape[0], -1, -1, -1, -1)
        trace_row: Dict[str, object] = {
            "sample_id": int(sample_id),
            "step": int(step),
            "t": t_value,
            "w_t": guidance_weight,
            "tau": tau,
            **_loss_trace_values(loss_diagnostics),
            "raw_grad_norm": _batch_l2(raw_gradient),
            "used_grad_norm": _batch_l2(used_gradient),
            "prior_velocity_norm": _batch_l2(prior_velocity),
            "guidance_velocity_norm": guidance_diagnostics["guidance_velocity_norm"],
            "guided_velocity_norm": _batch_l2(guided_velocity),
            "requested_guidance_ratio": guidance_diagnostics[
                "requested_guidance_ratio"
            ],
            "used_guidance_ratio": guidance_diagnostics["used_guidance_ratio"],
            "uncapped_guidance_ratio": guidance_diagnostics[
                "uncapped_guidance_ratio"
            ],
            "guidance_cap_fraction": guidance_diagnostics["guidance_cap_fraction"],
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
            "hard_change_inside_confidence": int((changed & region).sum().item()),
            "hard_change_outside_confidence": int((changed & ~region).sum().item()),
            "alpha": float(alpha),
            "property_loss_mode": loss_mode,
            "guidance_scaling_mode": guidance_scaling_mode,
            "guidance_reference_policy": (
                GUIDANCE_GRADIENT_REFERENCE_POLICY
                if guidance_scaling_mode == REFERENCE_GUIDANCE_SCALING_MODE
                else None
            ),
        }
        trace.append(trace_row)
        state = next_state.detach()

    return state, trace
