"""Paired frozen-flow trajectory isolation for Stage6Q D4."""

from __future__ import annotations

from typing import Callable, Mapping, Sequence

import torch

from .causality_gradient_audit import cosine_decode_categories
from .generator_posterior import project_conditions
from .probability_sampling import (
    build_probability_guidance_velocity,
    probability_guidance_weight,
    temperature_at_time,
)
from .soft_hard_transfer_audit import paired_attainment, response_distance_geometry
from guided_geophysical_sampling import clip_gradient_by_norm


FROZEN_FLOW_CAUSALITY_VERSION = "phase6q_frozen_flow_causality_v1"


def _norm(value: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(value.detach()))


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    denominator = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denominator) <= eps:
        return float("nan")
    return float(torch.sum(left * right) / denominator)


def run_base_trajectory(
    *,
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    n_steps: int,
) -> dict[str, object]:
    """Run BASE once and retain its exogenous norm/direction/state schedules."""
    state = project_conditions(initial_state.detach(), embedded_conditions, condition_mask)
    condition = conditioning.to(state).expand(state.shape[0], -1, -1, -1, -1)
    dt = 1.0 / n_steps
    states = [state.detach().cpu().clone()]
    velocities = []
    norms = []
    with torch.no_grad():
        for step in range(n_steps):
            time = torch.full(
                (state.shape[0],), (step + 0.5) / n_steps, device=state.device, dtype=state.dtype
            )
            velocity = model.net(state, condition, time)
            velocities.append(velocity.detach().cpu().clone())
            norms.append(_norm(velocity))
            state = project_conditions(
                state + dt * velocity, embedded_conditions, condition_mask
            )
            states.append(state.detach().cpu().clone())
    return {"states": states, "velocities": velocities, "velocity_norms": norms}


def run_paired_trajectory(
    *,
    mode: str,
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    soft_loss: Callable[[torch.Tensor, float], torch.Tensor],
    soft_response: Callable[[torch.Tensor, float], torch.Tensor],
    hard_response: Callable[[torch.Tensor], torch.Tensor],
    observation: torch.Tensor,
    truth_response: torch.Tensor,
    target_label: int,
    base_schedule: Mapping[str, Sequence[torch.Tensor] | Sequence[float]],
    n_steps: int,
    alpha: float,
    max_ratio: float,
    tau_start: float,
    tau_end: float,
    tau_schedule: str,
    guidance_start: float,
    guidance_schedule: str,
    gradient_clip_norm: float,
    scaling_mode: str,
    late_start: float,
) -> dict[str, object]:
    """Run one trajectory arm; physics-only consumes only the frozen BASE norm schedule."""
    allowed = {"BASE", "MATCHED_NORM_PHYSICS_ONLY", "BASE_PLUS_PHYSICS", "LATE_PHYSICS"}
    if mode not in allowed:
        raise ValueError(f"unsupported trajectory mode: {mode}")
    state = project_conditions(initial_state.detach(), embedded_conditions, condition_mask)
    condition = conditioning.to(state).expand(state.shape[0], -1, -1, -1, -1)
    dt = 1.0 / n_steps
    trace = []
    best_hard_loss = float("inf")
    best_hard_step = 0
    best_hard_state = state.detach().cpu().clone()
    best_soft_loss = float("inf")
    best_soft_step = 0
    reference_gradient_norm = None
    for step in range(n_steps):
        t = (step + 0.5) / n_steps
        tau = temperature_at_time(t, tau_start, tau_end, tau_schedule)
        baseline_state = base_schedule["states"][step + 1].to(state)
        paired_base_velocity = base_schedule["velocities"][step].to(state)
        if mode == "MATCHED_NORM_PHYSICS_ONLY":
            base_velocity = torch.zeros_like(state)
            reference_velocity = torch.zeros_like(state)
            reference_velocity.reshape(reference_velocity.shape[0], -1)[:, 0] = float(
                base_schedule["velocity_norms"][step]
            )
        else:
            with torch.no_grad():
                time = torch.full((state.shape[0],), t, device=state.device, dtype=state.dtype)
                base_velocity = model.net(state, condition, time)
            reference_velocity = base_velocity
        weight = probability_guidance_weight(t, guidance_schedule, guidance_start)
        active = mode != "BASE" and weight > 0 and (
            mode != "LATE_PHYSICS" or t >= late_start
        )
        differentiable = state.detach().requires_grad_(active)
        pre_soft_loss = soft_loss(differentiable, tau)
        if active:
            raw_gradient = torch.autograd.grad(pre_soft_loss, differentiable)[0]
            used_gradient = clip_gradient_by_norm(raw_gradient, gradient_clip_norm)
            guidance, controller, reference_gradient_norm = build_probability_guidance_velocity(
                used_gradient,
                reference_velocity,
                requested_ratio=alpha * weight,
                max_ratio=max_ratio,
                scaling_mode=scaling_mode,
                reference_gradient_norm=reference_gradient_norm,
            )
            physics_velocity = -guidance
        else:
            raw_gradient = torch.zeros_like(state)
            physics_velocity = torch.zeros_like(state)
            controller = {"guidance_cap_fraction": 0.0, "used_guidance_ratio": 0.0}
        total_velocity = base_velocity + physics_velocity
        physics_state = state + dt * physics_velocity
        total_state = state + dt * total_velocity
        projected = project_conditions(total_state, embedded_conditions, condition_mask)
        base_labels = (cosine_decode_categories(baseline_state, embedding_weight) - 1).unsqueeze(1)
        before_labels = (cosine_decode_categories(state, embedding_weight) - 1).unsqueeze(1)
        physics_labels = (cosine_decode_categories(physics_state, embedding_weight) - 1).unsqueeze(1)
        total_labels = (cosine_decode_categories(total_state, embedding_weight) - 1).unsqueeze(1)
        projected_labels = (cosine_decode_categories(projected, embedding_weight) - 1).unsqueeze(1)
        baseline_soft = soft_response(baseline_state, tau).detach()
        baseline_hard = hard_response(base_labels).detach()
        projected_soft = soft_response(projected, tau).detach()
        projected_hard = hard_response(projected_labels).detach()
        truth_loss = float((truth_response - observation).square().mean())
        soft_baseline_loss = float((baseline_soft - observation).square().mean())
        hard_baseline_loss = float((baseline_hard - observation).square().mean())
        post_soft_loss = float((projected_soft - observation).square().mean())
        post_hard_loss = float((projected_hard - observation).square().mean())
        attainment = paired_attainment(
            soft_baseline_loss=soft_baseline_loss,
            guided_soft_loss=post_soft_loss,
            soft_truth_loss=truth_loss,
            hard_baseline_loss=hard_baseline_loss,
            guided_hard_loss=post_hard_loss,
            hard_truth_loss=truth_loss,
        )
        geometry = response_distance_geometry(
            baseline=baseline_hard, guided=projected_hard, truth=truth_response
        )
        changed = projected_labels != base_labels
        row = {
            "mode": mode,
            "step": step + 1,
            "t": t,
            "dt": dt,
            "base_velocity_norm": float(base_schedule["velocity_norms"][step])
            if mode == "MATCHED_NORM_PHYSICS_ONLY" else _norm(base_velocity),
            "raw_physics_gradient_norm": _norm(raw_gradient),
            "applied_physics_velocity_norm": _norm(physics_velocity),
            "total_velocity_norm": _norm(total_velocity),
            "cosine_base_velocity_applied_physics_velocity": _cosine(
                paired_base_velocity if mode == "MATCHED_NORM_PHYSICS_ONLY" else base_velocity,
                physics_velocity,
            ),
            "pre_update_soft_loss": float(pre_soft_loss.detach()),
            "post_physics_soft_loss": float(soft_loss(physics_state, tau).detach()),
            "post_total_soft_loss": float(soft_loss(total_state, tau).detach()),
            "post_projection_soft_loss": post_soft_loss,
            "pre_update_hard_loss": float((hard_response(before_labels) - observation).square().mean()),
            "post_physics_hard_loss": float((hard_response(physics_labels) - observation).square().mean()),
            "post_total_hard_loss": float((hard_response(total_labels) - observation).square().mean()),
            "post_projection_hard_loss": post_hard_loss,
            **attainment,
            "distance_to_baseline": geometry["distance_to_baseline"],
            "distance_to_truth": geometry["distance_to_truth"],
            "truth_direction_fraction": geometry["truth_direction_fraction"],
            "orthogonal_response_fraction": geometry["orthogonal_response_fraction"],
            "hard_label_flip_count": int(changed.sum()),
            "target_flip_in": int((changed & (projected_labels == target_label)).sum()),
            "target_flip_out": int((changed & (base_labels == target_label)).sum()),
            "wrong_class_flips": int((changed & (projected_labels != target_label)).sum()),
            "condition_projection_change_norm": _norm(projected - total_state),
            "guidance_clipping_fraction": controller["guidance_cap_fraction"],
            "used_guidance_ratio": controller["used_guidance_ratio"],
        }
        counts = torch.bincount((projected_labels[changed] + 1).long(), minlength=15)
        row.update({f"flip_into_raw_label_{i-1}": int(v) for i, v in enumerate(counts.tolist())})
        trace.append(row)
        if post_soft_loss < best_soft_loss:
            best_soft_loss, best_soft_step = post_soft_loss, step + 1
        if post_hard_loss < best_hard_loss:
            best_hard_loss, best_hard_step = post_hard_loss, step + 1
            best_hard_state = projected.detach().cpu().clone()
        state = projected.detach()
    return {
        "trace": trace,
        "final_state": state.detach().cpu(),
        "best_hard_state": best_hard_state,
        "best_hard_step": best_hard_step,
        "best_soft_step": best_soft_step,
        "best_hard_loss": best_hard_loss,
    }
