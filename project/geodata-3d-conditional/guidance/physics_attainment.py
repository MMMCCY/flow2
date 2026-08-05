"""Truth-blind endpoint optimization for the Phase-6P attainment audit.

This module deliberately has no geology-truth metric dependency.  It optimizes
one continuous endpoint state against a caller-supplied differentiable physics
loss, projects exact conditions after every update, and selects checkpoints by
the caller-supplied *hard* physics loss only.
"""

from __future__ import annotations

import math
from typing import Callable, Dict, Mapping, Sequence

import torch

from guidance.generator_posterior import project_conditions


PHYSICS_ATTAINMENT_OPTIMIZER_VERSION = "endpoint_adam_hard_selection_v1"
ENDPOINT_CONDITION_POLICY = "project_after_every_adam_update_v1"
ENDPOINT_SELECTION_POLICY = "minimum_hard_physics_loss_only_v1"


SoftLoss = Callable[
    [torch.Tensor, float], tuple[torch.Tensor, Mapping[str, object]]
]
HardEvaluation = Callable[
    [torch.Tensor], tuple[Mapping[str, object], Mapping[str, torch.Tensor]]
]


def expand_temperature_schedule(
    schedule: Sequence[Mapping[str, object]],
) -> list[float]:
    """Expand frozen ``temperature/steps`` segments into one value per update."""
    if not schedule:
        raise ValueError("temperature schedule must be nonempty")
    values: list[float] = []
    for index, segment in enumerate(schedule):
        if not isinstance(segment, Mapping):
            raise ValueError(f"temperature segment {index} must be a mapping")
        temperature = float(segment.get("temperature", float("nan")))
        steps = int(segment.get("steps", 0))
        if not math.isfinite(temperature) or temperature <= 0:
            raise ValueError("temperature values must be finite and positive")
        if steps <= 0:
            raise ValueError("temperature segment steps must be positive")
        values.extend([temperature] * steps)
    return values


def field_attainment_diagnostics(
    observed: torch.Tensor,
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    sample_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> Dict[str, float | bool]:
    """Quantify whether a candidate field actually approaches the observation.

    ``attainment`` is one minus the candidate-to-baseline RMSE ratio.  The
    update/residual norm ratio and cosine diagnose whether the field update is
    both large enough and aligned with the baseline-to-observation residual.
    """
    if observed.shape != baseline.shape or observed.shape != candidate.shape:
        raise ValueError("observed, baseline and candidate fields must match")
    if sample_mask.shape != observed.shape:
        raise ValueError("sample mask must match field shape")
    if not all(
        torch.isfinite(value).all()
        for value in (observed, baseline, candidate, sample_mask)
    ):
        raise ValueError("field diagnostics require finite tensors")
    mask = sample_mask.to(device=observed.device, dtype=observed.dtype)
    if bool((mask < 0).any()) or float(mask.sum()) <= 0:
        raise ValueError("sample mask must be non-negative and nonempty")
    residual = (observed - baseline) * mask
    update = (candidate - baseline) * mask
    remaining = (observed - candidate) * mask
    denominator = mask.sum().clamp_min(eps)
    baseline_rmse = torch.sqrt(residual.square().sum() / denominator)
    candidate_rmse = torch.sqrt(remaining.square().sum() / denominator)
    residual_norm = torch.linalg.vector_norm(residual.reshape(-1))
    update_norm = torch.linalg.vector_norm(update.reshape(-1))
    cosine_denominator = (residual_norm * update_norm).clamp_min(eps)
    alignment = (
        (residual.reshape(-1) * update.reshape(-1)).sum() / cosine_denominator
    ).clamp(-1.0, 1.0)
    baseline_value = float(baseline_rmse.detach().cpu())
    candidate_value = float(candidate_rmse.detach().cpu())
    residual_value = float(residual_norm.detach().cpu())
    update_value = float(update_norm.detach().cpu())
    return {
        "baseline_rmse": baseline_value,
        "candidate_rmse": candidate_value,
        "attainment": (
            1.0 - candidate_value / baseline_value
            if baseline_value > eps
            else float("nan")
        ),
        "update_to_required_residual_norm_ratio": (
            update_value / residual_value if residual_value > eps else float("nan")
        ),
        "update_residual_cosine": float(alignment.detach().cpu()),
        "candidate_closer_to_observation_than_baseline": candidate_value
        < baseline_value,
        "candidate_closer_to_observation_than_to_baseline": candidate_value
        < float(
            torch.sqrt(update.square().sum() / denominator).detach().cpu()
        ),
    }


def project_and_clip_state(
    state: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    max_voxel_norm: float,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Apply a per-voxel norm guard and then restore exact conditions."""
    limit = float(max_voxel_norm)
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("max_voxel_norm must be finite and positive")
    if state.ndim != 5 or not state.is_floating_point():
        raise ValueError("state must be a floating [B,E,X,Y,Z] tensor")
    norm = torch.linalg.vector_norm(state, dim=1, keepdim=True)
    scale = torch.clamp(limit / norm.clamp_min(eps), max=1.0)
    clipped = state * scale
    return project_conditions(
        clipped, embedded_conditions, condition_mask
    ).contiguous()


def _numeric_diagnostics(values: Mapping[str, object]) -> Dict[str, object]:
    output: Dict[str, object] = {}
    for name, value in values.items():
        if isinstance(value, torch.Tensor):
            if value.numel() == 1:
                output[name] = float(value.detach().cpu())
        elif isinstance(value, (bool, int, float)) or value is None:
            output[name] = value
    return output


def optimize_endpoint_state(
    *,
    initial_state: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    soft_loss: SoftLoss,
    hard_evaluate: HardEvaluation,
    temperature_schedule: Sequence[Mapping[str, object]],
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    hard_check_interval: int,
    max_voxel_norm: float,
    hard_loss_key: str = "hard_loss",
    improvement_tolerance: float = 0.0,
) -> dict[str, object]:
    """Optimize one endpoint and retain the minimum hard-physics checkpoint.

    The function never sees geological truth and never selects on a soft loss.
    It returns CPU copies of the initial/best/final states and best hard payload.
    """
    temperatures = expand_temperature_schedule(temperature_schedule)
    rate = float(learning_rate)
    decay = float(weight_decay)
    clip = float(gradient_clip_norm)
    tolerance = float(improvement_tolerance)
    if not math.isfinite(rate) or rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(decay) or decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(clip) or clip <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    if hard_check_interval <= 0:
        raise ValueError("hard_check_interval must be positive")
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and non-negative")
    if not torch.isfinite(initial_state).all():
        raise ValueError("initial_state contains NaN or Inf")

    projected_initial = project_and_clip_state(
        initial_state.detach(),
        embedded_conditions,
        condition_mask,
        max_voxel_norm=max_voxel_norm,
    )
    parameter = torch.nn.Parameter(projected_initial.clone())
    optimizer = torch.optim.Adam([parameter], lr=rate, weight_decay=decay)
    if optimizer.param_groups[0]["lr"] != rate:
        raise RuntimeError("Adam learning rate was not applied")

    initial_metrics_raw, initial_payload_raw = hard_evaluate(parameter.detach())
    initial_metrics = _numeric_diagnostics(initial_metrics_raw)
    if hard_loss_key not in initial_metrics:
        raise ValueError(f"hard evaluator lacks selection key: {hard_loss_key}")
    best_loss = float(initial_metrics[hard_loss_key])
    if not math.isfinite(best_loss) or best_loss < 0:
        raise ValueError("initial hard physics loss must be finite and non-negative")
    best_state = parameter.detach().cpu().clone()
    best_metrics = dict(initial_metrics)
    best_payload = {
        name: value.detach().cpu().clone()
        for name, value in initial_payload_raw.items()
    }
    initial_payload = {
        name: value.detach().cpu().clone()
        for name, value in initial_payload_raw.items()
    }
    best_step = 0
    trace: list[Dict[str, object]] = [
        {
            "step": 0,
            "temperature": temperatures[0],
            "soft_loss": None,
            "gradient_norm_before_clip": None,
            "hard_checked": True,
            "hard_improved": True,
            "selected_best_step": 0,
            **initial_metrics,
        }
    ]

    condition = condition_mask.to(device=parameter.device, dtype=torch.bool)
    condition = condition.expand(parameter.shape[0], -1, -1, -1, -1)
    for step, temperature in enumerate(temperatures, start=1):
        optimizer.zero_grad(set_to_none=True)
        projected = project_conditions(
            parameter, embedded_conditions, condition_mask
        )
        loss, soft_diagnostics_raw = soft_loss(projected, temperature)
        if loss.ndim != 0 or not torch.isfinite(loss):
            raise FloatingPointError("soft physics loss must be one finite scalar")
        loss.backward()
        if parameter.grad is None or not torch.isfinite(parameter.grad).all():
            raise FloatingPointError("endpoint gradient is absent or non-finite")
        parameter.grad.masked_fill_(condition.expand_as(parameter.grad), 0.0)
        gradient_norm = torch.nn.utils.clip_grad_norm_([parameter], clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError("endpoint gradient norm is non-finite")
        optimizer.step()
        with torch.no_grad():
            parameter.copy_(
                project_and_clip_state(
                    parameter,
                    embedded_conditions,
                    condition_mask,
                    max_voxel_norm=max_voxel_norm,
                )
            )
        if not torch.isfinite(parameter).all():
            raise FloatingPointError("endpoint state contains NaN or Inf")

        hard_checked = (
            step == 1
            or step % int(hard_check_interval) == 0
            or step == len(temperatures)
        )
        row: Dict[str, object] = {
            "step": step,
            "temperature": temperature,
            "soft_loss": float(loss.detach().cpu()),
            "gradient_norm_before_clip": float(gradient_norm.detach().cpu()),
            "hard_checked": hard_checked,
            "hard_improved": False,
            "selected_best_step": best_step,
            **{
                f"soft_{name}": value
                for name, value in _numeric_diagnostics(
                    soft_diagnostics_raw
                ).items()
            },
        }
        if hard_checked:
            hard_metrics_raw, hard_payload_raw = hard_evaluate(parameter.detach())
            hard_metrics = _numeric_diagnostics(hard_metrics_raw)
            if hard_loss_key not in hard_metrics:
                raise ValueError(
                    f"hard evaluator lacks selection key: {hard_loss_key}"
                )
            hard_loss = float(hard_metrics[hard_loss_key])
            if not math.isfinite(hard_loss) or hard_loss < 0:
                raise FloatingPointError("hard physics loss is invalid")
            improved = hard_loss < best_loss - tolerance
            if improved:
                best_loss = hard_loss
                best_state = parameter.detach().cpu().clone()
                best_metrics = dict(hard_metrics)
                best_payload = {
                    name: value.detach().cpu().clone()
                    for name, value in hard_payload_raw.items()
                }
                best_step = step
            row.update(hard_metrics)
            row["hard_improved"] = improved
            row["selected_best_step"] = best_step
        trace.append(row)

    return {
        "optimizer_version": PHYSICS_ATTAINMENT_OPTIMIZER_VERSION,
        "condition_policy": ENDPOINT_CONDITION_POLICY,
        "selection_policy": ENDPOINT_SELECTION_POLICY,
        "initial_state": projected_initial.detach().cpu(),
        "final_state": parameter.detach().cpu(),
        "best_state": best_state,
        "initial_metrics": initial_metrics,
        "initial_payload": initial_payload,
        "best_metrics": best_metrics,
        "best_payload": best_payload,
        "best_step": best_step,
        "updates_completed": len(temperatures),
        "hard_evaluations": sum(bool(row["hard_checked"]) for row in trace),
        "trace": trace,
    }
