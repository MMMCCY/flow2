"""Finite-difference and applied-update audits for Stage6Q causal physics."""

from __future__ import annotations

import math
from typing import Callable, Sequence

import torch
import torch.nn.functional as F


GRADIENT_AUDIT_VERSION = "phase6q_causality_gradient_audit_v1"
ScalarObjective = Callable[[torch.Tensor], torch.Tensor]
ResponseFunction = Callable[[torch.Tensor], torch.Tensor]
HardLossFunction = Callable[[torch.Tensor], float]


def cosine_decode_categories(
    state: torch.Tensor, embedding_weight: torch.Tensor
) -> torch.Tensor:
    """Return the exact category argmax used by ``Geo3DStochInterp.decode``."""
    state_norm = F.normalize(state, dim=1)
    embedding_norm = F.normalize(
        embedding_weight.to(device=state.device, dtype=state.dtype), dim=1
    )
    logits = torch.einsum("bexyz,ce->bcxyz", state_norm, embedding_norm)
    return logits.argmax(dim=1)


def normalize_direction(direction: torch.Tensor, *, eps: float = 1e-15) -> torch.Tensor:
    norm = torch.linalg.vector_norm(direction)
    if not torch.isfinite(norm) or float(norm) <= eps:
        raise ValueError("direction must have a positive finite norm")
    return direction / norm


def finite_difference_directional_audit(
    *,
    name: str,
    objective: ScalarObjective,
    state: torch.Tensor,
    direction: torch.Tensor,
    epsilons: Sequence[float],
    negative_gradient_step_norm: float,
    relative_floor: float = 1e-15,
) -> tuple[dict[str, object], list[dict[str, object]], torch.Tensor]:
    """Compare central finite differences with autograd along one unit direction."""
    if not epsilons:
        raise ValueError("epsilons must be non-empty")
    if state.dtype != torch.float64:
        raise ValueError("the primary finite-difference state must be float64")
    x = state.detach().clone().requires_grad_(True)
    p = normalize_direction(direction.detach().to(device=x.device, dtype=x.dtype))
    loss = objective(x)
    if loss.ndim != 0 or not torch.isfinite(loss):
        raise FloatingPointError(f"{name}: objective must be one finite scalar")
    gradient = torch.autograd.grad(loss, x)[0]
    if not torch.isfinite(gradient).all():
        raise FloatingPointError(f"{name}: gradient is non-finite")
    gradient_norm = torch.linalg.vector_norm(gradient)
    autograd_directional = torch.sum(gradient * p)
    rows: list[dict[str, object]] = []
    with torch.no_grad():
        for epsilon_value in epsilons:
            epsilon = float(epsilon_value)
            if not math.isfinite(epsilon) or epsilon <= 0:
                raise ValueError("epsilons must be positive and finite")
            plus = objective(x + epsilon * p)
            minus = objective(x - epsilon * p)
            finite_difference = (plus - minus) / (2.0 * epsilon)
            absolute_error = torch.abs(finite_difference - autograd_directional)
            scale = max(
                abs(float(finite_difference)),
                abs(float(autograd_directional)),
                float(relative_floor),
            )
            rows.append(
                {
                    "chain": name,
                    "epsilon": epsilon,
                    "finite_difference": float(finite_difference),
                    "autograd_directional_derivative": float(autograd_directional),
                    "absolute_error": float(absolute_error),
                    "relative_error": float(absolute_error) / scale,
                    "sign_match": bool(
                        float(finite_difference) == 0.0
                        or float(autograd_directional) == 0.0
                        or math.copysign(1.0, float(finite_difference))
                        == math.copysign(1.0, float(autograd_directional))
                    ),
                    "grad_norm": float(gradient_norm),
                    "loss_before": float(loss.detach()),
                }
            )
        if float(gradient_norm) > relative_floor:
            negative_step = (
                float(negative_gradient_step_norm) * gradient / gradient_norm
            )
            loss_after = objective(x - negative_step)
        else:
            loss_after = loss
    best = min(rows, key=lambda row: float(row["relative_error"]))
    summary = {
        "chain": name,
        "loss_before": float(loss.detach()),
        "loss_after_small_negative_gradient_step": float(loss_after),
        "negative_gradient_local_descent": bool(
            float(loss_after.detach()) < float(loss.detach())
        ),
        "grad_norm": float(gradient_norm),
        "best_epsilon": best["epsilon"],
        "best_relative_error": best["relative_error"],
        "all_sign_match": all(bool(row["sign_match"]) for row in rows),
    }
    for row in rows:
        row["loss_after_small_negative_gradient_step"] = float(loss_after)
    return summary, rows, gradient.detach()


def response_truth_direction_fraction(
    baseline: torch.Tensor,
    candidate: torch.Tensor,
    truth: torch.Tensor,
    *,
    eps: float = 1e-15,
) -> tuple[float, float]:
    """Project a response update onto the baseline-to-truth direction."""
    if not (baseline.shape == candidate.shape == truth.shape):
        raise ValueError("response tensors must share a shape")
    b = baseline.detach().double().reshape(-1)
    g = candidate.detach().double().reshape(-1)
    t = truth.detach().double().reshape(-1)
    direction = t - b
    update = g - b
    denominator = torch.dot(direction, direction)
    if float(denominator) <= eps:
        return float("nan"), float("nan")
    fraction = torch.dot(update, direction) / denominator
    orthogonal = update - fraction * direction
    orthogonal_fraction = torch.linalg.vector_norm(orthogonal) / torch.linalg.vector_norm(
        direction
    ).clamp_min(eps)
    return float(fraction), float(orthogonal_fraction)


def update_semantics_row(
    *,
    name: str,
    state: torch.Tensor,
    update: torch.Tensor,
    objective: ScalarObjective,
    soft_response: ResponseFunction,
    hard_loss: HardLossFunction,
    baseline_response: torch.Tensor,
    truth_response: torch.Tensor,
) -> dict[str, object]:
    """Evaluate one concrete state update in soft, hard and response geometry."""
    with torch.no_grad():
        before_soft_loss = objective(state)
        candidate = state + update
        after_soft_loss = objective(candidate)
        response = soft_response(candidate)
        truth_fraction, orthogonal_fraction = response_truth_direction_fraction(
            baseline_response, response, truth_response
        )
    return {
        "update": name,
        "update_norm": float(torch.linalg.vector_norm(update)),
        "soft_loss_before": float(before_soft_loss),
        "soft_loss_after": float(after_soft_loss),
        "soft_loss_delta": float(after_soft_loss - before_soft_loss),
        "soft_loss_improved": bool(float(after_soft_loss) < float(before_soft_loss)),
        "hard_loss_before": float(hard_loss(state)),
        "hard_loss_after": float(hard_loss(candidate)),
        "truth_direction_response_fraction": truth_fraction,
        "orthogonal_response_fraction": orthogonal_fraction,
    }
