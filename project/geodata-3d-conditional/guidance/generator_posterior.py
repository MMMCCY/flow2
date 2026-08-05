"""Direct posterior search in the frozen conditional-flow initial-noise space.

The utilities here deliberately know nothing about geological truth metrics.
They provide the deterministic projected generator, a pCN proposal that
preserves the standard-Gaussian initial-state prior, and the Metropolis
decision used by the Phase-5c runner.
"""

from __future__ import annotations

import math
from typing import Dict

import torch


GENERATOR_POSTERIOR_VERSION = "pcn_full_initial_noise_hard_seismic_v1"
CONDITION_PROJECTION_POLICY = (
    "project_clean_embedding_before_first_and_after_every_euler_step_v1"
)


def project_conditions(
    state: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    """Insert exact clean embeddings at all hard-condition voxels."""
    if state.ndim != 5:
        raise ValueError("state must have shape [B,E,X,Y,Z]")
    values = embedded_conditions.to(device=state.device, dtype=state.dtype)
    mask = condition_mask.to(device=state.device, dtype=torch.bool)
    if values.ndim != 5 or values.shape[1:] != state.shape[1:]:
        raise ValueError("embedded_conditions must match state channel/spatial shape")
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("condition_mask must have shape [B,1,X,Y,Z]")
    if mask.shape[2:] != state.shape[2:]:
        raise ValueError("condition_mask spatial shape must match state")
    if values.shape[0] not in (1, state.shape[0]) or mask.shape[0] not in (
        1,
        state.shape[0],
    ):
        raise ValueError("condition batch must be one or match state batch")
    values = values.expand(state.shape[0], -1, -1, -1, -1)
    mask = mask.expand(state.shape[0], -1, -1, -1, -1)
    return torch.where(mask.expand_as(state), values, state)


def projected_fixed_euler_prior_sample(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    n_steps: int,
) -> torch.Tensor:
    """Evaluate the frozen conditional generator with exact hard projection.

    This is the explicit no-guidance trajectory used by Phase 2/4.  Keeping it
    here prevents pCN from paying for a diagnostic loss at every Euler step.
    """
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    if initial_state.ndim != 5:
        raise ValueError("initial_state must have shape [B,E,X,Y,Z]")
    if conditioning.ndim != 5 or conditioning.shape[1:] != initial_state.shape[1:]:
        raise ValueError("conditioning must match initial_state channel/spatial shape")

    state = project_conditions(
        initial_state.detach(), embedded_conditions, condition_mask
    )
    condition = conditioning.to(device=state.device, dtype=state.dtype)
    if condition.shape[0] not in (1, state.shape[0]):
        raise ValueError("conditioning batch must be one or match state batch")
    condition = condition.expand(state.shape[0], -1, -1, -1, -1)
    dt = 1.0 / int(n_steps)
    with torch.no_grad():
        for step in range(int(n_steps)):
            time = torch.full(
                (state.shape[0],),
                (step + 0.5) / int(n_steps),
                device=state.device,
                dtype=state.dtype,
            )
            velocity = model.net(state, condition, time)
            state = project_conditions(
                state + dt * velocity,
                embedded_conditions,
                condition_mask,
            )
    return state


def pcn_proposal(
    current: torch.Tensor,
    *,
    beta: float,
    generator: torch.Generator,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return a pCN proposal and its standard-Gaussian innovation.

    Phase 5c intentionally creates innovations on CPU so runs remain auditable
    across the same CPU random generator policy as existing paired baselines.
    ``beta == 0`` is an explicit exact-regression branch and does not advance
    the random generator.
    """
    beta_value = float(beta)
    if not math.isfinite(beta_value) or not 0.0 <= beta_value <= 1.0:
        raise ValueError("beta must be finite and in [0,1]")
    if current.device.type != "cpu":
        raise ValueError("Phase-5c pCN latent tensors must remain on CPU")
    if not torch.is_floating_point(current) or not torch.isfinite(current).all():
        raise ValueError("current latent must be a finite floating tensor")
    if beta_value == 0.0:
        return current.clone(), torch.zeros_like(current)
    innovation = torch.randn(
        current.shape,
        generator=generator,
        device="cpu",
        dtype=current.dtype,
    )
    retained = math.sqrt(max(0.0, 1.0 - beta_value * beta_value))
    proposal = retained * current + beta_value * innovation
    return proposal, innovation


def posterior_energy(hard_seismic_loss: float, likelihood_weight: float) -> float:
    """Convert normalized hard seismic MSE to the frozen Phase-5c energy."""
    loss = float(hard_seismic_loss)
    weight = float(likelihood_weight)
    if not math.isfinite(loss) or loss < 0:
        raise ValueError("hard_seismic_loss must be finite and non-negative")
    if not math.isfinite(weight) or weight <= 0:
        raise ValueError("likelihood_weight must be finite and positive")
    return 0.5 * weight * loss


def metropolis_decision(
    current_energy: float,
    proposed_energy: float,
    uniform: float,
) -> Dict[str, float | bool]:
    """Apply the likelihood-only MH decision for the prior-preserving pCN move."""
    current = float(current_energy)
    proposed = float(proposed_energy)
    draw = float(uniform)
    if not math.isfinite(current) or not math.isfinite(proposed):
        raise ValueError("posterior energies must be finite")
    if not math.isfinite(draw) or not 0.0 < draw < 1.0:
        raise ValueError("uniform must lie strictly inside (0,1)")
    log_ratio = current - proposed
    log_uniform = math.log(draw)
    accepted = log_uniform < min(0.0, log_ratio)
    return {
        "accepted": accepted,
        "log_acceptance_ratio": log_ratio,
        "log_uniform": log_uniform,
        "acceptance_probability": math.exp(min(0.0, log_ratio)),
    }


def latent_diagnostics(value: torch.Tensor) -> Dict[str, float]:
    """Return scale diagnostics without treating a huge norm as a loss term."""
    if not torch.is_floating_point(value) or not torch.isfinite(value).all():
        raise ValueError("latent must be a finite floating tensor")
    flattened = value.detach().double().reshape(-1)
    return {
        "latent_l2": float(torch.linalg.vector_norm(flattened).item()),
        "latent_rms": float(flattened.square().mean().sqrt().item()),
        "latent_mean": float(flattened.mean().item()),
        "latent_std": float(flattened.std(unbiased=False).item()),
    }

