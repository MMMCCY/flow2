"""Projected fixed-Euler sampling adapter for Phase-4 gravity guidance."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .gravity import (
    GRAVITY_LOSS_MODE,
    RectangularPrismGravity,
    gravity_volume_loss,
)
from .property_sampling import fixed_euler_property_sample
from .probability_sampling import REFERENCE_GUIDANCE_SCALING_MODE


GRAVITY_SAMPLER_VERSION = "projected_fixed_euler_gravity_v1"


def _gravity_loss_adapter(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_properties: torch.Tensor,
    property_table: torch.Tensor,
    confidence: torch.Tensor,
    *,
    tau: float,
    sigmas: Sequence[float],
    scale_weights: Sequence[float],
    channel_weights: torch.Tensor,
    condition_mask: torch.Tensor,
    forward_operator: RectangularPrismGravity,
    observed_mgal: torch.Tensor,
    survey_mask: torch.Tensor,
    uncertainty_mgal: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the proven sampler loss interface without changing its solver."""
    del confidence, sigmas, scale_weights, channel_weights
    if property_table.ndim != 2 or property_table.shape[0] != 1:
        raise ValueError("gravity property_table adapter requires shape [1,C]")
    return gravity_volume_loss(
        state,
        embedding_weight,
        property_table[0],
        target_properties,
        condition_mask,
        forward_operator,
        observed_mgal,
        survey_mask,
        uncertainty_mgal,
        tau=tau,
    )


def fixed_euler_gravity_sample(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_truth: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
    target_density: torch.Tensor,
    density_table: torch.Tensor,
    guidance_confidence: torch.Tensor,
    forward_operator: RectangularPrismGravity,
    observed_mgal: torch.Tensor,
    survey_mask: torch.Tensor,
    uncertainty_mgal: torch.Tensor,
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
) -> tuple[torch.Tensor, List[dict[str, object]]]:
    """Run gravity guidance through the existing paired projected solver.

    The gravity-specific tensors enter only through the injected loss.  The
    ``alpha=0`` branch, time grid, controller and condition projection are the
    same implementation used by the validated Phase-2/3 sampler.
    """
    if density_table.ndim != 1:
        raise ValueError("density_table must have shape [C]")
    return fixed_euler_property_sample(
        model=model,
        initial_state=initial_state,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth_model,
        condition_mask=condition_mask,
        target_properties=target_density,
        property_table=density_table.unsqueeze(0),
        confidence=guidance_confidence,
        property_sigmas=(0.0,),
        property_scale_weights=(1.0,),
        property_channel_weights=torch.ones(
            1, device=density_table.device, dtype=density_table.dtype
        ),
        n_steps=n_steps,
        alpha=alpha,
        max_guidance_ratio=max_guidance_ratio,
        tau_start=tau_start,
        tau_end=tau_end,
        tau_schedule=tau_schedule,
        guidance_start=guidance_start,
        guidance_schedule=guidance_schedule,
        grad_clip_norm=grad_clip_norm,
        guidance_scaling_mode=guidance_scaling_mode,
        sample_id=sample_id,
        loss_function=_gravity_loss_adapter,
        loss_extra_kwargs={
            "condition_mask": condition_mask,
            "forward_operator": forward_operator,
            "observed_mgal": observed_mgal,
            "survey_mask": survey_mask,
            "uncertainty_mgal": uncertainty_mgal,
        },
        loss_mode=GRAVITY_LOSS_MODE,
    )
