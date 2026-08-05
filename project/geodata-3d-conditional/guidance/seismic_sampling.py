"""Projected fixed-Euler sampling adapter for Phase-4c seismic guidance."""

from __future__ import annotations

from typing import List, Sequence

import torch

from .property_sampling import fixed_euler_property_sample
from .probability_sampling import REFERENCE_GUIDANCE_SCALING_MODE
from .seismic import (
    SEISMIC_LOSS_MODE,
    ConvolutionalSeismic,
    seismic_volume_loss,
)


SEISMIC_SAMPLER_VERSION = "projected_fixed_euler_seismic_v1"


def _seismic_loss_adapter(
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
    subsurface_mask: torch.Tensor,
    forward_operator: ConvolutionalSeismic,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Match the validated injected-loss sampler without changing its solver."""
    del confidence, sigmas, scale_weights, channel_weights
    if property_table.ndim != 2 or property_table.shape[0] != 2:
        raise ValueError("seismic property_table adapter requires shape [2,C]")
    return seismic_volume_loss(
        state,
        embedding_weight,
        property_table,
        target_properties,
        condition_mask,
        subsurface_mask,
        forward_operator,
        observed,
        sample_mask,
        uncertainty,
        tau=tau,
    )


def fixed_euler_seismic_sample(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_truth: torch.Tensor,
    truth_model: torch.Tensor,
    condition_mask: torch.Tensor,
    target_acoustic: torch.Tensor,
    property_table: torch.Tensor,
    guidance_confidence: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator: ConvolutionalSeismic,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
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
    """Run seismic guidance through the existing projected fixed-Euler solver."""
    if property_table.ndim != 2 or property_table.shape[0] != 2:
        raise ValueError("property_table must have shape [2,C]")
    return fixed_euler_property_sample(
        model=model,
        initial_state=initial_state,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_model=truth_model,
        condition_mask=condition_mask,
        target_properties=target_acoustic,
        property_table=property_table,
        confidence=guidance_confidence,
        property_sigmas=(0.0,),
        property_scale_weights=(1.0,),
        property_channel_weights=torch.ones(
            2, device=property_table.device, dtype=property_table.dtype
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
        loss_function=_seismic_loss_adapter,
        loss_extra_kwargs={
            "condition_mask": condition_mask,
            "subsurface_mask": subsurface_mask,
            "forward_operator": forward_operator,
            "observed": observed,
            "sample_mask": sample_mask,
            "uncertainty": uncertainty,
        },
        loss_mode=SEISMIC_LOSS_MODE,
    )
