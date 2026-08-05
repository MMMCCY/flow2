"""Lightweight geophysics-conditioned residual velocity adapter for Phase 6."""

from __future__ import annotations

import math
from typing import Dict, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from .generator_posterior import project_conditions


RESIDUAL_ADAPTER_VERSION = "frozen_flow_residual_velocity_adapter_v1"


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2):
        if channels % groups == 0:
            return groups
    return 1


class _TimeResidualBlock(nn.Module):
    def __init__(self, channels: int, time_channels: int, dilation: int) -> None:
        super().__init__()
        if dilation <= 0:
            raise ValueError("dilation must be positive")
        groups = _group_count(channels)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv1 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )
        self.norm2 = nn.GroupNorm(groups, channels)
        self.time_affine = nn.Linear(time_channels, 2 * channels)
        self.conv2 = nn.Conv3d(
            channels,
            channels,
            kernel_size=3,
            padding=dilation,
            dilation=dilation,
        )

    def forward(self, value: torch.Tensor, time_embedding: torch.Tensor) -> torch.Tensor:
        hidden = self.conv1(F.silu(self.norm1(value)))
        scale, shift = self.time_affine(time_embedding).chunk(2, dim=1)
        hidden = self.norm2(hidden)
        hidden = hidden * (1.0 + scale[:, :, None, None, None])
        hidden = hidden + shift[:, :, None, None, None]
        hidden = self.conv2(F.silu(hidden))
        return value + hidden


class ResidualVelocityAdapter(nn.Module):
    """Predict an external velocity correction without modifying the base U-Net."""

    def __init__(
        self,
        embedding_channels: int,
        *,
        geophysics_channels: int,
        base_width: int = 12,
        dilations: Sequence[int] = (1, 2, 4, 1),
    ) -> None:
        super().__init__()
        if embedding_channels <= 0 or geophysics_channels <= 0 or base_width <= 0:
            raise ValueError("adapter channel counts must be positive")
        if not dilations:
            raise ValueError("adapter requires at least one residual block")
        self.embedding_channels = int(embedding_channels)
        self.geophysics_channels = int(geophysics_channels)
        self.base_width = int(base_width)
        self.dilations = tuple(int(value) for value in dilations)
        input_channels = 3 * self.embedding_channels + 1 + self.geophysics_channels
        self.input_conv = nn.Conv3d(input_channels, self.base_width, kernel_size=3, padding=1)
        time_channels = 4 * self.base_width
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_channels),
            nn.SiLU(),
            nn.Linear(time_channels, time_channels),
        )
        self.blocks = nn.ModuleList(
            _TimeResidualBlock(self.base_width, time_channels, dilation)
            for dilation in self.dilations
        )
        self.output_norm = nn.GroupNorm(_group_count(self.base_width), self.base_width)
        self.output_conv = nn.Conv3d(
            self.base_width, self.embedding_channels, kernel_size=1
        )
        nn.init.zeros_(self.output_conv.weight)
        nn.init.zeros_(self.output_conv.bias)

    def forward(
        self,
        state: torch.Tensor,
        base_velocity: torch.Tensor,
        conditioning: torch.Tensor,
        condition_mask: torch.Tensor,
        geophysics: torch.Tensor,
        time: torch.Tensor,
    ) -> torch.Tensor:
        if state.ndim != 5 or state.shape[1] != self.embedding_channels:
            raise ValueError("state has incompatible adapter shape")
        for name, value in (
            ("base_velocity", base_velocity),
            ("conditioning", conditioning),
        ):
            if value.shape != state.shape:
                raise ValueError(f"{name} must match state")
        if condition_mask.ndim != 5 or condition_mask.shape[1] != 1:
            raise ValueError("condition_mask must have shape [B,1,X,Y,Z]")
        if geophysics.ndim != 5 or geophysics.shape[1] != self.geophysics_channels:
            raise ValueError("geophysics has incompatible adapter shape")
        if condition_mask.shape[0] not in (1, state.shape[0]):
            raise ValueError("condition_mask batch must be one or match state")
        if geophysics.shape[0] not in (1, state.shape[0]):
            raise ValueError("geophysics batch must be one or match state")
        if condition_mask.shape[2:] != state.shape[2:] or geophysics.shape[2:] != state.shape[2:]:
            raise ValueError("adapter spatial inputs must match state")
        if time.ndim != 1 or time.shape[0] != state.shape[0]:
            raise ValueError("time must have shape [B]")

        mask = condition_mask.to(device=state.device, dtype=state.dtype).expand(
            state.shape[0], -1, -1, -1, -1
        )
        geo = geophysics.to(device=state.device, dtype=state.dtype).expand(
            state.shape[0], -1, -1, -1, -1
        )
        combined = torch.cat(
            (state, base_velocity, conditioning, mask, geo), dim=1
        )
        hidden = self.input_conv(combined)
        time_embedding = self.time_mlp(time[:, None].to(dtype=state.dtype))
        for block in self.blocks:
            hidden = block(hidden, time_embedding)
        correction = self.output_conv(F.silu(self.output_norm(hidden)))
        return correction * (1.0 - mask)

    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def cap_residual_velocity(
    correction: torch.Tensor,
    base_velocity: torch.Tensor,
    condition_mask: torch.Tensor,
    *,
    max_ratio: float,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Cap each sample's free-region correction norm relative to EMA velocity."""
    ratio = float(max_ratio)
    if not math.isfinite(ratio) or ratio < 0:
        raise ValueError("max_ratio must be finite and non-negative")
    if correction.shape != base_velocity.shape:
        raise ValueError("correction and base_velocity must match")
    mask = condition_mask.to(device=correction.device, dtype=torch.bool)
    if mask.ndim != 5 or mask.shape[1] != 1 or mask.shape[2:] != correction.shape[2:]:
        raise ValueError("condition_mask shape is invalid")
    free = (~mask).expand_as(correction)
    correction_free = torch.where(free, correction, torch.zeros_like(correction))
    base_free = torch.where(free, base_velocity, torch.zeros_like(base_velocity))
    correction_norm = correction_free.flatten(1).norm(dim=1)
    base_norm = base_free.flatten(1).norm(dim=1)
    if ratio == 0.0:
        factors = torch.zeros_like(correction_norm)
    else:
        factors = torch.minimum(
            torch.ones_like(correction_norm),
            ratio * base_norm / correction_norm.clamp_min(eps),
        )
    capped = correction_free * factors[:, None, None, None, None]
    used_ratio = capped.flatten(1).norm(dim=1) / base_norm.clamp_min(eps)
    return capped, used_ratio


def class_balancing_weights(
    truth: torch.Tensor,
    active_mask: torch.Tensor,
    num_categories: int,
    *,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Return inverse-square-root weights normalized over present classes."""
    labels = truth.squeeze(1).long() + 1
    active = active_mask.squeeze(1).bool()
    counts = torch.bincount(labels[active], minlength=num_categories).float()
    present = counts > 0
    weights = torch.zeros(num_categories, device=truth.device, dtype=torch.float32)
    weights[present] = counts[present].clamp_min(eps).rsqrt()
    if bool(present.any()):
        weights[present] /= weights[present].mean()
    return weights


def cosine_class_logits(
    endpoint: torch.Tensor,
    embedding_weight: torch.Tensor,
    *,
    temperature: float,
) -> torch.Tensor:
    value = float(temperature)
    if not math.isfinite(value) or value <= 0:
        raise ValueError("logit temperature must be finite and positive")
    normalized = F.normalize(endpoint, dim=1)
    embeddings = F.normalize(
        embedding_weight.to(device=endpoint.device, dtype=endpoint.dtype), dim=1
    )
    return torch.einsum("bexyz,ce->bcxyz", normalized, embeddings) / value


def residual_adapter_losses(
    *,
    state: torch.Tensor,
    target_velocity: torch.Tensor,
    base_velocity: torch.Tensor,
    correction: torch.Tensor,
    truth: torch.Tensor,
    condition_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    time: torch.Tensor,
    class_weights: torch.Tensor,
    logit_temperature: float,
    flow_weight: float,
    cross_entropy_weight: float,
    dice_weight: float,
    residual_regularizer_weight: float,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
    """Compute flow and all-class endpoint objectives on free subsurface voxels."""
    if not (state.shape == target_velocity.shape == base_velocity.shape == correction.shape):
        raise ValueError("adapter loss velocity/state tensors must match")
    if time.ndim != 1 or time.shape[0] != state.shape[0]:
        raise ValueError("time must have shape [B]")
    truth_value = truth.to(device=state.device).long()
    condition = condition_mask.to(device=state.device, dtype=torch.bool)
    active = (~condition) & (truth_value != -1)
    if not bool(active.any()):
        raise ValueError("adapter loss has no unconstrained subsurface voxels")
    active_channels = active.expand_as(state)
    adapted_velocity = base_velocity + correction
    squared_error = (adapted_velocity - target_velocity).square()
    target_energy = target_velocity.square()
    flow_loss = squared_error[active_channels].mean() / (
        target_energy[active_channels].mean() + eps
    )
    residual_loss = correction.square()[active_channels].mean() / (
        base_velocity.square()[active_channels].mean() + eps
    )
    time_broadcast = time.view(-1, 1, 1, 1, 1).to(dtype=state.dtype)
    endpoint = state + (1.0 - time_broadcast) * adapted_velocity
    logits = cosine_class_logits(
        endpoint, embedding_weight, temperature=logit_temperature
    )
    labels = truth_value.squeeze(1) + 1
    active_flat = active.squeeze(1)
    selected_logits = logits.permute(0, 2, 3, 4, 1)[active_flat]
    selected_labels = labels[active_flat]
    ce_loss = F.cross_entropy(
        selected_logits,
        selected_labels,
        weight=class_weights.to(device=state.device, dtype=state.dtype),
    )

    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(labels.clamp_min(0), num_classes=logits.shape[1]).permute(
        0, 4, 1, 2, 3
    ).to(dtype=probabilities.dtype)
    active_float = active.to(dtype=probabilities.dtype)
    intersection = (probabilities * one_hot * active_float).sum(dim=(0, 2, 3, 4))
    denominator = ((probabilities + one_hot) * active_float).sum(dim=(0, 2, 3, 4))
    present = one_hot.mul(active_float).sum(dim=(0, 2, 3, 4)) > 0
    dice = (2.0 * intersection + eps) / (denominator + eps)
    dice_loss = 1.0 - dice[present].mean()
    endpoint_prediction = logits.argmax(dim=1)
    endpoint_accuracy = (
        endpoint_prediction[active_flat] == selected_labels
    ).float().mean()
    total = (
        float(flow_weight) * flow_loss
        + float(cross_entropy_weight) * ce_loss
        + float(dice_weight) * dice_loss
        + float(residual_regularizer_weight) * residual_loss
    )
    diagnostics = {
        "total_loss": total,
        "flow_loss": flow_loss,
        "cross_entropy_loss": ce_loss,
        "dice_loss": dice_loss,
        "residual_regularizer": residual_loss,
        "endpoint_accuracy": endpoint_accuracy,
        "endpoint": endpoint,
        "adapted_velocity": adapted_velocity,
    }
    return total, diagnostics


def fixed_euler_adapter_sample(
    *,
    model,
    adapter: ResidualVelocityAdapter,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    geophysics: torch.Tensor,
    n_steps: int,
    adapter_scale: float,
    max_residual_ratio: float,
) -> tuple[torch.Tensor, list[Dict[str, float]]]:
    """Sample the frozen flow with an external condition-zero residual adapter."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    scale = float(adapter_scale)
    if not math.isfinite(scale) or scale < 0:
        raise ValueError("adapter_scale must be finite and non-negative")
    state = project_conditions(
        initial_state.detach(), embedded_conditions, condition_mask
    )
    condition = conditioning.to(device=state.device, dtype=state.dtype).expand(
        state.shape[0], -1, -1, -1, -1
    )
    dt = 1.0 / int(n_steps)
    trace: list[Dict[str, float]] = []
    with torch.no_grad():
        for step in range(int(n_steps)):
            time = torch.full(
                (state.shape[0],),
                (step + 0.5) / int(n_steps),
                device=state.device,
                dtype=state.dtype,
            )
            base_velocity = model.net(state, condition, time)
            if scale == 0.0:
                correction = torch.zeros_like(base_velocity)
                used_ratio = torch.zeros(state.shape[0], device=state.device)
            else:
                raw = adapter(
                    state,
                    base_velocity,
                    condition,
                    condition_mask,
                    geophysics,
                    time,
                )
                correction, used_ratio = cap_residual_velocity(
                    raw,
                    base_velocity,
                    condition_mask,
                    max_ratio=max_residual_ratio,
                )
                correction = scale * correction
            state = project_conditions(
                state + dt * (base_velocity + correction),
                embedded_conditions,
                condition_mask,
            )
            trace.append(
                {
                    "step": float(step),
                    "t": float(time[0].item()),
                    "adapter_scale": scale,
                    "used_residual_ratio": float(used_ratio.mean().item()),
                }
            )
    return state, trace

