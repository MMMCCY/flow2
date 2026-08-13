"""Truth-blind hard-binary seismic inversion primitives for Stage15.

The optimized state is a binary label-9 occupancy.  Its forward value is
always hard (zero or one), while the straight-through estimator carries
gradients through the associated sigmoid probability.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Mapping

import torch
import torch.nn.functional as F


STAGE15_BINARY_ACOUSTIC_SCHEMA = "stage15_binary_acoustic_upper_bound_v1"
TAU_SCHEDULES = ("linear", "cosine")


class _ExactHardSigmoidSTE(torch.autograd.Function):
    """Bit-exact hard forward with the derivative of ``sigmoid(logits/tau)``."""

    @staticmethod
    def forward(ctx, logits: torch.Tensor, tau: float) -> torch.Tensor:
        probability = torch.sigmoid(logits / float(tau))
        ctx.save_for_backward(probability)
        ctx.tau = float(tau)
        return (probability >= 0.5).to(probability.dtype)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        (probability,) = ctx.saved_tensors
        derivative = probability * (1.0 - probability) / ctx.tau
        return gradient * derivative, None


@dataclass(frozen=True)
class BinaryAcousticProperties:
    air_density: float
    air_velocity: float
    background_density: float
    background_velocity: float
    target_density: float
    target_velocity: float

    @property
    def air_impedance(self) -> float:
        return self.air_density * self.air_velocity

    @property
    def background_impedance(self) -> float:
        return self.background_density * self.background_velocity

    @property
    def target_impedance(self) -> float:
        return self.target_density * self.target_velocity


def binary_acoustic_properties_from_configs(
    binary_config: Mapping[str, object],
    source_config: Mapping[str, object],
) -> BinaryAcousticProperties:
    """Select raw -1/0/9 properties from the immutable Phase4 codebook."""
    if binary_config.get("schema") != STAGE15_BINARY_ACOUSTIC_SCHEMA:
        raise ValueError(f"binary acoustic schema must be {STAGE15_BINARY_ACOUSTIC_SCHEMA!r}")
    if binary_config.get("mapping_policy") != "air_raw_minus1_background_raw0_target_raw9_v1":
        raise ValueError("invalid Stage15 binary mapping policy")
    labels = binary_config.get("selected_raw_labels")
    if labels != {"air": -1, "background": 0, "target": 9}:
        raise ValueError("Stage15 binary labels must be exactly air=-1/background=0/target=9")
    values = source_config.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("source acoustic config has no values mapping")

    def pair(raw_label: int) -> tuple[float, float]:
        entry = values.get(str(raw_label))
        if not isinstance(entry, Mapping):
            raise ValueError(f"source acoustic config is missing raw label {raw_label}")
        density = float(entry["density"])
        velocity = float(entry["vp"])
        if not all(math.isfinite(value) and value > 0 for value in (density, velocity)):
            raise ValueError("binary acoustic properties must be finite and positive")
        return density, velocity

    air = pair(-1)
    background = pair(0)
    target = pair(9)
    return BinaryAcousticProperties(
        air_density=air[0],
        air_velocity=air[1],
        background_density=background[0],
        background_velocity=background[1],
        target_density=target[0],
        target_velocity=target[1],
    )


def _validate_single_channel(value: torch.Tensor, name: str) -> None:
    if value.ndim != 5 or value.shape[1] != 1:
        raise ValueError(f"{name} must have shape [B,1,X,Y,Z]")


def binary_occupancy_to_acoustic(
    occupancy: torch.Tensor,
    subsurface_mask: torch.Tensor,
    properties: BinaryAcousticProperties,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Map binary occupancy to impedance/slowness, using air outside support."""
    _validate_single_channel(occupancy, "occupancy")
    _validate_single_channel(subsurface_mask, "subsurface_mask")
    if subsurface_mask.shape[0] not in (1, occupancy.shape[0]) or subsurface_mask.shape[2:] != occupancy.shape[2:]:
        raise ValueError("subsurface_mask must broadcast over occupancy")
    if not occupancy.is_floating_point() or not torch.isfinite(occupancy).all():
        raise ValueError("occupancy must be finite floating point")
    if bool(((occupancy < 0) | (occupancy > 1)).any()):
        raise ValueError("occupancy must lie in [0,1]")
    rock = subsurface_mask.to(device=occupancy.device, dtype=torch.bool).expand_as(occupancy)
    target_impedance = occupancy.new_tensor(properties.target_impedance)
    background_impedance = occupancy.new_tensor(properties.background_impedance)
    air_impedance = occupancy.new_tensor(properties.air_impedance)
    target_slowness = occupancy.new_tensor(1.0 / properties.target_velocity)
    background_slowness = occupancy.new_tensor(1.0 / properties.background_velocity)
    air_slowness = occupancy.new_tensor(1.0 / properties.air_velocity)
    impedance_rock = background_impedance + occupancy * (
        target_impedance - background_impedance
    )
    slowness_rock = background_slowness + occupancy * (
        target_slowness - background_slowness
    )
    impedance = torch.where(rock, impedance_rock, air_impedance)
    slowness = torch.where(rock, slowness_rock, air_slowness)
    return impedance.contiguous(), slowness.contiguous()


def smooth_initial_logits(logits: torch.Tensor, passes: int) -> torch.Tensor:
    """Apply a pre-registered light 3x3x3 mean filter to initialization only."""
    if passes < 0:
        raise ValueError("initialization smoothing passes must be non-negative")
    value = logits
    for _ in range(int(passes)):
        value = F.avg_pool3d(value, kernel_size=3, stride=1, padding=1)
    return value


def straight_through_binary(
    logits: torch.Tensor,
    tau: float,
    subsurface_mask: torch.Tensor,
    binary_well_values: torch.Tensor,
    binary_well_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return soft probability, exact hard occupancy, and hard-valued STE state."""
    for value, name in (
        (logits, "logits"),
        (subsurface_mask, "subsurface_mask"),
        (binary_well_values, "binary_well_values"),
        (binary_well_mask, "binary_well_mask"),
    ):
        _validate_single_channel(value, name)
        if value.shape != logits.shape:
            raise ValueError(f"{name} must match logits")
    if not math.isfinite(float(tau)) or tau <= 0:
        raise ValueError("tau must be finite and positive")
    if not torch.isfinite(logits).all():
        raise ValueError("logits must be finite")
    rock = subsurface_mask.bool()
    wells = binary_well_mask.bool()
    if bool((wells & ~rock).any()):
        raise ValueError("binary well mask must be inside subsurface")
    if bool(((binary_well_values < 0) | (binary_well_values > 1))[wells].any()):
        raise ValueError("binary well values must be zero or one")
    probability = torch.sigmoid(logits / float(tau))
    free_hard = (probability >= 0.5).to(probability.dtype)
    hard = torch.where(wells, binary_well_values.to(probability), free_hard)
    hard = torch.where(rock, hard, torch.zeros_like(hard))
    # A custom autograd function avoids the float32 cancellation residual in
    # ``hard + p - p.detach()`` and makes the forward value bit-exactly hard.
    ste_free = _ExactHardSigmoidSTE.apply(logits, float(tau))
    ste = torch.where(wells, binary_well_values.to(probability), ste_free)
    ste = torch.where(rock, ste, torch.zeros_like(ste))
    return probability, hard, ste


def masked_first_difference_tv(
    probability: torch.Tensor,
    subsurface_mask: torch.Tensor,
) -> torch.Tensor:
    """Mean absolute 3-D first difference over within-subsurface edges."""
    _validate_single_channel(probability, "probability")
    _validate_single_channel(subsurface_mask, "subsurface_mask")
    if probability.shape != subsurface_mask.shape:
        raise ValueError("probability and subsurface_mask must match")
    rock = subsurface_mask.bool()
    total = probability.new_zeros(())
    count = probability.new_zeros(())
    for axis in (-3, -2, -1):
        left = [slice(None)] * probability.ndim
        right = [slice(None)] * probability.ndim
        left[axis] = slice(None, -1)
        right[axis] = slice(1, None)
        edge = rock[tuple(left)] & rock[tuple(right)]
        difference = (probability[tuple(right)] - probability[tuple(left)]).abs()
        total = total + (difference * edge.to(difference.dtype)).sum()
        count = count + edge.sum()
    return total / count.clamp_min(1).to(total.dtype)


def inversion_tau(step: int, n_steps: int, start: float, end: float, schedule: str) -> float:
    if n_steps <= 0 or step < 0 or step >= n_steps:
        raise ValueError("invalid optimization step")
    if start <= 0 or end <= 0:
        raise ValueError("tau endpoints must be positive")
    if schedule not in TAU_SCHEDULES:
        raise ValueError(f"tau schedule must be one of {TAU_SCHEDULES}")
    fraction = 1.0 if n_steps == 1 else step / (n_steps - 1)
    if schedule == "cosine":
        fraction = 0.5 - 0.5 * math.cos(math.pi * fraction)
    return float(start) + fraction * (float(end) - float(start))
