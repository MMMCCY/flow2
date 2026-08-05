"""Differentiable spatial observations for Phase-3 property guidance.

Phase 3 deliberately remains a three-dimensional inversion surrogate.  This
module degrades a complete property volume with a fixed spatial operator; it
does not implement gravity, magnetics, seismic acquisition, or measured data.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from guided_geophysical_sampling import soft_decode_to_probs

from .property_volume import (
    gaussian_blur_property_channels,
    matched_multiscale_property_loss,
    probabilities_to_expected_properties,
)


SPATIAL_PROPERTY_PROTOCOL_VERSION = 1
SPATIAL_PROPERTY_CONFIG_SCHEMA = "spatial_property_observation_v1"
SPATIAL_PROPERTY_LOSS_MODE = "matched_spatial_property_observation_mse_v1"
SPATIAL_OPERATOR_TYPES = ("identity", "gaussian_blur", "average_pool")
SPATIAL_CONFIDENCE_TYPES = (
    "base",
    "depth_exponential",
    "axis_aligned_missing",
)
SPATIAL_NOISE_TYPES = ("none", "relative_gaussian")


def tensor_sha256(value: torch.Tensor) -> str:
    """Return a stable hash over tensor dtype, shape, and contiguous bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


@dataclass(frozen=True)
class SpatialPropertyObservation:
    """Immutable tensors and audit metadata produced from one truth volume."""

    values: torch.Tensor
    noiseless_values: torch.Tensor
    confidence: torch.Tensor
    noise: torch.Tensor
    metadata: dict[str, object]


def _require_mapping(value: object, name: str) -> Mapping[str, object]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    return value


def _positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        parsed = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three integers")
        parsed = tuple(int(item) for item in value)
    else:
        raise ValueError(f"{name} must be an integer or three-integer array")
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def validate_spatial_property_config(
    config: Mapping[str, object],
    *,
    spatial_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Validate a Phase-3 observation config and return resolved metadata."""
    if config.get("schema") != SPATIAL_PROPERTY_CONFIG_SCHEMA:
        raise ValueError(
            f"spatial property config schema must be {SPATIAL_PROPERTY_CONFIG_SCHEMA!r}"
        )
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("spatial property config requires a non-empty id")

    operator = _require_mapping(config.get("operator"), "operator")
    operator_type = str(operator.get("type", ""))
    if operator_type not in SPATIAL_OPERATOR_TYPES:
        raise ValueError(f"operator type must be one of {SPATIAL_OPERATOR_TYPES}")
    resolved_operator: dict[str, object] = {"type": operator_type}
    if operator_type == "gaussian_blur":
        sigma = float(operator.get("sigma_voxels", float("nan")))
        if not math.isfinite(sigma) or sigma <= 0:
            raise ValueError("gaussian sigma_voxels must be finite and positive")
        resolved_operator["sigma_voxels"] = sigma
        resolved_operator["padding"] = "replicate"
    elif operator_type == "average_pool":
        factor = _positive_int_triple(operator.get("factor"), "average-pool factor")
        if spatial_shape is not None:
            if len(spatial_shape) != 3:
                raise ValueError("spatial_shape must have three values")
            if any(int(size) % step for size, step in zip(spatial_shape, factor)):
                raise ValueError(
                    "average-pool factor must divide every spatial dimension exactly"
                )
        resolved_operator["factor"] = list(factor)

    confidence = _require_mapping(config.get("confidence", {"type": "base"}), "confidence")
    confidence_type = str(confidence.get("type", ""))
    if confidence_type not in SPATIAL_CONFIDENCE_TYPES:
        raise ValueError(
            f"confidence type must be one of {SPATIAL_CONFIDENCE_TYPES}"
        )
    resolved_confidence: dict[str, object] = {"type": confidence_type}
    if confidence_type == "depth_exponential":
        e_folding = float(confidence.get("e_folding_depth_voxels", float("nan")))
        floor = float(confidence.get("floor", 0.0))
        if not math.isfinite(e_folding) or e_folding <= 0:
            raise ValueError("e_folding_depth_voxels must be finite and positive")
        if not math.isfinite(floor) or not 0 <= floor <= 1:
            raise ValueError("depth confidence floor must be in [0,1]")
        resolved_confidence.update(
            {"e_folding_depth_voxels": e_folding, "floor": floor}
        )
    elif confidence_type == "axis_aligned_missing":
        blocks = confidence.get("blocks")
        if not isinstance(blocks, Sequence) or isinstance(blocks, (str, bytes)):
            raise ValueError("axis_aligned_missing requires a blocks array")
        if not blocks:
            raise ValueError("axis_aligned_missing requires at least one block")
        resolved_blocks: list[dict[str, list[int]]] = []
        for index, block_value in enumerate(blocks):
            block = _require_mapping(block_value, f"missing block {index}")
            start = _positive_or_zero_int_triple(block.get("start"), f"block {index} start")
            stop = _positive_int_triple(block.get("stop"), f"block {index} stop")
            if any(left >= right for left, right in zip(start, stop)):
                raise ValueError(f"missing block {index} must have start < stop")
            if spatial_shape is not None and any(
                right > int(size) for right, size in zip(stop, spatial_shape)
            ):
                raise ValueError(f"missing block {index} exceeds the input grid")
            resolved_blocks.append({"start": list(start), "stop": list(stop)})
        resolved_confidence["blocks"] = resolved_blocks

    noise = _require_mapping(config.get("noise", {"type": "none"}), "noise")
    noise_type = str(noise.get("type", ""))
    if noise_type not in SPATIAL_NOISE_TYPES:
        raise ValueError(f"noise type must be one of {SPATIAL_NOISE_TYPES}")
    resolved_noise: dict[str, object] = {"type": noise_type}
    if noise_type == "relative_gaussian":
        relative_std = float(noise.get("relative_std", float("nan")))
        seed = int(noise.get("seed", -1))
        if not math.isfinite(relative_std) or relative_std <= 0:
            raise ValueError("relative Gaussian noise std must be finite and positive")
        if seed < 0:
            raise ValueError("relative Gaussian noise seed must be non-negative")
        resolved_noise.update({"relative_std": relative_std, "seed": seed})

    return {
        "schema": SPATIAL_PROPERTY_CONFIG_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "operator": resolved_operator,
        "confidence": resolved_confidence,
        "noise": resolved_noise,
        "truth_derived": True,
        "is_measured_geophysics": False,
        "vertical_axis": "last_spatial_axis_z_larger_index_upward",
        "known_property_policy": "exact_before_observation_operator_v1",
    }


def _positive_or_zero_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if isinstance(value, int):
        parsed = (value, value, value)
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three integers")
        parsed = tuple(int(item) for item in value)
    else:
        raise ValueError(f"{name} must be an integer or three-integer array")
    if any(item < 0 for item in parsed):
        raise ValueError(f"{name} values must be non-negative")
    return parsed


def overwrite_known_properties(
    predicted_properties: torch.Tensor,
    known_properties: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    """Use exact hard properties at surface/air/borehole condition voxels."""
    if predicted_properties.ndim != 5:
        raise ValueError("predicted_properties must have shape [B,P,X,Y,Z]")
    if known_properties.ndim != 5:
        raise ValueError("known_properties must have shape [1|B,P,X,Y,Z]")
    if condition_mask.ndim != 5 or condition_mask.shape[1] != 1:
        raise ValueError("condition_mask must have shape [1|B,1,X,Y,Z]")
    if known_properties.shape[1:] != predicted_properties.shape[1:]:
        raise ValueError("known and predicted property shapes must match")
    if condition_mask.shape[2:] != predicted_properties.shape[2:]:
        raise ValueError("condition mask and properties must share spatial shape")
    if known_properties.shape[0] not in (1, predicted_properties.shape[0]):
        raise ValueError("known property batch must be one or match prediction")
    if condition_mask.shape[0] not in (1, predicted_properties.shape[0]):
        raise ValueError("condition mask batch must be one or match prediction")
    known = known_properties.to(
        device=predicted_properties.device,
        dtype=predicted_properties.dtype,
    ).expand(predicted_properties.shape[0], -1, -1, -1, -1)
    mask = condition_mask.to(device=predicted_properties.device).bool()
    mask = mask.expand(predicted_properties.shape[0], predicted_properties.shape[1], -1, -1, -1)
    return torch.where(mask, known, predicted_properties)


def apply_spatial_property_operator(
    volume: torch.Tensor,
    operator: Mapping[str, object],
) -> torch.Tensor:
    """Apply the fixed differentiable Phase-3 spatial response."""
    if volume.ndim != 5:
        raise ValueError("volume must have shape [B,P,X,Y,Z]")
    operator_type = str(operator.get("type", ""))
    if operator_type == "identity":
        return volume
    if operator_type == "gaussian_blur":
        return gaussian_blur_property_channels(
            volume,
            float(operator["sigma_voxels"]),
        )
    if operator_type == "average_pool":
        factor = _positive_int_triple(operator.get("factor"), "average-pool factor")
        if any(size % step for size, step in zip(volume.shape[2:], factor)):
            raise ValueError("average-pool factor must divide the spatial shape exactly")
        return F.avg_pool3d(volume, kernel_size=factor, stride=factor)
    raise ValueError(f"unknown spatial property operator {operator_type!r}")


def depth_exponential_confidence(
    nonair_mask: torch.Tensor,
    *,
    e_folding_depth_voxels: float,
    floor: float = 0.0,
) -> torch.Tensor:
    """Return per-column confidence that decays downward from local surface."""
    if nonair_mask.ndim != 5 or nonair_mask.shape[1] != 1:
        raise ValueError("nonair_mask must have shape [B,1,X,Y,Z]")
    if e_folding_depth_voxels <= 0 or not math.isfinite(e_folding_depth_voxels):
        raise ValueError("e_folding_depth_voxels must be finite and positive")
    if not 0 <= floor <= 1 or not math.isfinite(floor):
        raise ValueError("floor must be finite and in [0,1]")
    mask = nonair_mask.bool()
    z_size = mask.shape[-1]
    z = torch.arange(z_size, device=mask.device, dtype=torch.float32)
    z = z.view(1, 1, 1, 1, z_size)
    surface = torch.where(mask, z, torch.full_like(z, -1.0)).amax(
        dim=-1,
        keepdim=True,
    )
    depth = (surface - z).clamp_min(0.0)
    decay = torch.exp(-depth / float(e_folding_depth_voxels))
    weights = floor + (1.0 - floor) * decay
    return torch.where(mask, weights, torch.zeros_like(weights))


def _apply_missing_blocks(
    confidence: torch.Tensor,
    blocks: Sequence[Mapping[str, object]],
) -> torch.Tensor:
    result = confidence.clone()
    for block in blocks:
        start = _positive_or_zero_int_triple(block.get("start"), "block start")
        stop = _positive_int_triple(block.get("stop"), "block stop")
        if any(right > size for right, size in zip(stop, result.shape[2:])):
            raise ValueError("missing block exceeds the resolved observation grid")
        result[..., start[0] : stop[0], start[1] : stop[1], start[2] : stop[2]] = 0
    return result


def _weighted_channel_std(
    values: torch.Tensor,
    confidence: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    weights = confidence.to(device=values.device, dtype=values.dtype)
    if weights.shape[1] == 1:
        weights = weights.expand(-1, values.shape[1], -1, -1, -1)
    denominator = weights.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
    mean = (weights * values).sum(dim=(0, 2, 3, 4)) / denominator
    variance = (
        weights * (values - mean.view(1, -1, 1, 1, 1)).square()
    ).sum(dim=(0, 2, 3, 4)) / denominator
    return variance.sqrt().clamp_min(eps)


def build_spatial_property_observation(
    target_properties: torch.Tensor,
    base_confidence: torch.Tensor,
    nonair_mask: torch.Tensor,
    config: Mapping[str, object],
) -> SpatialPropertyObservation:
    """Build one deterministic truth-derived observation and its confidence."""
    if target_properties.ndim != 5:
        raise ValueError("target_properties must have shape [1,P,X,Y,Z]")
    if target_properties.shape[0] != 1:
        raise ValueError("Phase-3 observation builder requires one truth volume")
    if base_confidence.ndim != 5 or base_confidence.shape[:2] != (1, 1):
        raise ValueError("base_confidence must have shape [1,1,X,Y,Z]")
    if nonair_mask.ndim != 5 or nonair_mask.shape[:2] != (1, 1):
        raise ValueError("nonair_mask must have shape [1,1,X,Y,Z]")
    if base_confidence.shape[2:] != target_properties.shape[2:]:
        raise ValueError("base confidence and target properties must share shape")
    if nonair_mask.shape != base_confidence.shape:
        raise ValueError("nonair mask and base confidence must share shape")
    if not torch.isfinite(target_properties).all():
        raise ValueError("target properties must be finite")
    base = base_confidence.to(dtype=target_properties.dtype, device=target_properties.device)
    if not torch.isfinite(base).all() or bool((base < 0).any()):
        raise ValueError("base confidence must be finite and non-negative")

    metadata = validate_spatial_property_config(
        config,
        spatial_shape=target_properties.shape[2:],
    )
    operator = _require_mapping(metadata["operator"], "resolved operator")
    confidence_config = _require_mapping(metadata["confidence"], "resolved confidence")
    confidence_type = str(confidence_config["type"])

    full_confidence = base
    if confidence_type == "depth_exponential":
        depth_weights = depth_exponential_confidence(
            nonair_mask.to(device=target_properties.device),
            e_folding_depth_voxels=float(confidence_config["e_folding_depth_voxels"]),
            floor=float(confidence_config["floor"]),
        ).to(dtype=target_properties.dtype)
        full_confidence = full_confidence * depth_weights

    noiseless = apply_spatial_property_operator(target_properties, operator)
    observation_confidence = apply_spatial_property_operator(full_confidence, operator)
    observation_confidence = observation_confidence.clamp_min(0.0)
    if confidence_type == "axis_aligned_missing":
        observation_confidence = _apply_missing_blocks(
            observation_confidence,
            confidence_config["blocks"],
        )
    if float(observation_confidence.sum().item()) <= 0:
        raise ValueError("resolved observation confidence contains no active values")

    noise_config = _require_mapping(metadata["noise"], "resolved noise")
    noise = torch.zeros_like(noiseless)
    if str(noise_config["type"]) == "relative_gaussian":
        generator = torch.Generator(device="cpu").manual_seed(int(noise_config["seed"]))
        random_cpu = torch.randn(
            noiseless.shape,
            generator=generator,
            dtype=torch.float32,
            device="cpu",
        )
        random_values = random_cpu.to(device=noiseless.device, dtype=noiseless.dtype)
        channel_std = _weighted_channel_std(noiseless, observation_confidence)
        noise = (
            random_values
            * channel_std.view(1, -1, 1, 1, 1)
            * float(noise_config["relative_std"])
        )
        active = observation_confidence > 0
        noise = torch.where(active.expand_as(noise), noise, torch.zeros_like(noise))
    values = noiseless + noise

    resolved_metadata = {
        **metadata,
        "input_spatial_shape": list(target_properties.shape[2:]),
        "observation_spatial_shape": list(values.shape[2:]),
        "property_channels": int(values.shape[1]),
        "active_confidence_values": int((observation_confidence > 0).sum().item()),
        "active_confidence_fraction": float(
            (observation_confidence > 0).float().mean().item()
        ),
        "target_properties_sha256": tensor_sha256(target_properties),
        "noiseless_observation_sha256": tensor_sha256(noiseless),
        "observation_confidence_sha256": tensor_sha256(observation_confidence),
        "observation_noise_sha256": tensor_sha256(noise),
        "observation_values_sha256": tensor_sha256(values),
        "noise_application": "immutable_observation_only_v1",
    }
    return SpatialPropertyObservation(
        values=values,
        noiseless_values=noiseless,
        confidence=observation_confidence,
        noise=noise,
        metadata=resolved_metadata,
    )


def spatial_property_volume_loss(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_properties: torch.Tensor,
    property_table: torch.Tensor,
    confidence: torch.Tensor,
    tau: float,
    sigmas: Sequence[float],
    scale_weights: Sequence[float],
    channel_weights: torch.Tensor,
    *,
    observed_properties: torch.Tensor,
    observation_confidence: torch.Tensor,
    observation_config: Mapping[str, object],
    condition_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft-decode state and compare the matched degraded 3-D observation."""
    del confidence  # The resolved observation confidence is authoritative.
    probabilities = soft_decode_to_probs(state, embedding_weight, tau=tau)
    predicted_properties = probabilities_to_expected_properties(
        probabilities,
        property_table,
    )
    predicted_known = overwrite_known_properties(
        predicted_properties,
        target_properties,
        condition_mask,
    )
    metadata = validate_spatial_property_config(
        observation_config,
        spatial_shape=predicted_properties.shape[2:],
    )
    predicted_observation = apply_spatial_property_operator(
        predicted_known,
        _require_mapping(metadata["operator"], "resolved operator"),
    )
    loss, diagnostics = matched_multiscale_property_loss(
        predicted_observation,
        observed_properties,
        observation_confidence,
        sigmas=sigmas,
        scale_weights=scale_weights,
        channel_weights=channel_weights,
    )
    diagnostics["observation_loss"] = loss
    diagnostics["predicted_observation_min"] = predicted_observation.min()
    diagnostics["predicted_observation_max"] = predicted_observation.max()
    entropy = -(
        probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()
    ).sum(dim=1, keepdim=True)
    unconditioned = (~condition_mask.to(device=entropy.device).bool()).to(entropy.dtype)
    if unconditioned.shape[0] == 1 and entropy.shape[0] > 1:
        unconditioned = unconditioned.expand(entropy.shape[0], -1, -1, -1, -1)
    diagnostics["soft_class_entropy_unconditioned_mean"] = (
        (entropy * unconditioned).sum() / unconditioned.sum().clamp_min(1e-6)
    )
    return loss, diagnostics


def hard_spatial_observation_loss(
    predicted_properties: torch.Tensor,
    target_properties: torch.Tensor,
    condition_mask: torch.Tensor,
    observation: SpatialPropertyObservation,
    observation_config: Mapping[str, object],
    *,
    sigmas: Sequence[float],
    scale_weights: Sequence[float],
    channel_weights: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate a decoded hard model in the same Phase-3 observation domain."""
    predicted_known = overwrite_known_properties(
        predicted_properties,
        target_properties,
        condition_mask,
    )
    metadata = validate_spatial_property_config(
        observation_config,
        spatial_shape=predicted_properties.shape[2:],
    )
    predicted_observation = apply_spatial_property_operator(
        predicted_known,
        _require_mapping(metadata["operator"], "resolved operator"),
    )
    return matched_multiscale_property_loss(
        predicted_observation,
        observation.values,
        observation.confidence,
        sigmas=sigmas,
        scale_weights=scale_weights,
        channel_weights=channel_weights,
    )

