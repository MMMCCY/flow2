"""Three-dimensional full-lithology property-volume guidance.

Phase 2 maps every soft categorical probability to one or more expected
physical-property channels and compares predicted and target property volumes
after applying the same three-dimensional scale operator to both sides.  The
target remains truth-derived in Phase 2a/2b; this module does not implement a
2-D gravity/magnetic forward operator and must not be described as field data.
"""

from __future__ import annotations

import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from guided_geophysical_sampling import soft_decode_to_probs


PROPERTY_PROTOCOL_VERSION = 1
PROPERTY_LOSS_MODE = "matched_multiscale_normalized_mse_v1"
PROPERTY_CONFIG_SCHEMA = "full_lithology_property_channels_v1"


def _tensor_sha256(value: torch.Tensor) -> str:
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def property_table_from_config(
    config: Mapping[str, object],
    num_categories: int,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, object]]:
    """Parse an explicit full-lithology multi-channel property table.

    Category index zero is raw label ``-1`` and category ``C-1`` is raw label
    ``C-2``.  Complete coverage is mandatory so absent model classes cannot
    silently receive a default physical value.
    """
    if num_categories <= 1:
        raise ValueError("num_categories must be greater than one")
    if config.get("schema") != PROPERTY_CONFIG_SCHEMA:
        raise ValueError(f"property config schema must be {PROPERTY_CONFIG_SCHEMA!r}")
    channels = config.get("channels")
    if not isinstance(channels, Sequence) or isinstance(channels, (str, bytes)):
        raise ValueError("property config must contain a channels array")
    if not channels:
        raise ValueError("at least one property channel is required")

    expected_labels = set(range(-1, num_categories - 1))
    names: list[str] = []
    units: list[str] = []
    rows: list[list[float]] = []
    weights: list[float] = []
    for channel_index, channel in enumerate(channels):
        if not isinstance(channel, Mapping):
            raise ValueError(f"channel {channel_index} must be an object")
        name = str(channel.get("name", "")).strip()
        if not name or name in names:
            raise ValueError("property channel names must be non-empty and unique")
        values = channel.get("values")
        if not isinstance(values, Mapping):
            raise ValueError(f"property channel {name!r} must contain values")
        try:
            parsed = {int(label): float(value) for label, value in values.items()}
        except (TypeError, ValueError) as exc:
            raise ValueError(f"property channel {name!r} contains invalid values") from exc
        missing = sorted(expected_labels - set(parsed))
        extra = sorted(set(parsed) - expected_labels)
        if missing or extra:
            raise ValueError(
                f"property channel {name!r} label coverage mismatch: "
                f"missing={missing}, extra={extra}"
            )
        row = [parsed[category - 1] for category in range(num_categories)]
        if not all(math.isfinite(value) for value in row):
            raise ValueError(f"property channel {name!r} must contain finite values")
        weight = float(channel.get("weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError(f"property channel {name!r} weight must be finite and non-negative")
        names.append(name)
        units.append(str(channel.get("unit", "arbitrary")))
        rows.append(row)
        weights.append(weight)
    if sum(weights) <= 0:
        raise ValueError("at least one property channel weight must be positive")

    table = torch.tensor(rows, dtype=torch.float32)
    channel_weights = torch.tensor(weights, dtype=torch.float32)
    channel_weights = channel_weights / channel_weights.sum()
    metadata: dict[str, object] = {
        "schema": PROPERTY_CONFIG_SCHEMA,
        "description": str(config.get("description", "")),
        "num_categories": num_categories,
        "raw_label_range": [-1, num_categories - 2],
        "channel_names": names,
        "channel_units": units,
        "channel_weights": channel_weights.tolist(),
        "property_table_sha256": _tensor_sha256(table),
        "truth_derived": True,
        "is_measured_geophysics": False,
    }
    return table, channel_weights, metadata


def property_codebook_diagnostics(
    property_table: torch.Tensor,
    channel_weights: torch.Tensor | None = None,
    target_raw_label: int = 9,
    eps: float = 1e-12,
) -> dict[str, object]:
    """Describe exact collisions and target separation in a property table.

    Distances use each channel's full codebook range and the normalized loss
    channel weights. They are audit diagnostics only and do not alter the
    matched property loss or its target-weighted normalization.
    """
    table = _validate_property_table(property_table, property_table.shape[1])
    table_cpu = table.detach().to(device="cpu", dtype=torch.float64)
    channel_count, category_count = table_cpu.shape
    if not -1 <= target_raw_label <= category_count - 2:
        raise ValueError(
            f"target_raw_label must be in [-1,{category_count - 2}]"
        )
    if channel_weights is None:
        weights = torch.ones(channel_count, dtype=torch.float64)
    else:
        if channel_weights.ndim != 1 or channel_weights.numel() != channel_count:
            raise ValueError("channel_weights must have one value per property channel")
        weights = channel_weights.detach().to(device="cpu", dtype=torch.float64)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("channel weights must be finite and non-negative")
    if float(weights.sum().item()) <= 0:
        raise ValueError("channel weights must sum to a positive value")
    weights = weights / weights.sum()

    grouped: dict[tuple[float, ...], list[int]] = {}
    for category in range(category_count):
        vector = tuple(float(value) for value in table_cpu[:, category])
        grouped.setdefault(vector, []).append(category - 1)
    duplicate_groups = [
        labels for labels in grouped.values() if len(labels) > 1
    ]

    channel_ranges = table_cpu.amax(dim=1) - table_cpu.amin(dim=1)
    normalized = table_cpu / channel_ranges.clamp_min(eps).view(-1, 1)
    target_category = target_raw_label + 1
    target_vector = normalized[:, target_category]
    distances = (
        weights.view(-1, 1)
        * (normalized - target_vector.view(-1, 1)).square()
    ).sum(dim=0).sqrt()
    distances[target_category] = float("inf")
    nearest_distance = float(distances.min().item())
    nearest_labels = [
        category - 1
        for category, distance in enumerate(distances.tolist())
        if math.isclose(distance, nearest_distance, rel_tol=1e-9, abs_tol=1e-12)
    ]
    target_exact_group = next(
        labels for labels in grouped.values() if target_raw_label in labels
    )
    return {
        "unique_property_vectors": len(grouped),
        "duplicate_property_groups": duplicate_groups,
        "target_raw_label": int(target_raw_label),
        "target_property_values": [
            float(value) for value in table_cpu[:, target_category]
        ],
        "target_exact_property_group": target_exact_group,
        "target_nearest_raw_labels": nearest_labels,
        "target_nearest_range_normalized_distance": nearest_distance,
        "distance_definition": "channel_range_normalized_weighted_euclidean_v1",
    }


def _validate_property_table(
    property_table: torch.Tensor,
    num_categories: int,
) -> torch.Tensor:
    if property_table.ndim != 2:
        raise ValueError("property_table must have shape [P,C]")
    if property_table.shape[1] != num_categories:
        raise ValueError(
            f"property_table has {property_table.shape[1]} categories, expected {num_categories}"
        )
    if property_table.shape[0] < 1:
        raise ValueError("property_table must contain at least one channel")
    if not torch.isfinite(property_table).all():
        raise ValueError("property_table must contain only finite values")
    return property_table


def hard_labels_to_properties(
    labels: torch.Tensor,
    property_table: torch.Tensor,
) -> torch.Tensor:
    """Map raw hard labels ``-1..C-2`` to ``[B,P,X,Y,Z]`` properties."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    table = _validate_property_table(property_table, property_table.shape[1])
    if not torch.equal(labels, labels.round()):
        raise ValueError("labels must be integer-valued")
    categories = labels.long() + 1
    if int(categories.min().item()) < 0 or int(categories.max().item()) >= table.shape[1]:
        raise ValueError(
            f"raw labels must be in [-1,{table.shape[1] - 2}] for the property table"
        )
    values = table.to(device=labels.device, dtype=torch.float32)
    one_hot = F.one_hot(categories[:, 0], num_classes=table.shape[1])
    one_hot = one_hot.permute(0, 4, 1, 2, 3).to(values.dtype)
    return torch.einsum("bcxyz,pc->bpxyz", one_hot, values)


def probabilities_to_expected_properties(
    probabilities: torch.Tensor,
    property_table: torch.Tensor,
) -> torch.Tensor:
    """Map full soft categorical probabilities to expected property channels."""
    if probabilities.ndim != 5:
        raise ValueError("probabilities must have shape [B,C,X,Y,Z]")
    table = _validate_property_table(property_table, probabilities.shape[1]).to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must contain only finite values")
    return torch.einsum("bcxyz,pc->bpxyz", probabilities, table)


def _gaussian_kernel_1d(
    sigma: float,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
    kernel = torch.exp(-0.5 * (coordinates / sigma).square())
    return kernel / kernel.sum()


def gaussian_blur_property_channels(volume: torch.Tensor, sigma: float) -> torch.Tensor:
    """Apply the same separable 3-D Gaussian to every property channel."""
    if volume.ndim != 5:
        raise ValueError("property volume must have shape [B,P,X,Y,Z]")
    if sigma < 0:
        raise ValueError("sigma must be non-negative")
    if sigma == 0:
        return volume
    kernel = _gaussian_kernel_1d(float(sigma), volume.device, volume.dtype)
    radius = kernel.numel() // 2
    channels = volume.shape[1]
    kernels = (
        kernel.view(1, 1, -1, 1, 1).expand(channels, 1, -1, 1, 1),
        kernel.view(1, 1, 1, -1, 1).expand(channels, 1, 1, -1, 1),
        kernel.view(1, 1, 1, 1, -1).expand(channels, 1, 1, 1, -1),
    )
    paddings = (
        (0, 0, 0, 0, radius, radius),
        (0, 0, radius, radius, 0, 0),
        (radius, radius, 0, 0, 0, 0),
    )
    result = volume
    for conv_kernel, padding in zip(kernels, paddings):
        result = F.conv3d(
            F.pad(result, padding, mode="replicate"),
            conv_kernel,
            groups=channels,
        )
    return result


def _expanded_spatial_weights(
    confidence: torch.Tensor,
    reference: torch.Tensor,
) -> torch.Tensor:
    if confidence.ndim != 5:
        raise ValueError("confidence must have shape [B,1|P,X,Y,Z]")
    if confidence.shape[0] not in (1, reference.shape[0]):
        raise ValueError("confidence batch must be one or match property volume batch")
    if confidence.shape[1] not in (1, reference.shape[1]):
        raise ValueError("confidence channels must be one or match property channels")
    if confidence.shape[2:] != reference.shape[2:]:
        raise ValueError("confidence and property volumes must have matching spatial shape")
    weights = confidence.to(device=reference.device, dtype=reference.dtype)
    if not torch.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("confidence must be finite and non-negative")
    return weights.expand(reference.shape[0], reference.shape[1], -1, -1, -1)


def matched_multiscale_property_loss(
    predicted_properties: torch.Tensor,
    target_properties: torch.Tensor,
    confidence: torch.Tensor,
    sigmas: Sequence[float] = (0.0, 1.5, 3.0),
    scale_weights: Sequence[float] = (0.50, 0.30, 0.20),
    channel_weights: torch.Tensor | None = None,
    eps: float = 1e-6,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Compare matched predicted/target 3-D properties at multiple scales.

    Each channel is normalized by its weighted target standard deviation, so
    density, susceptibility, or velocity units cannot dominate solely because
    of numerical scale.  The exact same Gaussian operator is applied to the
    current prediction and truth-derived target at every scale.
    """
    if predicted_properties.ndim != 5 or target_properties.ndim != 5:
        raise ValueError("property volumes must have shape [B,P,X,Y,Z]")
    if predicted_properties.shape[1:] != target_properties.shape[1:]:
        raise ValueError("predicted and target property channel/spatial shapes must match")
    if target_properties.shape[0] not in (1, predicted_properties.shape[0]):
        raise ValueError("target batch must be one or match predicted batch")
    if len(sigmas) == 0 or len(sigmas) != len(scale_weights):
        raise ValueError("sigmas and scale_weights must be non-empty and have equal length")
    sigma_values = [float(value) for value in sigmas]
    scale_values = [float(value) for value in scale_weights]
    if any(value < 0 or not math.isfinite(value) for value in sigma_values):
        raise ValueError("property sigmas must be finite and non-negative")
    if any(value < 0 or not math.isfinite(value) for value in scale_values) or sum(scale_values) <= 0:
        raise ValueError("scale weights must be finite, non-negative, and sum positive")
    scale_values = [value / sum(scale_values) for value in scale_values]

    target = target_properties.to(
        device=predicted_properties.device,
        dtype=predicted_properties.dtype,
    ).expand(predicted_properties.shape[0], -1, -1, -1, -1)
    spatial_weights = _expanded_spatial_weights(confidence, predicted_properties)
    channel_count = predicted_properties.shape[1]
    if channel_weights is None:
        channel_weight_values = torch.ones(
            channel_count,
            device=predicted_properties.device,
            dtype=predicted_properties.dtype,
        )
    else:
        if channel_weights.ndim != 1 or channel_weights.numel() != channel_count:
            raise ValueError("channel_weights must have one value per property channel")
        channel_weight_values = channel_weights.to(
            device=predicted_properties.device,
            dtype=predicted_properties.dtype,
        )
    if not torch.isfinite(channel_weight_values).all() or bool((channel_weight_values < 0).any()):
        raise ValueError("channel weights must be finite and non-negative")
    if float(channel_weight_values.sum().item()) <= 0:
        raise ValueError("channel weights must sum to a positive value")
    channel_weight_values = channel_weight_values / channel_weight_values.sum()

    total = torch.zeros((), device=predicted_properties.device, dtype=predicted_properties.dtype)
    diagnostics: dict[str, torch.Tensor] = {}
    for scale_index, (sigma, scale_weight) in enumerate(zip(sigma_values, scale_values)):
        predicted_scaled = gaussian_blur_property_channels(predicted_properties, sigma)
        target_scaled = gaussian_blur_property_channels(target, sigma)
        denominator = spatial_weights.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
        target_mean = (spatial_weights * target_scaled).sum(dim=(0, 2, 3, 4)) / denominator
        centered = target_scaled - target_mean.view(1, -1, 1, 1, 1)
        target_variance = (spatial_weights * centered.square()).sum(
            dim=(0, 2, 3, 4)
        ) / denominator
        channel_scale = target_variance.sqrt().clamp_min(eps)
        normalized_square_error = (
            (predicted_scaled - target_scaled)
            / channel_scale.view(1, -1, 1, 1, 1)
        ).square()
        channel_mse = (spatial_weights * normalized_square_error).sum(
            dim=(0, 2, 3, 4)
        ) / denominator
        scale_loss = (channel_weight_values * channel_mse).sum()
        total = total + scale_weight * scale_loss
        diagnostics[f"scale_{scale_index}_sigma"] = torch.as_tensor(
            sigma,
            device=total.device,
            dtype=total.dtype,
        )
        diagnostics[f"scale_{scale_index}_loss"] = scale_loss
        diagnostics[f"scale_{scale_index}_target_std_mean"] = channel_scale.mean()

    absolute_error = (spatial_weights * (predicted_properties - target).abs()).sum(
        dim=(0, 2, 3, 4)
    ) / spatial_weights.sum(dim=(0, 2, 3, 4)).clamp_min(eps)
    diagnostics.update(
        {
            "property_loss": total,
            "property_mae_mean": (channel_weight_values * absolute_error).sum(),
            "confidence_fraction": (spatial_weights > 0).float().mean(),
        }
    )
    return total, diagnostics


def property_volume_loss(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    target_properties: torch.Tensor,
    property_table: torch.Tensor,
    confidence: torch.Tensor,
    tau: float,
    sigmas: Sequence[float] = (0.0, 1.5, 3.0),
    scale_weights: Sequence[float] = (0.50, 0.30, 0.20),
    channel_weights: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Soft-decode a sampler state and evaluate the Phase-2 property loss."""
    probabilities = soft_decode_to_probs(state, embedding_weight, tau=tau)
    predicted_properties = probabilities_to_expected_properties(
        probabilities,
        property_table,
    )
    loss, diagnostics = matched_multiscale_property_loss(
        predicted_properties,
        target_properties,
        confidence,
        sigmas=sigmas,
        scale_weights=scale_weights,
        channel_weights=channel_weights,
    )
    entropy = -(probabilities.clamp_min(1e-8) * probabilities.clamp_min(1e-8).log()).sum(
        dim=1,
        keepdim=True,
    )
    confidence_one = confidence[:, :1].to(device=entropy.device, dtype=entropy.dtype)
    if confidence_one.shape[0] == 1 and entropy.shape[0] > 1:
        confidence_one = confidence_one.expand(entropy.shape[0], -1, -1, -1, -1)
    diagnostics["soft_class_entropy_confident_mean"] = (
        (entropy * confidence_one).sum() / confidence_one.sum().clamp_min(1e-6)
    )
    diagnostics["predicted_property_min"] = predicted_properties.min()
    diagnostics["predicted_property_max"] = predicted_properties.max()
    return loss, diagnostics
