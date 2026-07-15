"""Lightweight geophysical post-processing for decoded geology volumes.

This module is intentionally independent of model training and conditional
inference. It provides a small gravity-style forward proxy that can rank
already generated categorical realizations against observed data.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


Number = Union[int, float]


def _as_5d_volume(volume: torch.Tensor) -> torch.Tensor:
    """Normalize geology/density volumes to [B, 1, X, Y, Z]."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError("volume must be a torch.Tensor")

    if volume.dim() == 3:
        return volume.unsqueeze(0).unsqueeze(0)
    if volume.dim() == 4:
        return volume.unsqueeze(1) if volume.shape[0] > 1 else volume.unsqueeze(0)
    if volume.dim() == 5:
        if volume.shape[1] != 1:
            raise ValueError(
                "expected a single channel volume with shape [B, 1, X, Y, Z]"
            )
        return volume

    raise ValueError(
        "expected volume shape [X, Y, Z], [B, X, Y, Z], "
        "[1, X, Y, Z], or [B, 1, X, Y, Z]"
    )


def _as_4d_field(field: torch.Tensor) -> torch.Tensor:
    """Normalize gravity fields to [B, 1, X, Y]."""
    if not isinstance(field, torch.Tensor):
        raise TypeError("field must be a torch.Tensor")

    if field.dim() == 2:
        return field.unsqueeze(0).unsqueeze(0)
    if field.dim() == 3:
        return field.unsqueeze(1) if field.shape[0] > 1 else field.unsqueeze(0)
    if field.dim() == 4:
        if field.shape[1] != 1:
            raise ValueError(
                "expected a single channel field with shape [B, 1, X, Y]"
            )
        return field

    raise ValueError(
        "expected field shape [X, Y], [B, X, Y], "
        "[1, X, Y], or [B, 1, X, Y]"
    )


def _triple(value: Union[Number, Sequence[Number]]) -> Tuple[float, float, float]:
    if isinstance(value, (int, float)):
        v = float(value)
        return v, v, v
    if len(value) != 3:
        raise ValueError("cell_size must be a scalar or a length-3 sequence")
    return float(value[0]), float(value[1]), float(value[2])

# 离散岩性对应物理量
class LithologyPropertyMap:
    """Map categorical lithology ids to scalar density contrast values.

    The default values are relative density contrasts for screening only. They
    should be replaced with project-specific petrophysical  values before using
    this baseline for quantitative interpretation.
    """

    DEFAULT_DENSITY_CONTRASTS: Dict[int, float] = {
        -1: 0.0,
        0: 0.15,
        1: 0.18,
        2: 0.21,
        3: 0.24,
        4: 0.27,
        5: 0.30,
        6: 0.33,
        7: 0.36,
        8: 0.39,
        9: 0.42,
        10: 0.45,
        11: 0.48,
        12: 0.51,
        13: 0.54,
        14: 0.57,
    }

    def __init__(
        self,
        properties: Optional[Mapping[int, Number]] = None,
        default_value: Number = 0.0,
    ) -> None:
        source_properties = (
            self.DEFAULT_DENSITY_CONTRASTS if properties is None else properties
        )
        self.properties = dict(source_properties)
        self.default_value = float(default_value)

    def to_density(self, lithology: torch.Tensor) -> torch.Tensor:
        """Return a float density contrast volume with shape [B, 1, X, Y, Z]."""
        volume = _as_5d_volume(lithology)
        categories = volume.long()
        density = torch.full(
            categories.shape,
            self.default_value,
            dtype=torch.float32,
            device=categories.device,
        )

        for lithology_id, value in self.properties.items():
            density = torch.where(
                categories == int(lithology_id),
                torch.as_tensor(
                    float(value),
                    dtype=density.dtype,
                    device=density.device,
                ),
                density,
            )

        return density

    def __call__(self, lithology: torch.Tensor) -> torch.Tensor:
        return self.to_density(lithology)


class SimpleGravityForward:
    """Compute a small surface gravity proxy from a density contrast volume.

    For each depth slice, the density contrast is convolved with a symmetric
    vertical-component point-mass kernel and accumulated at the top surface.
    The output is a relative anomaly field with shape [B, 1, X, Y].
    """

    def __init__(
        self,
        cell_size: Union[Number, Sequence[Number]] = 1.0,
        observation_height: Number = 1.0,
        kernel_size: int = 9,
        gravitational_constant: Number = 1.0,
        remove_mean: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")

        self.cell_size = _triple(cell_size)
        self.observation_height = float(observation_height)
        self.kernel_size = int(kernel_size)
        self.gravitational_constant = float(gravitational_constant)
        self.remove_mean = bool(remove_mean)
        self.eps = float(eps)
        self._kernel_cache: Dict[
            Tuple[int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def _kernels(
        self,
        depth_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (depth_count, device, dtype)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        dx, dy, dz = self.cell_size
        half = self.kernel_size // 2
        x_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dx
        y_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dy
        xx, yy = torch.meshgrid(x_offsets, y_offsets, indexing="ij")

        kernels = []
        for z_index in range(depth_count):
            depth = (depth_count - z_index) * dz + self.observation_height
            depth_tensor = torch.as_tensor(depth, device=device, dtype=dtype)
            r2 = xx.square() + yy.square() + depth_tensor.square()
            kernel = depth_tensor / torch.pow(r2 + self.eps, 1.5)
            kernels.append(kernel)

        stacked = torch.stack(kernels, dim=0).unsqueeze(1).unsqueeze(1)
        self._kernel_cache[cache_key] = stacked
        return stacked

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Forward model density contrast [B, 1, X, Y, Z] to [B, 1, X, Y]."""
        volume = _as_5d_volume(density).to(dtype=torch.float32)
        _, _, _, _, depth_count = volume.shape
        kernels = self._kernels(depth_count, volume.device, volume.dtype)

        dx, dy, dz = self.cell_size
        scale = self.gravitational_constant * dx * dy * dz
        gravity = torch.zeros(
            volume.shape[0],
            1,
            volume.shape[2],
            volume.shape[3],
            dtype=volume.dtype,
            device=volume.device,
        )

        padding = self.kernel_size // 2
        for z_index in range(depth_count):
            gravity = gravity + F.conv2d(
                volume[..., z_index],
                kernels[z_index],
                padding=padding,
            )

        gravity = gravity * scale
        if self.remove_mean:
            gravity = gravity - gravity.mean(dim=(-2, -1), keepdim=True)
        return gravity

    def __call__(self, density: torch.Tensor) -> torch.Tensor:
        return self.forward(density)


class GravityGradientForward:
    """Compute a local gravity-gradient-style proxy from density contrast.

    This is still a lightweight proxy, not a full tensor gravity forward model.
    It uses a vertical second-derivative point-source style kernel, which is
    more local than :class:`SimpleGravityForward` and can make compact
    dike-like density contrasts more visible to inference-time guidance.
    """

    def __init__(
        self,
        cell_size: Union[Number, Sequence[Number]] = 1.0,
        observation_height: Number = 1.0,
        kernel_size: int = 9,
        gravitational_constant: Number = 1.0,
        remove_mean: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.cell_size = _triple(cell_size)
        self.observation_height = float(observation_height)
        self.kernel_size = int(kernel_size)
        self.gravitational_constant = float(gravitational_constant)
        self.remove_mean = bool(remove_mean)
        self.eps = float(eps)
        self._kernel_cache: Dict[
            Tuple[int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def _kernels(
        self,
        depth_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (depth_count, device, dtype)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        dx, dy, dz = self.cell_size
        half = self.kernel_size // 2
        x_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dx
        y_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dy
        xx, yy = torch.meshgrid(x_offsets, y_offsets, indexing="ij")

        kernels = []
        for z_index in range(depth_count):
            depth = (depth_count - z_index) * dz + self.observation_height
            z = torch.as_tensor(depth, device=device, dtype=dtype)
            r2 = xx.square() + yy.square() + z.square()
            kernel = (3.0 * z.square() - r2) / torch.pow(r2 + self.eps, 2.5)
            kernels.append(kernel)

        stacked = torch.stack(kernels, dim=0).unsqueeze(1).unsqueeze(1)
        self._kernel_cache[cache_key] = stacked
        return stacked

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        volume = _as_5d_volume(density).to(dtype=torch.float32)
        _, _, _, _, depth_count = volume.shape
        kernels = self._kernels(depth_count, volume.device, volume.dtype)

        dx, dy, dz = self.cell_size
        scale = self.gravitational_constant * dx * dy * dz
        gradient = torch.zeros(
            volume.shape[0],
            1,
            volume.shape[2],
            volume.shape[3],
            dtype=volume.dtype,
            device=volume.device,
        )
        padding = self.kernel_size // 2
        for z_index in range(depth_count):
            gradient = gradient + F.conv2d(
                volume[..., z_index],
                kernels[z_index],
                padding=padding,
            )
        gradient = gradient * scale
        if self.remove_mean:
            gradient = gradient - gradient.mean(dim=(-2, -1), keepdim=True)
        return gradient

    def __call__(self, density: torch.Tensor) -> torch.Tensor:
        return self.forward(density)


class MagneticTMIForward:
    """Compute a lightweight total magnetic intensity proxy.

    The input is a susceptibility-style scalar volume. The proxy assumes a
    vertical inducing field and vertical magnetization, then applies a dipole
    kernel at the top observation surface. This deliberately remains a
    lightweight magnetic-proxy for guidance and demo figures.
    """

    def __init__(
        self,
        cell_size: Union[Number, Sequence[Number]] = 1.0,
        observation_height: Number = 1.0,
        kernel_size: int = 9,
        field_strength: Number = 1.0,
        remove_mean: bool = True,
        eps: float = 1e-8,
    ) -> None:
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        self.cell_size = _triple(cell_size)
        self.observation_height = float(observation_height)
        self.kernel_size = int(kernel_size)
        self.field_strength = float(field_strength)
        self.remove_mean = bool(remove_mean)
        self.eps = float(eps)
        self._kernel_cache: Dict[
            Tuple[int, torch.device, torch.dtype],
            torch.Tensor,
        ] = {}

    def _kernels(
        self,
        depth_count: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        cache_key = (depth_count, device, dtype)
        if cache_key in self._kernel_cache:
            return self._kernel_cache[cache_key]

        dx, dy, dz = self.cell_size
        half = self.kernel_size // 2
        x_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dx
        y_offsets = (
            torch.arange(self.kernel_size, device=device, dtype=dtype) - half
        ) * dy
        xx, yy = torch.meshgrid(x_offsets, y_offsets, indexing="ij")

        kernels = []
        for z_index in range(depth_count):
            depth = (depth_count - z_index) * dz + self.observation_height
            z = torch.as_tensor(depth, device=device, dtype=dtype)
            r2 = xx.square() + yy.square() + z.square()
            kernel = (3.0 * z.square() - r2) / torch.pow(r2 + self.eps, 2.5)
            kernels.append(kernel)

        stacked = torch.stack(kernels, dim=0).unsqueeze(1).unsqueeze(1)
        self._kernel_cache[cache_key] = stacked
        return stacked

    def forward(self, susceptibility: torch.Tensor) -> torch.Tensor:
        volume = _as_5d_volume(susceptibility).to(dtype=torch.float32)
        _, _, _, _, depth_count = volume.shape
        kernels = self._kernels(depth_count, volume.device, volume.dtype)

        dx, dy, dz = self.cell_size
        scale = self.field_strength * dx * dy * dz
        anomaly = torch.zeros(
            volume.shape[0],
            1,
            volume.shape[2],
            volume.shape[3],
            dtype=volume.dtype,
            device=volume.device,
        )
        padding = self.kernel_size // 2
        for z_index in range(depth_count):
            anomaly = anomaly + F.conv2d(
                volume[..., z_index],
                kernels[z_index],
                padding=padding,
            )
        anomaly = anomaly * scale
        if self.remove_mean:
            anomaly = anomaly - anomaly.mean(dim=(-2, -1), keepdim=True)
        return anomaly

    def __call__(self, susceptibility: torch.Tensor) -> torch.Tensor:
        return self.forward(susceptibility)


def normalized_misfit(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    eps: float = 1e-8,
    reduction: str = "mean",
) -> torch.Tensor:
    """Compute normalized RMS misfit between predicted and observed fields.

    With ``reduction="none"``, returns one score per realization. Lower scores
    are better.
    """
    predicted_field = _as_4d_field(predicted)
    observed_field = _as_4d_field(observed).to(
        device=predicted_field.device,
        dtype=predicted_field.dtype,
    )
    observed_field = torch.broadcast_to(observed_field, predicted_field.shape)

    if mask is None:
        weight = torch.ones_like(predicted_field)
    else:
        weight = _as_4d_field(mask).to(
            device=predicted_field.device,
            dtype=predicted_field.dtype,
        )
        weight = torch.broadcast_to(weight, predicted_field.shape)

    residual = (predicted_field - observed_field) * weight
    observed_weighted = observed_field * weight
    count = weight.sum(dim=(1, 2, 3)).clamp_min(1.0)

    residual_rms = residual.square().sum(dim=(1, 2, 3)).div(count).sqrt()
    observed_rms = observed_weighted.square().sum(dim=(1, 2, 3)).div(count).sqrt()
    scores = residual_rms / (observed_rms + eps)

    if reduction == "none":
        return scores
    if reduction == "mean":
        return scores.mean()
    if reduction == "sum":
        return scores.sum()
    raise ValueError("reduction must be 'none', 'mean', or 'sum'")


def _prepare_label_tensors(
    predictions: torch.Tensor,
    target: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Normalize categorical predictions and target to matching 5D tensors."""
    predicted_labels = _as_5d_volume(predictions).long()
    target_labels = _as_5d_volume(target).to(predicted_labels.device).long()

    if predicted_labels.shape[1:] != target_labels.shape[1:]:
        raise ValueError(
            "predictions and target must have matching [1, X, Y, Z] dimensions"
        )
    if target_labels.shape[0] == 1 and predicted_labels.shape[0] > 1:
        target_labels = target_labels.expand(predicted_labels.shape[0], -1, -1, -1, -1)
    elif target_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("target batch size must be 1 or match predictions")

    return predicted_labels, target_labels


def voxel_accuracy(
    predictions: torch.Tensor,
    target: torch.Tensor,
    ignore_label: Optional[int] = -1,
) -> torch.Tensor:
    """Return per-sample categorical voxel accuracy with shape [B]."""
    predicted_labels, target_labels = _prepare_label_tensors(predictions, target)
    valid = torch.ones_like(target_labels, dtype=torch.bool)
    if ignore_label is not None:
        valid &= target_labels != int(ignore_label)

    correct = (predicted_labels == target_labels) & valid
    valid_count = valid.flatten(1).sum(dim=1)
    correct_count = correct.flatten(1).sum(dim=1)
    accuracy = correct_count.float() / valid_count.clamp_min(1).float()
    return torch.where(
        valid_count > 0,
        accuracy,
        torch.full_like(accuracy, torch.nan),
    )


def class_iou(
    predictions: torch.Tensor,
    target: torch.Tensor,
    class_ids: Optional[Sequence[int]] = None,
    ignore_label: Optional[int] = -1,
) -> torch.Tensor:
    """Return per-sample, per-class IoU with shape [B, C].

    Classes absent from both prediction and target for a sample receive NaN and
    are excluded by :func:`mean_iou`.
    """
    predicted_labels, target_labels = _prepare_label_tensors(predictions, target)
    valid = torch.ones_like(target_labels, dtype=torch.bool)
    if ignore_label is not None:
        valid &= target_labels != int(ignore_label)

    if class_ids is None:
        labels = torch.unique(
            torch.cat((predicted_labels[valid], target_labels[valid]))
        )
        if ignore_label is not None:
            labels = labels[labels != int(ignore_label)]
        labels = labels.sort().values
    else:
        labels = torch.as_tensor(
            list(class_ids),
            dtype=torch.long,
            device=predicted_labels.device,
        )
        if ignore_label is not None:
            labels = labels[labels != int(ignore_label)]

    if labels.numel() == 0:
        return torch.empty(
            (predicted_labels.shape[0], 0),
            dtype=torch.float32,
            device=predicted_labels.device,
        )

    per_class = []
    for class_id in labels:
        predicted_class = (predicted_labels == class_id) & valid
        target_class = (target_labels == class_id) & valid
        intersection = (predicted_class & target_class).flatten(1).sum(dim=1)
        union = (predicted_class | target_class).flatten(1).sum(dim=1)
        iou = intersection.float() / union.clamp_min(1).float()
        per_class.append(
            torch.where(union > 0, iou, torch.full_like(iou, torch.nan))
        )

    return torch.stack(per_class, dim=1)


def mean_iou(
    predictions: torch.Tensor,
    target: torch.Tensor,
    class_ids: Optional[Sequence[int]] = None,
    ignore_label: Optional[int] = -1,
) -> torch.Tensor:
    """Return mean IoU per sample with shape [B], excluding absent classes."""
    per_class = class_iou(
        predictions,
        target,
        class_ids=class_ids,
        ignore_label=ignore_label,
    )
    if per_class.shape[1] == 0:
        return torch.full(
            (per_class.shape[0],),
            torch.nan,
            dtype=per_class.dtype,
            device=per_class.device,
        )

    valid = torch.isfinite(per_class)
    valid_count = valid.sum(dim=1)
    total = torch.where(valid, per_class, torch.zeros_like(per_class)).sum(dim=1)
    result = total / valid_count.clamp_min(1)
    return torch.where(
        valid_count > 0,
        result,
        torch.full_like(result, torch.nan),
    )


def _average_ranks(values: torch.Tensor) -> torch.Tensor:
    """Assign one-based average ranks, including average ranks for ties."""
    order = torch.argsort(values, stable=True)
    sorted_values = values[order]
    _, inverse, counts = torch.unique_consecutive(
        sorted_values,
        return_inverse=True,
        return_counts=True,
    )
    starts = counts.cumsum(dim=0) - counts
    average = starts.to(torch.float64) + (counts.to(torch.float64) + 1.0) / 2.0
    ranks = torch.empty_like(values, dtype=torch.float64)
    ranks[order] = average[inverse]
    return ranks


def spearman_correlation(x: torch.Tensor, y: torch.Tensor) -> float:
    """Compute a tie-aware Spearman correlation without SciPy or sklearn."""
    x_values = torch.as_tensor(x, dtype=torch.float64).flatten()
    y_values = torch.as_tensor(y, dtype=torch.float64, device=x_values.device).flatten()
    if x_values.numel() != y_values.numel():
        raise ValueError("x and y must contain the same number of values")

    finite = torch.isfinite(x_values) & torch.isfinite(y_values)
    x_values = x_values[finite]
    y_values = y_values[finite]
    if x_values.numel() < 2:
        return float("nan")

    x_ranks = _average_ranks(x_values)
    y_ranks = _average_ranks(y_values)
    x_centered = x_ranks - x_ranks.mean()
    y_centered = y_ranks - y_ranks.mean()
    denominator = torch.sqrt(
        x_centered.square().sum() * y_centered.square().sum()
    )
    if denominator.item() == 0.0:
        return float("nan")
    return float((x_centered * y_centered).sum().div(denominator).item())


@dataclass(frozen=True)
class GeophysicsRanking:
    """Container returned by rank_realizations_by_geophysics."""

    ranked_indices: torch.Tensor
    ranked_misfits: torch.Tensor
    all_misfits: torch.Tensor
    predicted_gravity: torch.Tensor

    @property
    def best_index(self) -> int:
        return int(self.ranked_indices[0].item())


def rank_realizations_by_geophysics(
    realizations: torch.Tensor,
    observed_gravity: torch.Tensor,
    property_map: Optional[LithologyPropertyMap] = None,
    forward_model: Optional[SimpleGravityForward] = None,
    mask: Optional[torch.Tensor] = None,
) -> GeophysicsRanking:
    """Rank categorical geology realizations by normalized gravity misfit.

    Parameters
    ----------
    realizations:
        Categorical decoded volumes. Accepted shapes are [B, 1, X, Y, Z],
        [B, X, Y, Z], [1, X, Y, Z], or [X, Y, Z].
    observed_gravity:
        Observed or reference gravity field. Accepted shapes are [B, 1, X, Y],
        [B, X, Y], [1, X, Y], or [X, Y].
    property_map:
        Lithology-to-density mapping. Defaults to relative density contrasts.
    forward_model:
        Gravity proxy. Defaults to ``SimpleGravityForward()``.
    mask:
        Optional gravity observation mask where non-zero entries are used.
    """
    mapper = LithologyPropertyMap() if property_map is None else property_map
    forward = SimpleGravityForward() if forward_model is None else forward_model

    density = mapper(realizations)
    predicted_gravity = forward(density)
    all_misfits = normalized_misfit(
        predicted_gravity,
        observed_gravity,
        mask=mask,
        reduction="none",
    )
    ranked_misfits, ranked_indices = torch.sort(all_misfits)

    return GeophysicsRanking(
        ranked_indices=ranked_indices,
        ranked_misfits=ranked_misfits,
        all_misfits=all_misfits,
        predicted_gravity=predicted_gravity,
    )
