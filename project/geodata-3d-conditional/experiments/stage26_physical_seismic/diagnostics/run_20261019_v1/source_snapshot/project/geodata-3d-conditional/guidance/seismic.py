"""Differentiable normal-incidence convolutional seismic for Phase 4c.

The first Phase-4c operator is an explicitly synthetic post-stack upper bound,
not a wave-equation simulator.  It maps expected acoustic impedance and
slowness to primary reflection coefficients, deposits them on a regular
two-way-time grid and applies a fixed zero-phase Ricker wavelet independently
to every lateral trace.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from guided_geophysical_sampling import soft_decode_to_probs


SEISMIC_PROTOCOL_VERSION = 1
ACOUSTIC_CODEBOOK_SCHEMA = "phase4_acoustic_codebook_v1"
SEISMIC_OBSERVATION_CONFIG_SCHEMA = "phase4_seismic_observation_v1"
SEISMIC_FORWARD_MODE = "normal_incidence_twt_ricker_v1"
SEISMIC_LOSS_MODE = "diagonal_uncertainty_normalized_mse_v1"
SUBSURFACE_SOFT_ACOUSTIC_POLICY = "exclude_air_and_renormalize_known_subsurface_v1"


def tensor_sha256(value: torch.Tensor) -> str:
    """Return a stable hash over dtype, shape and contiguous tensor bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three integers")
    parsed = tuple(int(item) for item in value)
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def _positive_float_triple(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        parsed = (float(value),) * 3
    elif (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 3
    ):
        parsed = tuple(float(item) for item in value)
    else:
        raise ValueError(f"{name} must be a scalar or three-value array")
    if not all(math.isfinite(item) and item > 0 for item in parsed):
        raise ValueError(f"{name} values must be finite and positive")
    return parsed


@dataclass(frozen=True)
class AcousticTables:
    """Complete raw-label acoustic codebook in category-channel order."""

    density_kg_m3: torch.Tensor
    velocity_m_s: torch.Tensor
    impedance_kg_m2_s: torch.Tensor
    slowness_s_m: torch.Tensor

    @property
    def property_table(self) -> torch.Tensor:
        return torch.stack((self.impedance_kg_m2_s, self.slowness_s_m), dim=0)


def acoustic_tables_from_config(
    config: Mapping[str, object],
    num_categories: int,
) -> tuple[AcousticTables, dict[str, object]]:
    """Parse a complete positive density/P-wave-velocity codebook."""
    if config.get("schema") != ACOUSTIC_CODEBOOK_SCHEMA:
        raise ValueError(f"acoustic config schema must be {ACOUSTIC_CODEBOOK_SCHEMA!r}")
    if num_categories <= 1:
        raise ValueError("num_categories must be greater than one")
    if config.get("density_unit") not in {"kg m^-3", "kg/m^3"}:
        raise ValueError("density_unit must be 'kg m^-3' or 'kg/m^3'")
    if config.get("velocity_unit") not in {"m s^-1", "m/s"}:
        raise ValueError("velocity_unit must be 'm s^-1' or 'm/s'")
    values = config.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("acoustic config must contain a values object")

    parsed: dict[int, tuple[float, float]] = {}
    try:
        for label, entry in values.items():
            if not isinstance(entry, Mapping):
                raise TypeError
            parsed[int(label)] = (float(entry["density"]), float(entry["vp"]))
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("acoustic config contains invalid labels or properties") from exc
    expected = set(range(-1, num_categories - 1))
    missing = sorted(expected - set(parsed))
    extra = sorted(set(parsed) - expected)
    if missing or extra:
        raise ValueError(
            f"acoustic label coverage mismatch: missing={missing}, extra={extra}"
        )
    ordered = [parsed[category - 1] for category in range(num_categories)]
    density = torch.tensor([item[0] for item in ordered], dtype=torch.float32)
    velocity = torch.tensor([item[1] for item in ordered], dtype=torch.float32)
    if not torch.isfinite(density).all() or not torch.isfinite(velocity).all():
        raise ValueError("acoustic properties must be finite")
    if bool((density <= 0).any()) or bool((velocity <= 0).any()):
        raise ValueError("acoustic density and velocity must be positive")
    impedance = density * velocity
    slowness = velocity.reciprocal()
    tables = AcousticTables(
        density_kg_m3=density,
        velocity_m_s=velocity,
        impedance_kg_m2_s=impedance,
        slowness_s_m=slowness,
    )
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("acoustic config requires a non-empty id")
    metadata: dict[str, object] = {
        "schema": ACOUSTIC_CODEBOOK_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "scenario": str(config.get("scenario", "synthetic_acoustic")),
        "density_unit": "kg m^-3",
        "velocity_unit": "m s^-1",
        "impedance_unit": "kg m^-2 s^-1",
        "slowness_unit": "s m^-1",
        "num_categories": num_categories,
        "raw_label_range": [-1, num_categories - 2],
        "density_table_sha256": tensor_sha256(density),
        "velocity_table_sha256": tensor_sha256(velocity),
        "impedance_table_sha256": tensor_sha256(impedance),
        "slowness_table_sha256": tensor_sha256(slowness),
        "truth_derived": True,
        "site_calibrated_petrophysics": False,
    }
    return tables, metadata


def hard_labels_to_acoustic(
    labels: torch.Tensor,
    property_table: torch.Tensor,
) -> torch.Tensor:
    """Map raw labels ``-1..C-2`` to impedance/slowness ``[B,2,X,Y,Z]``."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if property_table.ndim != 2 or property_table.shape[0] != 2:
        raise ValueError("property_table must have shape [2,C]")
    if not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()):
        raise ValueError("labels must be finite and integer-valued")
    categories = labels.long() + 1
    if int(categories.min()) < 0 or int(categories.max()) >= property_table.shape[1]:
        raise ValueError(f"raw labels must be in [-1,{property_table.shape[1] - 2}]")
    table = property_table.to(device=labels.device)
    return table[:, categories[:, 0]].permute(1, 0, 2, 3, 4).contiguous()


def probabilities_to_acoustic(
    probabilities: torch.Tensor,
    property_table: torch.Tensor,
) -> torch.Tensor:
    """Map category probabilities to expected impedance and slowness."""
    if probabilities.ndim != 5:
        raise ValueError("probabilities must have shape [B,C,X,Y,Z]")
    if property_table.ndim != 2 or property_table.shape != (
        2,
        probabilities.shape[1],
    ):
        raise ValueError("property_table must have shape [2,C]")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    table = property_table.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return torch.einsum("bcxyz,pc->bpxyz", probabilities, table)


def probabilities_to_subsurface_acoustic(
    probabilities: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> torch.Tensor:
    """Use rock-conditional probabilities inside the known subsurface support.

    The air category is physically inadmissible below the already observed
    topographic surface.  Removing it from the soft acoustic mixture prevents
    transient air probability from creating nonphysical multi-second travel
    times, while decoded underground air remains visible to hard-label metrics.
    """
    expected_all = probabilities_to_acoustic(probabilities, property_table)
    if subsurface_mask.ndim != 5 or subsurface_mask.shape[1] != 1:
        raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
    if subsurface_mask.shape[2:] != probabilities.shape[2:]:
        raise ValueError("subsurface_mask spatial shape must match probabilities")
    if subsurface_mask.shape[0] not in (1, probabilities.shape[0]):
        raise ValueError("subsurface_mask batch must be one or match probabilities")
    rock_probabilities = probabilities[:, 1:]
    denominator = rock_probabilities.sum(dim=1, keepdim=True)
    if bool((denominator <= eps).any()):
        raise ValueError("non-air probability support is numerically empty")
    normalized_rock = rock_probabilities / denominator.clamp_min(eps)
    table = property_table.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    expected_rock = torch.einsum(
        "bcxyz,pc->bpxyz", normalized_rock, table[:, 1:]
    )
    mask = subsurface_mask.to(device=probabilities.device, dtype=torch.bool)
    mask = mask.expand(probabilities.shape[0], 2, *probabilities.shape[2:])
    return torch.where(mask, expected_rock, expected_all)


def overwrite_exact_condition_acoustic(
    predicted: torch.Tensor,
    target: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace known impedance/slowness before forward modeling."""
    if predicted.ndim != 5 or predicted.shape[1] != 2:
        raise ValueError("predicted must have shape [B,2,X,Y,Z]")
    if target.ndim != 5 or target.shape[1:] != predicted.shape[1:]:
        raise ValueError("target must match predicted channel/spatial shape")
    if target.shape[0] not in (1, predicted.shape[0]):
        raise ValueError("target batch must be one or match prediction")
    if condition_mask.ndim != 5 or condition_mask.shape[1] != 1:
        raise ValueError("condition_mask must have shape [B,1,X,Y,Z]")
    if condition_mask.shape[2:] != predicted.shape[2:]:
        raise ValueError("condition_mask spatial shape must match prediction")
    if condition_mask.shape[0] not in (1, predicted.shape[0]):
        raise ValueError("condition_mask batch must be one or match prediction")
    exact = target.to(device=predicted.device, dtype=predicted.dtype).expand_as(predicted)
    mask = condition_mask.to(device=predicted.device, dtype=torch.bool)
    mask = mask.expand(predicted.shape[0], 2, *predicted.shape[2:])
    return torch.where(mask, exact, predicted)


def validate_contiguous_subsurface_mask(mask: torch.Tensor) -> dict[str, object]:
    """Validate bottom-to-top columns containing rock followed only by air."""
    if mask.ndim != 5 or mask.shape[1] != 1:
        raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
    rock = mask.to(dtype=torch.bool)
    if bool((~rock).all(dim=-1).any()):
        raise ValueError("every lateral column must contain at least one subsurface cell")
    # Input z increases upward, so False -> True is a rock re-entry above air.
    if bool(((~rock[..., :-1]) & rock[..., 1:]).any()):
        raise ValueError("subsurface mask must be contiguous below the local surface")
    counts = rock.sum(dim=-1)
    return {
        "all_columns_nonempty": True,
        "contiguous_below_surface": True,
        "minimum_subsurface_cells": int(counts.min().item()),
        "maximum_subsurface_cells": int(counts.max().item()),
    }


def ricker_wavelet(
    peak_frequency_hz: float,
    sample_interval_ms: float,
    duration_ms: float,
    *,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Build an odd, peak-normalized zero-phase Ricker wavelet."""
    frequency = float(peak_frequency_hz)
    interval = float(sample_interval_ms)
    duration = float(duration_ms)
    if not all(math.isfinite(v) and v > 0 for v in (frequency, interval, duration)):
        raise ValueError("wavelet frequency, sample interval and duration must be positive")
    half_float = duration / (2.0 * interval)
    half_samples = int(round(half_float))
    if half_samples < 1 or not math.isclose(half_float, half_samples, abs_tol=1e-9):
        raise ValueError("wavelet duration must be an even multiple of sample_interval_ms")
    time_s = (
        torch.arange(-half_samples, half_samples + 1, device=device, dtype=dtype)
        * (interval / 1000.0)
    )
    squared = (math.pi * frequency * time_s).square()
    wavelet = (1.0 - 2.0 * squared) * torch.exp(-squared)
    wavelet = wavelet / wavelet.abs().max()
    if not torch.isfinite(wavelet).all():
        raise FloatingPointError("Ricker wavelet contains non-finite values")
    return wavelet.contiguous()


class ConvolutionalSeismic:
    """Independent normal-incidence convolutional traces on a regular TWT grid."""

    def __init__(
        self,
        grid_shape: Sequence[int],
        *,
        cell_size_m: float | Sequence[float],
        num_time_samples: int,
        sample_interval_ms: float,
        peak_frequency_hz: float,
        wavelet_duration_ms: float,
    ) -> None:
        self.grid_shape = _positive_int_triple(grid_shape, "grid_shape")
        self.cell_size_m = _positive_float_triple(cell_size_m, "cell_size_m")
        self.num_time_samples = int(num_time_samples)
        self.sample_interval_ms = float(sample_interval_ms)
        self.peak_frequency_hz = float(peak_frequency_hz)
        self.wavelet_duration_ms = float(wavelet_duration_ms)
        if self.num_time_samples <= 1:
            raise ValueError("num_time_samples must be greater than one")
        if not math.isfinite(self.sample_interval_ms) or self.sample_interval_ms <= 0:
            raise ValueError("sample_interval_ms must be finite and positive")
        self._wavelet_float64_cpu = ricker_wavelet(
            self.peak_frequency_hz,
            self.sample_interval_ms,
            self.wavelet_duration_ms,
            dtype=torch.float64,
        )

    @property
    def wavelet_num_samples(self) -> int:
        return int(self._wavelet_float64_cpu.numel())

    @property
    def recording_end_ms(self) -> float:
        return (self.num_time_samples - 1) * self.sample_interval_ms

    def wavelet(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        return self._wavelet_float64_cpu.to(device=device, dtype=dtype)

    def interface_response(
        self,
        impedance: torch.Tensor,
        slowness: torch.Tensor,
        subsurface_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return reflectivity, local-datum TWT and valid-interface mask."""
        for name, value in (("impedance", impedance), ("slowness", slowness)):
            if value.ndim != 5 or value.shape[1] != 1:
                raise ValueError(f"{name} must have shape [B,1,X,Y,Z]")
            if tuple(value.shape[2:]) != self.grid_shape:
                raise ValueError(f"{name} spatial shape must be {self.grid_shape}")
            if not value.is_floating_point() or not torch.isfinite(value).all():
                raise ValueError(f"{name} must be finite floating point")
            if bool((value <= 0).any()):
                raise ValueError(f"{name} must be positive")
        if slowness.shape != impedance.shape:
            raise ValueError("slowness must match impedance")
        if subsurface_mask.ndim != 5 or subsurface_mask.shape[1] != 1:
            raise ValueError("subsurface_mask must have shape [B,1,X,Y,Z]")
        if subsurface_mask.shape[2:] != impedance.shape[2:]:
            raise ValueError("subsurface_mask spatial shape must match impedance")
        if subsurface_mask.shape[0] not in (1, impedance.shape[0]):
            raise ValueError("subsurface_mask batch must be one or match impedance")

        rock = subsurface_mask.to(device=impedance.device, dtype=torch.bool)
        rock = rock.expand(impedance.shape[0], -1, -1, -1, -1)
        # Input z increases upward; traces and travel times start at local surface.
        z = impedance.flip(-1)
        s = slowness.flip(-1)
        rock_top_down = rock.flip(-1)
        valid = rock_top_down[..., :-1] & rock_top_down[..., 1:]
        above = z[..., :-1]
        below = z[..., 1:]
        reflectivity = (below - above) / (below + above).clamp_min(
            torch.finfo(z.dtype).tiny
        )
        layer_twt_ms = (
            2.0 * self.cell_size_m[2] * 1000.0 * s * rock_top_down.to(s.dtype)
        )
        interface_time_ms = layer_twt_ms.cumsum(dim=-1)[..., :-1]
        return reflectivity, interface_time_ms, valid

    def deposit_reflectivity(
        self,
        reflectivity: torch.Tensor,
        interface_time_ms: torch.Tensor,
        valid_interfaces: torch.Tensor,
        *,
        require_all_interfaces_in_window: bool = False,
    ) -> torch.Tensor:
        """Linearly deposit reflection coefficients onto regular TWT samples.

        Synthetic observation construction requires every truth interface to
        fit in the recording window. Predictions, like finite-duration field
        records, are cropped at the window rather than crashing when an
        incorrect slow model moves an arrival beyond the final sample.
        """
        if reflectivity.shape != interface_time_ms.shape:
            raise ValueError("reflectivity and interface_time_ms must match")
        if valid_interfaces.shape != reflectivity.shape:
            raise ValueError("valid_interfaces must match reflectivity")
        position = interface_time_ms / self.sample_interval_ms
        out_of_window = valid_interfaces & (
            (position < 0) | (position > self.num_time_samples - 1)
        )
        if require_all_interfaces_in_window and bool(out_of_window.any()):
            maximum = float(interface_time_ms[valid_interfaces].max().detach().cpu())
            raise ValueError(
                "valid seismic interface exceeds recording window: "
                f"max_twt_ms={maximum:.6g}, end_ms={self.recording_end_ms:.6g}"
            )
        left = torch.floor(position).long()
        right = left + 1
        fraction = position - left.to(position.dtype)
        left_valid = valid_interfaces & (left >= 0) & (left < self.num_time_samples)
        left_source = reflectivity * (1.0 - fraction) * left_valid.to(reflectivity.dtype)
        right_valid = valid_interfaces & (right < self.num_time_samples)
        right_source = reflectivity * fraction * right_valid.to(reflectivity.dtype)
        shape = (*reflectivity.shape[:-1], self.num_time_samples)
        spikes = torch.zeros(shape, device=reflectivity.device, dtype=reflectivity.dtype)
        spikes = spikes.scatter_add(-1, left.clamp(0, self.num_time_samples - 1), left_source)
        spikes = spikes.scatter_add(
            -1, right.clamp(0, self.num_time_samples - 1), right_source
        )
        return spikes

    def reflectivity_spikes(
        self,
        impedance: torch.Tensor,
        slowness: torch.Tensor,
        subsurface_mask: torch.Tensor,
        *,
        require_all_interfaces_in_window: bool = False,
    ) -> torch.Tensor:
        response = self.interface_response(impedance, slowness, subsurface_mask)
        return self.deposit_reflectivity(
            *response,
            require_all_interfaces_in_window=require_all_interfaces_in_window,
        )

    def convolve_reflectivity_spikes(self, spikes: torch.Tensor) -> torch.Tensor:
        """Apply the fixed wavelet independently with zero same-length padding."""
        if spikes.ndim != 5 or spikes.shape[1] != 1:
            raise ValueError("spikes must have shape [B,1,X,Y,T]")
        if tuple(spikes.shape[2:4]) != self.grid_shape[:2]:
            raise ValueError("spikes lateral shape must match the operator grid")
        if spikes.shape[-1] != self.num_time_samples:
            raise ValueError("spikes time length must match num_time_samples")
        if not spikes.is_floating_point() or not torch.isfinite(spikes).all():
            raise ValueError("spikes must be finite floating point")
        batch, _, nx, ny, nt = spikes.shape
        traces = spikes.reshape(batch * nx * ny, 1, nt)
        wavelet = self.wavelet(spikes.device, spikes.dtype).view(1, 1, -1)
        amplitudes = F.conv1d(
            traces,
            wavelet,
            padding=self.wavelet_num_samples // 2,
        )
        return amplitudes.reshape(batch, 1, nx, ny, nt).contiguous()

    def forward(
        self,
        impedance: torch.Tensor,
        slowness: torch.Tensor,
        subsurface_mask: torch.Tensor,
        *,
        require_all_interfaces_in_window: bool = False,
    ) -> torch.Tensor:
        """Return seismic amplitudes with shape ``[B,1,X,Y,T]``."""
        spikes = self.reflectivity_spikes(
            impedance,
            slowness,
            subsurface_mask,
            require_all_interfaces_in_window=require_all_interfaces_in_window,
        )
        amplitudes = self.convolve_reflectivity_spikes(spikes)
        if not torch.isfinite(amplitudes).all():
            raise FloatingPointError("seismic forward produced non-finite values")
        return amplitudes

    def __call__(
        self,
        impedance: torch.Tensor,
        slowness: torch.Tensor,
        subsurface_mask: torch.Tensor,
    ) -> torch.Tensor:
        return self.forward(impedance, slowness, subsurface_mask)

    def metadata(self) -> dict[str, object]:
        return {
            "forward_mode": SEISMIC_FORWARD_MODE,
            "grid_shape": list(self.grid_shape),
            "cell_size_m": list(self.cell_size_m),
            "axis_order": ["x", "y", "z"],
            "vertical_axis": "final spatial axis; larger index upward",
            "trace_order": "local_surface_to_depth",
            "local_surface_datum": True,
            "exclude_air_rock_interface": True,
            "num_time_samples": self.num_time_samples,
            "sample_interval_ms": self.sample_interval_ms,
            "recording_end_ms": self.recording_end_ms,
            "wavelet": {
                "type": "ricker_zero_phase",
                "peak_frequency_hz": self.peak_frequency_hz,
                "duration_ms": self.wavelet_duration_ms,
                "num_samples": self.wavelet_num_samples,
                "normalization": "peak_one",
                "sha256": tensor_sha256(self._wavelet_float64_cpu),
            },
            "lateral_mixing": False,
            "boundary": "zero_padding_same_length_no_wraparound",
            "prediction_recording_window_policy": "crop_arrivals_outside_fixed_window",
            "output_unit": "unscaled_convolutional_amplitude",
        }


def seismic_field_loss(
    predicted: torch.Tensor,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Diagonal-uncertainty normalized loss on valid trace samples."""
    if predicted.ndim != 5 or predicted.shape[1] != 1:
        raise ValueError("predicted must have shape [B,1,X,Y,T]")
    for name, value in (
        ("observed", observed),
        ("sample_mask", sample_mask),
        ("uncertainty", uncertainty),
    ):
        if value.ndim != 5 or value.shape[1:] != predicted.shape[1:]:
            raise ValueError(f"{name} must match predicted channel/spatial shape")
        if value.shape[0] not in (1, predicted.shape[0]):
            raise ValueError(f"{name} batch must be one or match prediction")
    observed_value = observed.to(device=predicted.device, dtype=predicted.dtype).expand_as(
        predicted
    )
    mask = sample_mask.to(device=predicted.device, dtype=predicted.dtype).expand_as(
        predicted
    )
    sigma = uncertainty.to(device=predicted.device, dtype=predicted.dtype).expand_as(
        predicted
    )
    if not all(torch.isfinite(value).all() for value in (predicted, observed_value, mask, sigma)):
        raise ValueError("seismic loss inputs must be finite")
    if bool((mask < 0).any()) or float(mask.sum()) <= 0:
        raise ValueError("sample_mask must be non-negative with positive support")
    if bool((sigma <= 0).any()):
        raise ValueError("uncertainty must be positive")
    difference = predicted - observed_value
    denominator = mask.sum().clamp_min(eps)
    loss = (mask * (difference / sigma.clamp_min(eps)).square()).sum() / denominator
    diagnostics = {
        "seismic_loss": loss,
        "seismic_rmse_amplitude": torch.sqrt((mask * difference.square()).sum() / denominator),
        "seismic_mae_amplitude": (mask * difference.abs()).sum() / denominator,
        "valid_seismic_sample_count": mask.sum(),
    }
    return loss, diagnostics


def seismic_volume_loss(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    property_table: torch.Tensor,
    target_acoustic: torch.Tensor,
    condition_mask: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator: ConvolutionalSeismic,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    *,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable soft decode -> acoustic properties -> seismic loss."""
    probabilities = soft_decode_to_probs(state, embedding_weight, tau=tau)
    predicted = probabilities_to_subsurface_acoustic(
        probabilities,
        property_table,
        subsurface_mask,
    )
    known = overwrite_exact_condition_acoustic(predicted, target_acoustic, condition_mask)
    predicted_seismic = forward_operator(
        known[:, 0:1],
        known[:, 1:2],
        subsurface_mask,
    )
    loss, diagnostics = seismic_field_loss(
        predicted_seismic,
        observed,
        sample_mask,
        uncertainty,
    )
    diagnostics.update(
        {
            "predicted_seismic": predicted_seismic,
            "predicted_acoustic": predicted,
            "known_acoustic": known,
        }
    )
    return loss, diagnostics


@dataclass(frozen=True)
class SeismicObservation:
    values: torch.Tensor
    noiseless: torch.Tensor
    noise: torch.Tensor
    sample_mask: torch.Tensor
    uncertainty: torch.Tensor
    subsurface_mask: torch.Tensor
    metadata: dict[str, object]


def build_seismic_observation(
    truth_acoustic: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator: ConvolutionalSeismic,
    *,
    sample_mask: torch.Tensor | None = None,
    uncertainty_amplitude: float | torch.Tensor = 0.01,
    noise_std_amplitude: float = 0.0,
    noise_seed: int = 0,
) -> SeismicObservation:
    """Build one immutable truth-derived convolutional seismic observation."""
    if truth_acoustic.ndim != 5 or truth_acoustic.shape[1] != 2:
        raise ValueError("truth_acoustic must have shape [B,2,X,Y,Z]")
    subsurface_report = validate_contiguous_subsurface_mask(subsurface_mask)
    noiseless = forward_operator.forward(
        truth_acoustic[:, 0:1],
        truth_acoustic[:, 1:2],
        subsurface_mask,
        require_all_interfaces_in_window=True,
    )
    if sample_mask is None:
        mask = torch.ones_like(noiseless)
    else:
        mask = sample_mask.to(device=noiseless.device, dtype=noiseless.dtype)
        if mask.shape != noiseless.shape:
            raise ValueError("sample_mask must match seismic data shape")
    if isinstance(uncertainty_amplitude, torch.Tensor):
        uncertainty = uncertainty_amplitude.to(
            device=noiseless.device, dtype=noiseless.dtype
        )
        if uncertainty.shape != noiseless.shape:
            raise ValueError("uncertainty tensor must match seismic data shape")
    else:
        uncertainty = torch.full_like(noiseless, float(uncertainty_amplitude))
    if not torch.isfinite(mask).all() or bool((mask < 0).any()) or float(mask.sum()) <= 0:
        raise ValueError("sample mask must be finite, non-negative and non-empty")
    if not torch.isfinite(uncertainty).all() or bool((uncertainty <= 0).any()):
        raise ValueError("uncertainty must be finite and positive")
    noise_std = float(noise_std_amplitude)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std_amplitude must be finite and non-negative")
    if noise_std == 0:
        noise = torch.zeros_like(noiseless)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        noise_cpu = torch.randn(
            noiseless.shape,
            generator=generator,
            device="cpu",
            dtype=noiseless.dtype,
        ) * noise_std
        noise = noise_cpu.to(noiseless.device) * mask
    values = noiseless + noise
    metadata = {
        **forward_operator.metadata(),
        "protocol_version": SEISMIC_PROTOCOL_VERSION,
        "loss_mode": SEISMIC_LOSS_MODE,
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "noise_model": "none" if noise_std == 0 else "fixed_additive_gaussian_amplitude",
        "noise_std_amplitude": noise_std,
        "noise_seed": int(noise_seed),
        "subsurface_mask_report": subsurface_report,
        "values_sha256": tensor_sha256(values),
        "noiseless_sha256": tensor_sha256(noiseless),
        "noise_sha256": tensor_sha256(noise),
        "sample_mask_sha256": tensor_sha256(mask),
        "uncertainty_sha256": tensor_sha256(uncertainty),
        "subsurface_mask_sha256": tensor_sha256(subsurface_mask.to(torch.bool)),
    }
    return SeismicObservation(
        values=values,
        noiseless=noiseless,
        noise=noise,
        sample_mask=mask,
        uncertainty=uncertainty,
        subsurface_mask=subsurface_mask.to(torch.bool),
        metadata=metadata,
    )


def validate_seismic_observation_config(
    config: Mapping[str, object],
    *,
    grid_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Validate the first full-cube Phase-4c observation configuration."""
    if config.get("schema") != SEISMIC_OBSERVATION_CONFIG_SCHEMA:
        raise ValueError(
            f"observation schema must be {SEISMIC_OBSERVATION_CONFIG_SCHEMA!r}"
        )
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("seismic observation config requires a non-empty id")
    configured_shape = _positive_int_triple(config.get("grid_shape"), "grid_shape")
    if grid_shape is not None and tuple(int(v) for v in grid_shape) != configured_shape:
        raise ValueError(
            f"configured grid_shape {configured_shape} does not match data {tuple(grid_shape)}"
        )
    cell_size = _positive_float_triple(config.get("cell_size_m"), "cell_size_m")
    if config.get("vertical_axis") != "final spatial axis; larger index upward":
        raise ValueError("vertical_axis must match the project tensor convention")
    if config.get("trace_order") != "local_surface_to_depth":
        raise ValueError("trace_order must be local_surface_to_depth")
    if config.get("local_surface_datum") is not True:
        raise ValueError("first Phase-4c config requires local_surface_datum=true")
    if config.get("exclude_air_rock_interface") is not True:
        raise ValueError("first Phase-4c config must exclude the air/rock interface")
    if config.get("trace_grid") != "all_xy_columns" or config.get("sample_mask") != "all":
        raise ValueError("first Phase-4c config requires all traces and samples")
    time_sampling = config.get("time_sampling")
    wavelet = config.get("wavelet")
    noise = config.get("noise")
    if not isinstance(time_sampling, Mapping):
        raise ValueError("observation config requires time_sampling")
    if not isinstance(wavelet, Mapping):
        raise ValueError("observation config requires wavelet")
    if not isinstance(noise, Mapping):
        raise ValueError("observation config requires noise")
    num_samples = int(time_sampling.get("num_samples", 0))
    sample_interval_ms = float(time_sampling.get("sample_interval_ms", float("nan")))
    if num_samples <= 1 or not math.isfinite(sample_interval_ms) or sample_interval_ms <= 0:
        raise ValueError("invalid seismic time sampling")
    if wavelet.get("type") != "ricker_zero_phase" or wavelet.get("normalization") != "peak_one":
        raise ValueError("first Phase-4c wavelet must be peak-one zero-phase Ricker")
    peak_frequency_hz = float(wavelet.get("peak_frequency_hz", float("nan")))
    duration_ms = float(wavelet.get("duration_ms", float("nan")))
    # Construction validates exact odd support and all positive values.
    ricker_wavelet(peak_frequency_hz, sample_interval_ms, duration_ms)
    uncertainty = float(config.get("uncertainty_amplitude", float("nan")))
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("uncertainty_amplitude must be finite and positive")
    noise_type = str(noise.get("type", ""))
    if noise_type not in {"none", "fixed_gaussian_amplitude"}:
        raise ValueError("noise type must be none or fixed_gaussian_amplitude")
    noise_std = 0.0 if noise_type == "none" else float(noise.get("std_amplitude", float("nan")))
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise std_amplitude must be finite and non-negative")
    if noise_type == "fixed_gaussian_amplitude" and noise_std <= 0:
        raise ValueError("fixed Gaussian noise std_amplitude must be positive")
    if config.get("truth_derived") is not True or config.get("measured_geophysics") is not False:
        raise ValueError("Phase-4c synthetic config must declare truth/measured status")
    if config.get("inverse_crime") is not True:
        raise ValueError("first Phase-4c config must declare inverse_crime=true")
    return {
        "schema": SEISMIC_OBSERVATION_CONFIG_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "grid_shape": list(configured_shape),
        "cell_size_m": list(cell_size),
        "axis_order": ["x", "y", "z"],
        "vertical_axis": "final spatial axis; larger index upward",
        "trace_order": "local_surface_to_depth",
        "local_surface_datum": True,
        "exclude_air_rock_interface": True,
        "time_sampling": {
            "num_samples": num_samples,
            "sample_interval_ms": sample_interval_ms,
        },
        "wavelet": {
            "type": "ricker_zero_phase",
            "peak_frequency_hz": peak_frequency_hz,
            "duration_ms": duration_ms,
            "normalization": "peak_one",
        },
        "trace_grid": "all_xy_columns",
        "sample_mask": "all",
        "uncertainty_amplitude": uncertainty,
        "noise": {
            "type": noise_type,
            "std_amplitude": noise_std,
            "seed": int(noise.get("seed", 0)),
        },
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
    }


def seismic_operator_from_config(
    config: Mapping[str, object],
    *,
    grid_shape: Sequence[int] | None = None,
) -> tuple[ConvolutionalSeismic, dict[str, object]]:
    resolved = validate_seismic_observation_config(config, grid_shape=grid_shape)
    time_sampling = resolved["time_sampling"]
    wavelet = resolved["wavelet"]
    operator = ConvolutionalSeismic(
        resolved["grid_shape"],
        cell_size_m=resolved["cell_size_m"],
        num_time_samples=int(time_sampling["num_samples"]),
        sample_interval_ms=float(time_sampling["sample_interval_ms"]),
        peak_frequency_hz=float(wavelet["peak_frequency_hz"]),
        wavelet_duration_ms=float(wavelet["duration_ms"]),
    )
    return operator, resolved
