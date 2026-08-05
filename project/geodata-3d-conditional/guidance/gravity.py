"""Full-support differentiable rectangular-prism gravity for Phase 4.

Unlike the historical local ``SimpleGravityForward`` proxy, this module uses
SI geometry, the analytic vertical attraction of rectangular prisms, every
cell at every aligned surface station, and reports downward-positive ``g_z``
in mGal.  The regular-grid implementation evaluates the exact translation-
invariant prism kernel with zero-padded FFT linear convolution; it never uses
circular wraparound or a truncated horizontal footprint.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Mapping, Sequence

import torch

from guided_geophysical_sampling import soft_decode_to_probs


GRAVITATIONAL_CONSTANT_SI = 6.67430e-11
SI_TO_MGAL = 1.0e5
GRAVITY_PROTOCOL_VERSION = 1
GRAVITY_DENSITY_CONFIG_SCHEMA = "phase4_density_contrast_v1"
GRAVITY_OBSERVATION_CONFIG_SCHEMA = "phase4_gravity_observation_v1"
GRAVITY_FORWARD_MODE = "full_support_rectangular_prism_fft_v1"
GRAVITY_LOSS_MODE = "diagonal_uncertainty_normalized_mse_v1"


def tensor_sha256(value: torch.Tensor) -> str:
    """Return a stable hash over dtype, shape and contiguous tensor bytes."""
    tensor = value.detach().cpu().contiguous()
    digest = hashlib.sha256()
    digest.update(str(tensor.dtype).encode("utf-8"))
    digest.update(str(tuple(tensor.shape)).encode("utf-8"))
    digest.update(tensor.view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def _float_triple(value: object, name: str) -> tuple[float, float, float]:
    if isinstance(value, (int, float)):
        parsed = (float(value),) * 3
    elif isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        if len(value) != 3:
            raise ValueError(f"{name} must contain exactly three values")
        parsed = tuple(float(item) for item in value)
    else:
        raise ValueError(f"{name} must be a scalar or three-value array")
    if not all(math.isfinite(item) for item in parsed):
        raise ValueError(f"{name} must contain finite values")
    return parsed


def _positive_float_triple(value: object, name: str) -> tuple[float, float, float]:
    parsed = _float_triple(value, name)
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def _positive_int_triple(value: object, name: str) -> tuple[int, int, int]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)) or len(value) != 3:
        raise ValueError(f"{name} must contain exactly three integers")
    parsed = tuple(int(item) for item in value)
    if any(item <= 0 for item in parsed):
        raise ValueError(f"{name} values must be positive")
    return parsed


def density_table_from_config(
    config: Mapping[str, object],
    num_categories: int,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Parse a complete raw-label ``kg m^-3`` density-contrast codebook."""
    if config.get("schema") != GRAVITY_DENSITY_CONFIG_SCHEMA:
        raise ValueError(
            f"density config schema must be {GRAVITY_DENSITY_CONFIG_SCHEMA!r}"
        )
    if num_categories <= 1:
        raise ValueError("num_categories must be greater than one")
    units = str(config.get("unit", "")).strip()
    if units not in {"kg m^-3", "kg/m^3"}:
        raise ValueError("density config unit must be 'kg m^-3' or 'kg/m^3'")
    values = config.get("values")
    if not isinstance(values, Mapping):
        raise ValueError("density config must contain a values object")
    try:
        parsed = {int(label): float(value) for label, value in values.items()}
    except (TypeError, ValueError) as exc:
        raise ValueError("density config contains invalid labels or values") from exc
    expected = set(range(-1, num_categories - 1))
    missing = sorted(expected - set(parsed))
    extra = sorted(set(parsed) - expected)
    if missing or extra:
        raise ValueError(
            f"density label coverage mismatch: missing={missing}, extra={extra}"
        )
    table = torch.tensor(
        [parsed[category - 1] for category in range(num_categories)],
        dtype=torch.float32,
    )
    if not torch.isfinite(table).all():
        raise ValueError("density values must be finite")
    metadata: dict[str, object] = {
        "schema": GRAVITY_DENSITY_CONFIG_SCHEMA,
        "id": str(config.get("id", "")).strip(),
        "description": str(config.get("description", "")),
        "scenario": str(config.get("scenario", "synthetic_density_contrast")),
        "unit": "kg m^-3",
        "num_categories": num_categories,
        "raw_label_range": [-1, num_categories - 2],
        "density_table_sha256": tensor_sha256(table),
        "truth_derived": True,
        "site_calibrated_petrophysics": False,
    }
    if not metadata["id"]:
        raise ValueError("density config requires a non-empty id")
    return table, metadata


def hard_labels_to_density(
    labels: torch.Tensor,
    density_table: torch.Tensor,
) -> torch.Tensor:
    """Map raw labels ``-1..C-2`` to density ``[B,1,X,Y,Z]``."""
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if density_table.ndim != 1 or density_table.numel() < 2:
        raise ValueError("density_table must have shape [C] with C > 1")
    if not torch.isfinite(labels).all() or not torch.equal(labels, labels.round()):
        raise ValueError("labels must be finite and integer-valued")
    categories = labels.long() + 1
    if int(categories.min()) < 0 or int(categories.max()) >= density_table.numel():
        raise ValueError(
            f"raw labels must be in [-1,{density_table.numel() - 2}]"
        )
    table = density_table.to(device=labels.device)
    return table[categories[:, 0]].unsqueeze(1)


def probabilities_to_density(
    probabilities: torch.Tensor,
    density_table: torch.Tensor,
) -> torch.Tensor:
    """Map categorical probabilities to expected density contrast."""
    if probabilities.ndim != 5:
        raise ValueError("probabilities must have shape [B,C,X,Y,Z]")
    if density_table.ndim != 1 or density_table.numel() != probabilities.shape[1]:
        raise ValueError("density_table must have one value per probability channel")
    if not torch.isfinite(probabilities).all():
        raise ValueError("probabilities must be finite")
    table = density_table.to(
        device=probabilities.device,
        dtype=probabilities.dtype,
    )
    return torch.einsum("bcxyz,c->bxyz", probabilities, table).unsqueeze(1)


def overwrite_exact_condition_density(
    predicted_density: torch.Tensor,
    target_density: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    """Replace known voxels before forward modeling and block their gradient."""
    if predicted_density.ndim != 5 or predicted_density.shape[1] != 1:
        raise ValueError("predicted_density must have shape [B,1,X,Y,Z]")
    if target_density.ndim != 5 or target_density.shape[1:] != predicted_density.shape[1:]:
        raise ValueError("target_density must match predicted channel/spatial shape")
    if target_density.shape[0] not in (1, predicted_density.shape[0]):
        raise ValueError("target_density batch must be one or match prediction")
    if condition_mask.ndim != 5 or condition_mask.shape[1:] != predicted_density.shape[1:]:
        raise ValueError("condition_mask must match predicted channel/spatial shape")
    if condition_mask.shape[0] not in (1, predicted_density.shape[0]):
        raise ValueError("condition_mask batch must be one or match prediction")
    target = target_density.to(
        device=predicted_density.device,
        dtype=predicted_density.dtype,
    ).expand_as(predicted_density)
    mask = condition_mask.to(device=predicted_density.device, dtype=torch.bool).expand_as(
        predicted_density
    )
    return torch.where(mask, target, predicted_density)


def prism_gz_kernel_mgal(
    stations_m: torch.Tensor,
    prism_lower_m: torch.Tensor,
    prism_upper_m: torch.Tensor,
    *,
    gravitational_constant: float = GRAVITATIONAL_CONSTANT_SI,
) -> torch.Tensor:
    """Return downward-positive unit-density rectangular-prism ``g_z``.

    ``stations_m`` has shape ``[S,3]`` and prism bounds have shape ``[N,3]``.
    The returned matrix ``[S,N]`` maps density in ``kg m^-3`` to mGal.  The
    analytic corner formula is evaluated with z positive upward; stations may
    not lie on a prism face/edge where the primitive is singular.
    """
    if stations_m.ndim != 2 or stations_m.shape[1] != 3:
        raise ValueError("stations_m must have shape [S,3]")
    if prism_lower_m.ndim != 2 or prism_lower_m.shape[1] != 3:
        raise ValueError("prism_lower_m must have shape [N,3]")
    if prism_upper_m.shape != prism_lower_m.shape:
        raise ValueError("prism_upper_m must match prism_lower_m")
    if stations_m.device != prism_lower_m.device or prism_upper_m.device != stations_m.device:
        raise ValueError("stations and prism bounds must share a device")
    if stations_m.dtype != prism_lower_m.dtype or prism_upper_m.dtype != stations_m.dtype:
        raise ValueError("stations and prism bounds must share a dtype")
    if not stations_m.is_floating_point():
        raise ValueError("geometry tensors must be floating point")
    if not (
        torch.isfinite(stations_m).all()
        and torch.isfinite(prism_lower_m).all()
        and torch.isfinite(prism_upper_m).all()
    ):
        raise ValueError("geometry tensors must be finite")
    if bool((prism_upper_m <= prism_lower_m).any()):
        raise ValueError("every prism upper bound must exceed its lower bound")
    if not math.isfinite(gravitational_constant) or gravitational_constant <= 0:
        raise ValueError("gravitational_constant must be finite and positive")

    lower = prism_lower_m.unsqueeze(0) - stations_m.unsqueeze(1)
    upper = prism_upper_m.unsqueeze(0) - stations_m.unsqueeze(1)
    total = torch.zeros(
        (stations_m.shape[0], prism_lower_m.shape[0]),
        device=stations_m.device,
        dtype=stations_m.dtype,
    )
    tiny = torch.finfo(stations_m.dtype).tiny
    for ix, x in enumerate((lower[..., 0], upper[..., 0])):
        for iy, y in enumerate((lower[..., 1], upper[..., 1])):
            for iz, z in enumerate((lower[..., 2], upper[..., 2])):
                radius = torch.sqrt(x.square() + y.square() + z.square())
                if bool((radius == 0).any()) or bool((z == 0).any()):
                    raise ValueError("station lies on a singular prism corner plane")
                primitive = (
                    x * torch.log((y + radius).clamp_min(tiny))
                    + y * torch.log((x + radius).clamp_min(tiny))
                    - z * torch.atan((x * y) / (z * radius))
                )
                total = total + ((-1.0) ** (ix + iy + iz)) * primitive
    return -float(gravitational_constant) * SI_TO_MGAL * total


def _next_power_of_two(value: int) -> int:
    return 1 << (int(value) - 1).bit_length()


class RectangularPrismGravity:
    """Exact full-support gravity on an aligned regular surface grid."""

    def __init__(
        self,
        grid_shape: Sequence[int],
        *,
        cell_size_m: float | Sequence[float],
        origin_m: Sequence[float] = (0.0, 0.0, 0.0),
        observation_height_m: float,
        gravitational_constant: float = GRAVITATIONAL_CONSTANT_SI,
    ) -> None:
        self.grid_shape = _positive_int_triple(grid_shape, "grid_shape")
        self.cell_size_m = _positive_float_triple(cell_size_m, "cell_size_m")
        self.origin_m = _float_triple(origin_m, "origin_m")
        self.observation_height_m = float(observation_height_m)
        self.gravitational_constant = float(gravitational_constant)
        if not math.isfinite(self.observation_height_m) or self.observation_height_m <= 0:
            raise ValueError("observation_height_m must be finite and positive")
        if not math.isfinite(self.gravitational_constant) or self.gravitational_constant <= 0:
            raise ValueError("gravitational_constant must be finite and positive")
        nx, ny, _ = self.grid_shape
        self.fft_shape = (
            _next_power_of_two(3 * nx - 2),
            _next_power_of_two(3 * ny - 2),
        )
        self._kernel_cache: dict[tuple[str, int | None, torch.dtype], torch.Tensor] = {}
        self._spatial_kernel_float64_cpu: torch.Tensor | None = None

    @property
    def station_height_m(self) -> float:
        return (
            self.origin_m[2]
            + self.grid_shape[2] * self.cell_size_m[2]
            + self.observation_height_m
        )

    def station_coordinates(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> torch.Tensor:
        """Return aligned station coordinates with x-major flattening."""
        nx, ny, _ = self.grid_shape
        dx, dy, _ = self.cell_size_m
        x = self.origin_m[0] + (torch.arange(nx, device=device, dtype=dtype) + 0.5) * dx
        y = self.origin_m[1] + (torch.arange(ny, device=device, dtype=dtype) + 0.5) * dy
        xx, yy = torch.meshgrid(x, y, indexing="ij")
        zz = torch.full_like(xx, self.station_height_m)
        return torch.stack((xx, yy, zz), dim=-1).reshape(-1, 3)

    def prism_bounds(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float64,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return lower/upper cell bounds with z as the fastest index."""
        nx, ny, nz = self.grid_shape
        dx, dy, dz = self.cell_size_m
        ix, iy, iz = torch.meshgrid(
            torch.arange(nx, device=device, dtype=dtype),
            torch.arange(ny, device=device, dtype=dtype),
            torch.arange(nz, device=device, dtype=dtype),
            indexing="ij",
        )
        lower = torch.stack(
            (
                self.origin_m[0] + ix * dx,
                self.origin_m[1] + iy * dy,
                self.origin_m[2] + iz * dz,
            ),
            dim=-1,
        ).reshape(-1, 3)
        size = torch.tensor(self.cell_size_m, device=device, dtype=dtype)
        return lower, lower + size

    def _build_spatial_kernel_float64_cpu(self) -> torch.Tensor:
        """Construct analytic geometry in float64 to avoid far-field cancellation."""
        device = torch.device("cpu")
        dtype = torch.float64
        nx, ny, nz = self.grid_shape
        dx, dy, dz = self.cell_size_m
        ox = torch.arange(-(nx - 1), nx, device=device, dtype=dtype) * dx
        oy = torch.arange(-(ny - 1), ny, device=device, dtype=dtype) * dy
        xx, yy = torch.meshgrid(ox, oy, indexing="ij")
        horizontal_lower = torch.stack(
            (xx.reshape(-1) - dx / 2, yy.reshape(-1) - dy / 2), dim=-1
        )
        horizontal_upper = torch.stack(
            (xx.reshape(-1) + dx / 2, yy.reshape(-1) + dy / 2), dim=-1
        )
        station = torch.tensor(
            [[0.0, 0.0, self.station_height_m]],
            device=device,
            dtype=dtype,
        )
        layers: list[torch.Tensor] = []
        for z_index in range(nz):
            z_lower = self.origin_m[2] + z_index * dz
            lower = torch.cat(
                (
                    horizontal_lower,
                    torch.full(
                        (horizontal_lower.shape[0], 1),
                        z_lower,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=1,
            )
            upper = torch.cat(
                (
                    horizontal_upper,
                    torch.full(
                        (horizontal_upper.shape[0], 1),
                        z_lower + dz,
                        device=device,
                        dtype=dtype,
                    ),
                ),
                dim=1,
            )
            layer = prism_gz_kernel_mgal(
                station,
                lower,
                upper,
                gravitational_constant=self.gravitational_constant,
            )[0]
            layers.append(layer.reshape(2 * nx - 1, 2 * ny - 1))
        return torch.stack(layers, dim=0)

    def _spatial_kernel(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        if self._spatial_kernel_float64_cpu is None:
            self._spatial_kernel_float64_cpu = self._build_spatial_kernel_float64_cpu()
        return self._spatial_kernel_float64_cpu.to(device=device, dtype=dtype)

    def _kernel_fft(self, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
        key = (device.type, device.index, dtype)
        cached = self._kernel_cache.get(key)
        if cached is not None:
            return cached
        kernel = self._spatial_kernel(device, dtype)
        kernel_fft = torch.fft.rfft2(kernel, s=self.fft_shape)
        self._kernel_cache[key] = kernel_fft
        return kernel_fft

    def forward(self, density: torch.Tensor) -> torch.Tensor:
        """Map ``[B,1,X,Y,Z]`` kg/m3 density contrast to ``[B,1,X,Y]`` mGal."""
        if density.ndim != 5 or density.shape[1] != 1:
            raise ValueError("density must have shape [B,1,X,Y,Z]")
        if tuple(density.shape[2:]) != self.grid_shape:
            raise ValueError(
                f"density spatial shape must be {self.grid_shape}, got {tuple(density.shape[2:])}"
            )
        if not density.is_floating_point():
            raise ValueError("density must be floating point")
        if density.dtype not in (torch.float32, torch.float64):
            raise ValueError("density dtype must be float32 or float64")
        if not torch.isfinite(density).all():
            raise ValueError("density must be finite")
        nx, ny, nz = self.grid_shape
        volume = density[:, 0].permute(0, 3, 1, 2).contiguous()
        volume_fft = torch.fft.rfft2(volume, s=self.fft_shape)
        field_fft = (volume_fft * self._kernel_fft(density.device, density.dtype)).sum(dim=1)
        full = torch.fft.irfft2(field_fft, s=self.fft_shape)
        # Materialize the valid linear-convolution window.  Keeping a view here
        # would make torch.save retain the much larger padded FFT storage.
        field = full[:, nx - 1 : nx - 1 + nx, ny - 1 : ny - 1 + ny].contiguous()
        if not torch.isfinite(field).all():
            raise FloatingPointError("gravity forward produced non-finite values")
        return field.unsqueeze(1)

    def __call__(self, density: torch.Tensor) -> torch.Tensor:
        return self.forward(density)

    def metadata(self) -> dict[str, object]:
        return {
            "forward_mode": GRAVITY_FORWARD_MODE,
            "grid_shape": list(self.grid_shape),
            "cell_size_m": list(self.cell_size_m),
            "origin_m": list(self.origin_m),
            "axis_order": ["x", "y", "z"],
            "vertical_axis": "final spatial axis; larger index upward",
            "station_grid": "x/y cell-centre aligned full surface grid",
            "station_height_m": self.station_height_m,
            "observation_height_above_top_m": self.observation_height_m,
            "gravitational_constant_si": self.gravitational_constant,
            "output_unit": "mGal",
            "fft_shape": list(self.fft_shape),
            "full_support": True,
            "circular_wraparound": False,
        }


def gravity_field_loss(
    predicted_mgal: torch.Tensor,
    observed_mgal: torch.Tensor,
    survey_mask: torch.Tensor,
    uncertainty_mgal: torch.Tensor,
    *,
    eps: float = 1e-12,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Diagonal-uncertainty normalized field loss on valid stations."""
    if predicted_mgal.ndim != 4 or predicted_mgal.shape[1] != 1:
        raise ValueError("predicted_mgal must have shape [B,1,X,Y]")
    for name, value in (
        ("observed_mgal", observed_mgal),
        ("survey_mask", survey_mask),
        ("uncertainty_mgal", uncertainty_mgal),
    ):
        if value.ndim != 4 or value.shape[1:] != predicted_mgal.shape[1:]:
            raise ValueError(f"{name} must match predicted channel/spatial shape")
        if value.shape[0] not in (1, predicted_mgal.shape[0]):
            raise ValueError(f"{name} batch must be one or match prediction")
    observed = observed_mgal.to(
        device=predicted_mgal.device, dtype=predicted_mgal.dtype
    ).expand_as(predicted_mgal)
    mask = survey_mask.to(
        device=predicted_mgal.device, dtype=predicted_mgal.dtype
    ).expand_as(predicted_mgal)
    uncertainty = uncertainty_mgal.to(
        device=predicted_mgal.device, dtype=predicted_mgal.dtype
    ).expand_as(predicted_mgal)
    if not (
        torch.isfinite(predicted_mgal).all()
        and torch.isfinite(observed).all()
        and torch.isfinite(mask).all()
        and torch.isfinite(uncertainty).all()
    ):
        raise ValueError("gravity loss inputs must be finite")
    if bool((mask < 0).any()) or float(mask.sum()) <= 0:
        raise ValueError("survey_mask must be non-negative with positive support")
    if bool((uncertainty <= 0).any()):
        raise ValueError("uncertainty_mgal must be positive")
    normalized = (predicted_mgal - observed) / uncertainty.clamp_min(eps)
    denominator = mask.sum().clamp_min(eps)
    loss = (mask * normalized.square()).sum() / denominator
    diagnostics = {
        "gravity_loss": loss,
        "gravity_rmse_mgal": torch.sqrt(
            (mask * (predicted_mgal - observed).square()).sum() / denominator
        ),
        "gravity_mae_mgal": (
            mask * (predicted_mgal - observed).abs()
        ).sum()
        / denominator,
        "valid_station_count": mask.sum(),
    }
    return loss, diagnostics


def gravity_volume_loss(
    state: torch.Tensor,
    embedding_weight: torch.Tensor,
    density_table: torch.Tensor,
    target_density: torch.Tensor,
    condition_mask: torch.Tensor,
    forward_operator: RectangularPrismGravity,
    observed_mgal: torch.Tensor,
    survey_mask: torch.Tensor,
    uncertainty_mgal: torch.Tensor,
    *,
    tau: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Differentiable soft decode -> density -> gravity guidance loss."""
    probabilities = soft_decode_to_probs(state, embedding_weight, tau=tau)
    predicted_density = probabilities_to_density(probabilities, density_table)
    known_density = overwrite_exact_condition_density(
        predicted_density, target_density, condition_mask
    )
    predicted_mgal = forward_operator(known_density)
    loss, diagnostics = gravity_field_loss(
        predicted_mgal,
        observed_mgal,
        survey_mask,
        uncertainty_mgal,
    )
    diagnostics.update(
        {
            "predicted_gravity_mgal": predicted_mgal,
            "predicted_density": predicted_density,
            "known_density": known_density,
        }
    )
    return loss, diagnostics


@dataclass(frozen=True)
class GravityObservation:
    values_mgal: torch.Tensor
    noiseless_mgal: torch.Tensor
    noise_mgal: torch.Tensor
    survey_mask: torch.Tensor
    uncertainty_mgal: torch.Tensor
    metadata: dict[str, object]


def build_gravity_observation(
    truth_density: torch.Tensor,
    forward_operator: RectangularPrismGravity,
    *,
    survey_mask: torch.Tensor | None = None,
    uncertainty_mgal: float | torch.Tensor = 0.01,
    noise_std_mgal: float = 0.0,
    noise_seed: int = 0,
) -> GravityObservation:
    """Build one immutable truth-derived surface observation on CPU RNG."""
    noiseless = forward_operator(truth_density)
    shape = noiseless.shape
    if survey_mask is None:
        mask = torch.ones(shape, device=noiseless.device, dtype=noiseless.dtype)
    else:
        mask = survey_mask.to(device=noiseless.device, dtype=noiseless.dtype)
        if mask.shape != shape:
            raise ValueError("survey_mask must match gravity field shape")
    if isinstance(uncertainty_mgal, torch.Tensor):
        uncertainty = uncertainty_mgal.to(
            device=noiseless.device, dtype=noiseless.dtype
        )
        if uncertainty.shape != shape:
            raise ValueError("uncertainty_mgal tensor must match gravity field shape")
    else:
        value = float(uncertainty_mgal)
        uncertainty = torch.full_like(noiseless, value)
    if not torch.isfinite(mask).all() or bool((mask < 0).any()) or float(mask.sum()) <= 0:
        raise ValueError("survey mask must be finite, non-negative and non-empty")
    if not torch.isfinite(uncertainty).all() or bool((uncertainty <= 0).any()):
        raise ValueError("uncertainty must be finite and positive")
    noise_std = float(noise_std_mgal)
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise_std_mgal must be finite and non-negative")
    if noise_std == 0:
        noise = torch.zeros_like(noiseless)
    else:
        generator = torch.Generator(device="cpu").manual_seed(int(noise_seed))
        noise_cpu = torch.randn(
            shape,
            generator=generator,
            device="cpu",
            dtype=noiseless.dtype,
        ) * noise_std
        noise = noise_cpu.to(noiseless.device) * mask
    values = noiseless + noise
    metadata = {
        **forward_operator.metadata(),
        "protocol_version": GRAVITY_PROTOCOL_VERSION,
        "loss_mode": GRAVITY_LOSS_MODE,
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "noise_model": "none" if noise_std == 0 else "fixed_additive_gaussian_mgal",
        "noise_std_mgal": noise_std,
        "noise_seed": int(noise_seed),
        "values_sha256": tensor_sha256(values),
        "noiseless_sha256": tensor_sha256(noiseless),
        "noise_sha256": tensor_sha256(noise),
        "survey_mask_sha256": tensor_sha256(mask),
        "uncertainty_sha256": tensor_sha256(uncertainty),
    }
    return GravityObservation(
        values_mgal=values,
        noiseless_mgal=noiseless,
        noise_mgal=noise,
        survey_mask=mask,
        uncertainty_mgal=uncertainty,
        metadata=metadata,
    )


def validate_gravity_observation_config(
    config: Mapping[str, object],
    *,
    grid_shape: Sequence[int] | None = None,
) -> dict[str, object]:
    """Validate the first full-grid Phase-4a observation configuration."""
    if config.get("schema") != GRAVITY_OBSERVATION_CONFIG_SCHEMA:
        raise ValueError(
            f"observation schema must be {GRAVITY_OBSERVATION_CONFIG_SCHEMA!r}"
        )
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("gravity observation config requires a non-empty id")
    configured_shape = _positive_int_triple(config.get("grid_shape"), "grid_shape")
    if grid_shape is not None and tuple(int(v) for v in grid_shape) != configured_shape:
        raise ValueError(
            f"configured grid_shape {configured_shape} does not match data {tuple(grid_shape)}"
        )
    cell_size = _positive_float_triple(config.get("cell_size_m"), "cell_size_m")
    origin = _float_triple(config.get("origin_m", (0, 0, 0)), "origin_m")
    height = float(config.get("observation_height_above_top_m", float("nan")))
    uncertainty = float(config.get("uncertainty_mgal", float("nan")))
    if not math.isfinite(height) or height <= 0:
        raise ValueError("observation height must be finite and positive")
    if not math.isfinite(uncertainty) or uncertainty <= 0:
        raise ValueError("uncertainty_mgal must be finite and positive")
    if config.get("station_grid") != "cell_center_aligned_full":
        raise ValueError("first Phase-4a station_grid must be cell_center_aligned_full")
    if config.get("survey_mask") != "all":
        raise ValueError("first Phase-4a survey_mask must be all")
    noise = config.get("noise")
    if not isinstance(noise, Mapping):
        raise ValueError("gravity observation config requires a noise object")
    noise_type = str(noise.get("type", ""))
    if noise_type not in {"none", "fixed_gaussian_mgal"}:
        raise ValueError("noise type must be none or fixed_gaussian_mgal")
    noise_std = 0.0 if noise_type == "none" else float(noise.get("std_mgal", float("nan")))
    if not math.isfinite(noise_std) or noise_std < 0:
        raise ValueError("noise std_mgal must be finite and non-negative")
    if noise_type == "fixed_gaussian_mgal" and noise_std <= 0:
        raise ValueError("fixed Gaussian noise std_mgal must be positive")
    if config.get("truth_derived") is not True or config.get("measured_geophysics") is not False:
        raise ValueError("Phase-4a synthetic config must declare truth/measured status")
    if config.get("inverse_crime") is not True:
        raise ValueError("first Phase-4a config must declare inverse_crime=true")
    return {
        "schema": GRAVITY_OBSERVATION_CONFIG_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "grid_shape": list(configured_shape),
        "cell_size_m": list(cell_size),
        "origin_m": list(origin),
        "axis_order": ["x", "y", "z"],
        "vertical_axis": "final spatial axis; larger index upward",
        "station_grid": "cell_center_aligned_full",
        "survey_mask": "all",
        "observation_height_above_top_m": height,
        "uncertainty_mgal": uncertainty,
        "noise": {
            "type": noise_type,
            "std_mgal": noise_std,
            "seed": int(noise.get("seed", 0)),
        },
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
    }


def gravity_operator_from_config(
    config: Mapping[str, object],
    *,
    grid_shape: Sequence[int] | None = None,
) -> tuple[RectangularPrismGravity, dict[str, object]]:
    resolved = validate_gravity_observation_config(config, grid_shape=grid_shape)
    operator = RectangularPrismGravity(
        resolved["grid_shape"],
        cell_size_m=resolved["cell_size_m"],
        origin_m=resolved["origin_m"],
        observation_height_m=float(resolved["observation_height_above_top_m"]),
    )
    return operator, resolved
