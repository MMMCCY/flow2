"""Generator-free causal benchmarks for Phase 6Q.

The module intentionally has no checkpoint or flow-model dependency.  It
constructs one deterministic two-material five-body case, evaluates exact hard
candidate combinations, and compares a soft occupancy relaxation with a
hard-forward/soft-backward straight-through estimator.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
import math
from typing import Mapping, Sequence

import torch
import torch.nn.functional as F

from guidance.gravity import probabilities_to_density
from guidance.property_volume import gaussian_blur_property_channels


PHASE6Q_SCHEMA = "phase6q_five_body_causality_v1"
PHASE6Q_IMPLEMENTATION_VERSION = "phase6q_simple_causality_v1"
OBSERVATION_MODES = (
    "property",
    "blurred_property",
    "reflectivity_spikes",
    "seismic",
    "gravity",
)
OBSERVATION_CONTROLS = ("correct", "zero", "shuffled_xy")
OPTIMIZATION_METHODS = ("soft", "ste_top2")
VOXEL_CONFIG_SCHEMA = "phase6q_voxel_reconstruction_v1"
VOXEL_METHODS = ("soft_voxel", "ste_voxel")
HARD_COORDINATE_SCHEMA = "phase6q_hard_coordinate_v1"
EMBEDDING_ENDPOINT_SCHEMA = "phase6q_embedding_endpoint_v1"
EMBEDDING_METHODS = (
    "soft_embedding",
    "ste_embedding_rock",
    "soft_embedding_binary",
    "ste_embedding_binary",
)


@dataclass(frozen=True)
class Cuboid:
    """Integer half-open cuboid in x/y/z order."""

    id: str
    start: tuple[int, int, int]
    stop: tuple[int, int, int]
    well_xy: tuple[int, int] | None = None

    @property
    def volume(self) -> int:
        return math.prod(right - left for left, right in zip(self.start, self.stop))

    def mask(self, shape: Sequence[int]) -> torch.Tensor:
        result = torch.zeros(tuple(shape), dtype=torch.bool)
        result[
            self.start[0] : self.stop[0],
            self.start[1] : self.stop[1],
            self.start[2] : self.stop[2],
        ] = True
        return result


@dataclass(frozen=True)
class SimpleCausalCase:
    truth_labels: torch.Tensor
    baseline_labels: torch.Tensor
    condition_mask: torch.Tensor
    subsurface_mask: torch.Tensor
    fixed_target_mask: torch.Tensor
    candidate_masks: torch.Tensor
    fixed_bodies: tuple[Cuboid, ...]
    candidate_bodies: tuple[Cuboid, ...]
    truth_candidate_indices: tuple[int, int]
    air_label: int
    background_label: int
    target_label: int
    validation: dict[str, object]

    @property
    def candidate_count(self) -> int:
        return int(self.candidate_masks.shape[0])

    def truth_coefficients(
        self,
        *,
        device: torch.device | str = "cpu",
        dtype: torch.dtype = torch.float32,
    ) -> torch.Tensor:
        values = torch.zeros((1, self.candidate_count), device=device, dtype=dtype)
        values[0, list(self.truth_candidate_indices)] = 1.0
        return values


def _int_triple(value: object, name: str) -> tuple[int, int, int]:
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes))
        or len(value) != 3
    ):
        raise ValueError(f"{name} must contain exactly three integers")
    return tuple(int(item) for item in value)


def _parse_cuboid(
    value: object,
    *,
    name: str,
    grid_shape: tuple[int, int, int],
    air_start_z: int,
    require_well: bool,
) -> Cuboid:
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must be an object")
    body_id = str(value.get("id", "")).strip()
    if not body_id:
        raise ValueError(f"{name} requires a non-empty id")
    start = _int_triple(value.get("start"), f"{name}.start")
    stop = _int_triple(value.get("stop"), f"{name}.stop")
    if any(left < 0 or right > size or left >= right for left, right, size in zip(start, stop, grid_shape)):
        raise ValueError(f"{name} bounds must satisfy 0 <= start < stop <= grid")
    if stop[2] > air_start_z:
        raise ValueError(f"{name} must lie completely below the air layer")
    well_xy: tuple[int, int] | None = None
    if require_well:
        raw_well = value.get("well_xy")
        if (
            not isinstance(raw_well, Sequence)
            or isinstance(raw_well, (str, bytes))
            or len(raw_well) != 2
        ):
            raise ValueError(f"{name}.well_xy must contain two integers")
        well_xy = (int(raw_well[0]), int(raw_well[1]))
        if not (
            start[0] <= well_xy[0] < stop[0]
            and start[1] <= well_xy[1] < stop[1]
        ):
            raise ValueError(f"{name} well must pass through its x/y footprint")
    return Cuboid(body_id, start, stop, well_xy)


def validate_simple_causality_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate and resolve the frozen Phase-6Q analytic configuration."""
    if config.get("schema") != PHASE6Q_SCHEMA:
        raise ValueError(f"config schema must be {PHASE6Q_SCHEMA!r}")
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("config requires a non-empty id")
    grid_shape = _int_triple(config.get("grid_shape"), "grid_shape")
    if any(size <= 1 for size in grid_shape):
        raise ValueError("grid dimensions must exceed one")
    air_start_z = int(config.get("air_start_z", -1))
    if not 1 <= air_start_z < grid_shape[2]:
        raise ValueError("air_start_z must lie inside the vertical grid")
    labels = {
        "air_label": int(config.get("air_label", -999)),
        "background_label": int(config.get("background_label", -999)),
        "target_label": int(config.get("target_label", -999)),
    }
    if len(set(labels.values())) != 3:
        raise ValueError("air, background and target labels must be distinct")
    if min(labels.values()) < -1 or max(labels.values()) > 13:
        raise ValueError("raw labels must lie in -1..13")

    fixed_raw = config.get("fixed_bodies")
    candidate_raw = config.get("candidate_bodies")
    if not isinstance(fixed_raw, Sequence) or isinstance(fixed_raw, (str, bytes)):
        raise ValueError("fixed_bodies must be an array")
    if not isinstance(candidate_raw, Sequence) or isinstance(candidate_raw, (str, bytes)):
        raise ValueError("candidate_bodies must be an array")
    if len(fixed_raw) != 3:
        raise ValueError("the benchmark requires exactly three fixed drilled bodies")
    if len(candidate_raw) < 3:
        raise ValueError("the benchmark requires at least three candidate bodies")
    fixed = tuple(
        _parse_cuboid(
            value,
            name=f"fixed_bodies[{index}]",
            grid_shape=grid_shape,
            air_start_z=air_start_z,
            require_well=True,
        )
        for index, value in enumerate(fixed_raw)
    )
    candidates = tuple(
        _parse_cuboid(
            value,
            name=f"candidate_bodies[{index}]",
            grid_shape=grid_shape,
            air_start_z=air_start_z,
            require_well=False,
        )
        for index, value in enumerate(candidate_raw)
    )
    ids = [body.id for body in (*fixed, *candidates)]
    if len(ids) != len(set(ids)):
        raise ValueError("all body ids must be unique")
    all_masks = [body.mask(grid_shape) for body in (*fixed, *candidates)]
    occupancy = torch.stack(all_masks).sum(dim=0)
    if int(occupancy.max()) > 1:
        raise ValueError("all fixed and candidate cuboids must be disjoint")
    candidate_volumes = {body.volume for body in candidates}
    if len(candidate_volumes) != 1:
        raise ValueError("Q0 candidate cuboids must have equal voxel volume")

    truth_indices_raw = config.get("truth_candidate_indices")
    if (
        not isinstance(truth_indices_raw, Sequence)
        or isinstance(truth_indices_raw, (str, bytes))
        or len(truth_indices_raw) != 2
    ):
        raise ValueError("truth_candidate_indices must contain exactly two indices")
    truth_indices = tuple(sorted(int(item) for item in truth_indices_raw))
    if len(set(truth_indices)) != 2 or any(
        index < 0 or index >= len(candidates) for index in truth_indices
    ):
        raise ValueError("truth candidate indices must be two unique valid indices")

    modes_raw = config.get("observation_modes")
    if not isinstance(modes_raw, Sequence) or isinstance(modes_raw, (str, bytes)):
        raise ValueError("observation_modes must be an array")
    modes = tuple(str(mode) for mode in modes_raw)
    if len(modes) != len(set(modes)) or any(mode not in OBSERVATION_MODES for mode in modes):
        raise ValueError(f"observation modes must be unique members of {OBSERVATION_MODES}")
    sigma = float(config.get("blur_sigma_voxels", float("nan")))
    if not math.isfinite(sigma) or sigma <= 0:
        raise ValueError("blur_sigma_voxels must be finite and positive")

    optimization = config.get("optimization")
    if not isinstance(optimization, Mapping):
        raise ValueError("optimization must be an object")
    methods_raw = optimization.get("methods")
    if not isinstance(methods_raw, Sequence) or isinstance(methods_raw, (str, bytes)):
        raise ValueError("optimization.methods must be an array")
    methods = tuple(str(method) for method in methods_raw)
    if len(methods) != len(set(methods)) or any(method not in OPTIMIZATION_METHODS for method in methods):
        raise ValueError(f"optimization methods must be unique members of {OPTIMIZATION_METHODS}")
    updates = int(optimization.get("updates", 0))
    learning_rate = float(optimization.get("learning_rate", float("nan")))
    weight_decay = float(optimization.get("weight_decay", float("nan")))
    initial_logit = float(optimization.get("initial_logit", float("nan")))
    cardinality_weight = float(optimization.get("cardinality_weight", float("nan")))
    hard_check_interval = int(optimization.get("hard_check_interval", 0))
    if updates <= 0 or hard_check_interval <= 0:
        raise ValueError("optimization updates and hard_check_interval must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(initial_logit):
        raise ValueError("initial_logit must be finite")
    if not math.isfinite(cardinality_weight) or cardinality_weight < 0:
        raise ValueError("cardinality_weight must be finite and non-negative")
    schedule_raw = optimization.get("temperature_schedule")
    if not isinstance(schedule_raw, Sequence) or isinstance(schedule_raw, (str, bytes)):
        raise ValueError("temperature_schedule must be an array")
    temperatures: list[float] = []
    resolved_schedule: list[dict[str, object]] = []
    for index, segment in enumerate(schedule_raw):
        if not isinstance(segment, Mapping):
            raise ValueError(f"temperature segment {index} must be an object")
        temperature = float(segment.get("temperature", float("nan")))
        steps = int(segment.get("steps", 0))
        if not math.isfinite(temperature) or temperature <= 0 or steps <= 0:
            raise ValueError("temperature segments require positive finite values")
        temperatures.extend([temperature] * steps)
        resolved_schedule.append({"temperature": temperature, "steps": steps})
    if len(temperatures) != updates:
        raise ValueError("temperature schedule steps must equal optimization updates")

    controls_raw = config.get("seismic_controls")
    if not isinstance(controls_raw, Sequence) or isinstance(controls_raw, (str, bytes)):
        raise ValueError("seismic_controls must be an array")
    controls = tuple(str(control) for control in controls_raw)
    if len(controls) != len(set(controls)) or any(control not in OBSERVATION_CONTROLS for control in controls):
        raise ValueError(f"seismic controls must be unique members of {OBSERVATION_CONTROLS}")
    if bool(config.get("formal_training_authorized", True)):
        raise ValueError("Phase 6Q config must explicitly forbid formal training")
    batch_size = int(config.get("enumeration_batch_size", 0))
    if batch_size <= 0:
        raise ValueError("enumeration_batch_size must be positive")

    return {
        "schema": PHASE6Q_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "grid_shape": list(grid_shape),
        "air_start_z": air_start_z,
        **labels,
        "fixed_bodies": fixed,
        "candidate_bodies": candidates,
        "truth_candidate_indices": list(truth_indices),
        "observation_modes": list(modes),
        "blur_sigma_voxels": sigma,
        "optimization": {
            "methods": list(methods),
            "updates": updates,
            "learning_rate": learning_rate,
            "weight_decay": weight_decay,
            "initial_logit": initial_logit,
            "cardinality_weight": cardinality_weight,
            "temperature_schedule": resolved_schedule,
            "temperatures": temperatures,
            "hard_check_interval": hard_check_interval,
        },
        "seismic_controls": list(controls),
        "shuffle_seed": int(config.get("shuffle_seed", -1)),
        "enumeration_batch_size": batch_size,
        "inverse_crime": bool(config.get("inverse_crime", False)),
        "measured_geophysics": bool(config.get("measured_geophysics", True)),
        "formal_training_authorized": False,
    }


def build_simple_causal_case(config: Mapping[str, object]) -> SimpleCausalCase:
    """Materialize the frozen analytic truth, missing-body baseline and conditions."""
    resolved = validate_simple_causality_config(config)
    shape = tuple(int(value) for value in resolved["grid_shape"])
    air_start = int(resolved["air_start_z"])
    air_label = int(resolved["air_label"])
    background_label = int(resolved["background_label"])
    target_label = int(resolved["target_label"])
    fixed = tuple(resolved["fixed_bodies"])
    candidates = tuple(resolved["candidate_bodies"])
    truth_indices = tuple(int(value) for value in resolved["truth_candidate_indices"])

    baseline = torch.full((1, 1, *shape), background_label, dtype=torch.long)
    baseline[..., air_start:] = air_label
    fixed_masks = torch.stack([body.mask(shape) for body in fixed])
    candidate_masks = torch.stack([body.mask(shape) for body in candidates])
    fixed_union = fixed_masks.any(dim=0)
    baseline[0, 0, fixed_union] = target_label
    truth = baseline.clone()
    truth_candidates = candidate_masks[list(truth_indices)].any(dim=0)
    truth[0, 0, truth_candidates] = target_label

    condition = torch.zeros((1, 1, *shape), dtype=torch.bool)
    condition[..., air_start:] = True
    condition[..., air_start - 1] = True
    well_reports: list[dict[str, object]] = []
    for body in fixed:
        if body.well_xy is None:
            raise RuntimeError("validated fixed body lacks a well")
        x, y = body.well_xy
        condition[0, 0, x, y, :air_start] = True
        fixed_hits = int(fixed_union[x, y].sum())
        candidate_hits = int(candidate_masks[:, x, y].sum())
        well_reports.append(
            {
                "body_id": body.id,
                "well_xy": [x, y],
                "fixed_target_voxels": fixed_hits,
                "candidate_target_voxels": candidate_hits,
            }
        )
    candidate_condition_overlap = int(
        (candidate_masks & condition[0, 0].unsqueeze(0)).sum()
    )
    if candidate_condition_overlap:
        raise ValueError("candidate bodies must not intersect hard conditions")
    difference = truth != baseline
    expected_difference = sum(candidates[index].volume for index in truth_indices)
    if int(difference.sum()) != expected_difference:
        raise RuntimeError("truth/baseline difference is not exactly the hidden bodies")
    if bool((truth[condition] != baseline[condition]).any()):
        raise RuntimeError("truth and baseline differ at conditioned voxels")
    subsurface = torch.zeros_like(condition)
    subsurface[..., :air_start] = True
    validation = {
        "truth_baseline_difference_voxels": int(difference.sum()),
        "expected_hidden_voxels": expected_difference,
        "condition_voxels": int(condition.sum()),
        "candidate_condition_overlap_voxels": candidate_condition_overlap,
        "truth_condition_mismatches": int((truth[condition] != baseline[condition]).sum()),
        "fixed_target_voxels": int(fixed_union.sum()),
        "candidate_voxels_each": int(candidate_masks[0].sum()),
        "truth_target_voxels": int((truth == target_label).sum()),
        "baseline_target_voxels": int((baseline == target_label).sum()),
        "well_reports": well_reports,
    }
    return SimpleCausalCase(
        truth_labels=truth,
        baseline_labels=baseline,
        condition_mask=condition,
        subsurface_mask=subsurface,
        fixed_target_mask=fixed_union.unsqueeze(0).unsqueeze(0),
        candidate_masks=candidate_masks,
        fixed_bodies=fixed,
        candidate_bodies=candidates,
        truth_candidate_indices=truth_indices,
        air_label=air_label,
        background_label=background_label,
        target_label=target_label,
        validation=validation,
    )


def coefficients_to_hard(
    probabilities: torch.Tensor,
    method: str,
    *,
    top_k: int = 2,
) -> torch.Tensor:
    """Convert candidate probabilities to the Phase-6Q hard decision."""
    if probabilities.ndim != 2:
        raise ValueError("candidate probabilities must have shape [B,C]")
    if method == "soft":
        return (probabilities >= 0.5).to(probabilities.dtype)
    if method == "ste_top2":
        if not 0 < top_k <= probabilities.shape[1]:
            raise ValueError("top_k must lie in 1..candidate_count")
        indices = probabilities.topk(top_k, dim=1).indices
        hard = torch.zeros_like(probabilities)
        return hard.scatter(1, indices, 1.0)
    raise ValueError(f"unknown optimization method {method!r}")


def optimization_coefficients(
    logits: torch.Tensor,
    temperature: float,
    method: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return forward coefficients, soft probabilities and hard audit values."""
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    probabilities = torch.sigmoid(logits / float(temperature))
    hard = coefficients_to_hard(probabilities, method)
    if method == "soft":
        forward = probabilities
    elif method == "ste_top2":
        forward = hard + probabilities - probabilities.detach()
    else:
        raise ValueError(f"unknown optimization method {method!r}")
    return forward, probabilities, hard


class AnalyticObservationSuite:
    """Compose candidate occupancies and apply the frozen Phase-3/4 operators."""

    def __init__(
        self,
        case: SimpleCausalCase,
        *,
        acoustic_property_table: torch.Tensor,
        density_table: torch.Tensor,
        seismic_operator: object,
        gravity_operator: object,
        blur_sigma_voxels: float,
    ) -> None:
        if acoustic_property_table.ndim != 2 or acoustic_property_table.shape[0] != 2:
            raise ValueError("acoustic_property_table must have shape [2,C]")
        if density_table.ndim != 1 or density_table.numel() != acoustic_property_table.shape[1]:
            raise ValueError("density table must match acoustic category count")
        self.case = case
        self.acoustic_property_table = acoustic_property_table.detach().float().contiguous()
        self.density_table = density_table.detach().float().contiguous()
        self.seismic_operator = seismic_operator
        self.gravity_operator = gravity_operator
        self.blur_sigma_voxels = float(blur_sigma_voxels)

    def occupancy(self, coefficients: torch.Tensor) -> torch.Tensor:
        if coefficients.ndim != 2 or coefficients.shape[1] != self.case.candidate_count:
            raise ValueError("coefficients must have shape [B,candidate_count]")
        masks = self.case.candidate_masks.to(
            device=coefficients.device, dtype=coefficients.dtype
        )
        candidate = torch.einsum("bc,cxyz->bxyz", coefficients, masks)
        fixed = self.case.fixed_target_mask.to(
            device=coefficients.device, dtype=coefficients.dtype
        ).expand(coefficients.shape[0], -1, -1, -1, -1)[:, 0]
        occupancy = fixed + candidate
        if bool((occupancy < -1e-6).any()) or bool((occupancy > 1.0 + 1e-6).any()):
            raise ValueError("candidate occupancies overlap or leave [0,1]")
        return occupancy.unsqueeze(1)

    def _two_material_volume(
        self,
        occupancy: torch.Tensor,
        table: torch.Tensor,
    ) -> torch.Tensor:
        background_index = self.case.background_label + 1
        target_index = self.case.target_label + 1
        air_index = self.case.air_label + 1
        values = table.to(device=occupancy.device, dtype=occupancy.dtype)
        background = values[..., background_index]
        target = values[..., target_index]
        air = values[..., air_index]
        spatial_dims = occupancy.ndim - 2
        prefix = (1, values.shape[0]) + (1,) * spatial_dims
        mixed = background.view(prefix) + occupancy * (
            target - background
        ).view(prefix)
        air_mask = (~self.case.subsurface_mask).to(
            device=occupancy.device, dtype=torch.bool
        ).expand(occupancy.shape[0], values.shape[0], *occupancy.shape[2:])
        return torch.where(air_mask, air.view(prefix).expand_as(mixed), mixed)

    def acoustic(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self._two_material_volume(
            self.occupancy(coefficients), self.acoustic_property_table
        )

    def density(self, coefficients: torch.Tensor) -> torch.Tensor:
        return self._two_material_volume(
            self.occupancy(coefficients), self.density_table.unsqueeze(0)
        )

    def field(self, coefficients: torch.Tensor, mode: str) -> torch.Tensor:
        if mode not in OBSERVATION_MODES:
            raise ValueError(f"unknown observation mode {mode!r}")
        occupancy = self.occupancy(coefficients)
        if mode == "property":
            return occupancy
        if mode == "blurred_property":
            return gaussian_blur_property_channels(occupancy, self.blur_sigma_voxels)
        if mode in {"reflectivity_spikes", "seismic"}:
            acoustic = self._two_material_volume(
                occupancy, self.acoustic_property_table
            )
            subsurface = self.case.subsurface_mask.to(device=coefficients.device)
            if mode == "reflectivity_spikes":
                return self.seismic_operator.reflectivity_spikes(
                    acoustic[:, 0:1], acoustic[:, 1:2], subsurface
                )
            return self.seismic_operator(
                acoustic[:, 0:1], acoustic[:, 1:2], subsurface
            )
        density = self._two_material_volume(
            occupancy, self.density_table.unsqueeze(0)
        )
        return self.gravity_operator(density)

    def field_from_occupancy(self, occupancy: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply an observation operator to a caller-supplied target occupancy."""
        expected_shape = self.case.truth_labels.shape
        if occupancy.ndim != 5 or occupancy.shape[1:] != expected_shape[1:]:
            raise ValueError(
                f"occupancy must have shape [B,1,{','.join(map(str, expected_shape[2:]))}]"
            )
        if not occupancy.is_floating_point() or not torch.isfinite(occupancy).all():
            raise ValueError("occupancy must be finite floating point")
        if bool((occupancy < -1e-6).any()) or bool((occupancy > 1.0 + 1e-6).any()):
            raise ValueError("occupancy must lie in [0,1]")
        if mode == "property":
            return occupancy
        if mode == "blurred_property":
            return gaussian_blur_property_channels(occupancy, self.blur_sigma_voxels)
        if mode in {"reflectivity_spikes", "seismic"}:
            acoustic = self._two_material_volume(
                occupancy, self.acoustic_property_table
            )
            subsurface = self.case.subsurface_mask.to(device=occupancy.device)
            if mode == "reflectivity_spikes":
                return self.seismic_operator.reflectivity_spikes(
                    acoustic[:, 0:1], acoustic[:, 1:2], subsurface
                )
            return self.seismic_operator(
                acoustic[:, 0:1], acoustic[:, 1:2], subsurface
            )
        if mode == "gravity":
            density = self._two_material_volume(
                occupancy, self.density_table.unsqueeze(0)
            )
            return self.gravity_operator(density)
        raise ValueError(f"unknown observation mode {mode!r}")

    def field_from_probabilities(
        self, probabilities: torch.Tensor, mode: str
    ) -> torch.Tensor:
        """Apply the deployed all-class soft property bridge and physics operator."""
        category_count = int(self.acoustic_property_table.shape[1])
        expected_shape = self.case.truth_labels.shape
        if probabilities.ndim != 5 or probabilities.shape[1] != category_count:
            raise ValueError("probabilities must have shape [B,C,X,Y,Z]")
        if probabilities.shape[2:] != expected_shape[2:]:
            raise ValueError("probability spatial shape must match the causal case")
        if not torch.isfinite(probabilities).all():
            raise ValueError("probabilities must be finite")
        target_index = self.case.target_label + 1
        target_occupancy = probabilities[:, target_index : target_index + 1]
        if mode == "property":
            return target_occupancy
        if mode == "blurred_property":
            return gaussian_blur_property_channels(
                target_occupancy, self.blur_sigma_voxels
            )
        if mode in {"reflectivity_spikes", "seismic"}:
            table = self.acoustic_property_table.to(
                device=probabilities.device, dtype=probabilities.dtype
            )
            expected_all = torch.einsum("bcxyz,pc->bpxyz", probabilities, table)
            rock = probabilities[:, 1:]
            rock_support = rock.sum(dim=1, keepdim=True)
            subsurface_one = self.case.subsurface_mask.to(
                device=probabilities.device, dtype=torch.bool
            ).expand(probabilities.shape[0], -1, -1, -1, -1)
            if bool((rock_support[subsurface_one] <= 1e-12).any()):
                raise ValueError("subsurface non-air probability support is empty")
            expected_rock = torch.einsum(
                "bcxyz,pc->bpxyz",
                rock / rock_support.clamp_min(1e-12),
                table[:, 1:],
            )
            acoustic = torch.where(
                subsurface_one.expand(probabilities.shape[0], 2, *probabilities.shape[2:]),
                expected_rock,
                expected_all,
            )
            subsurface = self.case.subsurface_mask.to(device=probabilities.device)
            if mode == "reflectivity_spikes":
                return self.seismic_operator.reflectivity_spikes(
                    acoustic[:, 0:1], acoustic[:, 1:2], subsurface
                )
            return self.seismic_operator(
                acoustic[:, 0:1], acoustic[:, 1:2], subsurface
            )
        if mode == "gravity":
            density = probabilities_to_density(probabilities, self.density_table)
            return self.gravity_operator(density)
        raise ValueError(f"unknown observation mode {mode!r}")

    def field_from_labels(self, labels: torch.Tensor, mode: str) -> torch.Tensor:
        """Apply an observation operator to raw hard labels in ``-1..13``."""
        expected_shape = self.case.truth_labels.shape
        if labels.ndim != 5 or labels.shape[1:] != expected_shape[1:]:
            raise ValueError("labels must have shape [B,1,X,Y,Z]")
        if not torch.equal(labels, labels.round()):
            raise ValueError("hard labels must be integer valued")
        categories = labels.long() + 1
        category_count = int(self.acoustic_property_table.shape[1])
        if int(categories.min()) < 0 or int(categories.max()) >= category_count:
            raise ValueError("hard labels lie outside the frozen codebook")
        occupancy = (labels == self.case.target_label).to(
            device=labels.device, dtype=self.acoustic_property_table.dtype
        )
        if mode == "property":
            return occupancy
        if mode == "blurred_property":
            return gaussian_blur_property_channels(occupancy, self.blur_sigma_voxels)
        if mode in {"reflectivity_spikes", "seismic"}:
            table = self.acoustic_property_table.to(device=labels.device)
            acoustic = table[:, categories[:, 0]].permute(1, 0, 2, 3, 4).contiguous()
            subsurface = self.case.subsurface_mask.to(device=labels.device)
            if mode == "reflectivity_spikes":
                return self.seismic_operator.reflectivity_spikes(
                    acoustic[:, 0:1], acoustic[:, 1:2], subsurface
                )
            return self.seismic_operator(
                acoustic[:, 0:1], acoustic[:, 1:2], subsurface
            )
        if mode == "gravity":
            table = self.density_table.to(device=labels.device)
            density = table[categories[:, 0]].unsqueeze(1)
            return self.gravity_operator(density)
        raise ValueError(f"unknown observation mode {mode!r}")

    def hard_labels(self, coefficients: torch.Tensor) -> torch.Tensor:
        if not torch.equal(coefficients, coefficients.round()):
            raise ValueError("hard coefficients must be binary")
        labels = self.case.baseline_labels.to(device=coefficients.device).expand(
            coefficients.shape[0], -1, -1, -1, -1
        ).clone()
        occupancy = self.occupancy(coefficients)[:, 0].bool()
        labels[:, 0][occupancy] = self.case.target_label
        return labels


def controlled_observation(
    observation: torch.Tensor,
    control: str,
    *,
    shuffle_seed: int,
) -> torch.Tensor:
    """Apply a deterministic correct/zero/x-y shuffled observation control."""
    if control not in OBSERVATION_CONTROLS:
        raise ValueError(f"unknown observation control {control!r}")
    if control == "correct":
        return observation.detach().clone()
    if control == "zero":
        return torch.zeros_like(observation)
    if observation.ndim not in (4, 5):
        raise ValueError("shuffled observation must have x/y dimensions at axes 2/3")
    generator = torch.Generator(device="cpu").manual_seed(int(shuffle_seed))
    x_order = torch.randperm(observation.shape[2], generator=generator).to(observation.device)
    y_order = torch.randperm(observation.shape[3], generator=generator).to(observation.device)
    return observation.index_select(2, x_order).index_select(3, y_order).contiguous()


def _rmse(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    return (predicted - observed).square().mean().sqrt()


def _attainment(baseline_rmse: float, candidate_rmse: float, eps: float = 1e-12) -> float:
    if baseline_rmse <= eps:
        return float("nan")
    return 1.0 - candidate_rmse / baseline_rmse


def _body_metrics(selected: Sequence[int], truth: Sequence[int]) -> dict[str, float | int]:
    selected_set = set(int(value) for value in selected)
    truth_set = set(int(value) for value in truth)
    true_positive = len(selected_set & truth_set)
    return {
        "selected_body_count": len(selected_set),
        "body_true_positive": true_positive,
        "body_precision": true_positive / len(selected_set) if selected_set else 0.0,
        "body_recall": true_positive / len(truth_set) if truth_set else 0.0,
    }


@torch.no_grad()
def enumerate_hard_pairs(
    suite: AnalyticObservationSuite,
    mode: str,
    *,
    batch_size: int,
) -> dict[str, object]:
    """Evaluate every exact two-candidate hard model against the truth field."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = suite.acoustic_property_table.device
    truth_coefficients = suite.case.truth_coefficients(device=device)
    observed = suite.field(truth_coefficients, mode)
    baseline_coefficients = torch.zeros_like(truth_coefficients)
    baseline_field = suite.field(baseline_coefficients, mode)
    baseline_rmse = float(_rmse(baseline_field, observed).detach().cpu())
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 0:
        raise ValueError(f"{mode} baseline must have a positive finite RMSE")
    pair_values = list(combinations(range(suite.case.candidate_count), 2))
    rows: list[dict[str, object]] = []
    for offset in range(0, len(pair_values), batch_size):
        pairs = pair_values[offset : offset + batch_size]
        coefficients = torch.zeros(
            (len(pairs), suite.case.candidate_count),
            device=device,
            dtype=observed.dtype,
        )
        for row_index, pair in enumerate(pairs):
            coefficients[row_index, list(pair)] = 1.0
        predicted = suite.field(coefficients, mode)
        target = observed.expand_as(predicted)
        batch_rmse = (predicted - target).square().flatten(1).mean(dim=1).sqrt()
        for pair, rmse_value in zip(pairs, batch_rmse.detach().cpu().tolist()):
            metrics = _body_metrics(pair, suite.case.truth_candidate_indices)
            rows.append(
                {
                    "candidate_0": pair[0],
                    "candidate_1": pair[1],
                    "candidate_ids": ",".join(
                        (suite.case.candidate_bodies[pair[0]].id, suite.case.candidate_bodies[pair[1]].id)
                    ),
                    "is_truth_pair": tuple(pair) == suite.case.truth_candidate_indices,
                    "hard_rmse": float(rmse_value),
                    "hard_attainment": _attainment(baseline_rmse, float(rmse_value)),
                    **metrics,
                }
            )
    rows.sort(key=lambda row: (float(row["hard_rmse"]), int(row["candidate_0"]), int(row["candidate_1"])))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    truth_row = next(row for row in rows if bool(row["is_truth_pair"]))
    numerical_tolerance = max(1e-7, baseline_rmse * 1e-6)
    second_rmse = min(
        float(row["hard_rmse"]) for row in rows if not bool(row["is_truth_pair"])
    )
    return {
        "mode": mode,
        "baseline_rmse": baseline_rmse,
        "truth_pair": list(suite.case.truth_candidate_indices),
        "truth_pair_rank": int(truth_row["rank"]),
        "truth_pair_rmse": float(truth_row["hard_rmse"]),
        "second_best_nontruth_rmse": second_rmse,
        "truth_to_second_gap": second_rmse - float(truth_row["hard_rmse"]),
        "numerical_tolerance": numerical_tolerance,
        "near_numerical_zero_count": sum(
            float(row["hard_rmse"]) <= numerical_tolerance for row in rows
        ),
        "candidate_pair_count": len(rows),
        "rows": rows,
        "observed": observed.detach().cpu(),
        "baseline_field": baseline_field.detach().cpu(),
    }


def _entropy(probabilities: torch.Tensor, eps: float = 1e-6) -> float:
    dtype_eps = torch.finfo(probabilities.dtype).eps
    bound = max(float(eps), float(dtype_eps))
    values = probabilities.clamp(bound, 1.0 - bound)
    entropy = -(values * values.log() + (1.0 - values) * (1.0 - values).log())
    return float(entropy.mean().detach().cpu())


def optimize_candidate_logits(
    suite: AnalyticObservationSuite,
    mode: str,
    *,
    control: str,
    method: str,
    temperatures: Sequence[float],
    learning_rate: float,
    weight_decay: float,
    initial_logit: float,
    cardinality_weight: float,
    hard_check_interval: int,
    shuffle_seed: int,
) -> dict[str, object]:
    """Optimize candidate logits and select checkpoints by hard physics only."""
    if method not in OPTIMIZATION_METHODS:
        raise ValueError(f"method must be one of {OPTIMIZATION_METHODS}")
    if not temperatures:
        raise ValueError("temperatures must be non-empty")
    device = suite.acoustic_property_table.device
    dtype = suite.acoustic_property_table.dtype
    truth_coefficients = suite.case.truth_coefficients(device=device, dtype=dtype)
    correct_observation = suite.field(truth_coefficients, mode).detach()
    observed = controlled_observation(
        correct_observation, control, shuffle_seed=shuffle_seed
    )
    zero_coefficients = torch.zeros_like(truth_coefficients)
    baseline_field = suite.field(zero_coefficients, mode).detach()
    baseline_rmse = float(_rmse(baseline_field, observed).detach().cpu())
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 1e-12:
        raise ValueError("controlled observation must differ from the baseline field")
    normalizer = baseline_rmse * baseline_rmse

    logits = torch.nn.Parameter(
        torch.full(
            (1, suite.case.candidate_count),
            float(initial_logit),
            device=device,
            dtype=dtype,
        )
    )
    optimizer = torch.optim.Adam(
        [logits], lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    trace: list[dict[str, object]] = []

    def audit(step: int, temperature: float, soft_loss_value: float | None) -> dict[str, object]:
        with torch.no_grad():
            _, probabilities, hard = optimization_coefficients(logits, temperature, method)
            soft_field = suite.field(probabilities, mode)
            hard_field = suite.field(hard, mode)
            soft_rmse = float(_rmse(soft_field, observed).detach().cpu())
            hard_rmse = float(_rmse(hard_field, observed).detach().cpu())
            selected = torch.nonzero(hard[0] > 0.5, as_tuple=False).flatten().cpu().tolist()
            return {
                "step": step,
                "temperature": float(temperature),
                "objective": soft_loss_value,
                "soft_rmse": soft_rmse,
                "hard_rmse": hard_rmse,
                "soft_attainment": _attainment(baseline_rmse, soft_rmse),
                "hard_attainment": _attainment(baseline_rmse, hard_rmse),
                "soft_hard_attainment_gap": _attainment(baseline_rmse, soft_rmse)
                - _attainment(baseline_rmse, hard_rmse),
                "probability_sum": float(probabilities.sum().detach().cpu()),
                "probability_entropy": _entropy(probabilities),
                "selected_indices": ",".join(str(value) for value in selected),
                "selected_count": len(selected),
                **_body_metrics(selected, suite.case.truth_candidate_indices),
            }

    initial_row = audit(0, float(temperatures[0]), None)
    trace.append(initial_row)
    with torch.no_grad():
        _, _, initial_hard = optimization_coefficients(
            logits, float(temperatures[0]), method
        )
    best_hard_rmse = float(initial_row["hard_rmse"])
    best_step = 0
    best_logits = logits.detach().cpu().clone()
    best_hard = initial_hard.detach().cpu().clone()
    best_selected = torch.nonzero(
        initial_hard[0] > 0.5, as_tuple=False
    ).flatten().cpu().tolist()
    for step, temperature in enumerate(temperatures, start=1):
        optimizer.zero_grad(set_to_none=True)
        forward_coefficients, probabilities, _ = optimization_coefficients(
            logits, float(temperature), method
        )
        predicted = suite.field(forward_coefficients, mode)
        physics = (predicted - observed).square().mean() / normalizer
        cardinality = (probabilities.sum() - 2.0).square()
        penalty = float(cardinality_weight) * cardinality if method == "soft" else cardinality * 0.0
        loss = physics + penalty
        if not torch.isfinite(loss):
            raise FloatingPointError("candidate objective is non-finite")
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("candidate gradients are absent or non-finite")
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("candidate logits are non-finite")
        should_check = (
            step == 1
            or step % int(hard_check_interval) == 0
            or step == len(temperatures)
        )
        if should_check:
            row = audit(step, float(temperature), float(loss.detach().cpu()))
            trace.append(row)
            hard_rmse = float(row["hard_rmse"])
            if hard_rmse < best_hard_rmse:
                with torch.no_grad():
                    _, _, hard = optimization_coefficients(logits, float(temperature), method)
                best_hard_rmse = hard_rmse
                best_step = step
                best_logits = logits.detach().cpu().clone()
                best_hard = hard.detach().cpu().clone()
                best_selected = torch.nonzero(
                    hard[0] > 0.5, as_tuple=False
                ).flatten().cpu().tolist()

    final_row = trace[-1]
    best_metrics = {
        "hard_rmse": best_hard_rmse,
        "hard_attainment": _attainment(baseline_rmse, best_hard_rmse),
        "best_step": best_step,
        "selected_indices": best_selected,
        **_body_metrics(best_selected, suite.case.truth_candidate_indices),
    }
    return {
        "mode": mode,
        "control": control,
        "method": method,
        "baseline_rmse": baseline_rmse,
        "best_metrics": best_metrics,
        "final_metrics": final_row,
        "best_logits": best_logits,
        "best_hard_coefficients": best_hard,
        "trace": trace,
        "observed": observed.detach().cpu(),
        "correct_observation": correct_observation.detach().cpu(),
        "baseline_field": baseline_field.detach().cpu(),
    }


def validate_voxel_reconstruction_config(
    config: Mapping[str, object],
    *,
    grid_shape: Sequence[int],
) -> dict[str, object]:
    """Validate the separately frozen Q2 free-voxel protocol."""
    if config.get("schema") != VOXEL_CONFIG_SCHEMA:
        raise ValueError(f"voxel config schema must be {VOXEL_CONFIG_SCHEMA!r}")
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("voxel config requires a non-empty id")
    shape = tuple(int(value) for value in grid_shape)
    region = config.get("search_region")
    if not isinstance(region, Mapping):
        raise ValueError("search_region must be an object")
    start = _int_triple(region.get("start"), "search_region.start")
    stop = _int_triple(region.get("stop"), "search_region.stop")
    if any(
        left < 0 or right > size or left >= right
        for left, right, size in zip(start, stop, shape)
    ):
        raise ValueError("search region must satisfy 0 <= start < stop <= grid")
    modes_raw = config.get("observation_modes")
    methods_raw = config.get("methods")
    controls_raw = config.get("seismic_controls")
    for values, allowed, name in (
        (modes_raw, OBSERVATION_MODES, "observation_modes"),
        (methods_raw, VOXEL_METHODS, "methods"),
        (controls_raw, OBSERVATION_CONTROLS, "seismic_controls"),
    ):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be an array")
        parsed = tuple(str(value) for value in values)
        if not parsed or len(parsed) != len(set(parsed)) or any(
            value not in allowed for value in parsed
        ):
            raise ValueError(f"{name} must be unique members of {allowed}")
    updates = int(config.get("updates", 0))
    hard_check_interval = int(config.get("hard_check_interval", 0))
    learning_rate = float(config.get("learning_rate", float("nan")))
    weight_decay = float(config.get("weight_decay", float("nan")))
    initial_logit = float(config.get("initial_logit", float("nan")))
    gradient_clip_norm = float(config.get("gradient_clip_norm", float("nan")))
    if updates <= 0 or hard_check_interval <= 0:
        raise ValueError("updates and hard_check_interval must be positive")
    if not math.isfinite(learning_rate) or learning_rate <= 0:
        raise ValueError("learning_rate must be finite and positive")
    if not math.isfinite(weight_decay) or weight_decay < 0:
        raise ValueError("weight_decay must be finite and non-negative")
    if not math.isfinite(initial_logit):
        raise ValueError("initial_logit must be finite")
    if not math.isfinite(gradient_clip_norm) or gradient_clip_norm <= 0:
        raise ValueError("gradient_clip_norm must be finite and positive")
    schedule = config.get("temperature_schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise ValueError("temperature_schedule must be an array")
    temperatures: list[float] = []
    resolved_schedule: list[dict[str, object]] = []
    for index, segment in enumerate(schedule):
        if not isinstance(segment, Mapping):
            raise ValueError(f"temperature segment {index} must be an object")
        temperature = float(segment.get("temperature", float("nan")))
        steps = int(segment.get("steps", 0))
        if not math.isfinite(temperature) or temperature <= 0 or steps <= 0:
            raise ValueError("temperature segments require positive finite values")
        temperatures.extend([temperature] * steps)
        resolved_schedule.append({"temperature": temperature, "steps": steps})
    if len(temperatures) != updates:
        raise ValueError("temperature schedule steps must equal updates")
    regularization = config.get("regularization")
    if not isinstance(regularization, Mapping):
        raise ValueError("regularization must be an object")
    if float(regularization.get("volume", float("nan"))) != 0.0:
        raise ValueError("Q2 forbids a volume regularizer")
    if float(regularization.get("smoothness", float("nan"))) != 0.0:
        raise ValueError("Q2 forbids a smoothness regularizer")
    if bool(regularization.get("truth_roi", True)):
        raise ValueError("Q2 forbids a truth ROI")
    if config.get("hard_selection") != "minimum_hard_physics_loss_only":
        raise ValueError("Q2 requires hard-physics-only selection")
    if bool(config.get("formal_training_authorized", True)):
        raise ValueError("Q2 must explicitly forbid formal training")
    return {
        "schema": VOXEL_CONFIG_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "search_region": {"start": list(start), "stop": list(stop)},
        "observation_modes": [str(value) for value in modes_raw],
        "methods": [str(value) for value in methods_raw],
        "updates": updates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "initial_logit": initial_logit,
        "gradient_clip_norm": gradient_clip_norm,
        "temperature_schedule": resolved_schedule,
        "temperatures": temperatures,
        "hard_check_interval": hard_check_interval,
        "seismic_controls": [str(value) for value in controls_raw],
        "shuffle_seed": int(config.get("shuffle_seed", -1)),
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "hard_selection": "minimum_hard_physics_loss_only",
        "formal_training_authorized": False,
    }


def build_voxel_search_mask(
    case: SimpleCausalCase,
    voxel_config: Mapping[str, object],
) -> tuple[torch.Tensor, dict[str, object]]:
    """Build the broad truth-blind Q2 search mask and validate its boundaries."""
    resolved = validate_voxel_reconstruction_config(
        voxel_config, grid_shape=case.truth_labels.shape[2:]
    )
    start = tuple(int(value) for value in resolved["search_region"]["start"])
    stop = tuple(int(value) for value in resolved["search_region"]["stop"])
    mask = torch.zeros_like(case.condition_mask)
    mask[
        ...,
        start[0] : stop[0],
        start[1] : stop[1],
        start[2] : stop[2],
    ] = True
    mask &= case.subsurface_mask
    mask &= ~case.condition_mask
    if bool((mask & case.fixed_target_mask).any()):
        raise ValueError("Q2 search region must not include fixed drilled bodies")
    hidden_truth = (case.truth_labels == case.target_label) & ~case.fixed_target_mask
    hidden_count = int(hidden_truth.sum())
    hidden_inside = int((hidden_truth & mask).sum())
    if hidden_inside != hidden_count:
        raise ValueError("Q2 search region must contain both complete hidden truth bodies")
    report = {
        "search_voxels": int(mask.sum()),
        "condition_overlap_voxels": int((mask & case.condition_mask).sum()),
        "fixed_target_overlap_voxels": int((mask & case.fixed_target_mask).sum()),
        "hidden_truth_voxels": hidden_count,
        "hidden_truth_inside_search_voxels": hidden_inside,
        "search_fraction_of_unconditioned_subsurface": float(
            mask.sum() / (case.subsurface_mask & ~case.condition_mask).sum()
        ),
    }
    return mask, report


def _voxel_geometry_metrics(
    hard_occupancy: torch.Tensor,
    case: SimpleCausalCase,
    search_mask: torch.Tensor,
) -> dict[str, object]:
    predicted = (hard_occupancy > 0.5) & search_mask.to(hard_occupancy.device)
    truth = (
        (case.truth_labels == case.target_label)
        & ~case.fixed_target_mask
        & search_mask
    ).to(hard_occupancy.device)
    true_positive = int((predicted & truth).sum())
    predicted_count = int(predicted.sum())
    truth_count = int(truth.sum())
    union = int((predicted | truth).sum())
    body_recalls: list[float] = []
    for index in case.truth_candidate_indices:
        body = case.candidate_masks[index].to(hard_occupancy.device)
        body_recalls.append(float(predicted[0, 0][body].float().mean()))
    return {
        "predicted_hidden_voxels": predicted_count,
        "truth_hidden_voxels": truth_count,
        "hidden_true_positive_voxels": true_positive,
        "hidden_precision": true_positive / predicted_count if predicted_count else 0.0,
        "hidden_recall": true_positive / truth_count if truth_count else 0.0,
        "hidden_iou": true_positive / union if union else 0.0,
        "hidden_false_positive_voxels": predicted_count - true_positive,
        "hidden_body_0_recall": body_recalls[0],
        "hidden_body_1_recall": body_recalls[1],
    }


def optimize_voxel_logits(
    suite: AnalyticObservationSuite,
    mode: str,
    *,
    search_mask: torch.Tensor,
    control: str,
    method: str,
    temperatures: Sequence[float],
    learning_rate: float,
    weight_decay: float,
    initial_logit: float,
    gradient_clip_norm: float,
    hard_check_interval: int,
    shuffle_seed: int,
) -> dict[str, object]:
    """Optimize unrestricted binary target logits inside the Q2 search region."""
    if method not in VOXEL_METHODS:
        raise ValueError(f"voxel method must be one of {VOXEL_METHODS}")
    if not temperatures:
        raise ValueError("temperatures must be non-empty")
    device = suite.acoustic_property_table.device
    dtype = suite.acoustic_property_table.dtype
    search = search_mask.to(device=device, dtype=torch.bool)
    flat_indices = torch.nonzero(search.reshape(-1), as_tuple=False).flatten()
    if not flat_indices.numel():
        raise ValueError("voxel search mask is empty")
    fixed = suite.case.fixed_target_mask.to(device=device, dtype=dtype)
    truth_occupancy = (suite.case.truth_labels == suite.case.target_label).to(
        device=device, dtype=dtype
    )
    observed_correct = suite.field_from_occupancy(truth_occupancy, mode).detach()
    observed = controlled_observation(
        observed_correct, control, shuffle_seed=shuffle_seed
    )
    baseline_occupancy = fixed.clone()
    baseline_field = suite.field_from_occupancy(baseline_occupancy, mode).detach()
    baseline_rmse = float(_rmse(baseline_field, observed).detach().cpu())
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 1e-12:
        raise ValueError("controlled observation must differ from the hard baseline")
    normalizer = baseline_rmse * baseline_rmse
    logits = torch.nn.Parameter(
        torch.full(
            (flat_indices.numel(),),
            float(initial_logit),
            device=device,
            dtype=dtype,
        )
    )
    optimizer = torch.optim.Adam(
        [logits], lr=float(learning_rate), weight_decay=float(weight_decay)
    )

    def occupancies(temperature: float) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        probabilities = torch.sigmoid(logits / float(temperature))
        hard_values = (probabilities >= 0.5).to(dtype)
        forward_values = (
            probabilities
            if method == "soft_voxel"
            else hard_values + probabilities - probabilities.detach()
        )
        soft_flat = torch.zeros(
            truth_occupancy.numel(), device=device, dtype=dtype
        ).scatter(0, flat_indices, probabilities)
        hard_flat = torch.zeros_like(soft_flat).scatter(0, flat_indices, hard_values)
        forward_flat = torch.zeros_like(soft_flat).scatter(0, flat_indices, forward_values)
        shape = truth_occupancy.shape
        return (
            fixed + forward_flat.reshape(shape),
            fixed + soft_flat.reshape(shape),
            fixed + hard_flat.reshape(shape),
        )

    def audit(
        step: int,
        temperature: float,
        objective: float | None,
        gradient_norm: float | None,
    ) -> tuple[dict[str, object], torch.Tensor]:
        with torch.no_grad():
            _, soft_occupancy, hard_occupancy = occupancies(temperature)
            soft_field = suite.field_from_occupancy(soft_occupancy, mode)
            hard_field = suite.field_from_occupancy(hard_occupancy, mode)
            soft_rmse = float(_rmse(soft_field, observed).detach().cpu())
            hard_rmse = float(_rmse(hard_field, observed).detach().cpu())
            probabilities = torch.sigmoid(logits / float(temperature))
            row = {
                "step": step,
                "temperature": float(temperature),
                "objective": objective,
                "gradient_norm_before_clip": gradient_norm,
                "soft_rmse": soft_rmse,
                "hard_rmse": hard_rmse,
                "soft_attainment": _attainment(baseline_rmse, soft_rmse),
                "hard_attainment": _attainment(baseline_rmse, hard_rmse),
                "soft_hard_attainment_gap": _attainment(baseline_rmse, soft_rmse)
                - _attainment(baseline_rmse, hard_rmse),
                "soft_target_mass_in_search": float(probabilities.sum().detach().cpu()),
                "probability_entropy": _entropy(probabilities),
                **_voxel_geometry_metrics(
                    hard_occupancy, suite.case, search_mask
                ),
            }
            return row, hard_occupancy.detach().cpu()

    trace: list[dict[str, object]] = []
    initial_row, initial_hard = audit(0, float(temperatures[0]), None, None)
    trace.append(initial_row)
    best_rmse = float(initial_row["hard_rmse"])
    best_step = 0
    best_logits = logits.detach().cpu().clone()
    best_hard_occupancy = initial_hard
    best_metrics = dict(initial_row)

    for step, temperature in enumerate(temperatures, start=1):
        optimizer.zero_grad(set_to_none=True)
        forward_occupancy, _, _ = occupancies(float(temperature))
        predicted = suite.field_from_occupancy(forward_occupancy, mode)
        loss = (predicted - observed).square().mean() / normalizer
        if not torch.isfinite(loss):
            raise FloatingPointError("voxel objective is non-finite")
        loss.backward()
        if logits.grad is None or not torch.isfinite(logits.grad).all():
            raise FloatingPointError("voxel gradients are absent or non-finite")
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [logits], float(gradient_clip_norm)
        )
        if not torch.isfinite(gradient_norm_tensor):
            raise FloatingPointError("voxel gradient norm is non-finite")
        optimizer.step()
        if not torch.isfinite(logits).all():
            raise FloatingPointError("voxel logits are non-finite")
        should_check = (
            step == 1
            or step % int(hard_check_interval) == 0
            or step == len(temperatures)
        )
        if should_check:
            row, hard_occupancy = audit(
                step,
                float(temperature),
                float(loss.detach().cpu()),
                float(gradient_norm_tensor.detach().cpu()),
            )
            trace.append(row)
            if float(row["hard_rmse"]) < best_rmse:
                best_rmse = float(row["hard_rmse"])
                best_step = step
                best_logits = logits.detach().cpu().clone()
                best_hard_occupancy = hard_occupancy
                best_metrics = dict(row)

    return {
        "mode": mode,
        "control": control,
        "method": method,
        "baseline_rmse": baseline_rmse,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "final_metrics": trace[-1],
        "best_logits": best_logits,
        "best_hard_occupancy": best_hard_occupancy,
        "trace": trace,
        "observed": observed.detach().cpu(),
        "correct_observation": observed_correct.detach().cpu(),
        "baseline_field": baseline_field.detach().cpu(),
    }


def validate_embedding_endpoint_config(
    config: Mapping[str, object], *, grid_shape: Sequence[int]
) -> dict[str, object]:
    """Validate the frozen Q3 checkpoint-embedding endpoint protocol."""
    if config.get("schema") != EMBEDDING_ENDPOINT_SCHEMA:
        raise ValueError(f"embedding endpoint schema must be {EMBEDDING_ENDPOINT_SCHEMA!r}")
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("embedding endpoint config requires a non-empty id")
    modes = config.get("observation_modes")
    methods = config.get("methods")
    controls = config.get("seismic_controls")
    for values, allowed, name in (
        (modes, OBSERVATION_MODES, "observation_modes"),
        (methods, EMBEDDING_METHODS, "methods"),
        (controls, OBSERVATION_CONTROLS, "seismic_controls"),
    ):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be an array")
        parsed = tuple(str(value) for value in values)
        if not parsed or len(parsed) != len(set(parsed)) or any(
            value not in allowed for value in parsed
        ):
            raise ValueError(f"{name} must be unique members of {allowed}")
    updates = int(config.get("updates", 0))
    learning_rate = float(config.get("learning_rate", float("nan")))
    weight_decay = float(config.get("weight_decay", float("nan")))
    gradient_clip_norm = float(config.get("gradient_clip_norm", float("nan")))
    hard_check_interval = int(config.get("hard_check_interval", 0))
    max_norm_ratio = float(
        config.get("max_state_norm_to_embedding_norm", float("nan"))
    )
    if updates <= 0 or hard_check_interval <= 0:
        raise ValueError("updates and hard_check_interval must be positive")
    for value, name, allow_zero in (
        (learning_rate, "learning_rate", False),
        (weight_decay, "weight_decay", True),
        (gradient_clip_norm, "gradient_clip_norm", False),
        (max_norm_ratio, "max_state_norm_to_embedding_norm", False),
    ):
        if not math.isfinite(value) or value < 0 or (not allow_zero and value == 0):
            raise ValueError(f"{name} has an invalid value")
    schedule = config.get("temperature_schedule")
    if not isinstance(schedule, Sequence) or isinstance(schedule, (str, bytes)):
        raise ValueError("temperature_schedule must be an array")
    temperatures: list[float] = []
    resolved_schedule: list[dict[str, object]] = []
    for index, segment in enumerate(schedule):
        if not isinstance(segment, Mapping):
            raise ValueError(f"temperature segment {index} must be an object")
        temperature = float(segment.get("temperature", float("nan")))
        steps = int(segment.get("steps", 0))
        if not math.isfinite(temperature) or temperature <= 0 or steps <= 0:
            raise ValueError("temperature segments require positive finite values")
        temperatures.extend([temperature] * steps)
        resolved_schedule.append({"temperature": temperature, "steps": steps})
    if len(temperatures) != updates:
        raise ValueError("temperature schedule steps must equal updates")
    if config.get("hard_selection") != "minimum_hard_physics_loss_only":
        raise ValueError("Q3 requires hard-physics-only selection")
    regularization = config.get("regularization")
    if not isinstance(regularization, Mapping):
        raise ValueError("regularization must be an object")
    if any(
        float(regularization.get(name, float("nan"))) != 0.0
        for name in ("volume", "smoothness")
    ) or bool(regularization.get("truth_roi", True)):
        raise ValueError("Q3 forbids volume, smoothness and truth-ROI regularization")
    if bool(config.get("flow_unet_loaded", True)):
        raise ValueError("Q3 must explicitly forbid loading the flow U-Net")
    if bool(config.get("formal_training_authorized", True)):
        raise ValueError("Q3 must explicitly forbid formal training")
    search_region = config.get("search_region")
    if not isinstance(search_region, Mapping):
        raise ValueError("search_region must be an object")
    start = _int_triple(search_region.get("start"), "search_region.start")
    stop = _int_triple(search_region.get("stop"), "search_region.stop")
    shape = tuple(int(value) for value in grid_shape)
    if any(left < 0 or right > size or left >= right for left, right, size in zip(start, stop, shape)):
        raise ValueError("search_region must lie inside the grid")
    return {
        "schema": EMBEDDING_ENDPOINT_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "search_region": {"start": list(start), "stop": list(stop)},
        "observation_modes": [str(value) for value in modes],
        "methods": [str(value) for value in methods],
        "updates": updates,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "gradient_clip_norm": gradient_clip_norm,
        "temperature_schedule": resolved_schedule,
        "temperatures": temperatures,
        "hard_check_interval": hard_check_interval,
        "max_state_norm_to_embedding_norm": max_norm_ratio,
        "seismic_controls": [str(value) for value in controls],
        "shuffle_seed": int(config.get("shuffle_seed", -1)),
        "hard_selection": "minimum_hard_physics_loss_only",
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "flow_unet_loaded": False,
        "formal_training_authorized": False,
    }


def optimize_embedding_endpoint(
    suite: AnalyticObservationSuite,
    mode: str,
    *,
    search_mask: torch.Tensor,
    embedding_weight: torch.Tensor,
    control: str,
    method: str,
    temperatures: Sequence[float],
    learning_rate: float,
    weight_decay: float,
    gradient_clip_norm: float,
    hard_check_interval: int,
    max_state_norm_to_embedding_norm: float,
    shuffle_seed: int,
) -> dict[str, object]:
    """Optimize only checkpoint-embedding vectors inside the broad Q2 region.

    ``soft_embedding`` exactly preserves the deployed all-class cosine-softmax
    bridge. ``ste_embedding_rock`` is a diagnostic hard-forward alternative
    whose underground hard choice excludes the physically inadmissible air
    class; its soft surrogate is normalized over the same fourteen rock classes.
    """
    if method not in EMBEDDING_METHODS:
        raise ValueError(f"embedding method must be one of {EMBEDDING_METHODS}")
    if not temperatures:
        raise ValueError("temperatures must be non-empty")
    device = suite.acoustic_property_table.device
    dtype = suite.acoustic_property_table.dtype
    embeddings = embedding_weight.detach().to(device=device, dtype=dtype).contiguous()
    if embeddings.ndim != 2 or embeddings.shape[0] != suite.acoustic_property_table.shape[1]:
        raise ValueError("embedding weight must have one row per category")
    if not torch.isfinite(embeddings).all() or bool(
        (torch.linalg.vector_norm(embeddings, dim=1) <= 1e-12).any()
    ):
        raise ValueError("embedding rows must be finite and nonzero")
    search = search_mask.to(device=device, dtype=torch.bool)
    flat_indices = torch.nonzero(search.reshape(-1), as_tuple=False).flatten()
    if not flat_indices.numel():
        raise ValueError("embedding search mask is empty")
    case = suite.case
    baseline_labels = case.baseline_labels.to(device=device)
    truth_labels = case.truth_labels.to(device=device)
    category_count, embedding_dim = embeddings.shape
    baseline_categories = baseline_labels.long() + 1
    baseline_probabilities = F.one_hot(
        baseline_categories[:, 0], num_classes=category_count
    ).permute(0, 4, 1, 2, 3).to(dtype=dtype)
    baseline_probability_flat = baseline_probabilities.reshape(category_count, -1)
    baseline_search = baseline_probability_flat[:, flat_indices]
    background_index = case.background_label + 1
    initial_vectors = embeddings[background_index].expand(flat_indices.numel(), -1)
    vectors = torch.nn.Parameter(initial_vectors.clone())
    optimizer = torch.optim.Adam(
        [vectors], lr=float(learning_rate), weight_decay=float(weight_decay)
    )
    max_embedding_norm = torch.linalg.vector_norm(embeddings, dim=1).max()
    max_state_norm = float(max_embedding_norm.detach().cpu()) * float(
        max_state_norm_to_embedding_norm
    )
    observed_correct = suite.field_from_labels(truth_labels, mode).detach()
    observed = controlled_observation(
        observed_correct, control, shuffle_seed=shuffle_seed
    )
    baseline_field = suite.field_from_labels(baseline_labels, mode).detach()
    baseline_rmse = float(_rmse(baseline_field, observed).detach().cpu())
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 1e-12:
        raise ValueError("controlled observation must differ from the hard baseline")
    normalizer = baseline_rmse * baseline_rmse
    normalized_embeddings = F.normalize(embeddings, dim=1)

    def decoded_values(
        temperature: float,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        similarities = F.normalize(vectors, dim=1) @ normalized_embeddings.T
        if method == "soft_embedding":
            soft = torch.softmax(similarities / float(temperature), dim=1)
            hard_categories = similarities.argmax(dim=1)
        elif method == "ste_embedding_rock":
            rock_soft = torch.softmax(
                similarities[:, 1:] / float(temperature), dim=1
            )
            soft = F.pad(rock_soft, (1, 0), value=0.0)
            hard_categories = similarities[:, 1:].argmax(dim=1) + 1
        else:
            binary_indices = torch.as_tensor(
                [background_index, case.target_label + 1], device=device
            )
            binary_soft = torch.softmax(
                similarities[:, binary_indices] / float(temperature), dim=1
            )
            soft = torch.zeros_like(similarities).scatter(
                1,
                binary_indices.unsqueeze(0).expand(similarities.shape[0], -1),
                binary_soft,
            )
            hard_categories = binary_indices[
                similarities[:, binary_indices].argmax(dim=1)
            ]
        hard = F.one_hot(hard_categories, num_classes=category_count).to(dtype)
        forward = (
            soft
            if method in {"soft_embedding", "soft_embedding_binary"}
            else hard + soft - soft.detach()
        )
        return forward, soft, hard, hard_categories

    def full_probabilities(values: torch.Tensor) -> torch.Tensor:
        delta = torch.zeros_like(baseline_probability_flat).scatter(
            1,
            flat_indices.unsqueeze(0).expand(category_count, -1),
            values.T - baseline_search,
        )
        return (baseline_probability_flat + delta).reshape_as(baseline_probabilities)

    def hard_labels(categories: torch.Tensor) -> torch.Tensor:
        labels_flat = baseline_labels.reshape(-1).clone()
        labels_flat.scatter_(0, flat_indices, categories.to(labels_flat.dtype) - 1)
        return labels_flat.reshape_as(baseline_labels)

    def audit(
        step: int,
        temperature: float,
        objective: float | None,
        gradient_norm: float | None,
    ) -> tuple[dict[str, object], torch.Tensor, torch.Tensor]:
        with torch.no_grad():
            _, soft, _, categories = decoded_values(temperature)
            soft_field = suite.field_from_probabilities(
                full_probabilities(soft), mode
            )
            labels = hard_labels(categories)
            hard_field = suite.field_from_labels(labels, mode)
            soft_rmse = float(_rmse(soft_field, observed).detach().cpu())
            hard_rmse = float(_rmse(hard_field, observed).detach().cpu())
            hard_occupancy = (labels == case.target_label).to(dtype=dtype)
            counts = torch.bincount(categories, minlength=category_count)
            changed = categories != background_index
            row = {
                "step": step,
                "temperature": float(temperature),
                "objective": objective,
                "gradient_norm_before_clip": gradient_norm,
                "soft_rmse": soft_rmse,
                "hard_rmse": hard_rmse,
                "soft_attainment": _attainment(baseline_rmse, soft_rmse),
                "hard_attainment": _attainment(baseline_rmse, hard_rmse),
                "soft_hard_attainment_gap": _attainment(baseline_rmse, soft_rmse)
                - _attainment(baseline_rmse, hard_rmse),
                "soft_target_mass_in_search": float(
                    soft[:, case.target_label + 1].sum().detach().cpu()
                ),
                "categorical_entropy": _entropy(soft),
                "hard_air_voxels_in_search": int(counts[0]),
                "hard_background_voxels_in_search": int(counts[background_index]),
                "hard_target_voxels_in_search": int(counts[case.target_label + 1]),
                "hard_changed_voxels_in_search": int(changed.sum()),
                "hard_non_target_changed_voxels_in_search": int(
                    (changed & (categories != case.target_label + 1)).sum()
                ),
                "hard_distinct_categories_in_search": int((counts > 0).sum()),
                "max_state_voxel_norm": float(
                    torch.linalg.vector_norm(vectors, dim=1).max().detach().cpu()
                ),
                **_voxel_geometry_metrics(hard_occupancy, case, search_mask),
            }
            row.update(
                {
                    f"hard_category_{index}_voxels_in_search": int(count)
                    for index, count in enumerate(counts.tolist())
                }
            )
            return row, hard_occupancy.detach().cpu(), labels.detach().cpu()

    trace: list[dict[str, object]] = []
    initial_row, initial_occupancy, initial_labels = audit(
        0, float(temperatures[0]), None, None
    )
    trace.append(initial_row)
    best_rmse = float(initial_row["hard_rmse"])
    best_step = 0
    best_vectors = vectors.detach().cpu().clone()
    best_hard_occupancy = initial_occupancy
    best_labels = initial_labels
    best_metrics = dict(initial_row)

    for step, temperature in enumerate(temperatures, start=1):
        optimizer.zero_grad(set_to_none=True)
        forward, _, _, _ = decoded_values(float(temperature))
        predicted = suite.field_from_probabilities(full_probabilities(forward), mode)
        loss = (predicted - observed).square().mean() / normalizer
        if not torch.isfinite(loss):
            raise FloatingPointError("embedding objective is non-finite")
        loss.backward()
        if vectors.grad is None or not torch.isfinite(vectors.grad).all():
            raise FloatingPointError("embedding gradients are absent or non-finite")
        gradient_norm_tensor = torch.nn.utils.clip_grad_norm_(
            [vectors], float(gradient_clip_norm)
        )
        if not torch.isfinite(gradient_norm_tensor):
            raise FloatingPointError("embedding gradient norm is non-finite")
        optimizer.step()
        with torch.no_grad():
            norms = torch.linalg.vector_norm(vectors, dim=1, keepdim=True)
            vectors.mul_(torch.clamp(max_state_norm / norms.clamp_min(1e-12), max=1.0))
        if not torch.isfinite(vectors).all():
            raise FloatingPointError("embedding endpoint state is non-finite")
        should_check = (
            step == 1
            or step % int(hard_check_interval) == 0
            or step == len(temperatures)
        )
        if should_check:
            row, hard_occupancy, labels = audit(
                step,
                float(temperature),
                float(loss.detach().cpu()),
                float(gradient_norm_tensor.detach().cpu()),
            )
            trace.append(row)
            if float(row["hard_rmse"]) < best_rmse:
                best_rmse = float(row["hard_rmse"])
                best_step = step
                best_vectors = vectors.detach().cpu().clone()
                best_hard_occupancy = hard_occupancy
                best_labels = labels
                best_metrics = dict(row)

    return {
        "mode": mode,
        "control": control,
        "method": method,
        "baseline_rmse": baseline_rmse,
        "best_step": best_step,
        "best_metrics": best_metrics,
        "final_metrics": trace[-1],
        "best_vectors": best_vectors,
        "best_hard_occupancy": best_hard_occupancy,
        "best_labels": best_labels,
        "trace": trace,
        "observed": observed.detach().cpu(),
        "correct_observation": observed_correct.detach().cpu(),
        "baseline_field": baseline_field.detach().cpu(),
    }


def validate_hard_coordinate_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen Q2b monotone hard-coordinate protocol."""
    if config.get("schema") != HARD_COORDINATE_SCHEMA:
        raise ValueError(f"hard-coordinate schema must be {HARD_COORDINATE_SCHEMA!r}")
    config_id = str(config.get("id", "")).strip()
    if not config_id:
        raise ValueError("hard-coordinate config requires a non-empty id")
    modes = config.get("observation_modes")
    controls = config.get("seismic_controls")
    for values, allowed, name in (
        (modes, OBSERVATION_MODES, "observation_modes"),
        (controls, OBSERVATION_CONTROLS, "seismic_controls"),
    ):
        if not isinstance(values, Sequence) or isinstance(values, (str, bytes)):
            raise ValueError(f"{name} must be an array")
        parsed = tuple(str(value) for value in values)
        if not parsed or len(parsed) != len(set(parsed)) or any(
            value not in allowed for value in parsed
        ):
            raise ValueError(f"{name} must be unique members of {allowed}")
    iterations = int(config.get("max_iterations", 0))
    if iterations <= 0:
        raise ValueError("max_iterations must be positive")
    counts_raw = config.get("proposal_flip_counts")
    if not isinstance(counts_raw, Sequence) or isinstance(counts_raw, (str, bytes)):
        raise ValueError("proposal_flip_counts must be an array")
    counts = tuple(int(value) for value in counts_raw)
    if not counts or tuple(sorted(set(counts))) != counts or any(value <= 0 for value in counts):
        raise ValueError("proposal flip counts must be unique increasing positive integers")
    tolerance = float(config.get("improvement_tolerance", float("nan")))
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("improvement_tolerance must be finite and non-negative")
    if config.get("allow_additions") is not True or config.get("allow_removals") is not True:
        raise ValueError("Q2b requires both additions and removals")
    if config.get("selection") != "minimum_hard_physics_rmse_only":
        raise ValueError("Q2b requires hard-physics-only selection")
    regularization = config.get("regularization")
    if not isinstance(regularization, Mapping):
        raise ValueError("regularization must be an object")
    if any(float(regularization.get(name, float("nan"))) != 0.0 for name in ("volume", "smoothness")):
        raise ValueError("Q2b forbids volume and smoothness regularization")
    if bool(regularization.get("truth_roi", True)):
        raise ValueError("Q2b forbids a truth ROI")
    if bool(config.get("formal_training_authorized", True)):
        raise ValueError("Q2b must explicitly forbid formal training")
    return {
        "schema": HARD_COORDINATE_SCHEMA,
        "id": config_id,
        "description": str(config.get("description", "")),
        "observation_modes": [str(value) for value in modes],
        "seismic_controls": [str(value) for value in controls],
        "max_iterations": iterations,
        "proposal_flip_counts": list(counts),
        "improvement_tolerance": tolerance,
        "allow_additions": True,
        "allow_removals": True,
        "selection": "minimum_hard_physics_rmse_only",
        "regularization": {"volume": 0.0, "smoothness": 0.0, "truth_roi": False},
        "formal_training_authorized": False,
    }


def optimize_hard_coordinates(
    suite: AnalyticObservationSuite,
    mode: str,
    *,
    search_mask: torch.Tensor,
    control: str,
    max_iterations: int,
    proposal_flip_counts: Sequence[int],
    improvement_tolerance: float,
    shuffle_seed: int,
) -> dict[str, object]:
    """Greedily accept only hard proposals that lower hard physics RMSE."""
    if max_iterations <= 0:
        raise ValueError("max_iterations must be positive")
    counts = tuple(int(value) for value in proposal_flip_counts)
    if not counts or any(value <= 0 for value in counts):
        raise ValueError("proposal_flip_counts must be positive")
    device = suite.acoustic_property_table.device
    dtype = suite.acoustic_property_table.dtype
    search = search_mask.to(device=device, dtype=torch.bool)
    flat_indices = torch.nonzero(search.reshape(-1), as_tuple=False).flatten()
    fixed = suite.case.fixed_target_mask.to(device=device, dtype=dtype)
    truth_occupancy = (suite.case.truth_labels == suite.case.target_label).to(
        device=device, dtype=dtype
    )
    correct_observation = suite.field_from_occupancy(truth_occupancy, mode).detach()
    observed = controlled_observation(
        correct_observation, control, shuffle_seed=shuffle_seed
    )

    def full_occupancy(values: torch.Tensor) -> torch.Tensor:
        flat = torch.zeros(
            (values.shape[0], truth_occupancy.numel()),
            device=device,
            dtype=dtype,
        ).scatter(1, flat_indices.unsqueeze(0).expand(values.shape[0], -1), values)
        return fixed.expand(values.shape[0], -1, -1, -1, -1) + flat.reshape(
            values.shape[0], *truth_occupancy.shape[1:]
        )

    current = torch.zeros((1, flat_indices.numel()), device=device, dtype=dtype)
    current_occupancy = full_occupancy(current)
    current_field = suite.field_from_occupancy(current_occupancy, mode).detach()
    baseline_rmse = float(_rmse(current_field, observed).detach().cpu())
    if not math.isfinite(baseline_rmse) or baseline_rmse <= 1e-12:
        raise ValueError("controlled observation must differ from the hard baseline")
    current_rmse = baseline_rmse
    initial_geometry = _voxel_geometry_metrics(
        current_occupancy, suite.case, search_mask
    )
    trace: list[dict[str, object]] = [
        {
            "iteration": 0,
            "accepted": True,
            "accepted_flip_count": 0,
            "hard_rmse": current_rmse,
            "hard_attainment": 0.0,
            "positive_action_count": 0,
            "gradient_norm": None,
            **initial_geometry,
        }
    ]

    for iteration in range(1, int(max_iterations) + 1):
        differentiable = current.detach().clone().requires_grad_(True)
        occupancy = full_occupancy(differentiable)
        predicted = suite.field_from_occupancy(occupancy, mode)
        loss = (predicted - observed).square().mean() / (baseline_rmse * baseline_rmse)
        gradient = torch.autograd.grad(loss, differentiable)[0]
        if not torch.isfinite(gradient).all():
            raise FloatingPointError("hard-coordinate gradient is non-finite")
        # A 0->1 change has first-order delta +grad; a 1->0 change has -grad.
        improvement_score = torch.where(current < 0.5, -gradient, gradient)
        positive = improvement_score[0] > 0
        positive_count = int(positive.sum())
        gradient_norm = float(torch.linalg.vector_norm(gradient).detach().cpu())
        if positive_count == 0:
            trace.append(
                {
                    "iteration": iteration,
                    "accepted": False,
                    "accepted_flip_count": 0,
                    "hard_rmse": current_rmse,
                    "hard_attainment": _attainment(baseline_rmse, current_rmse),
                    "positive_action_count": 0,
                    "gradient_norm": gradient_norm,
                    **_voxel_geometry_metrics(
                        full_occupancy(current), suite.case, search_mask
                    ),
                }
            )
            break
        ranked = torch.argsort(improvement_score[0], descending=True)
        ranked = ranked[positive[ranked]]
        proposals: list[torch.Tensor] = []
        actual_counts: list[int] = []
        for requested in counts:
            count = min(int(requested), positive_count)
            if actual_counts and count == actual_counts[-1]:
                continue
            proposal = current.clone()
            chosen = ranked[:count]
            proposal[0, chosen] = 1.0 - proposal[0, chosen]
            proposals.append(proposal)
            actual_counts.append(count)
        proposal_values = torch.cat(proposals, dim=0)
        proposal_fields = suite.field_from_occupancy(
            full_occupancy(proposal_values), mode
        )
        proposal_rmse = (proposal_fields - observed).square().flatten(1).mean(dim=1).sqrt()
        best_index = int(proposal_rmse.argmin())
        best_rmse = float(proposal_rmse[best_index].detach().cpu())
        accepted = best_rmse < current_rmse - float(improvement_tolerance)
        accepted_count = actual_counts[best_index] if accepted else 0
        if accepted:
            current = proposal_values[best_index : best_index + 1].detach()
            current_rmse = best_rmse
        hard_occupancy = full_occupancy(current)
        trace.append(
            {
                "iteration": iteration,
                "accepted": accepted,
                "accepted_flip_count": accepted_count,
                "hard_rmse": current_rmse,
                "hard_attainment": _attainment(baseline_rmse, current_rmse),
                "positive_action_count": positive_count,
                "gradient_norm": gradient_norm,
                **_voxel_geometry_metrics(
                    hard_occupancy, suite.case, search_mask
                ),
            }
        )
        if not accepted:
            break

    rmse_values = [float(row["hard_rmse"]) for row in trace]
    if any(right > left + 1e-12 for left, right in zip(rmse_values, rmse_values[1:])):
        raise RuntimeError("hard-coordinate accepted trace is not monotone")
    final_occupancy = full_occupancy(current).detach().cpu()
    return {
        "mode": mode,
        "control": control,
        "baseline_rmse": baseline_rmse,
        "iterations_completed": len(trace) - 1,
        "accepted_iterations": sum(bool(row["accepted"]) for row in trace[1:]),
        "final_metrics": trace[-1],
        "trace": trace,
        "final_hard_occupancy": final_occupancy,
        "observed": observed.detach().cpu(),
        "correct_observation": correct_observation.detach().cpu(),
        "baseline_field": current_field.detach().cpu(),
    }
