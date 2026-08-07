"""Derivative-free structured hard-geophysics inference for Stage 7B.

Selection in this module receives only a hard-forward response function and an
observed field.  Geological truth is intentionally absent from the search API.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
import time
from typing import Callable, Iterable, Sequence

import torch


STRUCTURED_HARD_VERSION = "stage7_structured_hard_geophysics_v1"


@dataclass(frozen=True)
class StructuredObject:
    """One low-dimensional categorical geological event."""

    object_id: str
    presence: bool
    center_x: float
    center_y: float
    center_z: float
    size_x: float
    size_y: float
    size_z: float
    orientation_deg: float
    shape: str
    material_label: int
    source_family: str

    def record(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class StructuredModel:
    objects: tuple[StructuredObject, ...]
    model_id: str
    parent_id: str | None
    proposal_move: str


def _coordinate_grid(shape: Sequence[int], device: torch.device) -> tuple[torch.Tensor, ...]:
    return torch.meshgrid(
        *(torch.arange(int(size), device=device, dtype=torch.float32) for size in shape),
        indexing="ij",
    )


def rasterize_object(value: StructuredObject, shape: Sequence[int], *, device: torch.device) -> torch.Tensor:
    """Rasterize a rotated cuboid/ellipsoid/native DikeHemisphere event."""
    result = torch.zeros(tuple(int(size) for size in shape), device=device, dtype=torch.bool)
    if not value.presence or value.material_label < 0:
        return result
    if min(value.size_x, value.size_y, value.size_z) <= 0:
        return result
    x, y, z = _coordinate_grid(shape, device)
    angle = math.radians(float(value.orientation_deg))
    dx = x - float(value.center_x)
    dy = y - float(value.center_y)
    dz = z - float(value.center_z)
    rotated_x = math.cos(angle) * dx + math.sin(angle) * dy
    rotated_y = -math.sin(angle) * dx + math.cos(angle) * dy
    if value.shape == "cuboid":
        return (
            (rotated_x.abs() < float(value.size_x) / 2.0)
            & (rotated_y.abs() < float(value.size_y) / 2.0)
            & (dz.abs() < float(value.size_z) / 2.0)
        )
    if value.shape == "ellipsoid":
        radius = (
            (rotated_x / (float(value.size_x) / 2.0)).square()
            + (rotated_y / (float(value.size_y) / 2.0)).square()
            + (dz / (float(value.size_z) / 2.0)).square()
        )
        return radius <= 1.0
    if value.shape == "dike_hemisphere":
        # Exact DikeHemisphere geometry: IntrusionSpec shifts the process origin
        # down by height/2 so center_z is a point inside the event.
        process_bottom = float(value.center_z) - 0.5 * float(value.size_z)
        scaled_x = rotated_x / (float(value.size_x) / 2.0)
        scaled_y = rotated_y / (float(value.size_y) / 2.0)
        scaled_z = (z - process_bottom) / float(value.size_z)
        surface = torch.sqrt((1.0 - scaled_x.square() - scaled_y.square()).clamp_min(0.0))
        return (scaled_z > 0) & (scaled_z < surface)
    raise ValueError(f"unsupported structured shape: {value.shape}")


def object_within_bounds(value: StructuredObject, shape: Sequence[int], air_start_z: int) -> bool:
    """Conservative event bounding-box validity check."""
    radius_xy = 0.5 * math.sqrt(float(value.size_x) ** 2 + float(value.size_y) ** 2)
    return (
        value.presence
        and value.center_x - radius_xy >= 0
        and value.center_x + radius_xy < int(shape[0])
        and value.center_y - radius_xy >= 0
        and value.center_y + radius_xy < int(shape[1])
        and value.center_z - 0.5 * value.size_z >= 0
        and value.center_z + 0.5 * value.size_z < int(air_start_z)
    )


def materialize_model(
    model: StructuredModel,
    *,
    baseline_labels: torch.Tensor,
    condition_mask: torch.Tensor,
    air_start_z: int,
    allowed_material_labels: Sequence[int],
) -> tuple[torch.Tensor | None, dict[str, object]]:
    """Create hard labels and reject any invalid or condition-violating proposal."""
    labels = baseline_labels.clone()
    shape = labels.shape[2:]
    occupied = torch.zeros(shape, device=labels.device, dtype=torch.bool)
    reasons: list[str] = []
    for value in model.objects:
        if value.material_label not in allowed_material_labels:
            reasons.append("invalid_material")
            continue
        if not object_within_bounds(value, shape, air_start_z):
            reasons.append("out_of_bounds_or_air")
            continue
        mask = rasterize_object(value, shape, device=labels.device)
        if bool((mask & occupied).any()):
            reasons.append("object_overlap")
            continue
        if bool((mask & condition_mask[0, 0]).any()):
            reasons.append("condition_intersection")
            continue
        occupied |= mask
        labels[0, 0, mask] = int(value.material_label)
    violations = int((labels[condition_mask] != baseline_labels[condition_mask]).sum())
    if violations:
        reasons.append("condition_violation")
    valid = not reasons
    return (labels if valid else None), {
        "valid": valid,
        "reasons": sorted(set(reasons)),
        "condition_violations": violations,
        "object_count": len(model.objects),
        "modified_voxels": int((labels != baseline_labels).sum()) if valid else 0,
    }


def _canonical_key(objects: Iterable[StructuredObject]) -> tuple[tuple[object, ...], ...]:
    return tuple(sorted((
        value.object_id,
        round(value.center_x, 4), round(value.center_y, 4), round(value.center_z, 4),
        round(value.size_x, 4), round(value.size_y, 4), round(value.size_z, 4),
        round(value.orientation_deg % 360.0, 4), value.shape, value.material_label,
    ) for value in objects))


def trust_region_mutations(
    parent: StructuredModel,
    *,
    generation: int,
    allowed_material_labels: Sequence[int],
) -> list[StructuredModel]:
    """Deterministic add/remove/translate/resize/rotate/shape/lithology moves."""
    children: list[StructuredModel] = []
    serial = 0
    def append(objects: tuple[StructuredObject, ...], move: str) -> None:
        nonlocal serial
        children.append(StructuredModel(objects, f"g{generation:02d}_{parent.model_id}_{serial:04d}", parent.model_id, move))
        serial += 1
    for index, value in enumerate(parent.objects):
        append(parent.objects[:index] + parent.objects[index + 1 :], "remove_object")
        for axis in ("center_x", "center_y", "center_z"):
            for delta in (-1.0, -0.25, 0.25, 1.0):
                changed = replace(value, **{axis: getattr(value, axis) + delta})
                append(parent.objects[:index] + (changed,) + parent.objects[index + 1 :], f"translate_{axis}_{delta:+g}")
        for axis in ("size_x", "size_y", "size_z"):
            for delta in (-1.0, -0.25, 0.25, 1.0):
                changed = replace(value, **{axis: getattr(value, axis) + delta})
                append(parent.objects[:index] + (changed,) + parent.objects[index + 1 :], f"resize_{axis}_{delta:+g}")
        for delta in (-15.0, -5.0, 5.0, 15.0):
            changed = replace(value, orientation_deg=(value.orientation_deg + delta) % 360.0)
            append(parent.objects[:index] + (changed,) + parent.objects[index + 1 :], f"rotate_{delta:+g}")
        if value.shape in {"cuboid", "ellipsoid"}:
            other = "ellipsoid" if value.shape == "cuboid" else "cuboid"
            append(parent.objects[:index] + (replace(value, shape=other),) + parent.objects[index + 1 :], "change_shape")
        for material in allowed_material_labels:
            if int(material) != value.material_label:
                append(parent.objects[:index] + (replace(value, material_label=int(material)),) + parent.objects[index + 1 :], "change_lithology")
    return children


@torch.no_grad()
def beam_evolutionary_search(
    *,
    baseline_labels: torch.Tensor,
    condition_mask: torch.Tensor,
    air_start_z: int,
    observation: torch.Tensor,
    hard_response: Callable[[torch.Tensor], torch.Tensor],
    proposal_library: Sequence[StructuredObject],
    allowed_material_labels: Sequence[int],
    kmax: int,
    beam_size: int,
    local_generations: int,
) -> dict[str, object]:
    """Top-K derivative-free search selected exclusively by hard observed RMSE."""
    if kmax <= 0 or beam_size <= 0:
        raise ValueError("kmax and beam_size must be positive")
    started = time.perf_counter()
    trace: list[dict[str, object]] = []
    forward_calls = 0
    def evaluate(model: StructuredModel, generation: int, parent_loss: float | None) -> float | None:
        nonlocal forward_calls
        labels, validation = materialize_model(
            model, baseline_labels=baseline_labels, condition_mask=condition_mask,
            air_start_z=air_start_z, allowed_material_labels=allowed_material_labels,
        )
        if labels is None:
            trace.append({
                "generation": generation, "model_id": model.model_id, "parent_id": model.parent_id,
                "proposal_move": model.proposal_move, "hard_seismic_rmse": float("inf"),
                "accepted_vs_parent": False, "selected_by": "hard_observed_physics_only",
                "cached": False, **validation,
                "objects": [value.record() for value in model.objects],
            })
            return None
        response = hard_response(labels)
        forward_calls += 1
        loss = float((response - observation).square().mean().sqrt().detach().cpu())
        accepted = parent_loss is None or loss < parent_loss - 1e-12
        trace.append({
            "generation": generation, "model_id": model.model_id, "parent_id": model.parent_id,
            "proposal_move": model.proposal_move, "hard_seismic_rmse": loss,
            "accepted_vs_parent": accepted, "selected_by": "hard_observed_physics_only",
            "cached": False, **validation,
            "objects": [value.record() for value in model.objects],
        })
        return loss

    empty = StructuredModel((), "empty_baseline", None, "initial")
    base_result = evaluate(empty, 0, None)
    if base_result is None:
        raise RuntimeError("baseline structured model is invalid")
    beams: list[tuple[float, StructuredModel]] = [(base_result, empty)]
    archive: list[tuple[float, StructuredModel]] = list(beams)
    # Add one object at a time.  The cuboid protocol uses beam_size == library size,
    # so all exact two-object combinations remain reachable without truth indices.
    for generation in range(1, kmax + 1):
        candidates: list[tuple[float, StructuredModel]] = []
        for parent_loss, parent in beams:
            used = {value.object_id for value in parent.objects}
            for proposal in proposal_library:
                if proposal.object_id in used:
                    continue
                child = StructuredModel(
                    parent.objects + (proposal,),
                    f"g{generation:02d}_{parent.model_id}_add_{proposal.object_id}",
                    parent.model_id,
                    "add_object",
                )
                result = evaluate(child, generation, parent_loss)
                # A population beam retains the best finite candidates at this
                # cardinality even when an individual add move crosses a hard-
                # physics barrier.  ``accepted_vs_parent`` remains false and no
                # accepted geology is replaced.  Final selection is still the
                # minimum hard loss in the global archive.
                if result is not None:
                    candidates.append((result, child))
        unique: dict[tuple[tuple[object, ...], ...], tuple[float, StructuredModel]] = {}
        for item in candidates:
            key = _canonical_key(item[1].objects)
            if key not in unique or item[0] < unique[key][0]:
                unique[key] = item
        beams = sorted(unique.values(), key=lambda item: (item[0], item[1].model_id))[:beam_size]
        archive.extend(beams)
    for local in range(local_generations):
        generation = kmax + local + 1
        candidates = list(beams)
        for parent_loss, parent in beams:
            for child in trust_region_mutations(parent, generation=generation, allowed_material_labels=allowed_material_labels):
                result = evaluate(child, generation, parent_loss)
                if result is not None and result < parent_loss - 1e-12:
                    candidates.append((result, child))
        unique = {}
        for item in candidates:
            key = _canonical_key(item[1].objects)
            if key not in unique or item[0] < unique[key][0]:
                unique[key] = item
        beams = sorted(unique.values(), key=lambda item: (item[0], item[1].model_id))[:beam_size]
        archive.extend(beams)
    archive_unique = {}
    for item in archive:
        key = _canonical_key(item[1].objects)
        if key not in archive_unique or item[0] < archive_unique[key][0]:
            archive_unique[key] = item
    best_loss, best_model = sorted(
        archive_unique.values(), key=lambda item: (item[0], item[1].model_id)
    )[0]
    best_labels, best_validation = materialize_model(
        best_model, baseline_labels=baseline_labels, condition_mask=condition_mask,
        air_start_z=air_start_z, allowed_material_labels=allowed_material_labels,
    )
    baseline_materialized, baseline_validation = materialize_model(
        empty, baseline_labels=baseline_labels, condition_mask=condition_mask,
        air_start_z=air_start_z, allowed_material_labels=allowed_material_labels,
    )
    if best_labels is None or baseline_materialized is None:
        raise RuntimeError("selected or baseline structured model became invalid")
    best_response = hard_response(best_labels)
    baseline_response = hard_response(baseline_materialized)
    return {
        "best_hard_rmse": best_loss,
        "best_model": best_model,
        "best_labels": best_labels.detach().cpu(),
        "best_response": best_response.detach().cpu(),
        "baseline_hard_rmse": base_result,
        "baseline_labels": baseline_materialized.detach().cpu(),
        "baseline_response": baseline_response.detach().cpu(),
        "hard_attainment": 1.0 - best_loss / base_result if base_result > 0 else float("nan"),
        "beam": [{"hard_rmse": item[0], "model": item[1]} for item in beams],
        "trace": trace,
        "forward_call_count": forward_calls,
        "runtime_seconds": time.perf_counter() - started,
        "selection_used_truth": False,
        "selection_criterion": "minimum hard observed seismic RMSE only",
    }
