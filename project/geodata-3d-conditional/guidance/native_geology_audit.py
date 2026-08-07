"""StructuralGeo-native target construction and prior-support diagnostics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np
import torch
from scipy import ndimage


NATIVE_GEOLOGY_AUDIT_VERSION = "phase6q_structuralgeo_native_audit_v1"


@dataclass(frozen=True)
class NativeGeologyCase:
    truth_labels: torch.Tensor
    condition_mask: torch.Tensor
    subsurface_mask: torch.Tensor
    body_masks: torch.Tensor
    event_labels: tuple[int, ...]
    event_roles: tuple[str, ...]
    well_xy: tuple[tuple[int, int], ...]
    air_label: int = -1
    background_label: int = 0
    target_label: int = 9


def build_structuralgeo_native_case(
    *,
    seed: int,
    grid_shape: Sequence[int] = (64, 64, 64),
    air_start_z: int = 56,
) -> tuple[NativeGeologyCase, dict[str, object]]:
    """Build five separate native IntrusionSpec events and merge their audit labels.

    Labels 9..13 are temporary event identities produced by StructuralGeo.  They are
    merged to audit label 9 only after the five body masks have been recovered from
    the computed event history; label 9 is therefore not assumed to mean "dike".
    """
    if tuple(int(v) for v in grid_shape) != (64, 64, 64):
        raise ValueError("the frozen model native audit uses its canonical 64^3 grid")
    if int(air_start_z) != 56:
        raise ValueError("the native audit requires the canonical air boundary z=56")
    from geogen.engine import GeoModelSpec, IntrusionSpec, ParametricGeoEngine, Uniform

    anchors = ((10, 10, 17), (26, 10, 37), (42, 10, 25), (12, 42, 21), (42, 42, 39))
    event_labels = (9, 10, 11, 12, 13)
    roles = ("drilled", "drilled", "drilled", "hidden", "hidden")
    events = [
        IntrusionSpec(
            origin=anchor,
            kind="hemisphere",
            value=label,
            anchor_to_present=False,
            clip=False,
            diameter=Uniform(8.0, 12.0),
            height=Uniform(7.0, 11.0),
            minor_axis_scale=Uniform(0.7, 1.2),
            rotation=Uniform(0.0, 360.0),
            upper=True,
        )
        for anchor, label in zip(anchors, event_labels)
    ]
    spec = GeoModelSpec(
        name="phase6q_five_native_intrusions",
        events=events,
        bounds=((0.0, 63.0), (0.0, 63.0), (0.0, 63.0)),
        resolution=(64, 64, 64),
        seed=int(seed),
        normalize=False,
        height_tracking=False,
        include_basement=True,
        basement_base=55.0,
        basement_value=0,
    )
    model = ParametricGeoEngine().generate(spec, keep_snapshots=True)
    native = model.get_data_grid().copy()
    native[np.isnan(native)] = -1
    native = native.astype(np.int64)
    body_masks_np = np.stack([native == label for label in event_labels])
    if any(int(mask.sum()) == 0 for mask in body_masks_np):
        raise RuntimeError("a StructuralGeo native event produced an empty body")
    if int(body_masks_np.sum(axis=0).max()) > 1:
        raise RuntimeError("native body event masks overlap after history evaluation")
    merged = native.copy()
    merged[np.isin(merged, event_labels)] = 9
    truth = torch.from_numpy(merged).view(1, 1, 64, 64, 64).long()
    body_masks = torch.from_numpy(body_masks_np).bool()
    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., air_start_z:] = True
    condition[..., air_start_z - 1] = True
    for x, y, _ in anchors[:3]:
        condition[0, 0, x, y, :air_start_z] = True
    if bool((body_masks[3:] & condition[0, 0]).any()):
        raise RuntimeError("hidden native bodies intersect hard conditions")
    subsurface = torch.zeros_like(condition)
    subsurface[..., :air_start_z] = True
    case = NativeGeologyCase(
        truth_labels=truth,
        condition_mask=condition,
        subsurface_mask=subsurface,
        body_masks=body_masks,
        event_labels=event_labels,
        event_roles=roles,
        well_xy=tuple((x, y) for x, y, _ in anchors[:3]),
    )
    history = model.get_history_string(unpacked=True)
    metadata = {
        "generator": "StructuralGeo ParametricGeoEngine",
        "generator_seed": int(seed),
        "model_name": spec.name,
        "bounds": [list(v) for v in spec.bounds],
        "resolution": list(spec.resolution),
        "normalize": spec.normalize,
        "height_tracking": spec.height_tracking,
        "event_types": ["IntrusionSpec(kind=hemisphere)" for _ in events],
        "event_audit_labels": list(event_labels),
        "event_roles": list(roles),
        "anchors_xyz": [list(v) for v in anchors],
        "merged_target_audit_label": 9,
        "audit_label_semantics": "union of five explicitly recorded StructuralGeo IntrusionSpec event masks",
        "body_voxel_counts": [int(mask.sum()) for mask in body_masks_np],
        "event_history": history,
    }
    return case, metadata


def connected_target_statistics(
    labels: torch.Tensor,
    *,
    target_label: int,
    condition_mask: torch.Tensor,
) -> dict[str, object]:
    """Return deterministic 6-connected size/location/topology statistics."""
    target = (labels.detach().cpu().numpy()[0, 0] == int(target_label))
    structure = ndimage.generate_binary_structure(3, 1)
    components, count = ndimage.label(target, structure=structure)
    conditioned_target = target & condition_mask.detach().cpu().numpy()[0, 0]
    rows = []
    for component_id in range(1, int(count) + 1):
        mask = components == component_id
        coordinates = np.argwhere(mask)
        touches_condition = bool((mask & conditioned_target).any())
        rows.append({
            "component_id": component_id,
            "voxel_count": int(mask.sum()),
            "centroid_xyz": coordinates.mean(axis=0).tolist(),
            "bbox_min_xyz": coordinates.min(axis=0).tolist(),
            "bbox_max_xyz": coordinates.max(axis=0).tolist(),
            "touches_conditioned_target": touches_condition,
        })
    labels_np = labels.detach().cpu().numpy()
    values, frequencies = np.unique(labels_np, return_counts=True)
    return {
        "target_voxel_count": int(target.sum()),
        "target_fraction": float(target.mean()),
        "component_count_6": int(count),
        "components": rows,
        "unconditioned_component_count": sum(not row["touches_conditioned_target"] for row in rows),
        "target_outside_condition_voxels": int((target & ~condition_mask.detach().cpu().numpy()[0, 0]).sum()),
        "raw_label_frequency": {str(int(v)): int(n) for v, n in zip(values, frequencies)},
    }
