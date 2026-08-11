"""Deterministic StructuralGeo cases for the pre-registered Stage11 benchmark."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np
import torch

from guidance.native_geology_audit import NativeGeologyCase


STAGE11_BUILDER_SCHEMA = "stage11_diverse_structuralgeo_builder_v1"


def build_stage11_diverse_case(
    case_config: Mapping[str, object],
    *,
    borehole_xy: Sequence[Sequence[int]],
    grid_shape: Sequence[int] = (64, 64, 64),
    air_start_z: int = 56,
) -> tuple[NativeGeologyCase, dict[str, object]]:
    """Build one explicit StructuralGeo recipe without any geophysical input."""
    if tuple(int(value) for value in grid_shape) != (64, 64, 64):
        raise ValueError("Stage11 uses the frozen 64^3 grid")
    if int(air_start_z) != 56:
        raise ValueError("Stage11 uses the frozen z=56 air boundary")
    from geogen.engine import GeoModelSpec, IntrusionSpec, ParametricGeoEngine

    registered_events = list(case_config["events"])
    if not 1 <= len(registered_events) <= 5:
        raise ValueError("Stage11 recipes require one to five target events")
    event_labels = tuple(9 + index for index in range(len(registered_events)))
    events = []
    for event, label in zip(registered_events, event_labels):
        kwargs = {
            "origin": tuple(float(value) for value in event["anchor_xyz"]),
            "kind": str(event["kind"]),
            "value": label,
            "anchor_to_present": False,
            "clip": False,
            "diameter": float(event["diameter"]),
            "minor_axis_scale": float(event["minor_axis_scale"]),
            "rotation": float(event["rotation_deg"]),
        }
        if event["kind"] == "hemisphere":
            kwargs.update(height=float(event["height"]), upper=bool(event["upper"]))
        elif event["kind"] == "column":
            kwargs.update(depth=float(event["depth"]))
        elif event["kind"] == "plug":
            kwargs.update(plug_shape=float(event["plug_shape"]))
        else:
            raise ValueError(f"unsupported Stage11 intrusion kind: {event['kind']}")
        events.append(IntrusionSpec(**kwargs))

    spec = GeoModelSpec(
        name=f"stage11_diverse_{case_config['case_id']}",
        events=events,
        bounds=((0.0, 63.0), (0.0, 63.0), (0.0, 63.0)),
        resolution=(64, 64, 64),
        seed=int(case_config["structuralgeo_seed"]),
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
        raise RuntimeError("a registered Stage11 target event is empty")
    merged = native.copy()
    merged[np.isin(merged, event_labels)] = 9
    truth = torch.from_numpy(merged).view(1, 1, 64, 64, 64).long()
    body_masks = torch.from_numpy(body_masks_np).bool()
    subsurface = torch.zeros_like(truth, dtype=torch.bool)
    subsurface[..., :air_start_z] = True
    if bool(((truth == 9) & ~subsurface).any()):
        raise RuntimeError("target body extends into the registered air domain")

    condition = torch.zeros_like(truth, dtype=torch.bool)
    condition[..., air_start_z:] = True
    condition[..., air_start_z - 1] = True
    wells = tuple((int(pair[0]), int(pair[1])) for pair in borehole_xy)
    for x, y in wells:
        condition[0, 0, x, y, :air_start_z] = True
    per_well_hits = []
    for x, y in wells:
        hit_z = torch.nonzero(truth[0, 0, x, y, :air_start_z] == 9).flatten().tolist()
        per_well_hits.append({"xy": [x, y], "target_hit": bool(hit_z), "target_hit_z": hit_z})

    case = NativeGeologyCase(
        truth_labels=truth,
        condition_mask=condition,
        subsurface_mask=subsurface,
        body_masks=body_masks,
        event_labels=event_labels,
        event_roles=tuple("hidden_target" for _ in event_labels),
        well_xy=wells,
    )
    metadata = {
        "schema": STAGE11_BUILDER_SCHEMA,
        "generator": "StructuralGeo ParametricGeoEngine",
        "case_id": str(case_config["case_id"]),
        "generator_seed": int(case_config["structuralgeo_seed"]),
        "model_name": spec.name,
        "grid_shape": [64, 64, 64],
        "air_start_z": air_start_z,
        "registered_events": registered_events,
        "event_audit_labels": list(event_labels),
        "body_voxel_counts": [int(mask.sum()) for mask in body_masks_np],
        "merged_target_label": 9,
        "borehole_layout_truth_dependent": False,
        "borehole_target_hits": per_well_hits,
        "event_history": model.get_history_string(unpacked=True),
    }
    return case, metadata
