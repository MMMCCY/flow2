#!/usr/bin/env python3
"""Truth-only preparation and hard diversity gate for Stage11."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.optimize import linear_sum_assignment
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guidance.stage11_diverse_geometry import build_stage11_diverse_case
from scripts.stage11.common import (
    DIVERSITY_SPEC_PATH,
    EXPERIMENT_DIR,
    PROTOCOL_PATH,
    REGISTRY_PATH,
    file_record,
    load_frozen_inputs,
    sha256,
    write_json,
)
from scripts.stage9.common import save_tensor_x, utc_now


def _centroid(mask: np.ndarray) -> np.ndarray:
    coordinates = np.argwhere(mask)
    if not len(coordinates):
        raise ValueError("target mask is empty")
    return coordinates.mean(axis=0)


def _component_count(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, ndimage.generate_binary_structure(3, 1))[1])


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, object]:
    records = {}
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs.update(dpi=600, metadata={"Software": "Stage11 geometry audit"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"Creator": "Stage11 geometry audit", "CreationDate": None, "ModDate": None}
        else:
            kwargs["metadata"] = {"Creator": "Stage11 geometry audit", "Date": "2000-01-01T00:00:00Z"}
        fig.savefig(path, **kwargs)
        records[suffix] = file_record(path)
    plt.close(fig)
    return records


def _geometry_figure(
    case_ids: list[str],
    masks: dict[str, np.ndarray],
    pairwise_iou: np.ndarray,
    centroids: dict[str, np.ndarray],
    volumes: dict[str, int],
) -> dict[str, object]:
    plt.rcParams.update({
        "font.family": "STIXGeneral", "font.size": 8, "axes.titlesize": 8,
        "svg.fonttype": "none", "svg.hashsalt": "stage11-geometry-v1", "pdf.fonttype": 42,
    })
    fig = plt.figure(figsize=(8.4, 4.5))
    grid = fig.add_gridspec(2, 5, left=0.05, right=0.97, top=0.92, bottom=0.10, hspace=0.42, wspace=0.20)
    for index, case_id in enumerate(case_ids):
        axis = fig.add_subplot(grid[0, index])
        projection = masks[case_id].max(axis=2)
        axis.imshow(projection.T, origin="lower", cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
        center = centroids[case_id]
        axis.plot(center[0], center[1], marker="+", color="#1F4E79", ms=7, mew=1.3)
        axis.set_title(f"C{index + 1:02d}  n={_component_count(masks[case_id])}\nV={volumes[case_id]}")
        axis.set_xticks((0, 32, 63)); axis.set_yticks((0, 32, 63))
        axis.set_xlabel("x");
        if index == 0:
            axis.set_ylabel("y")
        else:
            axis.set_yticklabels([])
    heat_axis = fig.add_subplot(grid[1, 0:2])
    image = heat_axis.imshow(pairwise_iou, cmap="YlOrRd", vmin=0.0, vmax=0.2)
    labels = [f"C{index + 1:02d}" for index in range(len(case_ids))]
    heat_axis.set_xticks(range(len(case_ids)), labels=labels)
    heat_axis.set_yticks(range(len(case_ids)), labels=labels)
    heat_axis.set_title("Pairwise label-9 IoU")
    for row in range(len(case_ids)):
        for column in range(len(case_ids)):
            heat_axis.text(column, row, f"{pairwise_iou[row, column]:.2f}", ha="center", va="center", fontsize=6.5)
    fig.colorbar(image, ax=heat_axis, fraction=0.046, pad=0.04)
    scatter_axis = fig.add_subplot(grid[1, 2:5])
    xyz = np.stack([centroids[case_id] for case_id in case_ids])
    sizes = np.asarray([volumes[case_id] for case_id in case_ids], dtype=float)
    sizes = 55.0 + 145.0 * sizes / sizes.max()
    scatter = scatter_axis.scatter(xyz[:, 0], xyz[:, 1], c=xyz[:, 2], s=sizes, cmap="viridis", edgecolor="black", linewidth=0.5)
    for index, point in enumerate(xyz):
        scatter_axis.annotate(labels[index], point[:2], xytext=(4, 3), textcoords="offset points")
    scatter_axis.set_xlim(0, 63); scatter_axis.set_ylim(0, 63)
    scatter_axis.set_xlabel("Union centroid x [voxel]"); scatter_axis.set_ylabel("Union centroid y [voxel]")
    scatter_axis.set_title("Centroid separation; colour = depth z; size = target volume")
    colorbar = fig.colorbar(scatter, ax=scatter_axis, fraction=0.035, pad=0.03)
    colorbar.set_label("Centroid z [voxel]")
    fig.suptitle("Stage11 pre-geophysical geometry diversity audit", fontsize=10)
    return _save_figure(fig, EXPERIMENT_DIR / "figures/stage11_geometry_diversity")


def main() -> None:
    registry, diversity, protocol = load_frozen_inputs()
    benchmark_root = EXPERIMENT_DIR / "benchmark"
    audit_root = EXPERIMENT_DIR / "audit"
    figure_root = EXPERIMENT_DIR / "figures"
    if benchmark_root.exists() or audit_root.exists() or figure_root.exists():
        raise FileExistsError("refusing to reuse an existing Stage11 geometry preparation")
    benchmark_root.mkdir(parents=True)
    audit_root.mkdir(parents=True)
    figure_root.mkdir(parents=True)

    boreholes = registry["observation_layout"]["borehole_xy"]
    cases = list(registry["cases"])
    case_ids = [str(case["case_id"]) for case in cases]
    masks: dict[str, np.ndarray] = {}
    bodies: dict[str, np.ndarray] = {}
    centroids: dict[str, np.ndarray] = {}
    volumes: dict[str, int] = {}
    component_counts: dict[str, int] = {}
    case_records = {}
    borehole_rows = []
    for case_config in cases:
        case_id = str(case_config["case_id"])
        case, metadata = build_stage11_diverse_case(case_config, borehole_xy=boreholes)
        mask = (case.truth_labels[0, 0] == int(registry["target_label"])).numpy()
        body_array = case.body_masks.numpy()
        masks[case_id] = mask
        bodies[case_id] = body_array
        centroids[case_id] = _centroid(mask)
        volumes[case_id] = int(mask.sum())
        component_counts[case_id] = _component_count(mask)
        case_dir = benchmark_root / case_id
        case_dir.mkdir()
        tensors = {
            "truth_labels": save_tensor_x(case_dir / "truth_labels.pt", case.truth_labels.long()),
            "body_masks": save_tensor_x(case_dir / "body_masks.pt", case.body_masks.bool()),
            "condition_mask": save_tensor_x(case_dir / "condition_mask.pt", case.condition_mask.bool()),
            "subsurface_mask": save_tensor_x(case_dir / "subsurface_mask.pt", case.subsurface_mask.bool()),
        }
        write_json(case_dir / "metadata.json", metadata)
        case_manifest = {
            "schema": "stage11_truth_only_case_v1",
            "status": "frozen_before_geophysical_computation",
            "case_id": case_id,
            "truth_role": "geometry audit and later retrospective evaluation only",
            "case_registry_sha256": sha256(REGISTRY_PATH),
            "tensors": tensors,
            "metadata": file_record(case_dir / "metadata.json"),
        }
        write_json(case_dir / "manifest.json", case_manifest)
        case_records[case_id] = file_record(case_dir / "manifest.json")
        for hit in metadata["borehole_target_hits"]:
            borehole_rows.append({"case_id": case_id, **hit})

    pair_rows = []
    iou_matrix = np.eye(len(case_ids), dtype=float)
    for left_index in range(len(case_ids)):
        for right_index in range(left_index + 1, len(case_ids)):
            left_id, right_id = case_ids[left_index], case_ids[right_index]
            left, right = masks[left_id], masks[right_id]
            intersection = int((left & right).sum())
            union = int((left | right).sum())
            iou = intersection / union if union else 0.0
            distance = float(np.linalg.norm(centroids[left_id] - centroids[right_id]))
            left_body_centroids = np.stack([_centroid(mask) for mask in bodies[left_id]])
            right_body_centroids = np.stack([_centroid(mask) for mask in bodies[right_id]])
            distances = np.linalg.norm(left_body_centroids[:, None] - right_body_centroids[None, :], axis=2)
            matched_left, matched_right = linear_sum_assignment(distances)
            matched = distances[matched_left, matched_right]
            row = {
                "case_i": left_id,
                "case_j": right_id,
                "label9_iou": iou,
                "centroid_distance_voxels": distance,
                "volume_i": volumes[left_id],
                "volume_j": volumes[right_id],
                "volume_ratio_min_over_max": min(volumes[left_id], volumes[right_id]) / max(volumes[left_id], volumes[right_id]),
                "component_count_i": component_counts[left_id],
                "component_count_j": component_counts[right_id],
                "matched_body_count": len(matched),
                "unmatched_body_count": abs(len(bodies[left_id]) - len(bodies[right_id])),
                "matched_body_centroid_distance_mean": float(matched.mean()),
                "matched_body_centroid_distance_min": float(matched.min()),
                "matched_body_centroid_distance_max": float(matched.max()),
            }
            pair_rows.append(row)
            iou_matrix[left_index, right_index] = iou
            iou_matrix[right_index, left_index] = iou
    _write_csv(audit_root / "geometry_diversity.csv", pair_rows)

    primary = diversity["primary_pairwise_criteria"]
    cohort = diversity["cohort_criteria"]
    centroid_stack = np.stack([centroids[case_id] for case_id in case_ids])
    registered_events = [event for case in cases for event in case["events"]]
    rotation_values = [float(event["rotation_deg"]) for event in registered_events]
    diameter_values = [float(event["diameter"]) for event in registered_events]
    minor_values = [float(event["minor_axis_scale"]) for event in registered_events]
    checks = {
        "all_pairwise_iou_below_limit": all(float(row["label9_iou"]) < float(primary["label9_iou_strictly_below"]) for row in pair_rows),
        "all_pairwise_centroid_distance_at_least_minimum": all(float(row["centroid_distance_voxels"]) >= float(primary["union_centroid_distance_voxels_at_least"]) for row in pair_rows),
        "unique_component_counts": len(set(component_counts.values())) >= int(cohort["unique_component_counts_at_least"]),
        "centroid_span_x": float(np.ptp(centroid_stack[:, 0])) >= float(cohort["union_centroid_span_x_voxels_at_least"]),
        "centroid_span_y": float(np.ptp(centroid_stack[:, 1])) >= float(cohort["union_centroid_span_y_voxels_at_least"]),
        "centroid_span_z": float(np.ptp(centroid_stack[:, 2])) >= float(cohort["union_centroid_span_z_voxels_at_least"]),
        "registered_intrusion_kinds": len(set(str(event["kind"]) for event in registered_events)) >= int(cohort["registered_intrusion_kinds_at_least"]),
        "registered_rotation_span": max(rotation_values) - min(rotation_values) >= float(cohort["registered_rotation_span_degrees_at_least"]),
        "registered_diameter_span": max(diameter_values) - min(diameter_values) >= float(cohort["registered_diameter_span_voxels_at_least"]),
        "registered_minor_axis_span": max(minor_values) - min(minor_values) >= float(cohort["registered_minor_axis_scale_span_at_least"]),
    }
    passed = all(checks.values())
    decision = "PASS_DIVERSE_BENCHMARK" if passed else "STOP_BENCHMARK_NOT_DIVERSE"
    layout_audit = {
        "schema": "stage11_borehole_layout_audit_v1",
        "status": "frozen_before_geophysical_computation",
        "layout": registry["observation_layout"],
        "selection_used_hidden_truth": False,
        "case_hit_records": borehole_rows,
        "cases_excluded_due_to_hits": [],
    }
    write_json(audit_root / "borehole_layout.json", layout_audit)
    figures = _geometry_figure(case_ids, masks, iou_matrix, centroids, volumes)
    freeze = {
        "schema": "stage11_geometry_freeze_v1",
        "status": "complete_before_geophysical_computation",
        "machine_decision": decision,
        "checks": checks,
        "case_ids": case_ids,
        "case_registry": file_record(REGISTRY_PATH),
        "diversity_spec": file_record(DIVERSITY_SPEC_PATH),
        "protocol": file_record(PROTOCOL_PATH),
        "geometry_csv": file_record(audit_root / "geometry_diversity.csv"),
        "borehole_layout": file_record(audit_root / "borehole_layout.json"),
        "case_manifests": case_records,
        "figures": figures,
        "summary": {
            "pairwise_iou_range": [min(float(row["label9_iou"]) for row in pair_rows), max(float(row["label9_iou"]) for row in pair_rows)],
            "centroid_distance_range": [min(float(row["centroid_distance_voxels"]) for row in pair_rows), max(float(row["centroid_distance_voxels"]) for row in pair_rows)],
            "component_counts": component_counts,
            "centroid_spans_xyz": np.ptp(centroid_stack, axis=0).tolist(),
        },
        "geophysical_computation_count_at_freeze": 0,
        "flow_forward_count_at_freeze": 0,
        "completed_at_utc": utc_now(),
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
    }
    write_json(audit_root / "geometry_freeze.json", freeze)
    write_json(benchmark_root / "manifest.json", {
        "schema": "stage11_truth_only_benchmark_v1",
        "status": "frozen_before_geophysical_computation",
        "case_ids": case_ids,
        "case_manifests": case_records,
        "geometry_freeze": file_record(audit_root / "geometry_freeze.json"),
    })
    print(json.dumps({"status": decision, "checks": checks, "summary": freeze["summary"]}, indent=2))


if __name__ == "__main__":
    main()
