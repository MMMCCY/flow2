#!/usr/bin/env python3
"""Retrospective evaluator and compact figure for the topology support audit."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs, binary_occupancy_to_acoustic
from guidance.seismic import seismic_operator_from_config
from guidance.topology_support import betti_numbers, binary_metrics, ring_diagnostics
from scripts.stage15.common import read_json, refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_topology_support"


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _median(rows: list[dict[str, object]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _centroid_distance(prediction: torch.Tensor, truth: torch.Tensor) -> float:
    p = torch.nonzero(prediction, as_tuple=False).float()
    t = torch.nonzero(truth, as_tuple=False).float()
    if len(p) == 0 or len(t) == 0:
        return float("inf")
    return float(torch.linalg.vector_norm(p.mean(0) - t.mean(0)))


def _crop(mask: torch.Tensor, truth: torch.Tensor, margin: int = 5) -> torch.Tensor:
    index = torch.nonzero(truth, as_tuple=False)
    lo = torch.clamp(index.min(0).values - margin, min=0)
    hi = torch.minimum(index.max(0).values + margin + 1, torch.tensor(mask.shape))
    return mask[lo[0] : hi[0], lo[1] : hi[1], lo[2] : hi[2]]


def _plot(rows_by_key: dict[tuple[str, str, int], torch.Tensor], truths: dict[str, torch.Tensor], seed: int, path: Path) -> None:
    arms = ["TRUTH", "FLOW_ONLY", "SEISMIC_GUIDED", "ORACLE_GUIDED"]
    fig = plt.figure(figsize=(12, 6.5), constrained_layout=True)
    for row_index, case_id in enumerate(("A_SOLID", "B_RING")):
        for column_index, arm in enumerate(arms):
            ax = fig.add_subplot(2, 4, row_index * 4 + column_index + 1, projection="3d")
            mask = truths[case_id] if arm == "TRUTH" else rows_by_key[(case_id, arm, seed)] == 9
            coords = torch.nonzero(mask, as_tuple=False).numpy()
            if len(coords):
                ax.scatter(coords[:, 0], coords[:, 1], coords[:, 2], s=2.2, c="#d95f02", alpha=0.78, linewidths=0)
            ax.set_xlim(5, 35); ax.set_ylim(20, 50); ax.set_zlim(12, 38)
            ax.view_init(elev=24, azim=-55)
            ax.set_box_aspect((1, 1, 0.72))
            ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
            ax.set_title((case_id.replace("_", " ") + "\n" if column_index == 0 else "") + arm.replace("_", " "), fontsize=9)
    fig.savefig(path, dpi=220, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/frozen_protocol_v1.json")
    parser.add_argument("--evaluation-protocol", type=Path, default=ROOT / "configs/evaluation_protocol_v2.json")
    parser.add_argument("--cases-dir", type=Path, default=ROOT / "cases_v1_fix1")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_v1_fix1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/topology_support_v1")
    args = parser.parse_args()
    refuse_nonempty(args.output_dir)
    protocol = read_json(args.protocol)
    evaluation = read_json(args.evaluation_protocol)
    if evaluation.get("status") not in {
        "frozen_before_retrospective_evaluation",
        "frozen_after_v1_metric_defect_before_v2_evaluation",
    }:
        raise ValueError("evaluation protocol is not frozen")
    for arm in protocol["arms"]:
        manifest = read_json(args.runs_dir / arm / "manifest.json")
        if manifest.get("run_status") != "completed" or int(manifest.get("sample_count", -1)) != 6:
            raise ValueError(f"incomplete arm: {arm}")
    binary_config = read_json(_path(protocol["binary_acoustic_config"]))
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(_path(binary_config["source_acoustic_config"]["path"])))
    operator, _ = seismic_operator_from_config(read_json(_path(protocol["seismic_config"])), grid_shape=(64, 64, 64))
    rows: list[dict[str, object]] = []
    predictions: dict[tuple[str, str, int], torch.Tensor] = {}
    truths: dict[str, torch.Tensor] = {}
    for case_id, definition in protocol["cases"].items():
        truth = runtime.load_tensor(args.cases_dir / case_id / "truth_restricted/binary_truth.pt").bool()[0, 0]
        support = runtime.load_tensor(args.cases_dir / case_id / "subsurface_mask.pt").bool()
        observed = runtime.load_tensor(args.cases_dir / case_id / "observed_seismic.pt").float()
        truths[case_id] = truth
        for arm in protocol["arms"]:
            for seed in protocol["source_seeds"]:
                prediction = runtime.load_tensor(args.runs_dir / arm / case_id / f"seed_{seed}.pt").long()
                predictions[(case_id, arm, int(seed))] = prediction
                target = prediction == 9
                metrics = binary_metrics(target, truth)
                impedance, slowness = binary_occupancy_to_acoustic(target.view(1, 1, 64, 64, 64).float(), support, properties)
                predicted_seismic = operator(impedance, slowness, support)
                row: dict[str, object] = {
                    "case_id": case_id, "arm": arm, "source_seed": int(seed), **metrics,
                    "predicted_label9_voxels": int(target.sum()),
                    "centroid_distance": _centroid_distance(target, truth),
                    "hard_seismic_rmse": float((predicted_seismic - observed).square().mean().sqrt()),
                }
                topology = betti_numbers(_crop(target, truth))
                row.update({f"target_crop_{key}": value for key, value in topology.items()})
                if case_id == "B_RING":
                    row.update(ring_diagnostics(target, definition["center"], definition["major_radius"], definition["tube_radius"]))
                rows.append(row)
    summary = {"schema": "stage15_topology_support_summary_v1", "run_status": "completed", "cases": {}}
    for case_id in protocol["cases"]:
        case_summary = {}
        for arm in protocol["arms"]:
            selected = [row for row in rows if row["case_id"] == case_id and row["arm"] == arm]
            fields = ["iou", "precision", "recall", "centroid_distance", "hard_seismic_rmse", "target_crop_beta1"]
            if case_id == "B_RING":
                fields += ["central_hole_preservation", "azimuthal_ring_coverage"]
            case_summary[arm] = {f"median_{field}": _median(selected, field) for field in fields}
            case_summary[arm]["beta1_equal_one_count"] = sum(int(row["target_crop_beta1"]) == 1 for row in selected)
        summary["cases"][case_id] = case_summary
    ring = summary["cases"]["B_RING"]
    prior_support = (
        ring["ORACLE_GUIDED"]["median_iou"] > ring["FLOW_ONLY"]["median_iou"]
        and ring["ORACLE_GUIDED"]["median_azimuthal_ring_coverage"] > ring["FLOW_ONLY"]["median_azimuthal_ring_coverage"]
        and ring["ORACLE_GUIDED"]["beta1_equal_one_count"] >= 1
    )
    seismic_correction = (
        prior_support
        and ring["SEISMIC_GUIDED"]["median_iou"] > ring["FLOW_ONLY"]["median_iou"]
        and ring["SEISMIC_GUIDED"]["median_azimuthal_ring_coverage"] > ring["FLOW_ONLY"]["median_azimuthal_ring_coverage"]
        and ring["SEISMIC_GUIDED"]["median_hard_seismic_rmse"] < ring["FLOW_ONLY"]["median_hard_seismic_rmse"]
    )
    decision = (
        "PRIOR_SUPPORTS_RING_AND_SEISMIC_CORRECTS" if seismic_correction else
        "PRIOR_SUPPORTS_RING_BUT_SEISMIC_BRIDGE_INSUFFICIENT" if prior_support else
        "FROZEN_PRIOR_RING_SUPPORT_NOT_DEMONSTRATED"
    )
    summary.update({
        "evaluation_protocol_schema": evaluation["schema"],
        "v1_metric_defect_corrected": evaluation["schema"] == "stage15_topology_support_evaluation_v2",
        "prior_support_gate": prior_support,
        "seismic_correction_gate": seismic_correction,
        "machine_decision": decision,
    })
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sample_metrics.csv", rows)
    write_json(args.output_dir / "summary.json", summary)
    _plot(predictions, truths, int(evaluation["figure_seed"]), args.output_dir / "topology_support_comparison.png")
    with (args.output_dir / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# Stage15 topology support audit\n\n")
        handle.write(f"Decision: **{decision}**\n\n")
        handle.write("This is a frozen-checkpoint topology stress test, not a certified historical held-out split.\n\n")
        for case_id, case_summary in summary["cases"].items():
            handle.write(f"## {case_id}\n\n")
            handle.write("| Arm | median IoU | median P/R | median seismic RMSE | median beta1 |\n|---|---:|---:|---:|---:|\n")
            for arm, values in case_summary.items():
                handle.write(f"| {arm} | {values['median_iou']:.4f} | {values['median_precision']:.4f}/{values['median_recall']:.4f} | {values['median_hard_seismic_rmse']:.6f} | {values['median_target_crop_beta1']:.1f} |\n")
            handle.write("\n")


if __name__ == "__main__":
    main()
