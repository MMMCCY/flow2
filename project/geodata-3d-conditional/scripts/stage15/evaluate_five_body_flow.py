#!/usr/bin/env python3
"""Retrospective evaluation of Stage15 guidance on the Stage7-style five bodies."""

from __future__ import annotations

import argparse
import csv
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
from guidance.topology_support import betti_numbers, binary_metrics
from scripts.stage15.common import read_json, refuse_nonempty, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_five_body_flow"


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _median(rows: list[dict[str, object]], field: str) -> float:
    return float(np.median([float(row[field]) for row in rows]))


def _components(mask: torch.Tensor) -> tuple[torch.Tensor, int]:
    from scipy import ndimage
    labels, count = ndimage.label(mask.cpu().numpy(), structure=ndimage.generate_binary_structure(3, 1))
    return torch.from_numpy(labels), int(count)


def _body_connectivity(prediction: torch.Tensor, body_masks: torch.Tensor) -> tuple[int, list[int]]:
    labels, _ = _components(prediction)
    body_components = []
    for mask in body_masks:
        values = labels[mask]
        values = values[values > 0]
        body_components.append(int(torch.mode(values).values) if len(values) else 0)
    positive = [value for value in body_components if value > 0]
    merged_pairs = sum(positive[i] == positive[j] for i in range(len(positive)) for j in range(i + 1, len(positive)))
    return merged_pairs, body_components


def _render(path: Path, truth: torch.Tensor, bodies: torch.Tensor, predictions: dict[tuple[str, int], torch.Tensor], seed: int) -> None:
    arms = ["TRUTH", "FLOW_ONLY", "SEISMIC_GUIDED", "ORACLE_GUIDED"]
    fig = plt.figure(figsize=(13.2, 4.2), constrained_layout=True)
    truth_xyz = torch.nonzero(truth).numpy()
    hidden_truth = bodies[3:].any(0)
    for index, arm in enumerate(arms):
        ax = fig.add_subplot(1, 4, index + 1, projection="3d")
        mask = truth if arm == "TRUTH" else predictions[(arm, seed)] == 9
        xyz = torch.nonzero(mask).numpy()
        if len(xyz):
            ax.scatter(xyz[:, 0], xyz[:, 1], xyz[:, 2], s=1.3, c="#d95f02", alpha=0.65, linewidths=0)
        if arm != "TRUTH":
            hx = torch.nonzero(hidden_truth).numpy()
            ax.scatter(hx[:, 0], hx[:, 1], hx[:, 2], s=4.0, facecolors="none", edgecolors="#2166ac", alpha=0.17, linewidths=0.35)
        for body_index, body in enumerate(bodies):
            coords = torch.nonzero(body).float()
            lo, hi = coords.min(0).values.numpy(), (coords.max(0).values + 1).numpy()
            color = "#2166ac" if body_index >= 3 else "#6a3d9a"
            for x in (lo[0], hi[0]):
                for y in (lo[1], hi[1]):
                    ax.plot([x, x], [y, y], [lo[2], hi[2]], color=color, alpha=.35, lw=.6)
        ax.set_xlim(0, 64); ax.set_ylim(0, 64); ax.set_zlim(0, 52)
        ax.set_box_aspect((1, 1, .78)); ax.view_init(25, -55)
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.set_title(arm.replace("_", " "), fontsize=10)
    fig.savefig(path, dpi=240, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/frozen_protocol_v1.json")
    parser.add_argument("--case-dir", type=Path, default=ROOT / "cases_v1/FIVE_BODY")
    parser.add_argument("--runs-dir", type=Path, default=ROOT / "runs_v1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/five_body_flow_v1")
    parser.add_argument("--figure-seed", type=int, default=142)
    args = parser.parse_args()
    refuse_nonempty(args.output_dir)
    protocol = read_json(args.protocol)
    case_manifest = read_json(args.case_dir / "manifest.json")
    if not case_manifest.get("risk_gate_passed"):
        raise ValueError("five-body risk gate did not pass")
    for arm in protocol["arms"]:
        manifest = read_json(args.runs_dir / arm / "manifest.json")
        if manifest.get("run_status") != "completed" or int(manifest.get("sample_count", -1)) != 3:
            raise ValueError(f"incomplete arm: {arm}")
    truth = runtime.load_tensor(args.case_dir / "truth_restricted/binary_truth.pt").bool()[0, 0]
    hidden = runtime.load_tensor(args.case_dir / "truth_restricted/hidden_binary_truth.pt").bool()[0, 0]
    bodies = runtime.load_tensor(args.case_dir / "truth_restricted/body_masks.pt").bool()
    support = runtime.load_tensor(args.case_dir / "subsurface_mask.pt").bool()
    observed = runtime.load_tensor(args.case_dir / "observed_seismic.pt").float()
    binary_config = read_json(_path(protocol["binary_acoustic_config"]))
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(_path(binary_config["source_acoustic_config"]["path"])))
    operator, _ = seismic_operator_from_config(read_json(_path(protocol["seismic_config"])), grid_shape=(64, 64, 64))
    rows, predictions = [], {}
    for arm in protocol["arms"]:
        for seed in protocol["source_seeds"]:
            prediction = runtime.load_tensor(args.runs_dir / arm / "FIVE_BODY" / f"seed_{seed}.pt").long()
            target = prediction == 9
            predictions[(arm, int(seed))] = prediction
            full = binary_metrics(target, truth)
            hidden_metrics = binary_metrics(target & ~bodies[:3].any(0), hidden)
            impedance, slowness = binary_occupancy_to_acoustic(target.view(1, 1, 64, 64, 64).float(), support, properties)
            seismic = operator(impedance, slowness, support)
            merged_pairs, component_ids = _body_connectivity(target, bodies)
            row = {
                "arm": arm, "source_seed": int(seed),
                **{f"full_{key}": value for key, value in full.items()},
                **{f"hidden_{key}": value for key, value in hidden_metrics.items()},
                "predicted_label9_voxels": int(target.sum()),
                "hard_seismic_rmse": float((seismic - observed).square().mean().sqrt()),
                "predicted_component_count": betti_numbers(target)["beta0"],
                "truth_body_merged_pair_count": merged_pairs,
            }
            for index, mask in enumerate(bodies):
                row[f"body_{index}_recall"] = float((target & mask).sum() / mask.sum())
                row[f"body_{index}_dominant_component"] = component_ids[index]
            rows.append(row)
    summary = {
        "schema": "stage15_five_body_flow_summary_v2",
        "run_status": "completed",
        "case_risk_gate_passed": bool(case_manifest["risk_gate_passed"]),
        "case_design": {
            "target_raw_label": 9,
            "body_count": 5,
            "drilled_body_count": 3,
            "hidden_body_count": 2,
            "hidden_body_condition_voxels": 0,
        },
        "arms": {},
    }
    for arm in protocol["arms"]:
        selected = [row for row in rows if row["arm"] == arm]
        summary["arms"][arm] = {
            f"median_{field}": _median(selected, field)
            for field in (
                "full_iou", "full_precision", "full_recall", "hidden_iou", "hidden_precision", "hidden_recall",
                "hard_seismic_rmse", "predicted_label9_voxels", "truth_body_merged_pair_count",
                "body_0_recall", "body_1_recall", "body_2_recall", "body_3_recall", "body_4_recall",
            )
        }
    baseline = summary["arms"]["FLOW_ONLY"]
    seismic = summary["arms"]["SEISMIC_GUIDED"]
    oracle = summary["arms"]["ORACLE_GUIDED"]
    summary["findings"] = {
        "oracle_hidden_improves": oracle["median_hidden_iou"] > baseline["median_hidden_iou"],
        "seismic_hidden_improves": seismic["median_hidden_iou"] > baseline["median_hidden_iou"],
        "seismic_fit_improves": seismic["median_hard_seismic_rmse"] < baseline["median_hard_seismic_rmse"],
        "seismic_retains_separated_five_body_geometry": seismic["median_truth_body_merged_pair_count"] == 0,
        "oracle_retains_separated_five_body_geometry": oracle["median_truth_body_merged_pair_count"] == 0,
        "interpretation": (
            "Stage15 seismic guidance restores hidden label9 support but does not preserve the five-body topology; "
            "oracle spatial guidance shows that the frozen Flow can express the separated target geometry."
        ),
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sample_metrics.csv", rows)
    write_json(args.output_dir / "summary.json", summary)
    _render(args.output_dir / "five_body_flow_comparison.png", truth, bodies, predictions, args.figure_seed)
    with (args.output_dir / "REPORT.md").open("w", encoding="utf-8") as handle:
        handle.write("# Stage15 guidance on the Stage7-style five-body case\n\n")
        handle.write("All five bodies are raw label9; bodies 0--2 are drilled and bodies 3--4 are hidden.\n\n")
        handle.write("| Arm | full IoU | hidden IoU | hidden P/R | seismic RMSE | merged truth-body pairs |\n|---|---:|---:|---:|---:|---:|\n")
        for arm, values in summary["arms"].items():
            handle.write(f"| {arm} | {values['median_full_iou']:.4f} | {values['median_hidden_iou']:.4f} | {values['median_hidden_precision']:.4f}/{values['median_hidden_recall']:.4f} | {values['median_hard_seismic_rmse']:.6f} | {values['median_truth_body_merged_pair_count']:.1f} |\n")
        handle.write("\nThe case-level risk gate passed: all five equal-volume, disjoint bodies use raw label9; "
                     "three are intersected by wells and two have no hard-condition overlap. The current checkpoint, "
                     "hard conditions, and frozen Stage15 binary seismic physics are reused.\n\n")
        handle.write("Flow-only occasionally intersects one hidden body in individual seeds, but does not recover both "
                     "hidden bodies consistently. Seismic guidance raises the aggregate hidden-body recall while "
                     "overproducing label9 and merging the five truth bodies into one connected system. Oracle guidance "
                     "recovers the separated bodies, so the result diagnoses an indirect-evidence/topology limitation "
                     "rather than an absolute inability of the frozen generator to represent the target.\n")


if __name__ == "__main__":
    main()
