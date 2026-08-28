#!/usr/bin/env python3
"""Truth-aware Stage17B evaluation with task-separated progression gates."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.probability_evaluation import sample_hard_metrics
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv, resolved_path, validate_asset_record


OPTIMIZER_ARMS = ("DFLOW_LEGACY_ADAM", "DFLOW_PAPER_ALIGNED_LBFGS")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _finite_median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def _gain(metrics: dict[str, object], baseline: dict[str, object], field: str, direction: str) -> float:
    value = float(metrics[field])
    base = float(baseline[field])
    if not math.isfinite(value) or not math.isfinite(base):
        return float("nan")
    return value - base if direction == "increase" else base - value


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "evaluation"
    refuse_nonempty(output_dir)
    manifest = read_json(args.run_dir / "run_manifest.json")
    if manifest.get("run_status") != "completed":
        raise ValueError("Stage17B run is not complete")
    if manifest.get("run_class") != "optimizer_calibration_confirmation":
        raise ValueError("engineering smoke cannot be evaluated as scientific calibration")
    if manifest.get("scientific_evidence_eligible") is not True:
        raise ValueError("run is not eligible as optimizer-calibration evidence")
    config = read_json(args.run_dir / "config.json")
    seeds = [int(seed) for seed in config["source_seeds"]]
    if len(seeds) != 3:
        raise ValueError("formal Stage17B confirmation requires three seeds")
    rows = read_csv(args.run_dir / "arm_manifest.csv")
    expected = len(config["tasks"]) * len(seeds) * len(config["arms"])
    if len(rows) != expected:
        raise ValueError(f"incomplete Stage17B arm set: expected {expected}, found {len(rows)}")

    row_index = {(row["task"], int(row["source_seed"]), row["arm"]): row for row in rows}
    metric_rows: list[dict[str, object]] = []
    metric_index: dict[tuple[str, int, str], dict[str, object]] = {}
    for task, task_record in config["tasks"].items():
        source_config_path = validate_asset_record(task_record["stage16_source_config"])
        source_config = read_json(source_config_path)
        assets = source_config["assets"]
        truth_path = validate_asset_record(assets["truth_model"])
        borehole_path = validate_asset_record(assets["boreholes"])
        target_mask_path = validate_asset_record(assets["target_mask"])
        roi_path = validate_asset_record(assets["target_roi_mask"])
        truth = normalize_volume(runtime.load_tensor(truth_path), "truth_model").long()
        boreholes = normalize_volume(runtime.load_tensor(borehole_path), "boreholes").long()
        condition_mask = (boreholes != -1) | (truth == -1)
        target_mask = normalize_volume(runtime.load_tensor(target_mask_path), "target_mask").bool()
        roi = normalize_volume(runtime.load_tensor(roi_path), "target_roi_mask").bool()
        for sample_id, seed in enumerate(seeds):
            baseline_prediction = None
            for arm in config["arms"]:
                row = row_index.get((task, seed, arm))
                if row is None:
                    raise ValueError(f"missing arm: {task}/{seed}/{arm}")
                path = args.run_dir / row["decoded_path"]
                if runtime.file_sha256(path) == "":
                    raise RuntimeError("unreachable empty hash")
                prediction = normalize_volume(runtime.load_tensor(path), arm).long()
                metrics = sample_hard_metrics(
                    prediction=prediction,
                    truth_model=truth,
                    target_mask=target_mask,
                    roi_mask=roi,
                    condition_mask=condition_mask,
                    target_label=int(config["target_label"]),
                    sample_id=sample_id,
                    baseline_prediction=baseline_prediction if arm != "FLOW_ONLY" else None,
                )
                metrics.update({"task": task, "source_seed": seed, "arm": arm})
                if int(metrics["condition_violation_count"]) != 0:
                    raise RuntimeError(f"hard-condition violation: {task}/{seed}/{arm}")
                metric_rows.append(metrics)
                metric_index[(task, seed, arm)] = metrics
                if arm == "FLOW_ONLY":
                    baseline_prediction = prediction

    gate = config["progression_gate"]
    paired_rows: list[dict[str, object]] = []
    task_summaries: list[dict[str, object]] = []
    summaries: dict[tuple[str, str], dict[str, object]] = {}
    for task in config["tasks"]:
        for arm in OPTIMIZER_ARMS:
            material_count = 0
            iou_recovery: list[float] = []
            recall_recovery: list[float] = []
            volume_recovery: list[float] = []
            centroid_recovery: list[float] = []
            mechanism_all = True
            endpoint_solves = 0
            runtime_seconds = 0.0
            for seed in seeds:
                baseline = metric_index[(task, seed, "FLOW_ONLY")]
                reference = metric_index[(task, seed, "REFERENCE_TRAJECTORY_GUIDANCE")]
                candidate = metric_index[(task, seed, arm)]
                iou_gain = _gain(candidate, baseline, "target_iou", "increase")
                recall_gain = _gain(candidate, baseline, "target_recall", "increase")
                volume_gain = _gain(candidate, baseline, "target_absolute_volume_error_fraction", "decrease")
                centroid_gain = _gain(candidate, baseline, "target_centroid_distance", "decrease")
                material = iou_gain >= float(gate["target_iou_improvement"]) and any(
                    (
                        recall_gain >= float(gate["secondary_any"]["target_recall_improvement"]),
                        centroid_gain >= float(gate["secondary_any"]["target_centroid_distance_reduction_voxels"]),
                        volume_gain >= float(gate["secondary_any"]["target_absolute_volume_error_fraction_reduction"]),
                    )
                )
                material_count += int(material)

                def recovery(field: str, direction: str, threshold: float) -> float:
                    ref_gain = _gain(reference, baseline, field, direction)
                    candidate_gain = _gain(candidate, baseline, field, direction)
                    if not math.isfinite(ref_gain) or ref_gain < threshold:
                        return float("nan")
                    return candidate_gain / ref_gain

                iou_rec = recovery("target_iou", "increase", float(gate["non_micro_reference_gain"]["target_iou"]))
                recall_rec = recovery("target_recall", "increase", float(gate["non_micro_reference_gain"]["target_recall"]))
                volume_rec = recovery("target_absolute_volume_error_fraction", "decrease", float(gate["non_micro_reference_gain"]["target_absolute_volume_error_fraction_reduction"]))
                centroid_rec = recovery("target_centroid_distance", "decrease", float(gate["non_micro_reference_gain"]["target_centroid_distance_reduction_voxels"]))
                iou_recovery.append(iou_rec)
                recall_recovery.append(recall_rec)
                volume_recovery.append(volume_rec)
                centroid_recovery.append(centroid_rec)
                candidate_row = row_index[(task, seed, arm)]
                baseline_row = row_index[(task, seed, "FLOW_ONLY")]
                objective_decreased = float(candidate_row["endpoint_primary_loss"]) < float(baseline_row["endpoint_primary_loss"])
                source_changed = float(candidate_row["source_update_norm"]) > 0.0
                mechanism_all = mechanism_all and objective_decreased and source_changed
                endpoint_solves += int(candidate_row["endpoint_solves"])
                runtime_seconds += float(candidate_row["runtime_seconds"])
                paired_rows.append(
                    {
                        "task": task,
                        "source_seed": seed,
                        "optimizer_arm": arm,
                        "delta_target_iou": iou_gain,
                        "delta_target_recall": recall_gain,
                        "target_centroid_distance_reduction": centroid_gain,
                        "target_absolute_volume_error_fraction_reduction": volume_gain,
                        "material_hard_control": material,
                        "target_iou_recovery_fraction": iou_rec,
                        "target_recall_recovery_fraction": recall_rec,
                        "target_centroid_recovery_fraction": centroid_rec,
                        "target_volume_error_recovery_fraction": volume_rec,
                        "objective_decreased": objective_decreased,
                        "source_changed": source_changed,
                    }
                )
            median_iou_recovery = _finite_median(iou_recovery)
            passed = (
                material_count >= int(gate["material_seed_count_required_per_task"])
                and math.isfinite(median_iou_recovery)
                and median_iou_recovery >= float(gate["minimum_median_target_iou_recovery_fraction_per_task"])
                and mechanism_all
                and manifest.get("all_model_state_dicts_unchanged") is True
                and manifest.get("all_condition_violations_zero") is True
            )
            summary = {
                "task": task,
                "optimizer_arm": arm,
                "material_seed_count": material_count,
                "source_seed_count": len(seeds),
                "median_target_iou_recovery_fraction": median_iou_recovery,
                "median_target_recall_recovery_fraction": _finite_median(recall_recovery),
                "median_target_centroid_recovery_fraction": _finite_median(centroid_recovery),
                "median_target_volume_error_recovery_fraction": _finite_median(volume_recovery),
                "mechanism_active_all_seeds": mechanism_all,
                "endpoint_solves": endpoint_solves,
                "runtime_seconds": runtime_seconds,
                "task_progression_gate_passed": passed,
            }
            summaries[(task, arm)] = summary
            task_summaries.append(summary)

    eligible: list[tuple[str, float, int, float]] = []
    optimizer_summaries: list[dict[str, object]] = []
    for arm in OPTIMIZER_ARMS:
        task_rows = [summaries[(task, arm)] for task in config["tasks"]]
        both_pass = all(bool(row["task_progression_gate_passed"]) for row in task_rows)
        min_recovery = min(float(row["median_target_iou_recovery_fraction"]) for row in task_rows)
        solves = sum(int(row["endpoint_solves"]) for row in task_rows)
        runtime_seconds = sum(float(row["runtime_seconds"]) for row in task_rows)
        optimizer_summaries.append(
            {
                "optimizer_arm": arm,
                "oracle_gate_passed": bool(summaries[("oracle_probability", arm)]["task_progression_gate_passed"]),
                "property_gate_passed": bool(summaries[("property", arm)]["task_progression_gate_passed"]),
                "both_task_gates_passed": both_pass,
                "minimum_task_median_iou_recovery_fraction": min_recovery,
                "endpoint_solves": solves,
                "runtime_seconds": runtime_seconds,
            }
        )
        if both_pass:
            eligible.append((arm, min_recovery, solves, runtime_seconds))
    eligible.sort(key=lambda item: (-item[1], item[2], item[3], item[0]))
    selected = eligible[0][0] if eligible else None
    decision = "DFLOW_HARD_CONTROL_MATERIAL" if selected else "DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL"
    summary = {
        "schema": "stage17_optimizer_calibration_evaluation_v1",
        "decision": decision,
        "selected_optimizer_arm": selected,
        "task_separated_gate": True,
        "pooled_compensation_permitted": False,
        "stage17c_dflow_authorized": selected is not None,
        "task_summaries": task_summaries,
        "optimizer_summaries": optimizer_summaries,
    }
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "sample_metrics.csv", metric_rows)
    write_csv(output_dir / "paired_dflow_deltas.csv", paired_rows)
    write_csv(output_dir / "task_progression_gates.csv", task_summaries)
    write_csv(output_dir / "optimizer_summary.csv", optimizer_summaries)
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Stage17B optimizer calibration confirmation",
        "",
        f"Decision: **{decision}**",
        "",
        f"Selected optimizer: `{selected}`" if selected else "Selected optimizer: none",
        "",
        "Oracle and property gates were evaluated independently; pooled compensation was not used.",
        "",
        "| Task | Optimizer | Material seeds | Median IoU recovery | Passed |",
        "|---|---|---:|---:|---|",
    ]
    for row in task_summaries:
        report.append(
            f"| {row['task']} | {row['optimizer_arm']} | {row['material_seed_count']}/3 | "
            f"{float(row['median_target_iou_recovery_fraction']):.6g} | {row['task_progression_gate_passed']} |"
        )
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
