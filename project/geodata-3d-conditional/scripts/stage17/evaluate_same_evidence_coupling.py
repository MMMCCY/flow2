#!/usr/bin/env python3
"""Case-primary retrospective evaluation for completed Stage17C runs."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.probability_evaluation import paired_metric_deltas, sample_hard_metrics
from guidance.probability_volume import build_target_mask, dilate_mask, tensor_sha256
from guidance.property_evaluation import per_class_hard_metrics, truth_component_recovery_rows
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv, validate_asset_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _median(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    return float(statistics.median(finite)) if finite else float("nan")


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "evaluation"
    refuse_nonempty(output_dir)
    manifest = read_json(args.run_dir / "run_manifest.json")
    if manifest.get("run_status") != "completed" or manifest.get("smoke_subset") is not False:
        raise ValueError("Stage17C scientific evaluation requires a complete formal run")
    if manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage17C truth firewall failed")
    config = read_json(args.run_dir / "config.json")
    if config.get("primary_statistical_unit") != "independent_geology_case":
        raise ValueError("geology case must be the primary statistical unit")
    seeds = [int(value) for value in config["source_seeds"]]
    if seeds != [42, 142, 242]:
        raise ValueError("formal Stage17C source seeds changed")
    records = read_csv(args.run_dir / "sample_manifest.csv")
    if len(records) != 5 * 3 * 2:
        raise ValueError("formal Stage17C requires 30 arm outputs")
    index = {(row["case_id"], int(row["source_seed"]), row["arm"]): row for row in records}

    metrics_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    pair_rows: list[dict[str, object]] = []
    for case in config["cases"]:
        case_id = str(case["case_id"])
        observation_manifest_path = validate_asset_record(
            case["observation_assets"]["observation_manifest"], f"{case_id}/observation_manifest"
        )
        observation_manifest = read_json(observation_manifest_path)
        truth_path = validate_asset_record(
            observation_manifest["input_assets"]["truth_model"], f"{case_id}/truth"
        )
        condition_mask_path = validate_asset_record(
            case["observation_assets"]["condition_mask"], f"{case_id}/condition_mask"
        )
        truth = normalize_volume(runtime.load_tensor(truth_path), "truth").long()
        condition_mask = normalize_volume(runtime.load_tensor(condition_mask_path), "condition_mask").bool()
        target_mask, _ = build_target_mask(truth, target_label=int(config["target_label"]), component_mode="all")
        roi = dilate_mask(target_mask, 6)
        for sample_id, seed in enumerate(seeds):
            arm_metrics: dict[str, dict[str, object]] = {}
            baseline_prediction = None
            for arm in config["authorized_arms"]:
                record = index[(case_id, seed, arm)]
                prediction = normalize_volume(
                    runtime.load_tensor(args.run_dir / record["decoded_path"]), arm
                ).long()
                if tensor_sha256(prediction) != record["decoded_geology_sha256"]:
                    raise ValueError(f"decoded output changed: {case_id}/{seed}/{arm}")
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
                pred_target = prediction == int(config["target_label"])
                truth_target = truth == int(config["target_label"])
                metrics.update(
                    {
                        "case_id": case_id,
                        "source_seed": seed,
                        "arm": arm,
                        "target_tp_voxels": int((pred_target & truth_target).sum()),
                        "target_fp_voxels": int((pred_target & ~truth_target).sum()),
                        "target_fn_voxels": int((~pred_target & truth_target).sum()),
                        "evidence_tensor_sha256": record["evidence_tensor_sha256"],
                        "initial_noise_sha256": record["initial_noise_sha256"],
                    }
                )
                metrics_rows.append(metrics)
                arm_metrics[arm] = metrics
                class_rows.extend(
                    {"case_id": case_id, "source_seed": seed, "arm": arm, **row}
                    for row in per_class_hard_metrics(prediction, truth, sample_id)
                )
                component_rows.extend(
                    {"case_id": case_id, "source_seed": seed, "arm": arm, **row}
                    for row in truth_component_recovery_rows(
                        prediction, truth, int(config["target_label"]), sample_id
                    )
                )
                if arm == "FLOW_ONLY":
                    baseline_prediction = prediction
            base = arm_metrics["FLOW_ONLY"]
            guided = arm_metrics["TRAJECTORY_EVIDENCE"]
            delta = paired_metric_deltas(base, guided)
            pair_rows.append(
                {
                    "case_id": case_id,
                    "source_seed": seed,
                    **delta,
                    "delta_target_tp_voxels": int(guided["target_tp_voxels"]) - int(base["target_tp_voxels"]),
                    "delta_target_fp_voxels": int(guided["target_fp_voxels"]) - int(base["target_fp_voxels"]),
                    "delta_target_fn_voxels": int(guided["target_fn_voxels"]) - int(base["target_fn_voxels"]),
                }
            )

    case_rows: list[dict[str, object]] = []
    for case in config["cases"]:
        case_id = str(case["case_id"])
        selected = [row for row in pair_rows if row["case_id"] == case_id]
        case_rows.append(
            {
                "case_id": case_id,
                "stochastic_replicate_count": len(selected),
                "median_delta_target_iou": _median([float(row["delta_target_iou"]) for row in selected]),
                "positive_seed_fraction_target_iou": sum(float(row["delta_target_iou"]) > 0 for row in selected) / len(selected),
                "median_delta_target_precision": _median([float(row["delta_target_precision"]) for row in selected]),
                "median_delta_target_recall": _median([float(row["delta_target_recall"]) for row in selected]),
                "median_delta_absolute_volume_error_fraction": _median([float(row["delta_target_absolute_volume_error_fraction"]) for row in selected]),
                "median_delta_target_centroid_distance": _median([float(row["delta_target_centroid_distance"]) for row in selected]),
                "median_delta_global_mean_iou": _median([float(row["delta_global_mean_iou"]) for row in selected]),
            }
        )
    positive_cases = sum(float(row["median_delta_target_iou"]) > 0 for row in case_rows)
    group_median = _median([float(row["median_delta_target_iou"]) for row in case_rows])
    rule = config["decision_rule"]
    positive = positive_cases >= int(rule["positive_case_count_required"]) and group_median > float(
        rule["group_median_delta_target_iou_required_gt"]
    )
    decision = "TRAJECTORY_COUPLING_POSITIVE" if positive else "TRAJECTORY_COUPLING_NOT_CONSISTENT"
    summary = {
        "schema": "stage17_same_evidence_coupling_evaluation_v1",
        "decision": decision,
        "primary_statistical_unit": "independent_geology_case",
        "case_count": len(case_rows),
        "source_seeds_per_case": 3,
        "positive_case_count_target_iou": positive_cases,
        "group_median_case_median_delta_target_iou": group_median,
        "pooled_pair_count": len(pair_rows),
        "pooled_15_pair_role": "auxiliary_only",
        "pooled_median_delta_target_iou_auxiliary": _median([float(row["delta_target_iou"]) for row in pair_rows]),
        "dflow_geophysical_arm_executed": False,
        "dflow_stop_reason": config["dflow_stop_reason"],
        "truth_role": "retrospective_evaluation_only_after_complete_run",
    }
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "sample_metrics.csv", metrics_rows)
    write_csv(output_dir / "per_class_metrics.csv", class_rows)
    write_csv(output_dir / "truth_component_recovery.csv", component_rows)
    write_csv(output_dir / "paired_deltas_all15_auxiliary.csv", pair_rows)
    write_csv(output_dir / "case_primary_summary.csv", case_rows)
    write_json(output_dir / "summary.json", summary)
    report = [
        "# Stage17C same-evidence coupling report",
        "",
        f"Decision: **{decision}**",
        "",
        f"Primary case-level result: {positive_cases}/5 cases have positive median paired target-IoU delta; group median is {group_median:.9g}.",
        "",
        "The three source seeds are within-case stochastic replicates. The pooled 15-pair table is auxiliary only.",
        "",
        "The D-Flow geophysical arm was not run because the independent Stage17B property gate failed.",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
