#!/usr/bin/env python3
"""Retrospective case-first Stage19 evaluator and frozen decision gates."""

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
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs
from guidance.property_evaluation import per_class_hard_metrics, size_stratified_component_metrics, truth_present_mean_iou
from guidance.probability_evaluation import sample_hard_metrics
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv
from scripts.stage18.common import hard_seismic_metrics
from scripts.stage19.common import CONFIG_DIR, ROOT, require_config, resolve_project_path, validate_asset
from scripts.stage19.run_inference import ARMS

DEFAULT_CONFIG = CONFIG_DIR / "inference_v1.json"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_EVIDENCE_CONFIG = CONFIG_DIR / "evidence_v1.json"
DEFAULT_RUN = ROOT / "formal/inference_v1"
DEFAULT_TRAINING = ROOT / "checkpoints/formal_v1"
DEFAULT_GATE = ROOT / "evidence_audit/summary.json"
DEFAULT_OUTPUT = ROOT / "reports/formal_v1"

METRICS = ("target_iou", "target_precision", "target_recall", "target_absolute_volume_error_fraction", "target_centroid_distance", "global_voxel_accuracy", "truth_present_mean_iou", "target_connected_components", "largest_component_fraction", "target_top4_component_mass_fraction", "target_top8_component_mass_fraction", "target_tiny_component_mass_fraction_le_5", "hard_seismic_mse", "hard_seismic_rmse")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--evidence-config", type=Path, default=DEFAULT_EVIDENCE_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--training-dir", type=Path, default=DEFAULT_TRAINING)
    parser.add_argument("--evidence-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def med(values) -> float:
    finite = [float(v) for v in values if math.isfinite(float(v))]
    return float(statistics.median(finite)) if finite else float("nan")


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage19_inference_v1")
    training_cfg = require_config(args.training_config, "stage19_learned_evidence_adapter_training_v1")
    evidence_cfg = require_config(args.evidence_config, "stage19_evidence_v1")
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    training_manifest = read_json(args.training_dir / "training_manifest.json")
    evidence_gate = read_json(args.evidence_gate)
    records = read_csv(args.run_dir / "sample_manifest.csv")
    if run_manifest.get("run_status") != "completed" or run_manifest.get("smoke_subset") is not False or len(records) != 180:
        raise ValueError("Stage19 formal inference is incomplete")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage19 inference truth firewall failed")
    registry = read_json(validate_asset(run_manifest["evidence_registry"], "evidence registry"))
    cases = [case for case in registry["cases"] if case["split"] == "test"]
    case_index = {str(case["case_id"]): case for case in cases}
    if len(cases) != 12:
        raise ValueError("Stage19 evaluator requires 12 TEST cases")
    binary_path = resolve_project_path(evidence_cfg["observation"]["binary_acoustic_config"])
    seismic_path = resolve_project_path(evidence_cfg["observation"]["seismic_config"])
    binary_cfg = read_json(binary_path)
    source_path = validate_asset(binary_cfg["source_acoustic_config"], "source acoustic config")
    properties = binary_acoustic_properties_from_configs(binary_cfg, read_json(source_path))
    seismic_cfg = read_json(seismic_path)
    device = torch.device(args.device)
    rows, class_rows = [], []
    for record in records:
        case_id, seed, arm = record["case_id"], int(record["source_seed"]), record["arm"]
        case = case_index[case_id]; obs = case["observation_assets"]
        decoded = normalize_volume(runtime.load_tensor(args.run_dir / record["decoded_path"]), "decoded").long()
        if tensor_sha256(decoded) != record["decoded_geology_sha256"]:
            raise ValueError(f"decoded hash mismatch: {case_id}/{seed}/{arm}")
        truth = normalize_volume(runtime.load_tensor(validate_asset(case["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long()
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        observed = runtime.load_tensor(validate_asset(obs["observed_seismic"], f"{case_id}/observed")).float()
        target, _ = build_target_mask(truth, target_label=int(cfg["target_label"]), component_mode="all")
        metrics = sample_hard_metrics(decoded, truth, target, dilate_mask(target, 6), condition, int(cfg["target_label"]), 0)
        metrics["truth_present_mean_iou"] = truth_present_mean_iou(decoded, truth)
        metrics.update(size_stratified_component_metrics(decoded == int(cfg["target_label"])))
        pred_target, true_target = decoded == int(cfg["target_label"]), truth == int(cfg["target_label"])
        operator, _ = seismic_operator_from_config(seismic_cfg, grid_shape=support.shape[2:])
        metrics.update(hard_seismic_metrics(decoded, support, observed, properties, operator, device))
        metrics.update({"case_id": case_id, "source_seed": seed, "arm": arm, "target_tp_voxels": int((pred_target & true_target).sum()), "target_fp_voxels": int((pred_target & ~true_target).sum()), "target_fn_voxels": int((~pred_target & true_target).sum())})
        rows.append(metrics)
        class_rows.extend({"case_id": case_id, "source_seed": seed, "arm": arm, **item} for item in per_class_hard_metrics(decoded, truth, 0))
        print(f"Stage19 evaluation {case_id} seed={seed} arm={arm}", flush=True)
    if any(int(row["condition_violation_count"]) for row in rows):
        raise RuntimeError("condition violation in formal outputs")
    summaries = []
    for case in cases:
        case_id = str(case["case_id"])
        for arm in ARMS:
            group = [row for row in rows if row["case_id"] == case_id and row["arm"] == arm]
            if len(group) != 3:
                raise ValueError("case/arm does not contain three seed replicates")
            summaries.append({"case_id": case_id, "arm": arm, **{name: med(row[name] for row in group) for name in METRICS}})
    def cross(arm: str, metric: str) -> float:
        return med(row[metric] for row in summaries if row["arm"] == arm)
    def count_better(metric: str, left: str, right: str, *, lower: bool = False) -> int:
        count = 0
        for case in cases:
            cid = str(case["case_id"])
            l = next(row for row in summaries if row["case_id"] == cid and row["arm"] == left)[metric]
            r = next(row for row in summaries if row["case_id"] == cid and row["arm"] == right)[metric]
            count += l < r if lower else l > r
        return count
    learned_count = count_better("target_iou", "ADAPTER_CORRECT", "FLOW_ONLY")
    learned_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["target_iou"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "FLOW_ONLY")["target_iou"] for case in cases)
    correct_zero_count = count_better("target_iou", "ADAPTER_CORRECT", "ADAPTER_ZERO")
    correct_wrong_count = count_better("target_iou", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE")
    physics_count = count_better("hard_seismic_rmse", "ADAPTER_CORRECT", "FLOW_ONLY", lower=True)
    physics_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["hard_seismic_rmse"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "FLOW_ONLY")["hard_seismic_rmse"] for case in cases)
    global_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["truth_present_mean_iou"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "FLOW_ONLY")["truth_present_mean_iou"] for case in cases)
    gates = {
        "engineering": bool(training_manifest.get("base_model_unchanged") and training_manifest.get("base_gradients_absent") and training_manifest.get("adapter_parameter_count", 10**9) < 100000 and run_manifest.get("all_condition_violations_zero") and run_manifest.get("output_count") == 180),
        "evidence_reuse": evidence_gate.get("adapter_training_authorized") is True,
        "learned_coupling": learned_count >= 9 and learned_delta > 0.05,
        "specificity": cross("ADAPTER_CORRECT", "target_iou") > cross("ADAPTER_ZERO", "target_iou") and cross("ADAPTER_CORRECT", "target_iou") > cross("ADAPTER_WRONG_CASE", "target_iou") and correct_zero_count >= 8 and correct_wrong_count >= 8,
        "hard_physics": physics_count >= 8 and physics_delta < 0,
        "volume": cross("ADAPTER_CORRECT", "target_absolute_volume_error_fraction") < cross("FLOW_ONLY", "target_absolute_volume_error_fraction"),
        "structure": cross("ADAPTER_CORRECT", "target_connected_components") < cross("STAGE18_CONTINUOUS_TARGET", "target_connected_components") and cross("ADAPTER_CORRECT", "target_top8_component_mass_fraction") > cross("STAGE18_CONTINUOUS_TARGET", "target_top8_component_mass_fraction"),
        "global_geology_preservation": global_delta >= -0.01,
    }
    if not gates["engineering"]:
        decision = "ENGINEERING_FAIL"
    elif not gates["evidence_reuse"]:
        decision = "EVIDENCE_REUSE_NOT_VALIDATED"
    elif not gates["specificity"]:
        decision = "ADAPTER_NOT_USING_CASE_SPECIFIC_EVIDENCE"
    elif not gates["learned_coupling"]:
        decision = "ADAPTER_CAPACITY_OR_TRAINING_INSUFFICIENT"
    elif all(gates.values()):
        decision = "LEARNED_EVIDENCE_ADAPTER_VALIDATED"
    elif all(gates[name] for name in ("engineering", "evidence_reuse", "learned_coupling", "specificity", "hard_physics", "volume", "global_geology_preservation")) and not gates["structure"]:
        decision = "LEARNED_COUPLING_POSITIVE_BUT_STRUCTURE_UNRESOLVED"
    else:
        decision = "STAGE19_PARTIAL_OR_NEGATIVE"
    cross_rows = [{"arm": arm, **{metric: cross(arm, metric) for metric in METRICS}} for arm in ARMS]
    summary = {"schema": "stage19_evaluation_v1", "run_status": "completed", "primary_statistical_unit": "independent_geology_case", "case_count": 12, "source_seeds_per_case": 3, "formal_output_count": 180, "gates": gates, "gate_diagnostics": {"learned_iou_positive_cases": learned_count, "median_delta_target_iou_correct_minus_flow": learned_delta, "correct_gt_zero_cases": correct_zero_count, "correct_gt_wrong_cases": correct_wrong_count, "hard_rmse_improved_cases": physics_count, "median_delta_hard_seismic_rmse": physics_delta, "median_delta_truth_present_mean_iou": global_delta}, "machine_decision": decision, "mechanism_boundary": "binary_high_contrast_noiseless_inverse_crime_upper_bound", "field_generalization_proven": False, "historical_pretraining_sample_overlap_excluded": False, "stop_after_stage19": True}
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sample_metrics.csv", rows)
    write_csv(args.output_dir / "per_class_iou.csv", class_rows)
    write_csv(args.output_dir / "case_arm_primary_summary.csv", summaries)
    write_csv(args.output_dir / "cross_case_summary.csv", cross_rows)
    write_json(args.output_dir / "summary.json", summary)
    report = ["# Stage19 learned evidence adapter report", "", f"Decision: **{decision}**", "", "## Frozen gates", ""] + [f"- `{name}`: `{value}`" for name, value in gates.items()] + ["", "This remains a binary high-contrast, noiseless inverse-crime mechanism upper bound. It does not establish realistic multiclass petrophysics, measured-field inversion, calibrated probabilities, or sample-level exclusion from historical streaming pretraining.", ""]
    (args.output_dir / "REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
