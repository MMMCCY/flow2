#!/usr/bin/env python3
"""Retrospective case-first Stage20 evaluator and frozen scientific gates."""

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
from guidance.property_evaluation import per_class_hard_metrics, size_stratified_component_metrics, truth_present_mean_iou
from guidance.probability_evaluation import sample_hard_metrics
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv
from scripts.stage20.acoustic_semantics import stage20_labels_to_acoustic
from scripts.stage20.common import CONFIG_DIR, ROOT, load_codebook, require_config, resolve_project_path, validate_asset
from scripts.stage20.run_inference import ARMS

DEFAULT_CONFIG = CONFIG_DIR / "inference_v1.json"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_EVIDENCE_CONFIG = CONFIG_DIR / "evidence_v1.json"
DEFAULT_EVALUATION_REGISTRY = ROOT / "evidence_fix2/evidence_registry.json"
DEFAULT_RUN = ROOT / "formal/inference_v1"
DEFAULT_TRAINING = ROOT / "checkpoints/formal_v1"
DEFAULT_GATE = ROOT / "evidence_audit_fix2/summary.json"
DEFAULT_OUTPUT = ROOT / "reports/formal_v1"

METRICS = ("target_iou", "target_precision", "target_recall", "predicted_target_volume", "target_absolute_volume_error_fraction", "target_centroid_distance", "global_voxel_accuracy", "truth_present_mean_iou", "target_connected_components", "largest_component_fraction", "target_top4_component_mass_fraction", "target_top8_component_mass_fraction", "hard_seismic_mse", "hard_seismic_rmse", "hard_seismic_mae")


@torch.no_grad()
def multiclass_seismic_metrics(decoded, support, observed, property_table, operator, device):
    acoustic, _ = stage20_labels_to_acoustic(normalize_volume(decoded, "decoded").long().to(device), support.to(device), property_table.to(device))
    predicted = operator(acoustic[:, 0:1], acoustic[:, 1:2], support.to(device))
    residual = predicted - observed.to(device=device, dtype=predicted.dtype)
    mse = float(residual.square().mean().cpu())
    return {"hard_seismic_mse": mse, "hard_seismic_rmse": math.sqrt(mse), "hard_seismic_mae": float(residual.abs().mean().cpu())}


def nearest_logz_labels(logz, property_table, support, values, condition):
    rock = property_table[0, 1:].log().to(logz)
    labels = (logz - rock.view(1, -1, 1, 1, 1)).abs().argmin(dim=1, keepdim=True).long()
    labels = torch.where(support, labels, torch.full_like(labels, -1))
    return torch.where(condition, values.long(), labels)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--evidence-config", type=Path, default=DEFAULT_EVIDENCE_CONFIG)
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--evaluation-registry", type=Path, default=DEFAULT_EVALUATION_REGISTRY)
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
    cfg = require_config(args.config, "stage20_inference_v1")
    training_cfg = require_config(args.training_config, "stage20_continuous_impedance_adapter_training_v1")
    evidence_cfg = require_config(args.evidence_config, "stage20_continuous_impedance_evidence_v1")
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    training_manifest = read_json(args.training_dir / "training_manifest.json")
    evidence_gate = read_json(args.evidence_gate)
    records = read_csv(args.run_dir / "sample_manifest.csv")
    if run_manifest.get("run_status") != "completed" or run_manifest.get("smoke_subset") is not False or len(records) != 144:
        raise ValueError("Stage20 formal inference is incomplete")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage19 inference truth firewall failed")
    registry = read_json(args.evaluation_registry)
    if registry.get("schema") != "stage20_evidence_registry_v1" or registry.get("run_status") != "completed":
        raise ValueError("invalid full evaluation registry")
    cases = [case for case in registry["cases"] if case["split"] == "test"]
    case_index = {str(case["case_id"]): case for case in cases}
    if len(cases) != 12:
        raise ValueError("Stage20 evaluator requires 12 TEST cases")
    acoustic_path = resolve_project_path(evidence_cfg["phase4c"]["acoustic_config"])
    seismic_path = resolve_project_path(evidence_cfg["phase4c"]["seismic_config"])
    property_table, _ = load_codebook(acoustic_path)
    seismic_cfg = read_json(seismic_path)
    full_registry = read_json(ROOT / "observations_fix2/observation_registry_full.json")
    full_index = {str(case["case_id"]): case for case in full_registry["cases"]}
    device = torch.device(args.device)
    rows, class_rows = [], []
    for record in records:
        case_id, seed, arm = record["case_id"], int(record["source_seed"]), record["arm"]
        case = case_index[case_id]; obs = case["observation_assets"]
        decoded = normalize_volume(runtime.load_tensor(args.run_dir / record["decoded_path"]), "decoded").long()
        if tensor_sha256(decoded) != record["decoded_geology_sha256"]:
            raise ValueError(f"decoded hash mismatch: {case_id}/{seed}/{arm}")
        truth = normalize_volume(runtime.load_tensor(validate_asset(full_index[case_id]["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long()
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        observed = runtime.load_tensor(validate_asset(obs["observed_seismic"], f"{case_id}/observed")).float()
        target, _ = build_target_mask(truth, target_label=int(cfg["target_label"]), component_mode="all")
        metrics = sample_hard_metrics(decoded, truth, target, dilate_mask(target, 6), condition, int(cfg["target_label"]), 0)
        metrics["truth_present_mean_iou"] = truth_present_mean_iou(decoded, truth)
        metrics.update(size_stratified_component_metrics(decoded == int(cfg["target_label"])))
        pred_target, true_target = decoded == int(cfg["target_label"]), truth == int(cfg["target_label"])
        operator, _ = seismic_operator_from_config(seismic_cfg, grid_shape=support.shape[2:])
        metrics.update(multiclass_seismic_metrics(decoded, support, observed, property_table, operator, device))
        metrics.update({"case_id": case_id, "source_seed": seed, "arm": arm, "target_tp_voxels": int((pred_target & true_target).sum()), "target_fp_voxels": int((pred_target & ~true_target).sum()), "target_fn_voxels": int((~pred_target & true_target).sum())})
        rows.append(metrics)
        class_rows.extend({"case_id": case_id, "source_seed": seed, "arm": arm, **item} for item in per_class_hard_metrics(decoded, truth, 0))
        print(f"Stage20 evaluation {case_id} seed={seed} arm={arm}", flush=True)
    # Deterministic property-only nearest-logZ diagnostic, once per TEST case.
    property_rows = []
    for case in cases:
        case_id = str(case["case_id"]); obs = case["observation_assets"]
        truth_case = full_index[case_id]
        truth = normalize_volume(runtime.load_tensor(validate_asset(truth_case["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long()
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        values = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/values")), "values").long()
        observed = runtime.load_tensor(validate_asset(obs["observed_seismic"], f"{case_id}/observed")).float()
        logz = runtime.load_tensor(validate_asset(case["assets"]["inverted_logz"], f"{case_id}/inverted logZ")).float()
        decoded = nearest_logz_labels(logz, property_table, support, values, condition)
        target, _ = build_target_mask(truth, target_label=int(cfg["target_label"]), component_mode="all")
        metrics = sample_hard_metrics(decoded, truth, target, dilate_mask(target, 6), condition, int(cfg["target_label"]), 0)
        metrics["truth_present_mean_iou"] = truth_present_mean_iou(decoded, truth); metrics.update(size_stratified_component_metrics(decoded == int(cfg["target_label"])))
        operator, _ = seismic_operator_from_config(seismic_cfg, grid_shape=support.shape[2:]); metrics.update(multiclass_seismic_metrics(decoded, support, observed, property_table, operator, device))
        property_rows.append({"case_id": case_id, "source_seed": -1, "arm": "PROPERTY_ONLY_NEAREST_LOGZ", **metrics})
        class_rows.extend({"case_id": case_id, "source_seed": -1, "arm": "PROPERTY_ONLY_NEAREST_LOGZ", **item} for item in per_class_hard_metrics(decoded, truth, 0))
    rows.extend(property_rows)
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
    correct_prior_count = count_better("target_iou", "ADAPTER_CORRECT", "ADAPTER_PRIOR_ONLY")
    correct_prior_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["target_iou"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_PRIOR_ONLY")["target_iou"] for case in cases)
    correct_wrong_count = count_better("target_iou", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE")
    correct_wrong_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["target_iou"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_WRONG_CASE")["target_iou"] for case in cases)
    physics_count = count_better("hard_seismic_rmse", "ADAPTER_CORRECT", "FLOW_ONLY", lower=True)
    physics_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["hard_seismic_rmse"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "FLOW_ONLY")["hard_seismic_rmse"] for case in cases)
    global_delta = med(next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "ADAPTER_CORRECT")["truth_present_mean_iou"] - next(row for row in summaries if row["case_id"] == str(case["case_id"]) and row["arm"] == "FLOW_ONLY")["truth_present_mean_iou"] for case in cases)
    gates = {
        "engineering": bool(training_manifest.get("base_model_unchanged") and training_manifest.get("base_gradients_absent") and training_manifest.get("adapter_parameter_count") == 54003 and run_manifest.get("all_condition_violations_zero") and run_manifest.get("output_count") == 144),
        "evidence": evidence_gate.get("adapter_training_authorized") is True,
        "A_learned_continuous_evidence_coupling": learned_count >= 9 and learned_delta > 0.05,
        "B_seismic_increment_beyond_boreholes": correct_prior_count >= 8 and correct_prior_delta > 0.02,
        "C_case_specificity": correct_wrong_count >= 8 and correct_wrong_delta > 0.02,
        "hard_physics": physics_count >= 8 and physics_delta < 0,
        "volume": cross("ADAPTER_CORRECT", "target_absolute_volume_error_fraction") < cross("FLOW_ONLY", "target_absolute_volume_error_fraction"),
        "global_geology_preservation": global_delta >= -0.01,
        "hard_conditions": all(int(row["condition_violation_count"]) == 0 for row in rows),
    }
    if not gates["engineering"]:
        decision = "ENGINEERING_FAIL"
    elif not gates["A_learned_continuous_evidence_coupling"]:
        decision = "CONTINUOUS_EVIDENCE_COUPLING_NOT_VALIDATED"
    elif not gates["B_seismic_increment_beyond_boreholes"]:
        decision = "WELL_PRIOR_DOMINATED_NO_SEISMIC_INCREMENT"
    elif not gates["C_case_specificity"]:
        decision = "CONTINUOUS_EVIDENCE_NOT_CASE_SPECIFIC"
    elif all(gates.values()):
        decision = "CONTINUOUS_IMPEDANCE_ADAPTER_VALIDATED"
    elif not gates["hard_physics"]:
        decision = "GEOLOGY_IMPROVES_WITHOUT_HARD_SEISMIC_SUPPORT"
    else:
        decision = "STAGE20_PARTIAL_OR_NEGATIVE"
    cross_rows = [{"arm": arm, **{metric: cross(arm, metric) for metric in METRICS}} for arm in ARMS]
    cross_rows.append({"arm": "PROPERTY_ONLY_NEAREST_LOGZ", **{metric: med(row[metric] for row in property_rows) for metric in METRICS}})
    summary = {"schema": "stage20_evaluation_v1", "run_status": "completed", "primary_statistical_unit": "independent_geology_case", "case_count": 12, "source_seeds_per_case": 3, "formal_output_count": 144, "gates": gates, "gate_diagnostics": {"learned_iou_positive_cases": learned_count, "median_delta_target_iou_correct_minus_flow": learned_delta, "correct_gt_prior_only_cases": correct_prior_count, "median_delta_target_iou_correct_minus_prior_only": correct_prior_delta, "correct_gt_wrong_cases": correct_wrong_count, "median_delta_target_iou_correct_minus_wrong": correct_wrong_delta, "hard_rmse_improved_cases": physics_count, "median_delta_hard_seismic_rmse": physics_delta, "median_delta_truth_present_mean_iou": global_delta}, "machine_decision": decision, "mechanism_boundary": "noiseless_inverse_crime_multiclass_acoustic_upper_bound", "field_generalization_proven": False, "exact_posterior_claimed": False, "stop_after_stage20": True}
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "sample_metrics.csv", rows)
    write_csv(args.output_dir / "per_class_iou.csv", class_rows)
    write_csv(args.output_dir / "case_arm_primary_summary.csv", summaries)
    write_csv(args.output_dir / "cross_case_summary.csv", cross_rows)
    write_json(args.output_dir / "summary.json", summary)
    report = ["# Stage20 continuous impedance adapter report", "", f"Decision: **{decision}**", "", "## Q1. Is deterministic inversion evidence valid?", "", f"VAL evidence decision: `{evidence_gate.get('machine_decision')}`.", "", "## Q2. Does continuous q improve hard categorical geology?", "", f"CORRECT minus FLOW median target-IoU: `{learned_delta:+.6f}` ({learned_count}/12 positive).", "", "## Q3. Is improvement beyond well interpolation?", "", f"CORRECT minus PRIOR_ONLY median target-IoU: `{correct_prior_delta:+.6f}` ({correct_prior_count}/12 positive).", "", "## Q4. Is evidence case-specific?", "", f"CORRECT minus WRONG_CASE median target-IoU: `{correct_wrong_delta:+.6f}` ({correct_wrong_count}/12 positive).", "", "## Q5. Is hard geology supported by acquisition-domain physics?", "", f"CORRECT minus FLOW hard-seismic RMSE: `{physics_delta:+.8f}` ({physics_count}/12 improved).", "", "## Frozen gates", ""] + [f"- `{name}`: `{value}`" for name, value in gates.items()] + ["", "`PROPERTY_ONLY_NEAREST_LOGZ` is reported as a deterministic diagnostic only. This noiseless inverse-crime upper bound does not establish field validation, noise robustness, realistic petrophysics, or exact posterior inference.", ""]
    (args.output_dir / "STAGE20_REPORT.md").write_text("\n".join(report), encoding="utf-8")


if __name__ == "__main__":
    main()
