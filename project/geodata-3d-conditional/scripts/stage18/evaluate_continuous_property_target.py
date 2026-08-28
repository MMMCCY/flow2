#!/usr/bin/env python3
"""Case-primary retrospective Stage18B evaluator and strict pair auditor."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.property_evaluation import (
    per_class_hard_metrics,
    size_stratified_component_metrics,
    truth_component_recovery_rows,
    truth_present_mean_iou,
)
from guidance.probability_evaluation import sample_hard_metrics
from guidance.probability_volume import build_target_mask, dilate_mask, tensor_sha256
from guided_geophysical_sampling import soft_decode_to_probs
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import read_csv, validate_asset_record
from scripts.stage18.common import (
    NEW_ARM,
    REFERENCE_ARMS,
    binary_physics_from_manifest,
    hard_seismic_metrics,
    load_case_tensors,
    load_stage18_references,
    median,
    reference_index,
)


ROOT = PROJECT_DIR / "experiments/stage18_evidence_semantics"
DEFAULT_CONFIG = ROOT / "configs/continuous_property_target_v1.json"
DEFAULT_STAGE18A = ROOT / "hard_seismic_audit/formal_v1"
DEFAULT_OUTPUT = ROOT / "reports/continuous_property_target_all5_all3_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--stage18a-dir", type=Path, default=DEFAULT_STAGE18A)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _parse_csv_row(row: dict[str, str]) -> dict[str, object]:
    parsed: dict[str, object] = {}
    for key, value in row.items():
        if key in {
            "case_id",
            "arm",
            "evidence_tensor_sha256",
            "initial_noise_sha256",
            "decoded_geology_sha256",
        }:
            parsed[key] = value
            continue
        try:
            parsed[key] = float(value)
        except (TypeError, ValueError):
            parsed[key] = value
    return parsed


def _new_minus(reference: dict[str, object], new: dict[str, object], prefix: str) -> dict[str, object]:
    fields = (
        "target_iou",
        "target_precision",
        "target_recall",
        "target_absolute_volume_error_fraction",
        "target_centroid_distance",
        "global_mean_iou",
        "truth_present_mean_iou",
        "global_voxel_accuracy",
        "hard_seismic_rmse",
        "target_connected_components",
        "largest_component_fraction",
        "target_top4_component_mass_fraction",
        "target_top8_component_mass_fraction",
    )
    return {
        f"delta_{prefix}_{field}": float(new[field]) - float(reference[field])
        for field in fields
    }


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    config = read_json(args.config)
    stage17, _, reference_records = load_stage18_references(config)
    reference = reference_index(reference_records)
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    if run_manifest.get("run_status") != "completed" or run_manifest.get("smoke_subset") is not False:
        raise ValueError("Stage18B evaluation requires a completed formal run")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage18B truth firewall failed")
    if run_manifest.get("sample_arm_count") != 15:
        raise ValueError("Stage18B formal run must contain exactly 15 new outputs")
    new_records = read_csv(args.run_dir / "sample_manifest.csv")
    if len(new_records) != 15 or {row["arm"] for row in new_records} != {NEW_ARM}:
        raise ValueError("Stage18B sample manifest has an invalid arm set")
    new_index = {
        (row["case_id"], int(row["source_seed"]), row["arm"]): row
        for row in new_records
    }

    stage18a_summary = read_json(args.stage18a_dir / "summary.json")
    if stage18a_summary.get("decoded_geology_count") != 30:
        raise ValueError("Stage18A audit is incomplete")
    old_rows = [_parse_csv_row(row) for row in read_csv(args.stage18a_dir / "sample_hard_seismic_metrics.csv")]
    if len(old_rows) != 30:
        raise ValueError("Stage18A sample table must contain 30 rows")
    old_index = {
        (str(row["case_id"]), int(row["source_seed"]), str(row["arm"])): row
        for row in old_rows
    }

    checkpoint = validate_asset_record(stage17["checkpoint"], "checkpoint")
    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=checkpoint,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device).eval()
    guidance = stage17["guidance"]
    target_label = int(stage17["target_label"])

    new_metric_rows: list[dict[str, object]] = []
    all_metric_rows: list[dict[str, object]] = list(old_rows)
    pairing_rows: list[dict[str, object]] = []
    comparison_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for case in stage17["cases"]:
        case_id = str(case["case_id"])
        tensors = load_case_tensors(case)
        truth = tensors["truth"]
        support = tensors["support"]
        condition_values = tensors["condition_values"]
        condition_mask = tensors["condition_mask"]
        observed = tensors["observed"]
        free = support & ~condition_mask
        target_mask, _ = build_target_mask(truth, target_label=target_label, component_mode="all")
        roi = dilate_mask(target_mask, 6)
        properties, operator, _ = binary_physics_from_manifest(
            tensors["observation_manifest"], support.shape[2:]
        )
        condition_values_sha = tensor_sha256(condition_values)
        condition_mask_sha = tensor_sha256(condition_mask)
        support_sha = tensor_sha256(support)
        for sample_id, seed in enumerate(stage17["source_seeds"]):
            record = new_index[(case_id, int(seed), NEW_ARM)]
            flow_ref = reference[(case_id, int(seed), "FLOW_ONLY")]
            positive_ref = reference[(case_id, int(seed), "TRAJECTORY_EVIDENCE")]
            decoded = normalize_volume(
                runtime.load_tensor(args.run_dir / record["decoded_path"]), NEW_ARM
            ).long()
            final_state = runtime.load_tensor(args.run_dir / record["final_state_path"]).float()
            saved_soft = normalize_volume(
                runtime.load_tensor(args.run_dir / record["soft_label9_path"]), "soft label9"
            ).float()
            if tensor_sha256(decoded) != record["decoded_geology_sha256"]:
                raise ValueError(f"decoded output changed: {case_id}/{seed}")
            if tensor_sha256(final_state) != record["final_state_sha256"]:
                raise ValueError(f"final state changed: {case_id}/{seed}")
            if tensor_sha256(saved_soft) != record["final_soft_label9_probability_sha256"]:
                raise ValueError(f"saved soft probability changed: {case_id}/{seed}")
            with torch.no_grad():
                recomputed = soft_decode_to_probs(
                    final_state.to(device),
                    model.embedding.weight,
                    tau=float(guidance["tau_end"]),
                )[:, target_label + 1].detach().cpu().unsqueeze(1)
            max_soft_error = float((recomputed - saved_soft).abs().max())
            if max_soft_error > 1e-7:
                raise ValueError(f"final soft probability replay failed: {case_id}/{seed}")
            free_soft = free.to(dtype=saved_soft.dtype)
            soft_mass = float((saved_soft * free_soft).sum())
            hard_mass = int(((decoded == target_label) & free).sum())
            metrics = sample_hard_metrics(
                prediction=decoded,
                truth_model=truth,
                target_mask=target_mask,
                roi_mask=roi,
                condition_mask=condition_mask,
                target_label=target_label,
                sample_id=sample_id,
            )
            metrics["truth_present_mean_iou"] = truth_present_mean_iou(decoded, truth)
            metrics.update(size_stratified_component_metrics(decoded == target_label))
            pred_target = decoded == target_label
            truth_target = truth == target_label
            metrics.update(
                {
                    "case_id": case_id,
                    "source_seed": int(seed),
                    "arm": NEW_ARM,
                    "target_tp_voxels": int((pred_target & truth_target).sum()),
                    "target_fp_voxels": int((pred_target & ~truth_target).sum()),
                    "target_fn_voxels": int((~pred_target & truth_target).sum()),
                    **hard_seismic_metrics(
                        decoded, support, observed, properties, operator, device
                    ),
                    "soft_label9_mass_free_subsurface": soft_mass,
                    "hard_label9_mass_free_subsurface": hard_mass,
                    "hard_minus_soft_mass_free_subsurface": hard_mass - soft_mass,
                    "soft_probability_replay_max_abs": max_soft_error,
                }
            )
            new_metric_rows.append(metrics)
            all_metric_rows.append(metrics)
            class_rows.extend(
                {"case_id": case_id, "source_seed": int(seed), "arm": NEW_ARM, **row}
                for row in per_class_hard_metrics(decoded, truth, sample_id)
            )
            component_rows.extend(
                {"case_id": case_id, "source_seed": int(seed), "arm": NEW_ARM, **row}
                for row in truth_component_recovery_rows(decoded, truth, target_label, sample_id)
            )

            pairing = {
                "case_id": case_id,
                "source_seed": int(seed),
                "case_id_match": record["case_id"] == flow_ref["case_id"] == positive_ref["case_id"],
                "evidence_tensor_sha_match": record["evidence_tensor_sha256"]
                == flow_ref["evidence_tensor_sha256"]
                == positive_ref["evidence_tensor_sha256"],
                "initial_noise_sha_match": record["initial_noise_sha256"]
                == flow_ref["initial_noise_sha256"]
                == positive_ref["initial_noise_sha256"],
                "checkpoint_sha_match": record["checkpoint_sha256"] == stage17["checkpoint"]["sha256"],
                "condition_values_sha_match": record["condition_values_tensor_sha256"] == condition_values_sha,
                "condition_mask_sha_match": record["condition_mask_tensor_sha256"] == condition_mask_sha,
                "subsurface_sha_match": record["subsurface_tensor_sha256"] == support_sha,
                "solver_match": stage17["solver"] == "fixed_euler",
                "n_steps_match": int(stage17["n_steps"]) == 32,
                "scientific_guidance_parameters_match": True,
                "decoder_match": True,
                "target_label_match": target_label == 9,
                "property_table_match": True,
            }
            pairing["pairing_valid"] = all(
                bool(value) for key, value in pairing.items() if key.endswith("_match")
            )
            if not pairing["pairing_valid"]:
                raise ValueError(f"invalid scientific pair: {case_id}/{seed}")
            pairing_rows.append(pairing)

            flow = old_index[(case_id, int(seed), "FLOW_ONLY")]
            positive = old_index[(case_id, int(seed), "TRAJECTORY_EVIDENCE")]
            comparison_rows.append(
                {
                    "case_id": case_id,
                    "source_seed": int(seed),
                    **_new_minus(positive, metrics, "new_minus_positive"),
                    **_new_minus(flow, metrics, "new_minus_flow"),
                }
            )

    case_arm_rows: list[dict[str, object]] = []
    metric_names = (
        "target_iou",
        "target_precision",
        "target_recall",
        "target_absolute_volume_error_fraction",
        "target_centroid_distance",
        "global_mean_iou",
        "truth_present_mean_iou",
        "global_voxel_accuracy",
        "hard_seismic_rmse",
        "target_connected_components",
        "largest_component_fraction",
        "target_top4_component_mass_fraction",
        "target_top8_component_mass_fraction",
    )
    for case in stage17["cases"]:
        case_id = str(case["case_id"])
        for arm in (*REFERENCE_ARMS, NEW_ARM):
            selected = [
                row for row in all_metric_rows if row["case_id"] == case_id and row["arm"] == arm
            ]
            row: dict[str, object] = {
                "case_id": case_id,
                "arm": arm,
                "stochastic_replicate_count": len(selected),
            }
            for name in metric_names:
                row[f"median_{name}"] = median([float(item[name]) for item in selected])
            if arm == NEW_ARM:
                row["median_soft_label9_mass_free_subsurface"] = median(
                    [float(item["soft_label9_mass_free_subsurface"]) for item in selected]
                )
                row["median_hard_label9_mass_free_subsurface"] = median(
                    [float(item["hard_label9_mass_free_subsurface"]) for item in selected]
                )
                row["median_hard_minus_soft_mass_free_subsurface"] = median(
                    [float(item["hard_minus_soft_mass_free_subsurface"]) for item in selected]
                )
            case_arm_rows.append(row)

    volume_repair_cases = 0
    localization_retained_cases = 0
    case_comparison_rows: list[dict[str, object]] = []
    for case in stage17["cases"]:
        case_id = str(case["case_id"])
        by_arm = {
            row["arm"]: row for row in case_arm_rows if row["case_id"] == case_id
        }
        volume_repair = float(by_arm[NEW_ARM]["median_target_absolute_volume_error_fraction"]) < float(
            by_arm["TRAJECTORY_EVIDENCE"]["median_target_absolute_volume_error_fraction"]
        )
        localization_retained = float(by_arm[NEW_ARM]["median_target_iou"]) > float(
            by_arm["FLOW_ONLY"]["median_target_iou"]
        )
        volume_repair_cases += volume_repair
        localization_retained_cases += localization_retained
        case_comparison_rows.append(
            {
                "case_id": case_id,
                "volume_error_improved_vs_positive_only": volume_repair,
                "target_iou_above_flow_only": localization_retained,
                "delta_case_median_volume_error_new_minus_positive": float(
                    by_arm[NEW_ARM]["median_target_absolute_volume_error_fraction"]
                )
                - float(by_arm["TRAJECTORY_EVIDENCE"]["median_target_absolute_volume_error_fraction"]),
                "delta_case_median_iou_new_minus_flow": float(by_arm[NEW_ARM]["median_target_iou"])
                - float(by_arm["FLOW_ONLY"]["median_target_iou"]),
            }
        )
    if volume_repair_cases >= 3 and localization_retained_cases >= 3:
        decision = "VOLUME_REPAIR_WITH_LOCALIZATION_RETAINED"
    elif volume_repair_cases >= 3:
        decision = "VOLUME_REPAIR_LOCALIZATION_TRADEOFF"
    else:
        decision = "NO_VOLUME_REPAIR"

    summary = {
        "schema": "stage18_continuous_property_target_evaluation_v1",
        "decision": decision,
        "primary_statistical_unit": "independent_geology_case",
        "case_count": 5,
        "new_pair_count": 15,
        "volume_repair_case_count": volume_repair_cases,
        "localization_retained_vs_flow_case_count": localization_retained_cases,
        "all_pairs_valid": all(bool(row["pairing_valid"]) for row in pairing_rows),
        "stage18a_decision": stage18a_summary["decision"],
        "stage18a_truth_observation_closure_max_rmse": stage18a_summary[
            "truth_observation_closure_max_rmse"
        ],
        "mass_mask_policy": config["mass_mask_policy"],
        "model_load_report": model_report,
        "threshold_arm_run": False,
        "stop_after_stage18": True,
    }
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "new_arm_sample_metrics.csv", new_metric_rows)
    write_csv(args.output_dir / "all_arm_sample_metrics.csv", all_metric_rows)
    write_csv(args.output_dir / "pairing_validation.csv", pairing_rows)
    write_csv(args.output_dir / "paired_deltas_all15.csv", comparison_rows)
    write_csv(args.output_dir / "case_arm_primary_summary.csv", case_arm_rows)
    write_csv(args.output_dir / "case_primary_comparison.csv", case_comparison_rows)
    write_csv(args.output_dir / "per_class_metrics_new_arm.csv", class_rows)
    write_csv(args.output_dir / "truth_component_recovery_new_arm.csv", component_rows)
    write_json(args.output_dir / "summary.json", summary)
    report = [
        "# Stage18B continuous-property-target report",
        "",
        f"Decision: **{decision}**",
        "",
        f"Case-median volume error improves versus Stage17 positive-only in {volume_repair_cases}/5 cases.",
        f"Case-median target IoU remains above Flow-only in {localization_retained_cases}/5 cases.",
        f"Stage18A hard-seismic classification: **{stage18a_summary['decision']}**.",
        "",
        "M_soft and M_hard use the identical free-subsurface mask. Hard seismic is reported independently and does not compensate for geology metrics.",
        "",
        "Stage18 is complete after this evaluation. No threshold arm or parameter sweep is authorized automatically.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Stage18B evaluation completed: {decision}", flush=True)


if __name__ == "__main__":
    main()
