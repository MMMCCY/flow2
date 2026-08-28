#!/usr/bin/env python3
"""Retrospective Stage17A evidence evaluation with AP-skill specificity."""

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
from guidance.binary_trace_boundary import vertical_boundary_strength
from guidance.seismic import tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import ap_skill, average_precision, median, require_frozen_config, validate_asset_record


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    return parser.parse_args()


def _score_stats(score: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    positive = score[target & mask]
    negative = score[~target & mask]
    if positive.numel() == 0 or negative.numel() == 0:
        raise ValueError("target/background score partitions must both be non-empty")
    return {
        "truth_mean_score": float(positive.mean()),
        "truth_median_score": float(positive.median()),
        "background_mean_score": float(negative.mean()),
        "background_median_score": float(negative.median()),
    }


def _binary_counts(prediction: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> dict[str, object]:
    p, t = prediction & mask, target & mask
    tp = int((p & t).sum())
    fp = int((p & ~t).sum())
    fn = int((~p & t).sum())
    return {
        "fixed_0p5_tp": tp,
        "fixed_0p5_fp": fp,
        "fixed_0p5_fn": fn,
        "fixed_0p5_precision": tp / (tp + fp) if tp + fp else 0.0,
        "fixed_0p5_recall": tp / (tp + fn) if tp + fn else 0.0,
        "fixed_0p5_iou": tp / (tp + fp + fn) if tp + fp + fn else 0.0,
    }


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.run_dir / "evaluation"
    refuse_nonempty(output_dir)
    run_manifest = read_json(args.run_dir / "run_manifest.json")
    if run_manifest.get("run_status") != "completed":
        raise ValueError("evidence run is incomplete")
    if run_manifest.get("smoke_subset") is not False:
        raise ValueError("engineering smoke cannot produce Stage17A scientific conclusions")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("truth firewall failed")
    config = read_json(args.run_dir / "config.json") if (args.run_dir / "config.json").exists() else read_json(Path(run_manifest["config"]["path"]))
    require_frozen_config(config, "stage17_evidence_multicase_v1")
    cases = config["cases"]
    if run_manifest.get("executed_case_order") != [case["case_id"] for case in cases]:
        raise ValueError("formal evidence run does not contain the complete case cohort")

    scores: dict[str, torch.Tensor] = {}
    truths: dict[str, torch.Tensor] = {}
    supports: dict[str, torch.Tensor] = {}
    condition_masks: dict[str, torch.Tensor] = {}
    rows: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        case_manifest = read_json(args.run_dir / case_id / "manifest.json")
        if case_manifest.get("run_status") != "completed" or case_manifest.get("truth_loaded_by_runner") is not False:
            raise ValueError(f"invalid evidence case manifest: {case_id}")
        score = normalize_volume(runtime.load_tensor(args.run_dir / case_id / "binary_impedance_score.pt"), "score", torch.float32)
        boundary = normalize_volume(runtime.load_tensor(args.run_dir / case_id / "vertical_boundary_strength.pt"), "boundary", torch.float32)
        if tensor_sha256(score) != case_manifest["output_tensor_sha256"]["binary_impedance_score.pt"]:
            raise ValueError(f"evidence tensor changed: {case_id}")
        assets = {name: validate_asset_record(record, f"{case_id}/{name}") for name, record in case["assets"].items()}
        observation_manifest = read_json(assets["observation_manifest"])
        truth_path = validate_asset_record(
            observation_manifest["input_assets"]["truth_model"],
            f"{case_id}/retrospective_truth",
        )
        truth = normalize_volume(runtime.load_tensor(truth_path), "truth").long()
        seismic_support = normalize_volume(runtime.load_tensor(assets["subsurface_mask"]), "support").bool()
        support = truth != -1
        condition_mask = normalize_volume(runtime.load_tensor(assets["condition_mask"]), "condition_mask").bool()
        target = (truth == int(config["target_label"])) & support
        evaluation_mask = support & ~condition_mask
        prevalence = float(target[evaluation_mask].float().mean())
        ap = average_precision(score[evaluation_mask], target[evaluation_mask])
        truth_edge = vertical_boundary_strength(target.float()).bool()
        boundary_ap = average_precision(boundary[evaluation_mask], truth_edge[evaluation_mask])
        xy_mask = evaluation_mask.any(dim=-1)
        footprint_ap = average_precision(
            score.max(dim=-1).values[xy_mask], target.any(dim=-1)[xy_mask]
        )
        row = {
            "case_id": case_id,
            "root_seed": int(case["root_seed"]),
            "voxel_auprc": ap,
            "prevalence": prevalence,
            "auprc_minus_prevalence": ap - prevalence,
            "ap_skill": ap_skill(ap, prevalence),
            "boundary_auprc": boundary_ap,
            "xy_footprint_auprc": footprint_ap,
            "evaluation_voxels": int(evaluation_mask.sum()),
            "seismic_support_voxels": int(seismic_support.sum()),
            "filled_enclosed_air_voxels": int((seismic_support & ~support).sum()),
            "hidden_truth_target_voxels": int((target & evaluation_mask).sum()),
            **_score_stats(score, target, evaluation_mask),
            **_binary_counts(score >= 0.5, target, evaluation_mask),
        }
        rows.append(row)
        scores[case_id], truths[case_id] = score, target
        supports[case_id], condition_masks[case_id] = support, condition_mask

    common_mask = torch.ones_like(next(iter(supports.values())), dtype=torch.bool)
    for case_id in scores:
        common_mask &= supports[case_id] & ~condition_masks[case_id]
    if not bool(common_mask.any()):
        raise ValueError("common specificity mask is empty")

    raw_matrix_rows: list[dict[str, object]] = []
    skill_matrix_rows: list[dict[str, object]] = []
    diagnostic_rows: list[dict[str, object]] = []
    deltas: list[dict[str, object]] = []
    case_ids = [str(case["case_id"]) for case in cases]
    common_prevalence = {
        truth_id: float(truths[truth_id][common_mask].float().mean()) for truth_id in case_ids
    }
    for evidence_id in case_ids:
        raw_row: dict[str, object] = {"evidence_case_id": evidence_id}
        skill_row: dict[str, object] = {"evidence_case_id": evidence_id}
        off_diagonal: list[float] = []
        for truth_id in case_ids:
            ap = average_precision(scores[evidence_id][common_mask], truths[truth_id][common_mask])
            skill = ap_skill(ap, common_prevalence[truth_id])
            raw_row[truth_id] = ap
            skill_row[truth_id] = skill
            if truth_id != evidence_id:
                off_diagonal.append(skill)
            truth_mask = supports[truth_id] & ~condition_masks[truth_id]
            diagnostic_ap = average_precision(scores[evidence_id][truth_mask], truths[truth_id][truth_mask])
            diagnostic_prevalence = float(truths[truth_id][truth_mask].float().mean())
            diagnostic_rows.append({
                "evidence_case_id": evidence_id,
                "truth_case_id": truth_id,
                "raw_auprc": diagnostic_ap,
                "prevalence": diagnostic_prevalence,
                "ap_skill": ap_skill(diagnostic_ap, diagnostic_prevalence),
                "comparison_domain": "truth_case_unconditioned_subsurface_auxiliary",
            })
        diagonal = float(skill_row[evidence_id])
        off_median = median(off_diagonal)
        deltas.append({
            "case_id": evidence_id,
            "diagonal_ap_skill": diagonal,
            "off_diagonal_median_ap_skill": off_median,
            "delta_ap_skill": diagonal - off_median,
        })
        raw_matrix_rows.append(raw_row)
        skill_matrix_rows.append(skill_row)

    positive = sum(float(row["delta_ap_skill"]) > 0 for row in deltas)
    group_median = median([float(row["delta_ap_skill"]) for row in deltas])
    informative = sum(float(row["auprc_minus_prevalence"]) > 0 for row in rows)
    if positive >= 4 and group_median > 0:
        decision = "EVIDENCE_CASE_SPECIFIC"
    elif informative >= 3:
        decision = "EVIDENCE_INFORMATIVE_BUT_NOT_CASE_SPECIFIC"
    else:
        decision = "EVIDENCE_NOT_SUPPORTED"

    summary = {
        "schema": "stage17_multicase_evidence_evaluation_v1",
        "run_status": "completed",
        "run_class": "formal_evidence_confirmation",
        "primary_specificity_metric": "prevalence_corrected_average_precision_skill_v1",
        "ap_skill_definition": "(AUPRC-prevalence)/(1-prevalence)",
        "primary_comparison_mask": "intersection_of_all_case_unconditioned_subsurface_masks",
        "primary_comparison_voxels": int(common_mask.sum()),
        "raw_auprc_matrix_role": "mandatory_auxiliary",
        "positive_delta_case_count": positive,
        "case_count": len(case_ids),
        "group_median_delta_ap_skill": group_median,
        "within_case_auprc_above_prevalence_count": informative,
        "machine_decision": decision,
        "stage17c_trajectory_authorized": decision == "EVIDENCE_CASE_SPECIFIC",
        "truth_role": "retrospective_evaluation_only_after_completed_hash_validation",
    }
    output_dir.mkdir(parents=True)
    write_csv(output_dir / "per_case_evidence_metrics.csv", rows)
    write_csv(output_dir / "raw_auprc_specificity_matrix.csv", raw_matrix_rows)
    write_csv(output_dir / "ap_skill_specificity_matrix.csv", skill_matrix_rows)
    write_csv(output_dir / "specificity_deltas.csv", deltas)
    write_csv(output_dir / "per_truth_domain_specificity_auxiliary.csv", diagnostic_rows)
    torch.save(common_mask, output_dir / "common_specificity_mask.pt")
    write_json(output_dir / "summary.json", summary)
    report_lines = [
        "# Stage17A multi-case evidence report", "",
        f"Decision: **{decision}**", "",
        "Primary specificity uses prevalence-corrected AP skill; raw AUPRC is auxiliary.", "",
        f"Positive diagonal advantages: {positive}/{len(case_ids)}; group median Delta skill: {group_median:.9f}.", "",
        "This report is retrospective. The inversion runner did not load truth or run Flow.", "",
    ]
    (output_dir / "REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    manifest = base_manifest("stage17_multicase_evidence_evaluation_v1", Path(__file__))
    manifest.update({
        "run_status": "completed",
        "source_run_manifest": runtime.asset_record(args.run_dir / "run_manifest.json"),
        "summary": runtime.asset_record(output_dir / "summary.json"),
        "common_specificity_mask_sha256": tensor_sha256(common_mask),
        "truth_loaded_after_run_completion": True,
        "truth_used_by_inversion_runner": False,
    })
    write_json(output_dir / "evaluation_manifest.json", manifest)


if __name__ == "__main__":
    main()
