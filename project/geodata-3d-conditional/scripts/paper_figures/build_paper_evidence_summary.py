#!/usr/bin/env python3
"""Build the paper evidence layer, compact Table 1, and writing fact sheet.

This script performs no scientific inference.  It only extracts values from
frozen machine summaries and emits deterministic paper-writing artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import subprocess
from typing import Any, Iterable


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
SCRIPT_PATH = Path(__file__).resolve()
FIGURE_DATA_DIR = PROJECT_DIR / "paper/figure_data"
TABLE_DIR = PROJECT_DIR / "paper/tables"
FACT_SHEET_PATH = PROJECT_DIR / "paper/PAPER_RESULT_FACTS.md"

P1_SUMMARY = PROJECT_DIR / "experiments/stage1_probability/reports/phase1b_v4_12pair/summary.json"
P1_REPORT = PROJECT_DIR / "docs/PHASE1_REPORT.md"
P1_PAIRS = P1_SUMMARY.with_name("paired_samples.csv")
P2_SUMMARY = PROJECT_DIR / "experiments/stage2_property/reports/phase2a_v1_12pair/summary.json"
P2_REPORT = PROJECT_DIR / "docs/PHASE2A_REPORT.md"
P2_PAIRS = P2_SUMMARY.with_name("paired_samples.csv")
S6_VERDICT = PROJECT_DIR / "experiments/stage6_inference_causality/diagnostic_verdict.json"
S6_REPORT = PROJECT_DIR / "experiments/stage6_inference_causality/DIAGNOSTIC_SYNTHESIS.md"
S6_D3_SUMMARY = PROJECT_DIR / "experiments/stage6_inference_causality/runs/five_body_cuboid_v1/d3_soft_hard_transfer_v1_provenance_85d5deb_clean/summary.json"
S7_SUMMARY = PROJECT_DIR / "experiments/stage6_inference_causality/reports/stage7_v1_final_v2/stage7_summary.json"
S7_REPORT = S7_SUMMARY.with_name("STAGE7_REPORT.md")
S7_NATIVE_CSV = S7_SUMMARY.with_name("native_cross_evaluation.csv")
S9_SUMMARY = PROJECT_DIR / "experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/summary.json"
S9_REPORT = S9_SUMMARY.with_name("STAGE9A_REPORT.md")
FULLGEO_REPORT = PROJECT_DIR / "experiments/full_structuralgeo_benchmark/FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md"
S12_SUMMARY = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge/evaluation/stage12b_a/summary.json"
S12_REPORT = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge/reports/STAGE12B_REPORT.md"
S14_SUMMARY = PROJECT_DIR / "experiments/stage14_gansim_style_geo_guidance/reports/pilot_v1/summary.json"
S14_REPORT = S14_SUMMARY.with_name("STAGE14_REPORT.md")
S14_DELTAS = S14_SUMMARY.with_name("paired_deltas.csv")

CAPTION_INTEGRITY_NOTE = (
    "Results summarize distinct diagnostic protocols and are not a single shared-case leaderboard."
)


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _relative(path: Path) -> str:
    return str(path.resolve().relative_to(REPOSITORY_ROOT))


def _source_file(path: Path, role: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(path)
    return {
        "path": _relative(path),
        "role": role,
        "sha256": _sha256(path),
        "size_bytes": path.stat().st_size,
    }


def _metric(
    metric_id: str,
    display_name: str,
    value: Any,
    unit: str,
    source_path: Path,
    source_key: str,
    *,
    scientific_role: str = "primary",
    qualification: str = "",
) -> dict[str, Any]:
    return {
        "metric_id": metric_id,
        "display_name": display_name,
        "value": value,
        "unit": unit,
        "scientific_role": scientific_role,
        "qualification": qualification,
        "source": {
            "path": _relative(source_path),
            "json_key": source_key,
            "sha256": _sha256(source_path),
        },
    }


def _mean_pair(metric: dict[str, Any]) -> dict[str, float]:
    return {
        "baseline": float(metric["baseline"]["mean"]),
        "guided": float(metric["guided"]["mean"]),
    }


def _experiment(
    experiment_id: str,
    display_name: str,
    evidence_type: str,
    inference_method: str,
    n_cases: int,
    n_pairs_or_samples: dict[str, int],
    truth_visible_during_inference: bool,
    observation_visible: bool,
    observation_type: str,
    selection_rule: str,
    primary_metrics: list[dict[str, Any]],
    machine_decision: str,
    scientific_interpretation: str,
    source_files: Iterable[tuple[Path, str]],
) -> dict[str, Any]:
    return {
        "experiment_id": experiment_id,
        "display_name": display_name,
        "evidence_type": evidence_type,
        "inference_method": inference_method,
        "n_cases": n_cases,
        "n_pairs_or_samples": n_pairs_or_samples,
        "truth_visible_during_inference": truth_visible_during_inference,
        "observation_visible": observation_visible,
        "observation_type": observation_type,
        "selection_rule": selection_rule,
        "primary_metrics": primary_metrics,
        "machine_decision": machine_decision,
        "scientific_interpretation": scientific_interpretation,
        "source_files": [_source_file(path, role) for path, role in source_files],
    }


def _find_hypothesis(verdict: dict[str, Any], hypothesis_id: str) -> dict[str, Any]:
    matches = [row for row in verdict["hypotheses"] if row["id"] == hypothesis_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one {hypothesis_id} diagnosis")
    return matches[0]


def _build_experiments() -> list[dict[str, Any]]:
    p1 = _read_json(P1_SUMMARY)
    p2 = _read_json(P2_SUMMARY)
    s6 = _read_json(S6_VERDICT)
    s7 = _read_json(S7_SUMMARY)
    s9 = _read_json(S9_SUMMARY)
    s12 = _read_json(S12_SUMMARY)
    s14 = _read_json(S14_SUMMARY)
    p2_pair_rows = _read_csv(P2_PAIRS)

    if p1.get("strict_pairing_validated") is not True or p1.get("n_pairs") != 12:
        raise ValueError("Phase-1 frozen pairing contract changed")
    if p2["gates"]["all_pair_gates_pass"] is not True or p2.get("n_pairs") != 12:
        raise ValueError("Phase-2a frozen pairing contract changed")
    if any(value != "PASS" for value in s6["gate_verdicts"].values()):
        raise ValueError("Stage-6 implementation gate changed")
    if s7.get("decision") != "STRUCTURED_HARD_INFERENCE_VALIDATED" or s7.get("truth_used_for_selection") is not False:
        raise ValueError("Stage-7 decision/truth firewall changed")
    if s9.get("NEXT_ACTION") != "STOP_REASSESS_FROZEN_INFERENCE_ROUTE":
        raise ValueError("Stage-9A decision changed")
    if s12.get("machine_decision") != "STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC":
        raise ValueError("Stage-12B decision changed")
    if s14.get("decision") != "GANSIM_STYLE_GEO_GUIDANCE_NOT_SUPPORTED":
        raise ValueError("Stage-14 decision changed")

    p1_metrics = p1["metrics"]
    p2_metrics = p2["metrics"]
    phase1_shared = (P1_SUMMARY, "authoritative Phase-1 machine summary"), (P1_REPORT, "formal Phase-1 interpretation"), (P1_PAIRS, "strict paired Phase-1 rows")
    phase2_shared = (P2_SUMMARY, "authoritative Phase-2a machine summary"), (P2_REPORT, "formal Phase-2a interpretation"), (P2_PAIRS, "strict paired Phase-2a rows")

    experiments: list[dict[str, Any]] = []
    experiments.append(
        _experiment(
            "sparse_cfm_baseline",
            "Sparse conditional Flow baseline",
            "reference prior sample",
            "frozen conditional Flow; alpha-zero paired baseline",
            1,
            {"paired_runs": int(p1["n_pairs"]), "samples_per_seed": int(p1["n_pairs"]) // len(p1["seeds"]), "seeds": len(p1["seeds"])},
            False,
            True,
            "surface and borehole categorical conditions",
            "paired alpha-zero baseline; no physics or truth ranking",
            [
                _metric("sparse_global_accuracy", "Global voxel accuracy", float(p1_metrics["global_voxel_accuracy"]["baseline"]["mean"]), "fraction", P1_SUMMARY, "metrics.global_voxel_accuracy.baseline.mean"),
                _metric("sparse_global_miou", "Global dynamic-union mIoU", float(p1_metrics["global_mean_iou"]["baseline"]["mean"]), "fraction", P1_SUMMARY, "metrics.global_mean_iou.baseline.mean"),
                _metric("sparse_label9_iou", "Label-9 IoU", float(p1_metrics["target_iou"]["baseline"]["mean"]), "fraction", P1_SUMMARY, "metrics.target_iou.baseline.mean"),
                _metric("sparse_condition_violations", "Condition violations", int(p1["gate"]["entries"]["zero_condition_violations"]["value"]), "voxels", P1_SUMMARY, "gate.entries.zero_condition_violations.value"),
            ],
            "REFERENCE_BASELINE",
            "Sparse categorical conditions under-constrain the hidden label-9 target in this registered case.",
            phase1_shared,
        )
    )
    experiments.append(
        _experiment(
            "phase1_oracle_probability",
            "Phase 1 oracle 3-D probability guidance",
            "truth-derived controllability upper bound",
            "gradient guidance through frozen Flow integration",
            1,
            {"paired_runs": int(p1["n_pairs"]), "samples_per_seed": int(p1["n_pairs"]) // len(p1["seeds"]), "seeds": len(p1["seeds"])},
            True,
            True,
            "truth-derived 3-D label-9 probability volume",
            "strict baseline/guided pairing; no sample selection",
            [
                _metric("p1_global_accuracy", "Global voxel accuracy", _mean_pair(p1_metrics["global_voxel_accuracy"]), "fraction", P1_SUMMARY, "metrics.global_voxel_accuracy.{baseline,guided}.mean"),
                _metric("p1_global_miou", "Global dynamic-union mIoU", _mean_pair(p1_metrics["global_mean_iou"]), "fraction", P1_SUMMARY, "metrics.global_mean_iou.{baseline,guided}.mean"),
                _metric("p1_label9_iou", "Label-9 IoU", _mean_pair(p1_metrics["target_iou"]), "fraction", P1_SUMMARY, "metrics.target_iou.{baseline,guided}.mean"),
                _metric("p1_label9_precision", "Label-9 precision", _mean_pair(p1_metrics["target_precision"]), "fraction", P1_SUMMARY, "metrics.target_precision.{baseline,guided}.mean"),
                _metric("p1_label9_recall", "Label-9 recall", _mean_pair(p1_metrics["target_recall"]), "fraction", P1_SUMMARY, "metrics.target_recall.{baseline,guided}.mean"),
                _metric("p1_centroid_distance", "Label-9 centroid distance", _mean_pair(p1_metrics["target_centroid_distance"]), "voxels", P1_SUMMARY, "metrics.target_centroid_distance.{baseline,guided}.mean"),
                _metric("p1_roi_iou", "Selected-ROI IoU", _mean_pair(p1_metrics["selected_roi_iou"]), "fraction", P1_SUMMARY, "metrics.selected_roi_iou.{baseline,guided}.mean"),
                _metric("p1_condition_violations", "Condition violations", int(p1["gate"]["entries"]["zero_condition_violations"]["value"]), "voxels", P1_SUMMARY, "gate.entries.zero_condition_violations.value"),
                _metric("p1_paired_runs", "Strict paired runs", int(p1["n_pairs"]), "pairs", P1_SUMMARY, "n_pairs"),
            ],
            str(p1["gate"]["phase_decision"]),
            "Strong frozen-Flow controllability is demonstrated under privileged truth-derived probability evidence, with retained topology and endpoint caveats.",
            phase1_shared,
        )
    )
    experiments.append(
        _experiment(
            "phase2a_ideal_property",
            "Phase 2a ideal 3-D property guidance",
            "truth-derived physical-property upper bound",
            "two-channel differentiable property guidance through frozen Flow",
            1,
            {"paired_runs": int(p2["n_pairs"]), "samples_per_seed": int(p2["n_samples_per_seed"]), "seeds": len(p2["seeds"])},
            True,
            True,
            "full-resolution noiseless truth-derived 3-D properties",
            "strict baseline/guided pairing; no sample selection",
            [
                _metric("p2_global_accuracy", "Global voxel accuracy", _mean_pair(p2_metrics["global_voxel_accuracy"]), "fraction", P2_SUMMARY, "metrics.global_voxel_accuracy.{baseline,guided}.mean"),
                _metric("p2_dynamic_miou", "Global dynamic-union mIoU", _mean_pair(p2_metrics["global_mean_iou"]), "fraction", P2_SUMMARY, "metrics.global_mean_iou.{baseline,guided}.mean"),
                _metric("p2_truth_present_miou", "Truth-present fixed-set mIoU", _mean_pair(p2_metrics["truth_present_mean_iou"]), "fraction", P2_SUMMARY, "metrics.truth_present_mean_iou.{baseline,guided}.mean"),
                _metric("p2_hard_property_loss", "Hard property loss", _mean_pair(p2_metrics["hard_property_loss"]), "loss", P2_SUMMARY, "metrics.hard_property_loss.{baseline,guided}.mean"),
                _metric("p2_label9_iou", "Label-9 IoU", _mean_pair(p2_metrics["target_iou"]), "fraction", P2_SUMMARY, "metrics.target_iou.{baseline,guided}.mean"),
                _metric("p2_label9_precision", "Label-9 precision", _mean_pair(p2_metrics["target_precision"]), "fraction", P2_SUMMARY, "metrics.target_precision.{baseline,guided}.mean"),
                _metric("p2_label9_recall", "Label-9 recall", _mean_pair(p2_metrics["target_recall"]), "fraction", P2_SUMMARY, "metrics.target_recall.{baseline,guided}.mean"),
                _metric("p2_centroid_distance", "Label-9 centroid distance", _mean_pair(p2_metrics["target_centroid_distance"]), "voxels", P2_SUMMARY, "metrics.target_centroid_distance.{baseline,guided}.mean"),
                _metric("p2_condition_violations", "Condition violations", sum(int(row["condition_violation_count"]) for row in p2_pair_rows), "voxels", P2_PAIRS, "sum(condition_violation_count) over all paired rows"),
                _metric("p2_paired_runs", "Strict paired runs", int(p2["n_pairs"]), "pairs", P2_SUMMARY, "n_pairs"),
            ],
            str(p2["decision"]),
            "Complete ideal properties control the frozen generator, but remain a truth-derived noiseless upper bound rather than acquisition-domain geophysics.",
            phase2_shared,
        )
    )

    h7 = _find_hypothesis(s6, "H7")["relevant_metric"]
    h13 = _find_hypothesis(s6, "H13")["relevant_metric"]
    experiments.append(
        _experiment(
            "stage6_direct_geophysical",
            "Direct geophysical gradient guidance",
            "synthetic acquisition-domain diagnostic",
            "direct continuous-state seismic gradient guidance",
            1,
            {"diagnostic_control_runs": len(_read_json(S6_D3_SUMMARY)["runs"])},
            False,
            True,
            "noiseless synthetic post-stack seismic",
            "observation-only best iterate; truth used for retrospective mechanism diagnostics",
            [
                _metric("s6_seismic_soft_attainment", "Seismic soft attainment (max/final)", [float(value) for value in h7["seismic_max_final_soft"]], "fraction", S6_VERDICT, "hypotheses[id=H7].relevant_metric.seismic_max_final_soft", scientific_role="diagnostic"),
                _metric("s6_seismic_hard_attainment", "Seismic hard attainment (max/final)", [float(value) for value in h7["seismic_max_final_hard"]], "fraction", S6_VERDICT, "hypotheses[id=H7].relevant_metric.seismic_max_final_hard", scientific_role="diagnostic"),
                _metric("s6_first_divergent_level", "First soft-hard divergent level", str(h7["first_divergent_level"]), "categorical level", S6_VERDICT, "hypotheses[id=H7].relevant_metric.first_divergent_level", scientific_role="diagnostic"),
                _metric("s6_best_hard_step", "Observation-selected best hard step", int(h13["D3_observation_selected_best_step"]), "step", S6_VERDICT, "hypotheses[id=H13].relevant_metric.D3_observation_selected_best_step", scientific_role="diagnostic"),
            ],
            "IMPLEMENTATION_GATES_PASS; DIRECT_GUIDANCE_LIMITED",
            "The continuous seismic objective improves, but hard categorical attainment is weak and degrades at the endpoint; the first reproducible divergence occurs at reflectivity/TWT.",
            ((S6_VERDICT, "authoritative causal diagnostic verdict"), (S6_REPORT, "formal Stage-6 diagnostic synthesis"), (S6_D3_SUMMARY, "direct soft/hard transfer machine summary")),
        )
    )

    correct_rows = [next(arm for arm in case["arms"] if arm["optimized_by"] == "correct") for case in s7["native_replicas"]]
    zero_rows = [next(arm for arm in case["arms"] if arm["optimized_by"] == "zero") for case in s7["native_replicas"]]
    hero_index = [case["case_id"] for case in s7["native_replicas"]].index("native_seed20260809")
    experiments.append(
        _experiment(
            "stage7_structured_hard_seismic",
            "Stage 7 structured hard-seismic inference",
            "bounded structured geophysical inference",
            "hard-seismic beam search over a registered structured hypothesis family",
            len(correct_rows),
            {"registered_native_cases": len(correct_rows)},
            False,
            True,
            "noiseless synthetic hard seismic",
            "minimum hard observed seismic RMSE only; truth metrics retrospective",
            [
                _metric("s7_correct_rank_first", "Correct observation ranks first", sum(bool(case["correct_optimized_is_best_against_correct"]) for case in s7["native_replicas"]), "cases", S7_SUMMARY, "native_replicas[*].correct_optimized_is_best_against_correct"),
                _metric("s7_registered_case_count", "Registered native cases", len(correct_rows), "cases", S7_SUMMARY, "len(native_replicas)"),
                _metric("s7_mean_hard_attainment", "Mean correct hard attainment", sum(float(row["hard_correct_observation_attainment"]) for row in correct_rows) / len(correct_rows), "fraction", S7_SUMMARY, "mean(native_replicas[*].arms[optimized_by=correct].hard_correct_observation_attainment)"),
                _metric("s7_hidden_iou_range", "Hidden label-9 IoU range", [min(float(row["hidden_target_iou"]) for row in correct_rows), max(float(row["hidden_target_iou"]) for row in correct_rows)], "fraction", S7_SUMMARY, "range(native_replicas[*].arms[optimized_by=correct].hidden_target_iou)"),
                _metric("s7_hidden_recall_range", "Hidden label-9 recall range", [min(float(row["hidden_target_recall"]) for row in correct_rows), max(float(row["hidden_target_recall"]) for row in correct_rows)], "fraction", S7_SUMMARY, "range(native_replicas[*].arms[optimized_by=correct].hidden_target_recall)"),
                _metric("s7_condition_violations", "Condition violations", sum(int(row["condition_violations"]) for row in correct_rows), "voxels", S7_SUMMARY, "sum(native_replicas[*].arms[optimized_by=correct].condition_violations)"),
                _metric("s7_hero_rmse", "Hero hard seismic RMSE", {"before": float(zero_rows[hero_index]["hard_correct_observation_rmse"]), "after": float(correct_rows[hero_index]["hard_correct_observation_rmse"])}, "RMSE", S7_SUMMARY, "native_replicas[case_id=native_seed20260809].arms[optimized_by=zero|correct].hard_correct_observation_rmse"),
                _metric("s7_hero_hidden_iou", "Hero hidden label-9 IoU", {"before": float(zero_rows[hero_index]["hidden_target_iou"]), "after": float(correct_rows[hero_index]["hidden_target_iou"])}, "fraction", S7_SUMMARY, "native_replicas[case_id=native_seed20260809].arms[optimized_by=zero|correct].hidden_target_iou"),
            ],
            str(s7["decision"]),
            "Synthetic seismic discriminates the correct observation within a bounded structured geological family; this is not a CFM posterior.",
            ((S7_SUMMARY, "authoritative Stage-7 machine summary"), (S7_REPORT, "formal Stage-7 report"), (S7_NATIVE_CSV, "native cross-evaluation matrix")),
        )
    )

    oracle_ious = {case["case_id"]: float(case["oracle_best_label9_iou"]["label9_iou"]) for case in s9["cases"]}
    experiments.append(
        _experiment(
            "stage9a_unrestricted_flow_prior",
            "Stage 9A unrestricted frozen-Flow ranking",
            "prior-support and discrimination audit",
            "unrestricted frozen-Flow sampling followed by hard-seismic ranking",
            int(s9["primary_case_count"]),
            {"samples_per_case": int(s9["formal_candidates_per_case"]), "total_samples": int(s9["formal_candidate_count_total"])},
            False,
            True,
            "surface/boreholes for generation; synthetic hard seismic for ranking",
            "ascending hard seismic RMSE with candidate-ID ties; truth loaded only after ranking",
            [
                _metric("s9_samples_per_case", "Frozen-Flow samples per case", int(s9["formal_candidates_per_case"]), "samples/case", S9_SUMMARY, "formal_candidates_per_case"),
                _metric("s9_case_count", "Primary registered cases", int(s9["primary_case_count"]), "cases", S9_SUMMARY, "primary_case_count"),
                _metric("s9_unique_hard_models", "Unique hard models", sum(int(case["ensemble"]["unique_hard_model_count"]) for case in s9["cases"]), "models", S9_SUMMARY, "sum(cases[*].ensemble.unique_hard_model_count)"),
                _metric("s9_flow_velocity_forwards", "Flow velocity forwards", int(s9["flow_velocity_forward_count_total"]), "forwards", S9_SUMMARY, "flow_velocity_forward_count_total"),
                _metric("s9_hard_seismic_forwards", "Hard seismic forwards", int(s9["hard_seismic_forward_count_total"]), "forwards", S9_SUMMARY, "hard_seismic_forward_count_total"),
                _metric("s9_support_pass_count", "Support pass count", int(s9["support_case_pass_count"]), "cases", S9_SUMMARY, "support_case_pass_count"),
                _metric("s9_discrimination_pass_count", "Discrimination pass count", int(s9["discrimination_case_pass_count"]), "cases", S9_SUMMARY, "discrimination_case_pass_count"),
                _metric("s9_oracle_best_label9_iou", "Oracle-best label-9 IoU by case", oracle_ious, "fraction", S9_SUMMARY, "cases[*].oracle_best_label9_iou.label9_iou", scientific_role="retrospective upper bound", qualification="Not a deployable selector; truth was used only after pool/ranking freeze."),
            ],
            str(s9["NEXT_ACTION"]),
            "The registered hidden-target tasks are outside adequate support/discrimination of the tested unrestricted frozen-Flow proposal pool.",
            ((S9_SUMMARY, "authoritative Stage-9A machine summary"), (S9_REPORT, "formal Stage-9A report")),
        )
    )

    fullgeo_manifests = [PROJECT_DIR / f"experiments/full_structuralgeo_benchmark/cases/fullgeo_case{index:02d}/manifest.json" for index in range(1, 6)]
    fullgeo_payloads = [_read_json(path) for path in fullgeo_manifests]
    fullgeo_counts = {payload["case_id"]: int(payload["raw_label_counts"]["9"]) for payload in fullgeo_payloads}
    fullgeo_count_metric = _metric("fullgeo_label9_voxel_counts", "Raw label-9 voxel counts", fullgeo_counts, "voxels", fullgeo_manifests[0], "raw_label_counts.9 for each registered case", scientific_role="cohort descriptor")
    fullgeo_count_metric["source_by_case"] = {
        payload["case_id"]: {
            "path": _relative(path),
            "json_key": "raw_label_counts.9",
            "sha256": _sha256(path),
        }
        for payload, path in zip(fullgeo_payloads, fullgeo_manifests)
    }
    experiments.append(
        _experiment(
            "full_structuralgeo_registered_cohort",
            "Full StructuralGeo registered benchmark",
            "prospectively registered geology-only cohort",
            "no inference",
            5,
            {"registered_cases": 5},
            False,
            False,
            "none; geology-only benchmark construction",
            "first five prospectively eligible seeds; no visual or downstream replacement",
            [
                fullgeo_count_metric,
                _metric("fullgeo_resolution", "Model resolution", [int(value) for value in fullgeo_payloads[0]["resolution"]], "voxels", fullgeo_manifests[0], "resolution", scientific_role="method"),
                _metric("fullgeo_well_count", "Fixed vertical wells", int(fullgeo_payloads[0]["condition_statistics"]["well_count"]), "wells", fullgeo_manifests[0], "condition_statistics.well_count", scientific_role="method"),
            ],
            "FULL_STRUCTURALGEO_BENCHMARK_READY",
            "The five-case cohort is a frozen, diverse synthetic benchmark; it is not an inference result and cannot establish training-set non-overlap at sample level.",
            ((FULLGEO_REPORT, "authoritative benchmark build report"), *((path, f"registered FullGeo case {index} manifest") for index, path in enumerate(fullgeo_manifests, start=1))),
        )
    )

    experiments.append(
        _experiment(
            "stage12b_fullgeo_probability_bridge",
            "Stage 12B FullGeo probability bridge",
            "synthetic bridge-specificity gate",
            "fixed seismic-to-property inversion and probability construction; no Flow forward",
            len(s12["case_ids"]),
            {"registered_cases": len(s12["case_ids"]), "flow_forwards": int(bool(s12["stage12b_b_authorized"]))},
            False,
            True,
            "noiseless synthetic seismic; derived post-seismic label-9 probabilities",
            "prospective 5x5 transfer and control gates; no case replacement",
            [
                _metric("s12_diagonal_mean_auprc", "Diagonal mean AUPRC", float(s12["diagonal_mean_auprc"]), "AUPRC", S12_SUMMARY, "diagonal_mean_auprc"),
                _metric("s12_case_count", "Registered FullGeo cases", len(s12["case_ids"]), "cases", S12_SUMMARY, "len(case_ids)"),
                _metric("s12_off_diagonal_mean_auprc", "Off-diagonal mean AUPRC", float(s12["off_diagonal_mean_auprc"]), "AUPRC", S12_SUMMARY, "off_diagonal_mean_auprc"),
                _metric("s12_diagonal_row_maxima", "Diagonal row maxima", int(s12["diagonal_row_maximum_count"]), "cases", S12_SUMMARY, "diagonal_row_maximum_count"),
                _metric("s12_correct_above_shuffled", "Correct above shuffled", int(s12["correct_above_shuffled_count"]), "cases", S12_SUMMARY, "correct_above_shuffled_count"),
                _metric("s12_correct_above_constant", "Correct above constant", int(s12["correct_above_constant_count"]), "cases", S12_SUMMARY, "correct_above_constant_count"),
                _metric("s12_post_above_prior", "Post above prior", int(s12["post_ap_above_prior_count"]), "cases", S12_SUMMARY, "post_ap_above_prior_count"),
            ],
            str(s12["machine_decision"]),
            "The current scalar log-impedance probability bridge is not case-specific on the five registered FullGeo cases; Flow was not authorized in Stage 12B.",
            ((S12_SUMMARY, "authoritative Stage-12B-A machine summary"), (S12_REPORT, "formal Stage-12B report")),
        )
    )

    per_case_delta = {case_id: float(row["paired_delta_medians"]["hidden_label9_iou"]) for case_id, row in s14["case_summaries"].items()}
    experiments.append(
        _experiment(
            "stage14_probability_guided_flow",
            "Stage 14 probability-guided frozen Flow",
            "paired synthetic end-to-end pilot",
            "Stage-12B probability volumes through the frozen Phase-1 guidance interface",
            int(s14["n_cases"]),
            {"registered_cases": int(s14["n_cases"]), "paired_runs": int(s14["n_pairs"]), "pairs_per_case": int(s14["n_pairs"]) // int(s14["n_cases"])},
            False,
            True,
            "synthetic post-seismic label-9 probability volume",
            "all preregistered pairs; no sweep, tuning, or best-sample selection",
            [
                _metric("s14_overall_median_hidden_iou_delta", "Overall median paired hidden-label9 IoU delta", float(s14["overall_paired_median_hidden_label9_iou_delta"]), "fraction", S14_SUMMARY, "overall_paired_median_hidden_label9_iou_delta"),
                _metric("s14_case_count", "Registered FullGeo cases", int(s14["n_cases"]), "cases", S14_SUMMARY, "n_cases"),
                _metric("s14_pair_count", "Paired runs", int(s14["n_pairs"]), "pairs", S14_SUMMARY, "n_pairs"),
                _metric("s14_positive_case_count", "Positive case medians", int(s14["positive_case_count"]), "cases", S14_SUMMARY, "positive_case_count"),
                _metric("s14_condition_violations", "Hard-condition violations", int(s14["total_condition_violations"]), "voxels", S14_SUMMARY, "total_condition_violations"),
                _metric("s14_per_case_hidden_iou_deltas", "Per-case median hidden-label9 IoU delta", per_case_delta, "fraction", S14_SUMMARY, "case_summaries.<case_id>.paired_delta_medians.hidden_label9_iou"),
            ],
            str(s14["decision"]),
            "The current seismic-probability guidance does not improve hidden label-9 recovery across the five registered FullGeo cases.",
            ((S14_SUMMARY, "authoritative Stage-14 machine summary"), (S14_REPORT, "formal Stage-14 report"), (S14_DELTAS, "all paired Stage-14 deltas")),
        )
    )
    return experiments


def _metric_index(experiments: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    metrics = [metric for experiment in experiments for metric in experiment["primary_metrics"]]
    index = {metric["metric_id"]: metric for metric in metrics}
    if len(index) != len(metrics):
        raise ValueError("duplicate metric_id in evidence schema")
    return index


def _pair_text(metric: dict[str, Any], digits: int = 3) -> str:
    value = metric["value"]
    return f"{value['baseline']:.{digits}f} -> {value['guided']:.{digits}f}"


def _build_table1(metrics: dict[str, dict[str, Any]]) -> dict[str, Any]:
    s7_range = metrics["s7_hidden_iou_range"]["value"]
    s9_unique = int(metrics["s9_unique_hard_models"]["value"])
    s14_delta = float(metrics["s14_overall_median_hidden_iou_delta"]["value"])
    rows = [
        {
            "evidence_inference": "Sparse CFM",
            "observation_type": "Surface + 9 boreholes",
            "truth_used_at_inference": "No",
            "target_geology_result": f"Label-9 IoU {metrics['sparse_label9_iou']['value']:.3f}",
            "physical_result": "Not used",
            "primary_interpretation": "Sparse conditions under-constrain target geometry.",
            "source_metric_ids": ["sparse_label9_iou"],
        },
        {
            "evidence_inference": "Oracle probability guidance",
            "observation_type": "Truth-derived 3-D P(label 9)",
            "truth_used_at_inference": "Yes (oracle)",
            "target_geology_result": f"Label-9 IoU {_pair_text(metrics['p1_label9_iou'])}",
            "physical_result": "Not acquisition-domain",
            "primary_interpretation": "Strong controllability upper bound.",
            "source_metric_ids": ["p1_label9_iou"],
        },
        {
            "evidence_inference": "Ideal property guidance",
            "observation_type": "Truth-derived noiseless 3-D properties",
            "truth_used_at_inference": "Yes (upper bound)",
            "target_geology_result": f"Label-9 IoU {_pair_text(metrics['p2_label9_iou'])}",
            "physical_result": f"Hard-property loss {_pair_text(metrics['p2_hard_property_loss'])}",
            "primary_interpretation": "Ideal property controllability is demonstrated.",
            "source_metric_ids": ["p2_label9_iou", "p2_hard_property_loss"],
        },
        {
            "evidence_inference": "Structured hard-seismic inference",
            "observation_type": "Synthetic hard seismic",
            "truth_used_at_inference": "No",
            "target_geology_result": f"Hidden IoU {s7_range[0]:.3f}-{s7_range[1]:.3f}",
            "physical_result": f"Correct ranks first {metrics['s7_correct_rank_first']['value']}/{metrics['s7_registered_case_count']['value']}",
            "primary_interpretation": "Seismic discriminates bounded hypotheses.",
            "source_metric_ids": ["s7_hidden_iou_range", "s7_correct_rank_first"],
        },
        {
            "evidence_inference": "Unrestricted frozen-Flow ranking",
            "observation_type": "Synthetic hard seismic",
            "truth_used_at_inference": "No; oracle audit after ranking",
            "target_geology_result": f"Support {metrics['s9_support_pass_count']['value']}/{metrics['s9_case_count']['value']}; discrimination {metrics['s9_discrimination_pass_count']['value']}/{metrics['s9_case_count']['value']}",
            "physical_result": f"{s9_unique} unique models ranked",
            "primary_interpretation": "Frozen-prior support is limiting.",
            "source_metric_ids": ["s9_support_pass_count", "s9_discrimination_pass_count", "s9_unique_hard_models"],
        },
        {
            "evidence_inference": "FullGeo probability bridge",
            "observation_type": "Synthetic seismic -> P(label 9)",
            "truth_used_at_inference": "No",
            "target_geology_result": f"Diagonal/off-diagonal AP {metrics['s12_diagonal_mean_auprc']['value']:.4f}/{metrics['s12_off_diagonal_mean_auprc']['value']:.4f}",
            "physical_result": f"Diagonal row maximum {metrics['s12_diagonal_row_maxima']['value']}/{metrics['s12_case_count']['value']}",
            "primary_interpretation": "Current bridge lacks case specificity.",
            "source_metric_ids": ["s12_diagonal_mean_auprc", "s12_off_diagonal_mean_auprc", "s12_diagonal_row_maxima"],
        },
        {
            "evidence_inference": "Seismic-probability-guided Flow",
            "observation_type": "Synthetic post-seismic P(label 9)",
            "truth_used_at_inference": "No",
            "target_geology_result": f"Median change in hidden IoU {s14_delta:.4f}; positive {metrics['s14_positive_case_count']['value']}/{metrics['s14_case_count']['value']}",
            "physical_result": f"{metrics['s14_condition_violations']['value']} hard-condition violations",
            "primary_interpretation": "Current end-to-end bridge is insufficient.",
            "source_metric_ids": ["s14_overall_median_hidden_iou_delta", "s14_positive_case_count", "s14_condition_violations"],
        },
    ]
    return {
        "caption": "Compact quantitative evidence summary for flow2. " + CAPTION_INTEGRITY_NOTE,
        "columns": [
            "Evidence / inference",
            "Observation type",
            "Truth used at inference?",
            "Target-geology result",
            "Physical result",
            "Primary interpretation",
        ],
        "rows": rows,
        "footnotes": [
            CAPTION_INTEGRITY_NOTE,
            "Oracle metrics are privileged controllability upper bounds, not measured-geophysics results.",
            "Direct gradient guidance is summarized in text: soft seismic attainment reached 0.1909, while maximum/final hard attainment was 0.0244/-0.0080; the first reproducible soft-hard divergence occurred at reflectivity/TWT.",
            "Stage-9 oracle-best candidates are retrospective support ceilings and are not deployable selectors.",
            "All seismic experiments are synthetic; no field-data validation is claimed.",
        ],
    }


def _figure_links() -> list[dict[str, Any]]:
    links = []
    for path in sorted((PROJECT_DIR / "paper/manifests").glob("*.json")):
        payload = _read_json(path)
        links.append(
            {
                "manifest_path": _relative(path),
                "manifest_sha256": _sha256(path),
                "figure_or_gallery_id": payload.get("figure_id", payload.get("gallery_id", payload.get("schema"))),
                "output_paths": [row["path"] for row in payload.get("outputs", [])],
            }
        )
    return links


def build_evidence_summary() -> dict[str, Any]:
    experiments = _build_experiments()
    metrics = _metric_index(experiments)
    headline_ids = [
        "p1_label9_iou",
        "p2_label9_iou",
        "s7_hidden_iou_range",
        "s7_correct_rank_first",
        "s9_support_pass_count",
        "s9_discrimination_pass_count",
        "s12_diagonal_mean_auprc",
        "s12_off_diagonal_mean_auprc",
        "s14_overall_median_hidden_iou_delta",
        "s14_positive_case_count",
    ]
    git_head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip()
    return {
        "schema": "flow2_paper_evidence_summary_v1",
        "metadata": {
            "project": "flow2 / geodata-3d-conditional",
            "git_head_at_generation": git_head,
            "generator": _source_file(SCRIPT_PATH, "deterministic evidence-package generator"),
            "scientific_inference_rerun": False,
            "metric_source_policy": "all numeric evidence is extracted from frozen JSON/CSV artifacts; reports supply interpretation only",
            "protocol_comparability": "distinct diagnostic protocols; not a shared-case leaderboard",
        },
        "experiments": experiments,
        "headline_metrics": [metrics[metric_id] for metric_id in headline_ids],
        "scientific_boundaries": {
            "synthetic_data_only": True,
            "field_validation": False,
            "phase1_probability_is_truth_derived_oracle": True,
            "phase2_properties_are_truth_derived_ideal_upper_bound": True,
            "stage6_seismic_is_inverse_crime_synthetic": True,
            "stage7_is_structured_inference_not_cfm_posterior": True,
            "stage7_truth_used_for_selection": False,
            "stage9_oracle_best_is_retrospective_upper_bound": True,
            "stage12_probability_bridge_is_synthetic": True,
            "stage14_is_paired_synthetic_not_measured_field_test": True,
            "calibrated_bayesian_posterior_claim": False,
            "cross_protocol_head_to_head_claim": False,
        },
        "table1": _build_table1(metrics),
        "figure_links": _figure_links(),
    }


def _json_dump(payload: Any) -> str:
    return json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n"


def render_evidence_csv(summary: dict[str, Any]) -> str:
    columns = [
        "experiment_id",
        "display_name",
        "evidence_type",
        "inference_method",
        "metric_id",
        "metric_name",
        "value_json",
        "unit",
        "scientific_role",
        "qualification",
        "source_path",
        "source_key",
        "source_sha256",
    ]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=columns, lineterminator="\n")
    writer.writeheader()
    for experiment in summary["experiments"]:
        for metric in experiment["primary_metrics"]:
            writer.writerow(
                {
                    "experiment_id": experiment["experiment_id"],
                    "display_name": experiment["display_name"],
                    "evidence_type": experiment["evidence_type"],
                    "inference_method": experiment["inference_method"],
                    "metric_id": metric["metric_id"],
                    "metric_name": metric["display_name"],
                    "value_json": json.dumps(metric["value"], sort_keys=True, ensure_ascii=False, allow_nan=False),
                    "unit": metric["unit"],
                    "scientific_role": metric["scientific_role"],
                    "qualification": metric["qualification"],
                    "source_path": metric["source"]["path"],
                    "source_key": metric["source"]["json_key"],
                    "source_sha256": metric["source"]["sha256"],
                }
            )
    return buffer.getvalue()


def render_table_csv(table: dict[str, Any]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=table["columns"], lineterminator="\n")
    writer.writeheader()
    keys = ["evidence_inference", "observation_type", "truth_used_at_inference", "target_geology_result", "physical_result", "primary_interpretation"]
    for row in table["rows"]:
        writer.writerow(dict(zip(table["columns"], (row[key] for key in keys))))
    return buffer.getvalue()


def render_table_markdown(table: dict[str, Any]) -> str:
    keys = ["evidence_inference", "observation_type", "truth_used_at_inference", "target_geology_result", "physical_result", "primary_interpretation"]
    lines = [
        "# Table 1. Quantitative evidence summary",
        "",
        table["caption"],
        "",
        "| " + " | ".join(table["columns"]) + " |",
        "| " + " | ".join("---" for _ in table["columns"]) + " |",
    ]
    for row in table["rows"]:
        lines.append("| " + " | ".join(str(row[key]).replace("|", "\\|") for key in keys) + " |")
    lines.extend(["", "Notes:"])
    lines.extend(f"{index}. {note}" for index, note in enumerate(table["footnotes"], start=1))
    return "\n".join(lines) + "\n"


def _latex_escape(value: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(char, char) for char in value)


def render_table_latex(table: dict[str, Any]) -> str:
    keys = ["evidence_inference", "observation_type", "truth_used_at_inference", "target_geology_result", "physical_result", "primary_interpretation"]
    lines = [
        "% Requires: \\usepackage{booktabs,tabularx,threeparttable}",
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        "\\begin{threeparttable}",
        f"\\caption{{{_latex_escape(table['caption'])}}}",
        "\\label{tab:evidence-summary}",
        "\\begin{tabularx}{\\textwidth}{@{}p{0.13\\textwidth}p{0.15\\textwidth}p{0.11\\textwidth}p{0.16\\textwidth}p{0.15\\textwidth}X@{}}",
        "\\toprule",
        "Evidence / inference & Observation type & Truth at inference? & Target-geology result & Physical result & Primary interpretation \\\\",
        "\\midrule",
    ]
    for row in table["rows"]:
        lines.append(" & ".join(_latex_escape(str(row[key])).replace(" -> ", r" $\rightarrow$ ") for key in keys) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            "\\begin{tablenotes}[flushleft]",
            *[f"\\item[{index}] {_latex_escape(note)}" for index, note in enumerate(table["footnotes"], start=1)],
            "\\end{tablenotes}",
            "\\end{threeparttable}",
            "\\end{table*}",
        ]
    )
    return "\n".join(lines) + "\n"


def render_fact_sheet(summary: dict[str, Any]) -> str:
    metric = _metric_index(summary["experiments"])
    p1 = metric["p1_label9_iou"]["value"]
    p2 = metric["p2_label9_iou"]["value"]
    s7_iou = metric["s7_hidden_iou_range"]["value"]
    s14_deltas = metric["s14_per_case_hidden_iou_deltas"]["value"]
    lines = [
        "# flow2 paper result facts",
        "",
        "> Machine-grounded writing aid generated from frozen reports. It is not manuscript prose and does not make cross-protocol leaderboard claims.",
        "",
        "## SAFE CLAIMS",
        "",
        f"- Frozen conditional Flow is strongly controllable under truth-derived 3-D probability evidence: mean label-9 IoU changed from {p1['baseline']:.4f} to {p1['guided']:.4f} across 12 strict pairs.",
        "- Structured hard-geophysics inference ranks the correct observation first in all three registered StructuralGeo-native replicas.",
        "- Stage 9A demonstrates limited unrestricted frozen-Flow prior support for the three registered hidden-target tasks: support and discrimination both pass in 0/3 cases.",
        f"- Current seismic-derived probability guidance through frozen Flow does not improve hidden label-9 recovery on the five Full StructuralGeo cases: overall median paired delta is {metric['s14_overall_median_hidden_iou_delta']['value']:.6f}, with one positive case median.",
        "- All tested hard surface and borehole conditions remain exact in the reported Phase 1, Phase 2a, Stage 7, and Stage 14 evaluations.",
        "",
        "## QUALIFIED CLAIMS",
        "",
        f"- Synthetic seismic contains sufficient information to discriminate hidden structures inside the bounded Stage-7 hypothesis family (hidden IoU range {s7_iou[0]:.4f}-{s7_iou[1]:.4f}); this is structured search, not a CFM posterior.",
        "- Acquisition-domain geophysical constraints remain limited by observability and representation in the current synthetic setup.",
        f"- Ideal, truth-derived 3-D properties provide a physical controllability upper bound (mean label-9 IoU {p2['baseline']:.4f} to {p2['guided']:.4f}), not evidence for realistic property inversion.",
        "- Stage 12B shows small constructive probability changes in some metrics, but fails the prospective case-specificity gate and therefore does not validate the bridge.",
        "- Full StructuralGeo cases are independently generated and prospectively registered from the same recipe, but historical sample-level non-overlap with the streaming training run cannot be certified.",
        "",
        "## PROHIBITED CLAIMS",
        "",
        "- CFM seismic inversion recovers hidden geology with IoU 0.987.",
        "- Phase 1 IoU 0.81 is a measured-geophysics result.",
        "- The method provides a calibrated Bayesian posterior.",
        "- Stage 7 truth was used for candidate selection.",
        "- Current results demonstrate field-data generalization.",
        "- Results from Phase 1, Phase 2, Stage 7, Stage 9, Stage 12, and Stage 14 form a shared-case head-to-head leaderboard.",
        "",
        "## A. Abstract-ready metrics",
        "",
        f"1. Oracle 3-D probability guidance: label-9 IoU {p1['baseline']:.3f} to {p1['guided']:.3f} over 12 paired runs.",
        f"2. Ideal property guidance: label-9 IoU {p2['baseline']:.3f} to {p2['guided']:.3f} over 12 paired runs.",
        f"3. Structured seismic: hidden IoU {s7_iou[0]:.3f}-{s7_iou[1]:.3f}; correct observation ranks first 3/3.",
        f"4. Frozen-Flow prior audit: {metric['s9_unique_hard_models']['value']} unique models, support 0/3 and discrimination 0/3.",
        f"5. Probability-guided Flow: median paired hidden-label9 IoU delta {metric['s14_overall_median_hidden_iou_delta']['value']:.4f}; positive case medians 1/5.",
        "",
        "## B. Introduction-ready problem statements",
        "",
        "- Sparse surface and borehole observations leave substantial ambiguity in hidden 3-D target geometry.",
        "- Controllability under privileged geological evidence does not imply observability from acquisition-domain geophysics.",
        "- Geophysical consistency must be assessed after hard categorical decoding, because soft physical improvement can fail to transfer across categorical boundaries.",
        "- Inference parameterization and proposal support determine whether an informative observation can affect the frozen generator.",
        "",
        "## C. Methods facts",
        "",
        "- Model resolution: 64 x 64 x 64 categorical voxels.",
        "- Geological conditioning: the categorical surface plus nine prospectively fixed vertical boreholes.",
        "- Inference uses frozen Flow weights; no checkpoint update is performed in the reported guidance experiments.",
        "- Continuous embeddings are integrated and then hard-decoded to categorical geology before geological and physical evaluation.",
        "- Label 9 is the registered pressure-test target; raw labels 10-13 are not merged into it in the Full StructuralGeo cohort.",
        "",
        "## D. Results facts",
        "",
        "### 3.1 Privileged-evidence controllability",
        "",
        f"- Phase 1: global accuracy {metric['p1_global_accuracy']['value']['baseline']:.4f} to {metric['p1_global_accuracy']['value']['guided']:.4f}; global mIoU {metric['p1_global_miou']['value']['baseline']:.4f} to {metric['p1_global_miou']['value']['guided']:.4f}; label-9 IoU {p1['baseline']:.4f} to {p1['guided']:.4f}.",
        f"- Phase 2a: truth-present mIoU {metric['p2_truth_present_miou']['value']['baseline']:.4f} to {metric['p2_truth_present_miou']['value']['guided']:.4f}; hard-property loss {metric['p2_hard_property_loss']['value']['baseline']:.4f} to {metric['p2_hard_property_loss']['value']['guided']:.4f}.",
        "",
        "### 3.2 Acquisition-domain observability and structured inference",
        "",
        f"- Direct seismic guidance: maximum/final soft attainment {metric['s6_seismic_soft_attainment']['value'][0]:.4f}/{metric['s6_seismic_soft_attainment']['value'][1]:.4f}, but maximum/final hard attainment {metric['s6_seismic_hard_attainment']['value'][0]:.4f}/{metric['s6_seismic_hard_attainment']['value'][1]:.4f}.",
        f"- Stage 7: mean correct hard attainment {metric['s7_mean_hard_attainment']['value']:.4f}; hidden recall range {metric['s7_hidden_recall_range']['value'][0]:.4f}-{metric['s7_hidden_recall_range']['value'][1]:.4f}; correct rank first 3/3.",
        "",
        "### 3.3 Prior support and probability bridge",
        "",
        f"- Stage 9A: {metric['s9_samples_per_case']['value']} frozen-Flow samples per case and {metric['s9_flow_velocity_forwards']['value']} Flow velocity forwards; support/discrimination 0/3.",
        f"- Stage 12B: diagonal/off-diagonal mean AUPRC {metric['s12_diagonal_mean_auprc']['value']:.6f}/{metric['s12_off_diagonal_mean_auprc']['value']:.6f}; diagonal row maximum 1/5.",
        "- Stage 14 per-case median hidden-IoU deltas: " + ", ".join(f"{case_id} {value:+.6f}" for case_id, value in s14_deltas.items()) + ".",
        "",
        "## E. Limitations",
        "",
        "- All geophysical observations are synthetic; no field validation has been performed.",
        "- The seismic studies use an inverse-crime forward configuration where stated.",
        "- The synthetic acoustic codebook deliberately gives label 9 distinctive impedance and is not site-calibrated petrophysics.",
        "- Phase-1 probability and Phase-2a property inputs are truth-derived oracle/upper-bound evidence.",
        "- Stage-7 structured search is a bounded inference mechanism, not a posterior sampled by CFM.",
        "- Protocols, cases, observations, and inference spaces differ across stages; their metrics are not directly exchangeable leaderboard scores.",
        "- Full StructuralGeo training-sample non-overlap cannot be certified because the historical streaming run retained no seed/sample manifest.",
        "",
        "## F. Recommended future work",
        "",
        "1. Establish a more identifiable observation model and realistic petrophysical likelihood under a new prospective protocol.",
        "2. Develop and validate a learned seismic-to-geology evidence representation before reconnecting it to Flow guidance.",
        "3. Consider D-Flow or source optimization only after the observation representation passes an independent case-specificity gate.",
        "4. Add noise, survey incompleteness, petrophysical ambiguity, and external/field validation without altering the frozen diagnostic conclusions.",
        "",
        "## Source policy",
        "",
        "Every number above is generated from `paper/figure_data/paper_evidence_summary.json`; consult each metric's `source.path` and `source.json_key` before reuse.",
    ]
    return "\n".join(lines) + "\n"


def _assert_finite(value: Any, path: str = "root") -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError(f"non-finite value at {path}: {value}")
    if isinstance(value, dict):
        for key, child in value.items():
            _assert_finite(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_finite(child, f"{path}[{index}]")


def render_all() -> dict[Path, str]:
    summary = build_evidence_summary()
    _assert_finite(summary)
    return {
        FIGURE_DATA_DIR / "paper_evidence_summary.json": _json_dump(summary),
        FIGURE_DATA_DIR / "paper_evidence_summary.csv": render_evidence_csv(summary),
        TABLE_DIR / "table01_evidence_summary.tex": render_table_latex(summary["table1"]),
        TABLE_DIR / "table01_evidence_summary.md": render_table_markdown(summary["table1"]),
        TABLE_DIR / "table01_evidence_summary.csv": render_table_csv(summary["table1"]),
        FACT_SHEET_PATH: render_fact_sheet(summary),
    }


def generate() -> dict[str, Any]:
    rendered = render_all()
    for path, content in rendered.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8", newline="")
    return {
        "status": "PASS",
        "outputs": [
            {"path": _relative(path), "sha256": _sha256(path), "size_bytes": path.stat().st_size}
            for path in rendered
        ],
        "scientific_inference_rerun": False,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Require checked-in outputs to match a fresh in-memory rendering.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.check:
        drift = []
        for path, expected in render_all().items():
            if not path.is_file() or path.read_text(encoding="utf-8") != expected:
                drift.append(_relative(path))
        if drift:
            raise SystemExit(f"paper evidence outputs drifted: {drift}")
        print({"status": "PASS", "drift": []})
    else:
        print(generate())


if __name__ == "__main__":
    main()
