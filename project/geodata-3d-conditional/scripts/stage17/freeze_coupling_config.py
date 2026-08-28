#!/usr/bin/env python3
"""Freeze Stage17C only after validating the Stage17A and Stage17B gates."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.stage15.common import read_json, write_json


ROOT = PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution"
DEFAULT_EVIDENCE_RUN = ROOT / "stage17a/formal_all5_v1"
DEFAULT_CALIBRATION_RUN = ROOT / "stage17b/formal_all3_v1"
DEFAULT_OUTPUT = ROOT / "configs/coupling_v1.json"
CHECKPOINT = PROJECT_DIR / "demo_model/conditional-weights.ckpt"
PROPERTY_CONFIG = ROOT.parents[0] / "stage15_binary_seismic_consensus/configs/binary_trace_property_indicator_v1.json"
REFERENCE_CONFIG = ROOT.parents[0] / "stage15_binary_seismic_consensus/trace_boundary/flow_property_seed42_v4/guided/config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-run", type=Path, default=DEFAULT_EVIDENCE_RUN)
    parser.add_argument("--calibration-run", type=Path, default=DEFAULT_CALIBRATION_RUN)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen coupling config: {args.output}")
    evidence_summary_path = args.evidence_run / "evaluation/summary.json"
    calibration_summary_path = args.calibration_run / "evaluation/summary.json"
    evidence_summary = read_json(evidence_summary_path)
    calibration_summary = read_json(calibration_summary_path)
    if evidence_summary.get("machine_decision") != "EVIDENCE_CASE_SPECIFIC":
        raise ValueError("Stage17A gate does not authorize Stage17C")
    dflow_authorized = calibration_summary.get("decision") == "DFLOW_HARD_CONTROL_MATERIAL"
    if dflow_authorized:
        raise ValueError("coupling_v1 is the trajectory-only protocol for the observed weak D-Flow gate")
    evidence_config = read_json(args.evidence_run / "config.json")
    reference = read_json(REFERENCE_CONFIG)
    cases = []
    for case in evidence_config["cases"]:
        case_id = str(case["case_id"])
        score_path = args.evidence_run / case_id / "binary_impedance_score.pt"
        case_manifest_path = args.evidence_run / case_id / "manifest.json"
        cases.append(
            {
                "case_id": case_id,
                "root_seed": int(case["root_seed"]),
                "evidence": runtime.asset_record(score_path),
                "evidence_case_manifest": runtime.asset_record(case_manifest_path),
                "observation_assets": case["assets"],
            }
        )
    config = {
        "schema": "stage17_same_evidence_coupling_v1",
        "status": "frozen_before_run",
        "authorized_arms": ["FLOW_ONLY", "TRAJECTORY_EVIDENCE"],
        "dflow_arm_authorized": False,
        "dflow_stop_reason": "Stage17B property task failed the independent progression gate",
        "source_seeds": [42, 142, 242],
        "primary_statistical_unit": "independent_geology_case",
        "source_seed_role": "within_case_stochastic_replicate",
        "pooled_15_pair_role": "auxiliary_only",
        "stage17a_gate": runtime.asset_record(evidence_summary_path),
        "stage17b_gate": runtime.asset_record(calibration_summary_path),
        "checkpoint": runtime.asset_record(CHECKPOINT),
        "property_config": runtime.asset_record(PROPERTY_CONFIG),
        "authoritative_stage15h_reference_config": runtime.asset_record(REFERENCE_CONFIG),
        "target_label": 9,
        "solver": "fixed_euler",
        "n_steps": 32,
        "device": "cuda",
        "guidance": {
            "alpha": float(reference["alpha"]),
            "max_guidance_ratio": float(reference["max_guidance_ratio"]),
            "tau_start": float(reference["tau_start"]),
            "tau_end": float(reference["tau_end"]),
            "tau_schedule": reference["tau_schedule"],
            "guidance_start": float(reference["guidance_start"]),
            "guidance_schedule": reference["guidance_schedule"],
            "grad_clip_norm": float(reference["grad_clip_norm"]),
            "guidance_scaling_mode": reference["guidance_scaling_mode"],
            "property_sigmas": reference["property_sigmas"],
            "property_scale_weights": reference["property_scale_weights"],
            "property_loss_mode": reference["property_loss_mode"]
        },
        "decision_rule": {
            "positive_case_count_required": 3,
            "positive_case_metric": "case_median_delta_target_iou_gt_zero",
            "group_median_delta_target_iou_required_gt": 0.0
        },
        "truth_loaded_by_runner": False,
        "sample_selection_performed": False,
        "parameter_sweep": False,
        "training_performed": False,
        "cases": cases
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, config)
    print(f"frozen Stage17C config: {args.output}")
    print(f"sha256: {runtime.file_sha256(args.output)}")


if __name__ == "__main__":
    main()
