#!/usr/bin/env python3
"""Retrospective VAL-only Stage19 evidence reuse and specificity gate."""

from __future__ import annotations

import argparse
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
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import ap_skill, average_precision
from scripts.stage19.common import CONFIG_DIR, ROOT, require_config, validate_asset

DEFAULT_CONFIG = CONFIG_DIR / "evidence_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_OUTPUT = ROOT / "evidence_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = require_config(args.config, "stage19_evidence_v1")
    registry = read_json(args.evidence_dir / "evidence_registry.json")
    val = [case for case in registry["cases"] if case["split"] == "val"]
    if len(val) != 8:
        raise ValueError("evidence audit requires exactly 8 VAL cases")
    scores, truths, masks = {}, {}, {}
    rows = []
    for case in val:
        case_id = str(case["case_id"])
        score = normalize_volume(runtime.load_tensor(validate_asset(case["evidence"], f"{case_id}/evidence")), "score", torch.float32)
        truth = normalize_volume(runtime.load_tensor(validate_asset(case["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long()
        support = normalize_volume(runtime.load_tensor(validate_asset(case["observation_assets"]["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(case["observation_assets"]["condition_mask"], f"{case_id}/condition")), "condition").bool()
        target, mask = truth == int(config["target_label"]), support & (truth != -1) & ~condition
        prevalence = float(target[mask].float().mean())
        ap = average_precision(score[mask], target[mask])
        rows.append({"case_id": case_id, "voxel_auprc": ap, "prevalence": prevalence, "ap_skill": ap_skill(ap, prevalence), "evaluation_voxels": int(mask.sum())})
        scores[case_id], truths[case_id], masks[case_id] = score, target, mask
    common = torch.ones_like(next(iter(masks.values())))
    for mask in masks.values():
        common &= mask
    specificity = []
    ids = list(scores)
    for evidence_id in ids:
        values = {}
        for truth_id in ids:
            prevalence = float(truths[truth_id][common].float().mean())
            values[truth_id] = ap_skill(average_precision(scores[evidence_id][common], truths[truth_id][common]), prevalence)
        diagonal = values[evidence_id]
        off = statistics.median(value for key, value in values.items() if key != evidence_id)
        specificity.append({"case_id": evidence_id, "diagonal_ap_skill": diagonal, "off_diagonal_median_ap_skill": off, "delta_ap_skill": diagonal - off})
    gate = config["reuse_gate"]
    positive_ap = sum(float(row["ap_skill"]) > 0 for row in rows)
    median_ap = float(statistics.median(float(row["ap_skill"]) for row in rows))
    specific = sum(float(row["delta_ap_skill"]) > 0 for row in specificity)
    passed = positive_ap >= int(gate["minimum_positive_ap_skill_cases"]) and median_ap >= float(gate["minimum_median_ap_skill"]) and specific >= int(gate["minimum_specificity_cases"])
    summary = {"schema": "stage19_evidence_reuse_audit_v1", "run_status": "completed", "case_count": 8, "positive_ap_skill_case_count": positive_ap, "median_ap_skill": median_ap, "specificity_case_count": specific, "common_mask_voxels": int(common.sum()), "gate_passed": passed, "machine_decision": "EVIDENCE_REUSE_VALIDATED" if passed else "EVIDENCE_REUSE_NOT_VALIDATED", "adapter_training_authorized": passed, "parameters_tuned": False}
    args.output_dir.mkdir(parents=True)
    write_csv(args.output_dir / "per_case_evidence_metrics.csv", rows)
    write_csv(args.output_dir / "specificity_deltas.csv", specificity)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(f"# Stage19 VAL evidence reuse audit\n\nDecision: **{summary['machine_decision']}**\n\nPositive AP skill: {positive_ap}/8; median AP skill: {median_ap:.6f}; correct-case specificity: {specific}/8.\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("STOP_BEFORE_ADAPTER_TRAINING: evidence reuse gate failed")


if __name__ == "__main__":
    main()
