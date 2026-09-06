#!/usr/bin/env python3
"""Retrospective eight-case VAL integrity and continuous-evidence gate."""

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
from scripts.stage20.acoustic_semantics import stage20_labels_to_acoustic
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, load_codebook, require_config, resolve_project_path, validate_asset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "evidence_v1.json")
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence_fix2")
    parser.add_argument("--full-observation-registry", type=Path, default=ROOT / "observations_fix2/observation_registry_full.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidence_audit_fix2")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage20_continuous_impedance_evidence_v1")
    registry = read_json(args.evidence_dir / "evidence_registry.json")
    full = read_json(args.full_observation_registry)
    val = [case for case in registry["cases"] if case["split"] == "val"]
    full_index = {str(case["case_id"]): case for case in full["cases"]}
    if len(val) != 8:
        raise ValueError("Stage20 evidence audit requires exactly 8 VAL cases")
    property_table, _ = load_codebook(resolve_project_path(cfg["phase4c"]["acoustic_config"]))
    rock_min, rock_max = property_table[0, 1:].min(), property_table[0, 1:].max()
    rows = []; integrity = True
    for case in val:
        case_id = str(case["case_id"]); assets = case["assets"]; obs = case["observation_assets"]
        prior = runtime.load_tensor(validate_asset(assets["low_frequency_acoustic"], f"{case_id}/prior")).float()
        post = runtime.load_tensor(validate_asset(assets["inverted_acoustic"], f"{case_id}/post")).float()
        fields = runtime.load_tensor(validate_asset(assets["seismic_fields_prior_post"], f"{case_id}/fields")).float()
        observed = runtime.load_tensor(validate_asset(obs["observed_seismic"], f"{case_id}/observed")).float()
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        values = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/values")), "values").long()
        truth_record = full_index[case_id]
        truth = normalize_volume(runtime.load_tensor(validate_asset(truth_record["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long()
        truth_acoustic, cleanup = stage20_labels_to_acoustic(truth, support, property_table)
        truth_logz = truth_acoustic[:, 0:1].log()
        free = support & ~condition
        exact, _ = stage20_labels_to_acoustic(values, support, property_table)
        finite = all(bool(torch.isfinite(t).all()) for t in (prior, post, fields))
        condition_violations = int((post != exact)[condition.expand_as(post)].sum())
        slowness_unchanged = torch.equal(post[:, 1:2][free], prior[:, 1:2][free])
        prior_bounds = bool(((prior[:, 0:1][support] >= rock_min) & (prior[:, 0:1][support] <= rock_max)).all())
        post_bounds = bool(((post[:, 0:1][support] >= rock_min) & (post[:, 0:1][support] <= rock_max)).all())
        bounds = prior_bounds and post_bounds
        closure = float(full_index[case_id]["forward_closure_max_abs"])
        case_integrity = finite and condition_violations == 0 and slowness_unchanged and bounds and closure <= 1e-7
        integrity &= case_integrity
        prior_seis = float((fields[0] - observed).square().mean().sqrt())
        post_seis = float((fields[1] - observed).square().mean().sqrt())
        prior_logz = float((prior[:, 0:1].log()[free] - truth_logz[free]).square().mean().sqrt())
        post_logz = float((post[:, 0:1].log()[free] - truth_logz[free]).square().mean().sqrt())
        rows.append({"case_id": case_id, "integrity_passed": case_integrity, "forward_closure_max_abs": closure, "condition_acoustic_violations": condition_violations, "free_slowness_unchanged": slowness_unchanged, "low_frequency_impedance_in_bounds": prior_bounds, "inverted_impedance_in_bounds": post_bounds, "impedance_in_bounds": bounds, "rmse_seismic_prior": prior_seis, "rmse_seismic_post": post_seis, "delta_seismic": post_seis - prior_seis, "rmse_logz_prior": prior_logz, "rmse_logz_post": post_logz, "delta_logz": post_logz - prior_logz})
    seismic_improved = sum(row["delta_seismic"] < 0 for row in rows); logz_improved = sum(row["delta_logz"] < 0 for row in rows)
    median_seismic = float(statistics.median(row["delta_seismic"] for row in rows)); median_logz = float(statistics.median(row["delta_logz"] for row in rows))
    gate = cfg["gate"]
    passed = integrity and seismic_improved >= int(gate["min_seismic_improved_cases"]) and median_seismic < 0 and logz_improved >= int(gate["min_logz_improved_cases"]) and median_logz < 0
    decision = "CONTINUOUS_IMPEDANCE_EVIDENCE_VALIDATED" if passed else "STOP_CONTINUOUS_IMPEDANCE_EVIDENCE_NOT_VALIDATED"
    summary = {"schema": "stage20_evidence_audit_v1", "run_status": "completed", "case_count": 8, "integrity_passed": integrity, "seismic_improved_cases": seismic_improved, "median_delta_seismic_rmse": median_seismic, "logz_improved_cases": logz_improved, "median_delta_logz_rmse": median_logz, "gate_passed": passed, "machine_decision": decision, "adapter_training_authorized": passed, "parameters_tuned": False}
    args.output_dir.mkdir(parents=True); write_csv(args.output_dir / "per_case_metrics.csv", rows); write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(f"# Stage20 VAL continuous evidence audit\n\nDecision: **{decision}**\n\nSeismic improved: {seismic_improved}/8 (median delta {median_seismic:+.8f}). LogZ improved: {logz_improved}/8 (median delta {median_logz:+.8f}). Integrity: {integrity}.\n", encoding="utf-8")
    if not integrity:
        raise RuntimeError("STOP_CONTINUOUS_EVIDENCE_INTEGRITY")
    if not passed:
        raise RuntimeError("STOP_CONTINUOUS_IMPEDANCE_EVIDENCE_NOT_VALIDATED")


if __name__ == "__main__":
    main()
