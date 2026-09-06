#!/usr/bin/env python3
"""Truth-blind 84-case CPU semantic preflight before Stage20 evidence v2."""

from __future__ import annotations

import argparse
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
from scripts.stage20.acoustic_semantics import enforce_stage20_acoustic_contract, stage20_labels_to_acoustic
from scripts.stage20.common import CONFIG_DIR, ROOT, assert_truth_firewall, build_low_frequency_acoustic, load_codebook, require_config, resolve_project_path, validate_asset, validate_wells


def validate_acoustic_preflight_tensor(acoustic: torch.Tensor, support: torch.Tensor, property_table: torch.Tensor) -> None:
    """Public focused-test seam for the strict preflight acoustic contract."""
    enforce_stage20_acoustic_contract(acoustic, support, property_table, context="Stage20 semantic preflight")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-registry", type=Path, default=ROOT / "inversion_input_registry_fix2/registry.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "acoustic_semantic_preflight")
    args = parser.parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(CONFIG_DIR / "evidence_v1.json", "stage20_continuous_impedance_evidence_v1")
    registry = read_json(args.input_registry); assert_truth_firewall(registry, "semantic preflight registry")
    if registry.get("schema") != "stage20_inversion_input_registry_v1" or registry.get("case_count") != 84:
        raise ValueError("semantic preflight requires the stripped 84-case registry")
    property_table, _ = load_codebook(resolve_project_path(cfg["phase4c"]["acoustic_config"]))
    zmin, zmax = property_table[0, 1:].min(), property_table[0, 1:].max(); air = property_table[:, 0]
    rows = []; passed = True
    for case in registry["cases"]:
        case_id = str(case["case_id"]); assets = case["assets"]; validate_wells(case["well_xy"])
        values = normalize_volume(runtime.load_tensor(validate_asset(assets["condition_values"], f"{case_id}/values")), "values").long()
        condition = normalize_volume(runtime.load_tensor(validate_asset(assets["condition_mask"], f"{case_id}/condition")), "condition").bool()
        support = normalize_volume(runtime.load_tensor(validate_asset(assets["subsurface_mask"], f"{case_id}/support")), "support").bool()
        labels_valid = int(values.min()) >= -1 and int(values.max()) <= 13
        acoustic, mapper = stage20_labels_to_acoustic(values, support, property_table)
        prior, fallback = build_low_frequency_acoustic(values, condition, support, case["well_xy"], property_table, power=2.0)
        validate_acoustic_preflight_tensor(acoustic, support, property_table)
        validate_acoustic_preflight_tensor(prior, support, property_table)
        impedance = acoustic[:, 0:1]; slowness = acoustic[:, 1:2]
        outside_air = (~support) & (values == -1)
        outside_air_exact = not bool(outside_air.any()) or bool((acoustic.permute(0, 2, 3, 4, 1)[outside_air[:, 0]] == air).all())
        subsurface_bounds = bool(((impedance[support] >= zmin) & (impedance[support] <= zmax)).all())
        no_air_impedance = not bool((impedance[support] == air[0]).any())
        condition_bounds = bool(((impedance[support & condition] >= zmin) & (impedance[support & condition] <= zmax)).all()) if bool((support & condition).any()) else True
        finite_positive = bool(torch.isfinite(acoustic).all() and torch.isfinite(prior).all() and (slowness > 0).all() and (prior[:, 1:2] > 0).all())
        case_pass = labels_valid and outside_air_exact and subsurface_bounds and no_air_impedance and condition_bounds and finite_positive
        passed &= case_pass
        rows.append({"case_id": case_id, "split": case["split"], "passed": case_pass, "subsurface_raw_minus1_count": int((support & (values == -1)).sum()), "condition_subsurface_raw_minus1_count": int((support & condition & (values == -1)).sum()), "neutral_replacement_count": mapper["underground_air_voxels_replaced"], "lf_fallback_depth_count": fallback["fallback_depth_count"], "lf_max_fallback_distance": max((row["distance"] for row in fallback["fallback_map"]), default=0), "acoustic_impedance_min": float(impedance.min()), "acoustic_impedance_max": float(impedance.max()), "subsurface_acoustic_impedance_min": float(impedance[support].min()), "subsurface_acoustic_impedance_max": float(impedance[support].max()), "labels_valid": labels_valid, "outside_air_exact": outside_air_exact, "subsurface_impedance_in_bounds": subsurface_bounds, "subsurface_air_impedance_absent": no_air_impedance, "condition_subsurface_impedance_in_bounds": condition_bounds, "finite_positive_slowness": finite_positive})
    decision = "STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT_VALIDATED" if passed and len(rows) == 84 else "STOP_STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT"
    result = {"schema": "stage20_acoustic_semantic_preflight_v1", "run_status": "completed", "case_count": len(rows), "truth_loaded": False, "flow_loaded": False, "adapter_loaded": False, "selection_performed": False, "parameter_sweep": False, "all_cases_passed": passed, "machine_decision": decision, "input_registry": runtime.asset_record(args.input_registry.resolve()), "rows": rows}
    args.output_dir.mkdir(parents=True); write_csv(args.output_dir / "per_case.csv", rows); write_json(args.output_dir / "summary.json", result); print(decision)
    if decision != "STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT_VALIDATED":
        raise RuntimeError("STOP_STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT")


if __name__ == "__main__":
    main()
