#!/usr/bin/env python3
"""Truth-blind Stage20 impedance-bound diagnostic for the frozen VAL failures."""

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
from scripts.stage15.common import normalize_volume, read_json, refuse_nonempty, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, load_codebook, require_config, resolve_project_path, validate_asset

CASES = ("stage19_val_case001", "stage19_val_case005")


def _stats(impedance: torch.Tensor, support: torch.Tensor, condition: torch.Tensor, zmin: float, zmax: float) -> dict[str, object]:
    values = impedance[:, 0:1]
    dtype = values.dtype
    tolerance = 8.0 * torch.finfo(dtype).eps * max(abs(zmin), abs(zmax))
    below = support & (values < zmin); above = support & (values > zmax); outside = below | above
    magnitude = torch.maximum((zmin - values).clamp_min(0), (values - zmax).clamp_min(0))
    free = support & ~condition
    return {
        "dtype": str(dtype), "zmin": zmin, "zmax": zmax, "tolerance": tolerance,
        "subsurface_min": float(values[support].min()), "subsurface_max": float(values[support].max()),
        "below_count": int(below.sum()), "above_count": int(above.sum()), "out_of_bounds_count": int(outside.sum()),
        "maximum_out_of_bounds_magnitude": float(magnitude[support].max()),
        "condition_out_of_bounds_count": int((outside & condition).sum()),
        "free_subsurface_out_of_bounds_count": int((outside & free).sum()),
        "maximum_condition_out_of_bounds_magnitude": float(magnitude[condition & support].max()) if bool((condition & support).any()) else 0.0,
        "maximum_free_subsurface_out_of_bounds_magnitude": float(magnitude[free].max()) if bool(free.any()) else 0.0,
        "numerical_boundary_reconstruction_only": float(magnitude[support].max()) <= tolerance,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence_fix1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "evidence_bound_diagnostic")
    args = parser.parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(CONFIG_DIR / "evidence_v1.json", "stage20_continuous_impedance_evidence_v1")
    property_table, _ = load_codebook(resolve_project_path(cfg["phase4c"]["acoustic_config"]))
    zmin, zmax = float(property_table[0, 1:].min()), float(property_table[0, 1:].max())
    registry = read_json(args.evidence_dir / "evidence_registry.json"); index = {str(case["case_id"]): case for case in registry["cases"]}
    rows = []
    for case_id in CASES:
        case = index[case_id]; assets = case["assets"]; obs = case["observation_assets"]
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        prior = runtime.load_tensor(validate_asset(assets["low_frequency_acoustic"], f"{case_id}/prior"))
        post = runtime.load_tensor(validate_asset(assets["inverted_acoustic"], f"{case_id}/post"))
        rows.append({"case_id": case_id, "low_frequency_prior": _stats(prior, support, condition, zmin, zmax), "inverted": _stats(post, support, condition, zmin, zmax)})
    material = any(not row[name]["numerical_boundary_reconstruction_only"] for row in rows for name in ("low_frequency_prior", "inverted"))
    decision = "STOP_MATERIAL_IMPEDANCE_BOUND_VIOLATION" if material else "NUMERICAL_BOUNDARY_RECONSTRUCTION_CONFIRMED"
    result = {"schema": "stage20_truth_blind_impedance_bound_diagnostic_v1", "run_status": "completed", "truth_loaded": False, "case_ids": list(CASES), "rows": rows, "material_violation": material, "machine_decision": decision}
    args.output_dir.mkdir(parents=True); write_json(args.output_dir / "summary.json", result); print(decision)
    if material:
        raise RuntimeError("STOP_MATERIAL_IMPEDANCE_BOUND_VIOLATION")


if __name__ == "__main__":
    main()
