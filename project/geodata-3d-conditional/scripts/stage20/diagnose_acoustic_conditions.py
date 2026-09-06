#!/usr/bin/env python3
"""Truth-blind validation of the Stage20 condition/acoustic mismatch hypothesis."""

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
from scripts.stage20.common import CONFIG_DIR, ROOT, canonical_tensor_sha256, load_codebook, require_config, resolve_project_path, validate_asset


def condition_subsurface_raw_air_mask(
    values: torch.Tensor, condition: torch.Tensor, support: torch.Tensor
) -> torch.Tensor:
    """Truth-blind geological mask used by both diagnostic and focused tests."""
    if values.shape != condition.shape or values.shape != support.shape:
        raise ValueError("diagnostic tensors must have matching shapes")
    return support.bool() & condition.bool() & (values == -1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--evidence-dir", type=Path, default=ROOT / "evidence_fix1")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "acoustic_condition_diagnostic")
    args = parser.parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(CONFIG_DIR / "evidence_v1.json", "stage20_continuous_impedance_evidence_v1")
    property_table, _ = load_codebook(resolve_project_path(cfg["phase4c"]["acoustic_config"]))
    zmin = float(property_table[0, 1:].min()); zmax = float(property_table[0, 1:].max())
    registry = read_json(args.evidence_dir / "evidence_registry.json")
    rows = []
    for case in (item for item in registry["cases"] if item["split"] == "val"):
        case_id = str(case["case_id"]); obs = case["observation_assets"]
        support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
        condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition")), "condition").bool()
        values = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/values")), "values").long()
        prior = runtime.load_tensor(validate_asset(case["assets"]["low_frequency_acoustic"], f"{case_id}/prior"))[:, 0:1]
        post = runtime.load_tensor(validate_asset(case["assets"]["inverted_acoustic"], f"{case_id}/post"))[:, 0:1]
        bad = condition_subsurface_raw_air_mask(values, condition, support)
        prior_bad = support & ((prior < zmin) | (prior > zmax))
        post_bad = support & ((post < zmin) | (post > zmax))
        coords = torch.nonzero(bad, as_tuple=False).to(torch.int64).contiguous()
        rows.append({"case_id": case_id, "bad_count": int(bad.sum()), "prior_out_of_bounds_count": int(prior_bad.sum()), "inverted_out_of_bounds_count": int(post_bad.sum()), "bad_equals_prior_out_of_bounds": torch.equal(bad, prior_bad), "bad_equals_inverted_out_of_bounds": torch.equal(bad, post_bad), "coordinates": coords.tolist(), "coordinates_sha256": canonical_tensor_sha256(coords)})
    expected = {"stage19_val_case001": 15, "stage19_val_case005": 58}
    matches = all(row["bad_count"] == expected.get(row["case_id"], 0) and row["bad_equals_prior_out_of_bounds"] and row["bad_equals_inverted_out_of_bounds"] for row in rows)
    decision = "STAGE20_ACOUSTIC_CONDITION_DIAGNOSIS_VALIDATED" if matches else "STOP_STAGE20_ACOUSTIC_CONDITION_DIAGNOSIS_MISMATCH"
    result = {"schema": "stage20_acoustic_condition_diagnostic_v1", "run_status": "completed", "truth_loaded": False, "target_metrics_used": False, "rock_impedance_bounds": [zmin, zmax], "rows": rows, "expected_nonzero_counts": expected, "machine_decision": decision}
    args.output_dir.mkdir(parents=True); write_json(args.output_dir / "summary.json", result); print(decision)
    if not matches:
        raise RuntimeError("STOP_STAGE20_ACOUSTIC_CONDITION_DIAGNOSIS_MISMATCH")


if __name__ == "__main__":
    main()
