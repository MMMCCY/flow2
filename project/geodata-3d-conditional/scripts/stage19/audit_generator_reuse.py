#!/usr/bin/env python3
"""Validate exact StructuralGeo generator/conditioning reuse for Stage19R."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for root in (PROJECT_DIR, REPOSITORY_ROOT, STRUCTURALGEO_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.full_structuralgeo_benchmark import canonical_tensor_sha256, prepare_candidate
from scripts.stage15.common import read_json, refuse_nonempty, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, require_config, validate_asset, validate_stage19_cohort_contract

DEFAULT_CONFIG = CONFIG_DIR / "cohort_v2.json"
DEFAULT_OUTPUT = ROOT / "generator_reuse_audit/v1"

CRITICAL_SOURCES = (
    "StructuralGeo-main/src/geogen/dataset/dataset.py",
    "StructuralGeo-main/src/geogen/generation/categorical_events.py",
    "StructuralGeo-main/src/geogen/generation/geowords.py",
    "StructuralGeo-main/src/geogen/generation/model_generators.py",
    "StructuralGeo-main/src/geogen/generation/rng_contract.py",
    "StructuralGeo-main/src/geogen/model/geomodel.py",
    "StructuralGeo-main/src/geogen/model/metaballs.py",
    "StructuralGeo-main/src/geogen/probability/random_varibles.py",
    "StructuralGeo-main/src/geogen/probability/sedimentbuilders.py",
    "StructuralGeo-main/src/geogen/probability/wavegenerators.py",
    "StructuralGeo-main/src/geogen/generation/markov_matrix/default_markov_matrix.csv",
    "project/geodata-3d-conditional/boreholes.py",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def run_audit(config_path: Path) -> dict[str, object]:
    config = require_config(config_path, "stage19_cohort_v2")
    reuse = config["authoritative_reuse"]
    reference_config_path = validate_asset(reuse["benchmark_config"], "benchmark config")
    reference_manifest_path = validate_asset(reuse["benchmark_manifest"], "benchmark manifest")
    case_manifest_path = validate_asset(reuse["historical_case01_manifest"], "historical case01 manifest")
    reference_config = read_json(reference_config_path)
    validate_stage19_cohort_contract(config, reference_config)
    benchmark = read_json(reference_manifest_path)
    source_rows = []
    for relative in CRITICAL_SOURCES:
        expected = str(benchmark["source_hashes"][relative])
        actual = runtime.file_sha256(REPOSITORY_ROOT / relative)
        source_rows.append({"path": relative, "expected_sha256": expected, "actual_sha256": actual, "match": actual == expected})
    if not all(row["match"] for row in source_rows):
        raise RuntimeError("STOP_GENERATOR_REUSE_MISMATCH: generator-critical source hash")
    historical = read_json(case_manifest_path)
    root_seed = int(reuse["replay_root_seed"])
    candidate = prepare_candidate(root_seed)
    replay = {
        "root_seed": root_seed,
        "eligible": bool(candidate["eligible"]),
        "truth_tensor_sha256": canonical_tensor_sha256(candidate["truth"]),
        "condition_values_tensor_sha256": canonical_tensor_sha256(candidate["conditions"]["condition_values"]),
        "condition_mask_tensor_sha256": canonical_tensor_sha256(candidate["conditions"]["condition_mask"]),
        "markov_sequence": candidate["metadata"]["markov_sequence"],
    }
    expected_replay = {
        "root_seed": int(historical["root_seed"]),
        "eligible": True,
        "truth_tensor_sha256": historical["tensor_content_hashes"]["truth/true_model.pt"],
        "condition_values_tensor_sha256": historical["tensor_content_hashes"]["condition/condition_values.pt"],
        "condition_mask_tensor_sha256": historical["tensor_content_hashes"]["condition/condition_mask.pt"],
        "markov_sequence": historical["markov_state_sequence"],
    }
    if replay != expected_replay:
        raise RuntimeError("STOP_GENERATOR_REUSE_MISMATCH: historical deterministic replay")
    return {
        "schema": "stage19r_generator_reuse_audit_v1",
        "run_status": "completed",
        "machine_decision": "GENERATOR_REUSE_VALIDATED",
        "generator_critical_sources": source_rows,
        "historical_replay": replay,
        "historical_replay_expected": expected_replay,
        "cohort_contract_exact": True,
        "reference_assets": {
            "benchmark_config": runtime.asset_record(reference_config_path),
            "benchmark_manifest": runtime.asset_record(reference_manifest_path),
            "historical_case01_manifest": runtime.asset_record(case_manifest_path),
        },
    }


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    args.output_dir.mkdir(parents=True)
    try:
        result = run_audit(args.config)
        write_json(args.output_dir / "summary.json", result)
        print(result["machine_decision"])
    except Exception as exc:
        write_json(args.output_dir / "summary.json", {"schema": "stage19r_generator_reuse_audit_v1", "run_status": "failed", "machine_decision": "STOP_GENERATOR_REUSE_MISMATCH", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
