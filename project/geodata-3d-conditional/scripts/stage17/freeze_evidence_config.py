#!/usr/bin/env python3
"""Freeze Stage17A inversion inputs only after observation generation completes."""

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
from scripts.stage17.common import require_frozen_config, validate_asset_record


ROOT = PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution"
DEFAULT_OBSERVATIONS = ROOT / "cases/observations_v2"
DEFAULT_OUTPUT = ROOT / "configs/evidence_multicase_v1.json"
OBSERVATION_CONFIG = ROOT / "configs/observation_generation_v2.json"
INVERSION_CONFIG = (
    PROJECT_DIR
    / "experiments/stage15_binary_seismic_consensus/configs/binary_trace_boundary_inversion_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output.exists():
        raise FileExistsError(f"refusing to overwrite frozen evidence config: {args.output}")
    generation = read_json(OBSERVATION_CONFIG)
    require_frozen_config(generation, "stage17_observation_generation_v2")
    master_path = args.observation_dir / "manifest.json"
    master = read_json(master_path)
    if master.get("run_status") != "completed" or int(master.get("case_count", 0)) != 5:
        raise ValueError("five-case observation generation is not complete")
    expected_order = [str(record["case_id"]) for record in master["cases"]]
    cases: list[dict[str, object]] = []
    for case_id in expected_order:
        case_dir = args.observation_dir / case_id
        manifest_path = case_dir / "manifest.json"
        manifest = read_json(manifest_path)
        if manifest.get("run_status") != "completed":
            raise ValueError(f"observation case is incomplete: {case_id}")
        if manifest.get("truth_loaded_by_inversion_runner") is not False:
            raise ValueError(f"truth-firewall declaration failed: {case_id}")
        assets = {
            name: runtime.asset_record(case_dir / filename)
            for name, filename in {
                "observation_manifest": "manifest.json",
                "observed_seismic": "observed_seismic.pt",
                "subsurface_mask": "subsurface_mask.pt",
                "binary_fixed_values": "binary_fixed_values.pt",
                "binary_fixed_mask": "binary_fixed_mask.pt",
                "condition_values": "condition_values.pt",
                "condition_mask": "condition_mask.pt",
            }.items()
        }
        for name, record in assets.items():
            validate_asset_record(record, f"{case_id}/{name}")
        cases.append(
            {
                "case_id": case_id,
                "root_seed": int(manifest["root_seed"]),
                "assets": assets,
            }
        )
    inversion = read_json(INVERSION_CONFIG)
    config = {
        "schema": "stage17_evidence_multicase_v1",
        "status": "frozen_before_run",
        "freeze_order": "observation_config_then_generated_hashed_observations_then_evidence_config",
        "observation_generation_config": runtime.asset_record(OBSERVATION_CONFIG),
        "completed_observation_manifest": runtime.asset_record(master_path),
        "binary_acoustic_config": generation["binary_acoustic_config"],
        "seismic_config": generation["seismic_config"],
        "inversion_config": runtime.asset_record(INVERSION_CONFIG),
        "inversion_parameters": {
            "refinement_passes": int(inversion["refinement_passes"]),
            "prior_relative_weight": float(inversion["prior_relative_weight"]),
            "vertical_smoothness_relative_weight": float(inversion["vertical_smoothness_relative_weight"]),
        },
        "target_label": int(generation["target_label"]),
        "full_vertical_trace_used": True,
        "trace_samples": int(generation["full_trace_samples"]),
        "lateral_filter_used": False,
        "truth_loaded_by_runner": False,
        "truth_role": "not_available_to_inversion_code_retrospective_evaluator_only",
        "cases": cases,
        "parameter_sweep": False,
        "training_performed": False,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    write_json(args.output, config)
    print(f"frozen Stage17 evidence config: {args.output}")
    print(f"sha256: {runtime.file_sha256(args.output)}")


if __name__ == "__main__":
    main()
