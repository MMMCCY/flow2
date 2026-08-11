#!/usr/bin/env python3
"""Read-only integrity and stop-policy verification for completed Stage12B-A."""

from __future__ import annotations

from pathlib import Path
import sys


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.prior_ensemble import file_sha256
from scripts.stage12b.common import (
    CASE_IDS,
    EXPERIMENT_DIR,
    load_bridge_case,
    load_config,
    load_inference_case,
    load_prior_bank,
    verify_stage12a_files,
)
from scripts.stage9.common import load_tensor_record, read_json


def main() -> None:
    config = load_config()
    verify_stage12a_files(config)
    load_prior_bank()
    for case_id in CASE_IDS:
        load_inference_case(case_id)
        load_bridge_case(case_id)
    controls_root = EXPERIMENT_DIR / "controls"
    controls = read_json(controls_root / "manifest.json")
    if controls.get("status") != "complete_frozen_before_truth_evaluation":
        raise ValueError("controls were not frozen before truth evaluation")
    if controls.get("truth_tensor_received") is not False:
        raise ValueError("controls received truth")
    for records in controls["tensors"].values():
        for name in (
            "constant_prior_label9",
            "shuffled_xy_all_classes",
            "shuffled_xy_label9",
            "xy_permutation",
        ):
            load_tensor_record(controls_root, records[name])
    decision = read_json(EXPERIMENT_DIR / "reports/STAGE12B_A_MACHINE_DECISION.json")
    if decision.get("machine_decision") != "STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC":
        raise ValueError("unexpected Stage12B-A decision")
    if decision.get("stage12b_b_authorized") is not False:
        raise ValueError("Stage12B-B was incorrectly authorized")
    summary = decision["summary"]
    summary_path = REPOSITORY_ROOT / summary["path"]
    if file_sha256(summary_path) != summary["sha256"]:
        raise ValueError("Stage12B-A summary hash mismatch")
    scope = read_json(EXPERIMENT_DIR / "audit/execution_scope.json")
    if scope.get("stage12b_flow_forwards") != 0 or scope.get("stage12b_b_executed") is not False:
        raise ValueError("Stage12B stop policy was violated")
    forbidden_outputs = (
        EXPERIMENT_DIR / "stage12b_b",
        EXPERIMENT_DIR / "flow",
        EXPERIMENT_DIR / "paired_flow_pilot",
    )
    if any(path.exists() for path in forbidden_outputs):
        raise ValueError("unauthorized Stage12B-B output exists")
    print("Stage12B-A outputs and stop policy verified")


if __name__ == "__main__":
    main()
