#!/usr/bin/env python3
"""Reuse Stage19R TRAIN/VAL and generate the independent Stage20 TEST cohort."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT, REPOSITORY_ROOT / "StructuralGeo-main/src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from guidance.full_structuralgeo_benchmark import CANONICAL_NINE_WELL_XY, canonical_json_sha256, canonical_tensor_sha256, prepare_candidate
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, asset, require_config, resolve_project_path, validate_wells

DEFAULT_OUTPUT = ROOT / "cohort"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "cohort_v1.json")
    parser.add_argument("--reuse-audit", type=Path, default=ROOT / "reuse_audit/summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _save_test_case(output: Path, index: int, candidate: dict[str, object]) -> dict[str, object]:
    case_id = f"stage20_test_case{index:03d}"
    case_dir = output / "cases" / case_id
    truth_dir, condition_dir = case_dir / "truth", case_dir / "condition"
    truth_dir.mkdir(parents=True); condition_dir.mkdir()
    truth = candidate["truth"].cpu(); conditions = candidate["conditions"]
    tensors = {
        truth_dir / "true_model.pt": truth,
        condition_dir / "condition_values.pt": conditions["condition_values"].cpu(),
        condition_dir / "condition_mask.pt": conditions["condition_mask"].cpu(),
        condition_dir / "surface_mask.pt": conditions["surface_mask"].cpu(),
        condition_dir / "borehole_mask.pt": conditions["borehole_mask"].cpu(),
    }
    for path, tensor in tensors.items():
        torch.save(tensor, path)
    history = {"schema": "stage20_structuralgeo_history_v1", "root_seed": int(candidate["root_seed"]), "markov_sequence": candidate["metadata"]["markov_sequence"], "event_subtypes": candidate["event_subtypes"], "events": candidate["metadata"]["events"], "packed_history": candidate["metadata"]["packed_history"], "unpacked_history": candidate["metadata"]["unpacked_history"]}
    write_json(truth_dir / "history.json", history)
    manifest = {
        "schema": "stage20_structuralgeo_case_v1", "run_status": "completed", "case_id": case_id,
        "split": "test", "root_seed": int(candidate["root_seed"]), "reused_from_stage19r": False,
        "markov_sequence": candidate["metadata"]["markov_sequence"], "event_subtypes": candidate["event_subtypes"],
        "history_sha256": canonical_json_sha256(history), "truth_tensor_sha256": canonical_tensor_sha256(truth),
        "condition_values_tensor_sha256": canonical_tensor_sha256(conditions["condition_values"]),
        "condition_mask_tensor_sha256": canonical_tensor_sha256(conditions["condition_mask"]),
        "well_xy": [list(point) for point in CANONICAL_NINE_WELL_XY],
        "selection_firewall": {"seismic": False, "evidence": False, "flow": False, "downstream_metrics": False},
        "assets": {"truth": asset(truth_dir / "true_model.pt"), "condition_values": asset(condition_dir / "condition_values.pt"), "condition_mask": asset(condition_dir / "condition_mask.pt"), "history": asset(truth_dir / "history.json")},
    }
    validate_wells(manifest["well_xy"]); write_json(case_dir / "manifest.json", manifest)
    return {**manifest, "manifest": asset(case_dir / "manifest.json")}


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage20_cohort_v1")
    audit = read_json(args.reuse_audit)
    if audit.get("run_status") != "completed" or audit.get("machine_decision") != "STAGE20_REUSE_VALIDATED":
        raise RuntimeError("STOP_STAGE20_REUSE_MISMATCH")
    args.output_dir.mkdir(parents=True)
    master = base_manifest("stage20_cohort_manifest_v1", Path(__file__), args.config)
    master.update({"run_status": "running", "selection_uses_seismic_evidence_flow_or_downstream_metrics": False})
    write_json(args.output_dir / "cohort_manifest.json", master)
    registries = {}
    try:
        for split in ("train", "val"):
            source_path = resolve_project_path(cfg[split]["reuse_registry"]); source = read_json(source_path)
            cases = []
            for case in source["cases"]:
                validate_wells(case["well_xy"])
                cases.append({**case, "reused_from_stage19r": True, "source_registry": asset(source_path)})
            registry = {"schema": "stage20_split_registry_v1", "run_status": "completed", "split": split, "accepted_count": len(cases), "reused_from_stage19r": True, "source_registry": asset(source_path), "cases": cases}
            path = args.output_dir / f"{split}_registry.json"; write_json(path, registry); registries[split] = asset(path)
        frozen = cfg["test"]; accepted, trace = [], []
        for offset in range(int(frozen["max_candidates"])):
            seed = int(frozen["start"]) + offset * int(frozen["step"])
            candidate = prepare_candidate(seed)
            trace.append({"root_seed": seed, "eligible": bool(candidate["eligible"]), "rejection_reasons": list(candidate["rejection_reasons"])})
            if candidate["eligible"]:
                accepted.append(_save_test_case(args.output_dir, len(accepted) + 1, candidate))
                print(f"Stage20 TEST accepted {len(accepted)}/{frozen['accepted']} seed={seed}", flush=True)
                if len(accepted) == int(frozen["accepted"]):
                    break
        test_registry = {"schema": "stage20_split_registry_v1", "run_status": "completed" if len(accepted) == int(frozen["accepted"]) else "failed", "split": "test", "accepted_count": len(accepted), "candidate_count": len(trace), "seed_rule": frozen, "reused_from_stage19r": False, "cases": accepted, "candidate_trace": trace}
        test_path = args.output_dir / "test_registry.json"; write_json(test_path, test_registry); registries["test"] = asset(test_path)
        if test_registry["run_status"] != "completed":
            raise RuntimeError("STOP_STAGE20_TEST_COHORT_UNAVAILABLE")
        master.update({"run_status": "completed", "registries": registries, "total_accepted": 84, "train_val_exact_stage19r_reuse": True, "test_reused_from_stage19r": False})
        write_json(args.output_dir / "cohort_manifest.json", master)
    except Exception as exc:
        master.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}); write_json(args.output_dir / "cohort_manifest.json", master); raise


if __name__ == "__main__":
    main()
