#!/usr/bin/env python3
"""Build and freeze the independent 64/8/12 Stage19 StructuralGeo cohort."""

from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for root in (PROJECT_DIR, REPOSITORY_ROOT, STRUCTURALGEO_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.full_structuralgeo_benchmark import (
    CANONICAL_NINE_WELL_XY,
    MODEL_BOUNDS,
    MODEL_RESOLUTION,
    canonical_json_sha256,
    canonical_tensor_sha256,
    prepare_candidate,
)
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, require_config


DEFAULT_CONFIG = CONFIG_DIR / "cohort_v1.json"
DEFAULT_OUTPUT = ROOT / "cohort"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _save_case(output: Path, split: str, index: int, candidate: dict[str, object]) -> dict[str, object]:
    case_id = f"stage19_{split}_case{index:03d}"
    case_dir = output / "cases" / case_id
    truth_dir, condition_dir = case_dir / "truth", case_dir / "condition"
    truth_dir.mkdir(parents=True)
    condition_dir.mkdir()
    truth = candidate["truth"].cpu()
    conditions = candidate["conditions"]
    tensors = {
        truth_dir / "true_model.pt": truth,
        condition_dir / "condition_values.pt": conditions["condition_values"].cpu(),
        condition_dir / "condition_mask.pt": conditions["condition_mask"].cpu(),
        condition_dir / "surface_mask.pt": conditions["surface_mask"].cpu(),
        condition_dir / "borehole_mask.pt": conditions["borehole_mask"].cpu(),
    }
    for path, tensor in tensors.items():
        torch.save(tensor, path)
    history = {
        "schema": "stage19_structuralgeo_history_v1",
        "root_seed": int(candidate["root_seed"]),
        "markov_sequence": candidate["metadata"]["markov_sequence"],
        "event_subtypes": candidate["event_subtypes"],
        "events": candidate["metadata"]["events"],
        "packed_history": candidate["metadata"]["packed_history"],
        "unpacked_history": candidate["metadata"]["unpacked_history"],
    }
    write_json(truth_dir / "history.json", history)
    target = truth == 9
    hidden = target & ~conditions["condition_mask"].cpu()
    manifest = {
        "schema": "stage19_structuralgeo_case_v1",
        "run_status": "completed",
        "case_id": case_id,
        "split": split,
        "root_seed": int(candidate["root_seed"]),
        "markov_sequence": candidate["metadata"]["markov_sequence"],
        "event_subtypes": candidate["event_subtypes"],
        "history_sha256": canonical_json_sha256(history),
        "truth_tensor_sha256": canonical_tensor_sha256(truth),
        "condition_values_tensor_sha256": canonical_tensor_sha256(conditions["condition_values"]),
        "condition_mask_tensor_sha256": canonical_tensor_sha256(conditions["condition_mask"]),
        "target_voxel_count": int(target.sum()),
        "hidden_target_voxel_count": int(hidden.sum()),
        "well_xy": [list(point) for point in CANONICAL_NINE_WELL_XY],
        "selection_firewall": {"seismic": False, "evidence": False, "flow": False, "downstream_metrics": False},
        "assets": {
            "truth": asset(truth_dir / "true_model.pt"),
            "condition_values": asset(condition_dir / "condition_values.pt"),
            "condition_mask": asset(condition_dir / "condition_mask.pt"),
            "history": asset(truth_dir / "history.json"),
        },
    }
    write_json(case_dir / "manifest.json", manifest)
    return {**manifest, "manifest": asset(case_dir / "manifest.json")}


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = require_config(args.config, "stage19_cohort_v1")
    if tuple(tuple(v) for v in config["recipe"]["bounds"]) != MODEL_BOUNDS or tuple(config["recipe"]["resolution"]) != MODEL_RESOLUTION:
        raise ValueError("Stage19 StructuralGeo recipe changed")
    if tuple(tuple(v) for v in config["fixed_well_xy"]) != CANONICAL_NINE_WELL_XY:
        raise ValueError("Stage19 fixed wells changed")
    split_ranges: list[set[int]] = []
    for split in config["splits"].values():
        seeds = {int(split["start"]) + i * int(split["step"]) for i in range(int(split["max_candidates"]))}
        if any(seeds & prior for prior in split_ranges):
            raise ValueError("Stage19 seed ranges overlap")
        split_ranges.append(seeds)

    args.output_dir.mkdir(parents=True)
    master = base_manifest("stage19_cohort_manifest_v1", Path(__file__), args.config)
    master.update({"run_status": "running", "selection_uses_seismic_evidence_flow_or_downstream_metrics": False})
    write_json(args.output_dir / "cohort_manifest.json", master)
    registries = {}
    try:
        for split_name in ("train", "val", "test"):
            frozen = config["splits"][split_name]
            accepted, examined = [], []
            for offset in range(int(frozen["max_candidates"])):
                root_seed = int(frozen["start"]) + offset * int(frozen["step"])
                candidate = prepare_candidate(root_seed)
                examined.append({"root_seed": root_seed, "eligible": bool(candidate["eligible"]), "rejection_reasons": candidate["rejection_reasons"]})
                if candidate["eligible"]:
                    accepted.append(_save_case(args.output_dir, split_name, len(accepted) + 1, candidate))
                    print(f"Stage19 cohort accepted {split_name} {len(accepted)}/{frozen['accepted']} seed={root_seed}", flush=True)
                    if len(accepted) == int(frozen["accepted"]):
                        break
            if len(accepted) != int(frozen["accepted"]):
                raise RuntimeError(f"STOP: insufficient eligible {split_name} cases")
            registry = {
                "schema": "stage19_split_registry_v1", "run_status": "completed", "split": split_name,
                "accepted_count": len(accepted), "candidate_count": len(examined), "seed_rule": frozen,
                "cases": accepted, "candidate_trace": examined,
            }
            registry_path = args.output_dir / f"{split_name}_registry.json"
            write_json(registry_path, registry)
            registries[split_name] = asset(registry_path)
        master.update({"run_status": "completed", "registries": registries, "total_accepted": 84, "historical_cases_excluded": True, "training_overlap_wording": config["training_overlap_wording"]})
        write_json(args.output_dir / "cohort_manifest.json", master)
    except Exception as exc:
        master.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "cohort_manifest.json", master)
        raise


if __name__ == "__main__":
    main()
