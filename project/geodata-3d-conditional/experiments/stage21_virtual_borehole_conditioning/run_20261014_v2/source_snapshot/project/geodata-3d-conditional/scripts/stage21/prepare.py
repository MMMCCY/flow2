"""Offline oracle preparation and historical asset audit; never imported by runner."""
from __future__ import annotations

import argparse
import platform
import sys
from pathlib import Path

import numpy as np
import scipy
import torch

PROJECT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import (
    CASES, CHECKPOINT, CHECKPOINT_SHA, EXPERIMENT, LEVELS, REPO, SEEDS, W9,
    condition_assets, file_hash, git_snapshot, json_hash, load_tensor, plan_layout,
    read_json, save_tensor, source_hashes, tensor_hash, validate_registry, write_json,
)


def protocol():
    return {
        "schema": "stage21_step1_protocol_v1", "scope": "step1_accurate_extra_wells_only",
        "cases": CASES, "source_seeds": SEEDS, "well_counts": LEVELS, "shape": [64, 64, 64],
        "checkpoint": CHECKPOINT, "checkpoint_sha256": CHECKPOINT_SHA, "weight_source": "ema",
        "n_steps": 32, "integrator": "fixed_euler_midpoint_v1",
        "precision": "float32_no_autocast_no_tf32",
        "determinism": "torch_deterministic_algorithms_cublas4096_8_cudnn_benchmark_false",
        "layout_seed": 21001, "smoke_case": "fullgeo_case03", "smoke_seed": 142,
        "figure_seed": 142,
        "objects": {"raw_label": 9, "connectivity": 26, "major_min_voxels": 20,
                    "matching": "one_to_one_maximum_common_domain_iou_positive_edges_only",
                    "edge_min_intersection": 5, "edge_min_truth_coverage": 0.05,
                    "detection_min_iou": 0.1, "detection_min_coverage": 0.1,
                    "diagnostic_thresholds_are_not_scientific_success_gates": True},
        "evaluation": {"domain": "(truth!=-1)&~M25", "class_set": "truth_present_in_common_domain",
                       "case_aggregation": "median_of_three_seeds_then_median_across_cases",
                       "paired_aggregation": "median_within_case_of_same_seed_differences_then_cross_case_median",
                       "undefined": "null_with_denominator_and_reason",
                       "figure_rule": "all_five_cases_seed142_full_domain_fixed_cameras"},
        "budget": {"formal_outputs": 45, "formal_velocity_evaluations": 1440,
                   "cuda_smoke_outputs": 3, "independent_n9_regression_velocity_evaluations": 32,
                   "max_concurrent_gpu_jobs": 1},
    }


def prepare(out):
    if out.exists():
        raise FileExistsError(f"immutable output already exists: {out}")
    benchmark = PROJECT / "experiments/full_structuralgeo_benchmark"
    index = read_json(benchmark / "benchmark_manifest.json")
    if [c["case_id"] for c in index["cases"]] != CASES:
        raise ValueError("historical cohort mismatch")
    if file_hash(REPO / CHECKPOINT) != CHECKPOINT_SHA:
        raise ValueError("checkpoint mismatch: stop before preparation")
    layout = plan_layout()
    p = protocol()
    out.mkdir(parents=True)
    registry = {"schema": "stage21_inference_registry_v1", "protocol": p,
                "protocol_sha256": json_hash(p), "layout": layout, "layout_sha256": json_hash(layout),
                "conditions": {}, "source_sha256": source_hashes()}
    evaluation = {"schema": "stage21_evaluation_registry_v1", "role": "offline_retrospective_only",
                  "cases": {}, "benchmark_manifest_sha256": file_hash(benchmark / "benchmark_manifest.json")}
    audits = []
    for c in index["cases"]:
        case = c["case_id"]
        d = benchmark / "cases" / case
        m = read_json(d / "manifest.json")
        if file_hash(d / "manifest.json") != c["manifest_sha256"]:
            raise ValueError(f"historical manifest mismatch: {case}")
        tensors = {}
        for rel in ["truth/true_model.pt", *[f"condition/{k}.pt" for k in
                    ("condition_values", "condition_mask", "surface_mask", "borehole_mask")]]:
            t = load_tensor(d / rel)
            if file_hash(d / rel) != m["saved_file_sha256"][rel] or tensor_hash(t) != m["tensor_content_hashes"][rel]:
                raise ValueError(f"historical tensor mismatch: {case}/{rel}")
            tensors[rel] = t
        for rel in ("condition/well_xy.json", "geometry/connected_components.json"):
            if file_hash(d / rel) != m["saved_file_sha256"][rel]:
                raise ValueError(f"historical metadata mismatch: {case}/{rel}")
        if read_json(d / "condition/well_xy.json")["well_xy"] != [list(x) for x in W9]:
            raise ValueError("original well coordinates mismatch")
        truth = tensors["truth/true_model.pt"]
        surface = tensors["condition/surface_mask.pt"]
        registry["conditions"][case] = {}
        levels_audit = {}
        for n in LEVELS:
            assets = condition_assets(truth, surface, layout["levels"][str(n)])
            if n == 9 and any(not torch.equal(t, tensors[f"condition/{key}.pt"]) for key, t in assets.items()):
                raise ValueError(f"N9 tensor-content baseline mismatch: {case}")
            record = {"schema": "categorical_borehole_conditions_v1",
                      "origin": "synthetic_truth_sampled_at_frozen_wells_and_original_surface",
                      "oracle": True, "sequence_uncertainty": False,
                      "well_xy": layout["levels"][str(n)], "tensors": {}}
            for key, t in assets.items():
                record["tensors"][key] = save_tensor(out / "conditions" / case / f"n{n}" / f"{key}.pt", t)
            original = tensors["condition/condition_mask.pt"]
            if not torch.equal(assets["condition_values"][original], truth[original]):
                raise ValueError("original conditions changed")
            registry["conditions"][case][str(n)] = record
            levels_audit[str(n)] = {"condition_voxels": int(assets["condition_mask"].sum()),
                                   "well_voxels": int(assets["borehole_mask"].sum()),
                                   "surface_voxels": int(surface.sum()),
                                   "outside_mask_all_placeholder": bool(torch.all(assets["condition_values"][~assets["condition_mask"]] == -1)),
                                   "original_conditions_preserved": True,
                                   "n9_matches_registered_tensors": True if n == 9 else None}
        evaluation["cases"][case] = {
            "truth": {"path": str((d / "truth/true_model.pt").relative_to(REPO)),
                      "file_sha256": file_hash(d / "truth/true_model.pt"), "tensor_sha256": tensor_hash(truth)},
            "root_seed": c["root_seed"], "historical_manifest_sha256": c["manifest_sha256"]}
        audits.append({"case_id": case, "historical_assets_verified": True, "levels": levels_audit})
    validate_registry(registry)
    write_json(out / "layout.json", layout)
    write_json(out / "protocol.json", p)
    write_json(out / "inference_registry.json", registry)
    write_json(out / "evaluation_registry.json", evaluation)
    prep_sources = {str(f.relative_to(REPO)): file_hash(f) for f in
                    [Path(__file__).resolve(), PROJECT / "guidance/full_structuralgeo_benchmark.py"]}
    write_json(out / "condition_audit.json", {"status": "passed", "cases": audits,
               "truth_read_role": "offline_oracle_well_and_original_surface_preparation_only",
               "preparation_source_sha256": prep_sources})
    write_json(out / "preflight.json", {"status": "passed", "git": git_snapshot(),
               "checkpoint_sha256": CHECKPOINT_SHA, "checkpoint_size": (REPO / CHECKPOINT).stat().st_size,
               "python": platform.python_version(), "torch": torch.__version__,
               "numpy": np.__version__, "scipy": scipy.__version__,
               "cuda_available": torch.cuda.is_available(),
               "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None})
    print(f"Prepared {out}; inference registry contains conditions only", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.out.resolve())
