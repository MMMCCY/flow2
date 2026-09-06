#!/usr/bin/env python3
"""Build multiclass Phase4c observations and a stripped inversion registry."""

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
from guidance.seismic import build_seismic_observation, seismic_operator_from_config
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json
from scripts.stage20.acoustic_semantics import stage20_labels_to_acoustic
from scripts.stage20.common import CONFIG_DIR, ROOT, asset, assert_truth_firewall, canonical_tensor_sha256, columnwise_support, load_codebook, require_config, resolve_project_path, validate_asset, validate_wells

DEFAULT_OUTPUT = ROOT / "observations_fix2"
DEFAULT_INVERSION_REGISTRY = ROOT / "inversion_input_registry_fix2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "evidence_v1.json")
    parser.add_argument("--cohort-dir", type=Path, default=ROOT / "cohort")
    parser.add_argument("--reuse-audit", type=Path, default=ROOT / "reuse_audit/summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--inversion-registry-dir", type=Path, default=DEFAULT_INVERSION_REGISTRY)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _cases(cohort_dir: Path) -> list[dict[str, object]]:
    cases = []
    for split, count in (("train", 64), ("val", 8), ("test", 12)):
        registry = read_json(cohort_dir / f"{split}_registry.json")
        if registry.get("schema") != "stage20_split_registry_v1" or registry.get("run_status") != "completed" or len(registry.get("cases", [])) != count:
            raise ValueError(f"invalid Stage20 {split} registry")
        cases.extend(registry["cases"])
    return cases


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir); refuse_nonempty(args.inversion_registry_dir)
    cfg = require_config(args.config, "stage20_continuous_impedance_evidence_v1")
    audit = read_json(args.reuse_audit)
    if audit.get("machine_decision") != "STAGE20_REUSE_VALIDATED":
        raise RuntimeError("STOP_STAGE20_REUSE_MISMATCH")
    acoustic_path = resolve_project_path(cfg["phase4c"]["acoustic_config"])
    seismic_path = resolve_project_path(cfg["phase4c"]["seismic_config"])
    property_table, acoustic_metadata = load_codebook(acoustic_path)
    seismic_cfg = read_json(seismic_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    cases = _cases(args.cohort_dir)
    args.output_dir.mkdir(parents=True); args.inversion_registry_dir.mkdir(parents=True)
    torch.save(property_table, args.output_dir / "acoustic_property_table.pt")
    manifest = base_manifest("stage20_observation_run_v1", Path(__file__), args.config)
    manifest.update({"run_status": "running", "truth_loaded_for_synthetic_observation_only": True})
    write_json(args.output_dir / "run_manifest.json", manifest)
    full_records, stripped_records = [], []
    try:
        for case in cases:
            case_id = str(case["case_id"]); wells = validate_wells(case["well_xy"]); assets = case["assets"]
            truth = normalize_volume(runtime.load_tensor(validate_asset(assets["truth"], f"{case_id}/truth")), "truth").long()
            values = normalize_volume(runtime.load_tensor(validate_asset(assets["condition_values"], f"{case_id}/condition values")), "condition values").long()
            condition = normalize_volume(runtime.load_tensor(validate_asset(assets["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool()
            if int(((truth != values) & condition).sum()):
                raise ValueError(f"condition mismatch: {case_id}")
            support = columnwise_support(truth)
            operator, operator_metadata = seismic_operator_from_config(seismic_cfg, grid_shape=truth.shape[2:])
            acoustic, cleanup = stage20_labels_to_acoustic(truth.to(device), support.to(device), property_table.to(device))
            observation = build_seismic_observation(acoustic, support.to(device), operator, uncertainty_amplitude=float(seismic_cfg["uncertainty_amplitude"]), noise_std_amplitude=0.0, noise_seed=int(seismic_cfg["noise"]["seed"]))
            closure = operator(acoustic[:, 0:1], acoustic[:, 1:2], support.to(device))
            closure_error = float((closure - observation.values).abs().max().cpu())
            if closure_error > float(cfg["phase4c"]["forward_closure_max_abs"]):
                raise RuntimeError(f"STOP_PHASE4C_FORWARD_REUSE_MISMATCH: {case_id}={closure_error}")
            case_dir = args.output_dir / case_id; case_dir.mkdir()
            tensors = {
                "observed_seismic": observation.values.cpu(), "sample_mask": observation.sample_mask.cpu(),
                "uncertainty": observation.uncertainty.cpu(), "subsurface_mask": support.cpu(),
                "condition_values": values.cpu(), "condition_mask": condition.cpu(),
            }
            for name, tensor in tensors.items():
                torch.save(tensor, case_dir / f"{name}.pt")
            obs_assets = {name: asset(case_dir / f"{name}.pt") for name in tensors}
            case_manifest = {
                "schema": "stage20_observation_case_v1", "run_status": "completed", "case_id": case_id,
                "split": case["split"], "root_seed": int(case["root_seed"]), "well_xy": [list(p) for p in wells],
                "truth_role": "synthetic_observation_generation_and_retrospective_evaluation_only",
                "forward_closure_max_abs": closure_error, "enclosed_subsurface_air_cleanup": cleanup, "assets": obs_assets,
                "truth_assets": {"truth": assets["truth"]}, "operator_metadata": operator_metadata,
                "tensor_content_hashes": {name: canonical_tensor_sha256(tensor) for name, tensor in tensors.items()},
            }
            write_json(case_dir / "manifest.json", case_manifest)
            full_records.append({**case_manifest, "manifest": asset(case_dir / "manifest.json")})
            stripped_records.append({"case_id": case_id, "split": case["split"], "root_seed": int(case["root_seed"]), "well_xy": [list(p) for p in wells], "assets": obs_assets, "acoustic_config": asset(acoustic_path), "seismic_config": asset(seismic_path)})
            print(f"Stage20 observation {case_id} completed", flush=True)
        full_registry = {"schema": "stage20_observation_registry_full_v1", "run_status": "completed", "case_count": len(full_records), "cases": full_records}
        stripped = {"schema": "stage20_inversion_input_registry_v1", "run_status": "completed", "case_count": len(stripped_records), "cases": stripped_records}
        assert_truth_firewall(stripped, "inversion-only registry")
        write_json(args.output_dir / "observation_registry_full.json", full_registry)
        write_json(args.inversion_registry_dir / "registry.json", stripped)
        manifest.update({"run_status": "completed", "case_count": len(full_records), "all_forward_closures_passed": True, "full_registry": asset(args.output_dir / "observation_registry_full.json"), "inversion_input_registry": asset(args.inversion_registry_dir / "registry.json"), "acoustic_config": asset(acoustic_path), "seismic_config": asset(seismic_path), "acoustic_property_table": asset(args.output_dir / "acoustic_property_table.pt"), "acoustic_metadata": acoustic_metadata})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}); write_json(args.output_dir / "run_manifest.json", manifest); raise


if __name__ == "__main__":
    main()
