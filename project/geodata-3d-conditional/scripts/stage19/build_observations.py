#!/usr/bin/env python3
"""Build immutable binary seismic observations for all frozen Stage19 cases."""

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
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs, binary_occupancy_to_acoustic
from guidance.seismic import build_seismic_observation, seismic_operator_from_config
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, canonical_tensor_sha256, registry_cases, require_config, validate_asset, validate_stage17a_reuse

DEFAULT_CONFIG = CONFIG_DIR / "evidence_v2.json"
DEFAULT_COHORT = ROOT / "cohort_v2"
DEFAULT_OUTPUT = ROOT / "observations_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--cohort-dir", type=Path, default=DEFAULT_COHORT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def columnwise_support(truth: torch.Tensor) -> torch.Tensor:
    nonair = truth != -1
    if not bool(nonair.any(dim=-1).all()):
        raise ValueError("every column must contain non-air")
    z = torch.arange(truth.shape[-1]).view(1, 1, 1, 1, -1)
    highest = torch.where(nonair, z, torch.full_like(z, -1)).amax(dim=-1, keepdim=True)
    return z <= highest


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = require_config(args.config, "stage19_evidence_v2")
    reused = validate_stage17a_reuse(config)
    obs_cfg = config["observation"]
    binary_path = reused["binary_acoustic_config"]
    seismic_path = reused["seismic_config"]
    binary_config = read_json(binary_path)
    source_path = validate_asset(binary_config["source_acoustic_config"], "source acoustic config")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    seismic_config = read_json(seismic_path)
    device = torch.device(args.device)
    all_cases = []
    for split in ("train", "val", "test"):
        all_cases.extend(registry_cases(args.cohort_dir / f"{split}_registry.json", split))
    if len(all_cases) != 84:
        raise ValueError("Stage19 observation build requires 84 cases")

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage19r_observation_run_v2", Path(__file__), args.config)
    manifest.update({"run_status": "running", "truth_loaded_for_synthetic_observation_only": True})
    write_json(args.output_dir / "run_manifest.json", manifest)
    records = []
    try:
        for case in all_cases:
            case_id = str(case["case_id"])
            truth = normalize_volume(runtime.load_tensor(validate_asset(case["assets"]["truth"], f"{case_id}/truth")), "truth").long()
            values = normalize_volume(runtime.load_tensor(validate_asset(case["assets"]["condition_values"], f"{case_id}/condition values")), "condition values").long()
            condition = normalize_volume(runtime.load_tensor(validate_asset(case["assets"]["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool()
            if int(((values != truth) & condition).sum()):
                raise ValueError(f"condition mismatch: {case_id}")
            support = columnwise_support(truth)
            binary_truth = ((truth == int(config["target_label"])) & support).float()
            fixed_mask = condition & support
            fixed_values = torch.zeros_like(binary_truth)
            fixed_values[fixed_mask] = (values[fixed_mask] == int(config["target_label"])).float()
            operator, metadata = seismic_operator_from_config(seismic_config, grid_shape=truth.shape[2:])
            impedance, slowness = binary_occupancy_to_acoustic(binary_truth.to(device), support.to(device), properties)
            observation = build_seismic_observation(
                torch.cat((impedance, slowness), dim=1), support.to(device), operator,
                uncertainty_amplitude=float(metadata["uncertainty_amplitude"]),
                noise_std_amplitude=float(metadata["noise"]["std_amplitude"]),
                noise_seed=int(metadata["noise"]["seed"]),
            )
            closure = operator(impedance, slowness, support.to(device))
            closure_error = float((closure - observation.values).abs().max().cpu())
            if closure_error > float(obs_cfg["forward_closure_max_abs"]):
                raise RuntimeError(f"forward closure failed: {case_id}={closure_error}")
            case_dir = args.output_dir / case_id
            restricted = case_dir / "truth_restricted"
            restricted.mkdir(parents=True)
            tensors = {
                "observed_seismic.pt": observation.values.cpu(), "sample_mask.pt": observation.sample_mask.cpu(),
                "uncertainty.pt": observation.uncertainty.cpu(), "subsurface_mask.pt": support.cpu(),
                "condition_values.pt": values.cpu(), "condition_mask.pt": condition.cpu(),
                "binary_fixed_values.pt": fixed_values.cpu(), "binary_fixed_mask.pt": fixed_mask.cpu(),
            }
            for name, tensor in tensors.items():
                torch.save(tensor, case_dir / name)
            torch.save(binary_truth.cpu(), restricted / "binary_truth.pt")
            case_manifest = {
                "schema": "stage19r_observation_case_v2", "run_status": "completed", "case_id": case_id,
                "split": case["split"], "root_seed": int(case["root_seed"]),
                "truth_role": "synthetic_observation_generation_and_retrospective_evaluation_only",
                "forward_closure_max_abs": closure_error,
                "support_policy": obs_cfg["support_policy"], "enclosed_air_acoustic_policy": obs_cfg["enclosed_air_acoustic_policy"],
                "input_case_manifest": case["manifest"],
                "assets": {name.removesuffix(".pt"): asset(case_dir / name) for name in tensors},
                "truth_assets": {"binary_truth": asset(restricted / "binary_truth.pt"), "truth": case["assets"]["truth"]},
                "tensor_content_hashes": {name: canonical_tensor_sha256(tensor) for name, tensor in tensors.items()},
                "operator_metadata": metadata,
            }
            write_json(case_dir / "manifest.json", case_manifest)
            records.append({"case_id": case_id, "split": case["split"], "root_seed": case["root_seed"], "manifest": asset(case_dir / "manifest.json"), "assets": case_manifest["assets"], "truth_assets": case_manifest["truth_assets"]})
            print(f"Stage19 observation {case_id} completed", flush=True)
        registry = {"schema": "stage19r_observation_registry_v2", "run_status": "completed", "case_count": len(records), "cases": records}
        write_json(args.output_dir / "observation_registry.json", registry)
        manifest.update({"run_status": "completed", "case_count": len(records), "observation_registry": asset(args.output_dir / "observation_registry.json"), "binary_acoustic_config": asset(binary_path), "source_acoustic_config": asset(source_path), "seismic_config": asset(seismic_path)})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
