#!/usr/bin/env python3
"""Run the frozen Stage17A full-trace inversion on all Stage19 observations."""

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
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs
from guidance.binary_trace_boundary import refine_binary_trace_volume, vertical_boundary_strength
from guidance.seismic import seismic_operator_from_config
from guidance.seismic_inversion import ModelBasedInversionConfig
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, canonical_tensor_sha256, require_config, resolve_project_path, validate_asset

DEFAULT_CONFIG = CONFIG_DIR / "evidence_v1.json"
DEFAULT_OBSERVATIONS = ROOT / "observations"
DEFAULT_OUTPUT = ROOT / "evidence"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATIONS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = require_config(args.config, "stage19_evidence_v1")
    inv = config["inversion"]
    if inv["thresholding"] is not False or inv["lateral_filter_used"] is not False or inv["full_vertical_trace_used"] is not True:
        raise ValueError("Stage19 evidence semantics changed")
    observation_registry_path = args.observation_dir / "observation_registry.json"
    observation_registry = read_json(observation_registry_path)
    if observation_registry.get("run_status") != "completed" or observation_registry.get("case_count") != 84:
        raise ValueError("Stage19 observations are incomplete")
    binary_path = resolve_project_path(config["observation"]["binary_acoustic_config"])
    seismic_path = resolve_project_path(config["observation"]["seismic_config"])
    inversion_path = resolve_project_path(inv["inversion_config"])
    frozen_inversion = read_json(inversion_path)
    for key in ("refinement_passes", "prior_relative_weight", "vertical_smoothness_relative_weight"):
        if float(inv[key]) != float(frozen_inversion[key]):
            raise ValueError(f"Stage17A inversion parameter changed: {key}")
    binary_config = read_json(binary_path)
    source_path = validate_asset(binary_config["source_acoustic_config"], "source acoustic config")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    seismic_config = read_json(seismic_path)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage19_evidence_run_v1", Path(__file__), args.config)
    manifest.update({"run_status": "running", "truth_loaded_by_runner": False, "thresholding_performed": False})
    write_json(args.output_dir / "run_manifest.json", manifest)
    records = []
    try:
        for case in observation_registry["cases"]:
            case_id = str(case["case_id"])
            assets = case["assets"]
            observed = runtime.load_tensor(validate_asset(assets["observed_seismic"], f"{case_id}/observed")).to(device=device, dtype=torch.float32)
            support = normalize_volume(runtime.load_tensor(validate_asset(assets["subsurface_mask"], f"{case_id}/support")), "support").bool()
            fixed_values = normalize_volume(runtime.load_tensor(validate_asset(assets["binary_fixed_values"], f"{case_id}/fixed values")), "fixed values", torch.float32)
            fixed_mask = normalize_volume(runtime.load_tensor(validate_asset(assets["binary_fixed_mask"], f"{case_id}/fixed mask")), "fixed mask").bool()
            if observed.shape[-1] != 320:
                raise ValueError("full 320-sample traces required")
            operator, metadata = seismic_operator_from_config(seismic_config, grid_shape=support.shape[2:])
            inversion_config = ModelBasedInversionConfig("stage19_binary_trace_boundary_v1", float(inv["prior_relative_weight"]), float(inv["vertical_smoothness_relative_weight"]))
            score, acoustic, predicted, trace = refine_binary_trace_volume(observed, support.to(device), operator, properties, inversion_config, int(inv["refinement_passes"]), fixed_values, fixed_mask)
            boundary = vertical_boundary_strength(score)
            case_dir = args.output_dir / case_id
            case_dir.mkdir()
            outputs = {"binary_impedance_score.pt": score.cpu(), "binary_acoustic_volume.pt": acoustic.cpu(), "predicted_seismic.pt": predicted.cpu(), "vertical_boundary_strength.pt": boundary.cpu()}
            for name, tensor in outputs.items():
                torch.save(tensor, case_dir / name)
            write_csv(case_dir / "refinement_trace.csv", trace)
            case_manifest = {"schema": "stage19_evidence_case_v1", "run_status": "completed", "case_id": case_id, "split": case["split"], "truth_loaded_by_runner": False, "thresholding_performed": False, "full_vertical_trace_used": True, "trace_samples": 320, "input_observation_manifest": case["manifest"], "assets": {name.removesuffix(".pt"): asset(case_dir / name) for name in outputs}, "tensor_content_hashes": {name: canonical_tensor_sha256(tensor) for name, tensor in outputs.items()}, "operator_metadata": metadata}
            write_json(case_dir / "manifest.json", case_manifest)
            records.append({"case_id": case_id, "split": case["split"], "root_seed": case["root_seed"], "manifest": asset(case_dir / "manifest.json"), "evidence": case_manifest["assets"]["binary_impedance_score"], "observation_manifest": case["manifest"], "observation_assets": case["assets"], "truth_assets": case["truth_assets"]})
            print(f"Stage19 evidence {case_id} completed", flush=True)
        registry = {"schema": "stage19_evidence_registry_v1", "run_status": "completed", "case_count": len(records), "cases": records, "thresholding_performed": False}
        write_json(args.output_dir / "evidence_registry.json", registry)
        manifest.update({"run_status": "completed", "case_count": len(records), "evidence_registry": asset(args.output_dir / "evidence_registry.json"), "truth_loaded_by_runner": False})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
