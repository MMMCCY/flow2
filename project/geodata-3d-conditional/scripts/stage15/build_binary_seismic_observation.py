#!/usr/bin/env python3
"""Build the truth-derived Stage15 binary synthetic seismic observation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
)
from guidance.seismic import build_seismic_observation, seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    validate_asset,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_PHASE1_CONFIG = (
    PROJECT_DIR
    / "experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4"
    / "calibrated_reference_windowed/seed42_n4_s32/baseline/config.json"
)
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = (
    PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--phase1-config", type=Path, default=DEFAULT_PHASE1_CONFIG)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _resolved_repo_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    manifest = base_manifest("stage15_binary_observation_v1", Path(__file__))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        phase1 = read_json(args.phase1_config)
        if phase1.get("run_status") != "completed" or phase1.get("stage") != "phase1_oracle_3d_probability":
            raise ValueError("authoritative Phase1 config is not a completed Phase1 run")
        if phase1.get("samples_dir") != "project/geodata-3d-conditional/samples/jupyter-demo/cond_generation_0":
            raise ValueError("Phase1 case identity is not cond_generation_0")
        truth_path = _resolved_repo_path(phase1["truth_model"])
        boreholes_path = _resolved_repo_path(phase1["boreholes"])
        checkpoint_path = _resolved_repo_path(phase1["ckpt_path"])
        phase1_assets = {
            "truth_model": validate_asset(truth_path, str(phase1["truth_model_sha256"])),
            "boreholes": validate_asset(boreholes_path, str(phase1["boreholes_sha256"])),
            "checkpoint": validate_asset(checkpoint_path, str(phase1["checkpoint_sha256"])),
            "authoritative_config": runtime.asset_record(args.phase1_config),
        }
        truth = normalize_volume(runtime.load_tensor(truth_path), "true_model").long()
        boreholes = normalize_volume(runtime.load_tensor(boreholes_path), "boreholes").long()
        conditioning_report = runtime.validate_conditioning_pair(
            truth, boreholes, num_categories=15, target_label=9
        )

        binary_config = read_json(args.binary_acoustic_config)
        source_record = binary_config.get("source_acoustic_config")
        if not isinstance(source_record, dict):
            raise TypeError("binary acoustic config has no source record")
        source_path = _resolved_repo_path(source_record["path"])
        validate_asset(source_path, str(source_record["sha256"]))
        source_config = read_json(source_path)
        properties = binary_acoustic_properties_from_configs(binary_config, source_config)

        seismic_config = read_json(args.seismic_config)
        operator, resolved_seismic = seismic_operator_from_config(
            seismic_config, grid_shape=truth.shape[2:]
        )
        device = torch.device(args.device)
        subsurface = truth != -1
        binary_truth = ((truth == 9) & subsurface).float()
        flow_condition_values = boreholes.clone()
        flow_condition_mask = (~subsurface) | (boreholes != -1)
        binary_well_mask = subsurface & (boreholes != -1)
        binary_well_values = torch.zeros_like(binary_truth)
        binary_well_values[binary_well_mask] = (boreholes[binary_well_mask] == 9).float()
        impedance, slowness = binary_occupancy_to_acoustic(
            binary_truth.to(device), subsurface.to(device), properties
        )
        observation = build_seismic_observation(
            torch.cat((impedance, slowness), dim=1),
            subsurface.to(device),
            operator,
            uncertainty_amplitude=float(resolved_seismic["uncertainty_amplitude"]),
            noise_std_amplitude=float(resolved_seismic["noise"]["std_amplitude"]),
            noise_seed=int(resolved_seismic["noise"]["seed"]),
        )
        closure = operator(impedance, slowness, subsurface.to(device))
        closure_max_abs = float((closure - observation.values).abs().max().cpu())
        if closure_max_abs > 1e-7:
            raise RuntimeError(f"observation forward closure failed: {closure_max_abs}")

        saved = {
            "observed_seismic.pt": observation.values.cpu(),
            "sample_mask.pt": observation.sample_mask.cpu(),
            "uncertainty.pt": observation.uncertainty.cpu(),
            "subsurface_mask.pt": subsurface.cpu(),
            "flow_condition_values.pt": flow_condition_values.cpu(),
            "flow_condition_mask.pt": flow_condition_mask.cpu(),
            "binary_well_values.pt": binary_well_values.cpu(),
            "binary_well_mask.pt": binary_well_mask.cpu(),
        }
        for filename, tensor in saved.items():
            torch.save(tensor, args.output_dir / filename)
        truth_dir = args.output_dir / "truth_restricted"
        truth_dir.mkdir(parents=True)
        torch.save(binary_truth.cpu(), truth_dir / "binary_truth.pt")

        tensor_hashes = {name: tensor_sha256(value) for name, value in saved.items()}
        tensor_hashes["truth_restricted/binary_truth.pt"] = tensor_sha256(binary_truth)
        resolved = {
            "phase1_case_id": "cond_generation_0",
            "target_label": 9,
            "grid_shape": list(truth.shape[2:]),
            "binary_acoustic_config": runtime.asset_record(args.binary_acoustic_config),
            "source_acoustic_config": runtime.asset_record(source_path),
            "seismic_config": runtime.asset_record(args.seismic_config),
            "seismic_parameters": resolved_seismic,
            "binary_property_values": properties.__dict__,
            "binary_property_policy": "all_non_label9_subsurface_uses_raw_label0_reference",
            "scientific_role": "deliberately_binary_high_contrast_full_cube_noiseless_synthetic_upper_bound",
            "realistic_multiclass_petrophysics": False,
        }
        write_json(args.output_dir / "resolved_config.json", resolved)
        manifest.update(
            {
                "run_status": "completed",
                "phase1_assets": phase1_assets,
                "resolved_config": runtime.asset_record(args.output_dir / "resolved_config.json"),
                "input_tensor_sha256": {
                    "true_model": tensor_sha256(truth),
                    "boreholes": tensor_sha256(boreholes),
                },
                "output_tensor_sha256": tensor_hashes,
                "conditioning_report": conditioning_report,
                "observation_metadata": observation.metadata,
                "forward_closure_max_abs": closure_max_abs,
                "flow_condition_voxels": int(flow_condition_mask.sum()),
                "binary_well_voxels": int(binary_well_mask.sum()),
                "random_seeds": {"seismic_noise": int(resolved_seismic["noise"]["seed"])},
                "truth_allowed_for_observation_generation": True,
                "truth_allowed_for_inversion": False,
                "truth_allowed_for_consensus": False,
                "truth_allowed_for_flow": False,
                "truth_allowed_for_retrospective_evaluation": True,
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
