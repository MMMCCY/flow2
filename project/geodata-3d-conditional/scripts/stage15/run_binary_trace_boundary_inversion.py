#!/usr/bin/env python3
"""Truth-blind complete-trace binary boundary inversion for Stage15-H."""

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
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from guidance.seismic_inversion import ModelBasedInversionConfig
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, validate_asset, write_csv, write_json

ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION = ROOT / "observations/cond_generation_0"
DEFAULT_CONFIG = ROOT / "configs/binary_trace_boundary_inversion_v1.json"
DEFAULT_BINARY_CONFIG = ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
DEFAULT_OUTPUT = ROOT / "trace_boundary/cond_generation_0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    if config.get("schema") != "stage15_binary_trace_boundary_inversion_v1":
        raise ValueError("invalid trace-boundary config")
    if config.get("truth_used_by_runner") is not False or config.get("parameter_sweep") is not False:
        raise ValueError("runner must remain truth-blind and fixed")
    observation_manifest = read_json(args.observation_dir / "manifest.json")
    names = ("observed_seismic.pt", "subsurface_mask.pt", "binary_well_values.pt", "binary_well_mask.pt")
    inputs = {name: runtime.load_tensor(args.observation_dir / name) for name in names}
    for name, value in inputs.items():
        if tensor_sha256(value) != observation_manifest["output_tensor_sha256"][name]:
            raise ValueError(f"observation tensor changed: {name}")
    device = torch.device(args.device)
    observed = inputs["observed_seismic.pt"].to(device=device, dtype=torch.float32)
    support = normalize_volume(inputs["subsurface_mask.pt"], "subsurface").bool()
    wells = normalize_volume(inputs["binary_well_values.pt"], "well values", torch.float32)
    well_mask = normalize_volume(inputs["binary_well_mask.pt"], "well mask").bool()
    binary_config = read_json(args.binary_config)
    source_record = binary_config["source_acoustic_config"]
    source_path = REPOSITORY_ROOT / str(source_record["path"])
    validate_asset(source_path, str(source_record["sha256"]))
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    operator, operator_metadata = seismic_operator_from_config(read_json(args.seismic_config), grid_shape=support.shape[2:])
    inversion_config = ModelBasedInversionConfig(
        "stage15_binary_trace_boundary_v1",
        float(config["prior_relative_weight"]),
        float(config["vertical_smoothness_relative_weight"]),
    )
    score, acoustic, predicted, trace = refine_binary_trace_volume(
        observed,
        support.to(device),
        operator,
        properties,
        inversion_config,
        int(config["refinement_passes"]),
        wells,
        well_mask,
    )
    boundary = vertical_boundary_strength(score)
    args.output_dir.mkdir(parents=True)
    outputs = {
        "binary_impedance_score.pt": score,
        "binary_acoustic_volume.pt": acoustic,
        "predicted_seismic.pt": predicted,
        "vertical_boundary_strength.pt": boundary,
    }
    for name, value in outputs.items():
        torch.save(value, args.output_dir / name)
    write_csv(args.output_dir / "refinement_trace.csv", trace)
    manifest = base_manifest("stage15_binary_trace_boundary_inversion_run_v1", Path(__file__), args.config)
    manifest.update({
        "run_status": "completed",
        "truth_loaded_by_runner": False,
        "full_vertical_trace_used": True,
        "lateral_mixing": False,
        "output_tensor_sha256": {name: tensor_sha256(value) for name, value in outputs.items()},
        "input_assets": {name: runtime.asset_record(args.observation_dir / name) for name in names},
        "binary_acoustic_config": runtime.asset_record(args.binary_config),
        "seismic_config": runtime.asset_record(args.seismic_config),
        "operator_metadata": operator_metadata,
        "refinement_trace": trace,
    })
    write_json(args.output_dir / "manifest.json", manifest)


if __name__ == "__main__":
    main()
