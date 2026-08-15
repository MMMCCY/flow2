#!/usr/bin/env python3
"""Build paired solid/ring truths and immutable binary seismic observations."""

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
from guidance.seismic import build_seismic_observation, seismic_operator_from_config, tensor_sha256
from guidance.topology_support import betti_numbers, ellipsoid_mask, torus_mask
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json

ROOT = PROJECT_DIR / "experiments/stage15_topology_support"


def _path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=ROOT / "configs/frozen_protocol_v1.json")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "cases_v1")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    refuse_nonempty(args.output_dir)
    protocol = read_json(args.protocol)
    if protocol.get("schema") != "stage15_topology_support_protocol_v1" or protocol.get("status") != "frozen_before_execution":
        raise ValueError("topology-support protocol is not frozen")
    base = _path(protocol["base_case"])
    original_truth = normalize_volume(runtime.load_tensor(base / "true_model.pt"), "truth").long()
    original_boreholes = normalize_volume(runtime.load_tensor(base / "boreholes.pt"), "boreholes").long()
    support = original_truth != -1
    background = original_truth.clone()
    background[background == int(protocol["target_label"])] = int(protocol["background_replacement_label"])
    boreholes = original_boreholes.clone()
    boreholes[boreholes == int(protocol["target_label"])] = int(protocol["background_replacement_label"])
    binary_config_path = _path(protocol["binary_acoustic_config"])
    binary_config = read_json(binary_config_path)
    source_path = _path(binary_config["source_acoustic_config"]["path"])
    if runtime.file_sha256(source_path) != binary_config["source_acoustic_config"]["sha256"]:
        raise ValueError("binary acoustic source changed")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    seismic_config_path = _path(protocol["seismic_config"])
    operator, seismic_metadata = seismic_operator_from_config(read_json(seismic_config_path), grid_shape=(64, 64, 64))
    device = torch.device(args.device)
    manifest = base_manifest("stage15_topology_support_case_builder_v1", Path(__file__), args.protocol)
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "manifest.json", manifest)
    cases = {}
    try:
        for case_id, definition in protocol["cases"].items():
            center = definition["center"]
            if definition["family"] == "solid_oblate_ellipsoid":
                target = ellipsoid_mask((64, 64, 64), center, definition["axes"])
            elif definition["family"] == "z_axis_torus":
                target = torus_mask((64, 64, 64), center, definition["major_radius"], definition["tube_radius"])
            else:
                raise ValueError(f"unknown family: {definition['family']}")
            target = target.view(1, 1, 64, 64, 64) & support
            truth = background.clone()
            truth[target] = int(protocol["target_label"])
            # The registered target is concealed from all full boreholes.
            full_well_mask = (boreholes != -1) & support
            if bool((target & full_well_mask).any()):
                raise RuntimeError(f"registered target intersects a hard borehole: {case_id}")
            condition_values = boreholes.clone()
            condition_mask = (~support) | (boreholes != -1)
            conditioning = runtime.validate_conditioning_pair(truth, boreholes, num_categories=15, target_label=9)
            occupancy = target.float().to(device)
            impedance, slowness = binary_occupancy_to_acoustic(occupancy, support.to(device), properties)
            observation = build_seismic_observation(
                torch.cat((impedance, slowness), dim=1), support.to(device), operator,
                uncertainty_amplitude=float(seismic_metadata["uncertainty_amplitude"]),
                noise_std_amplitude=float(seismic_metadata["noise"]["std_amplitude"]),
                noise_seed=int(seismic_metadata["noise"]["seed"]),
            )
            case_dir = args.output_dir / case_id
            truth_dir = case_dir / "truth_restricted"
            truth_dir.mkdir(parents=True)
            tensors = {
                "condition_values.pt": condition_values,
                "condition_mask.pt": condition_mask,
                "subsurface_mask.pt": support,
                "observed_seismic.pt": observation.values.cpu(),
            }
            for name, value in tensors.items():
                torch.save(value, case_dir / name)
            torch.save(truth.cpu(), truth_dir / "true_model.pt")
            torch.save(target.cpu(), truth_dir / "binary_truth.pt")
            cases[case_id] = {
                "case_definition": definition,
                "target_voxels": int(target.sum()),
                "target_topology": betti_numbers(target[0, 0]),
                "conditioning_report": conditioning,
                "inference_tensor_sha256": {name: tensor_sha256(value) for name, value in tensors.items()},
                "truth_tensor_sha256": {"true_model.pt": tensor_sha256(truth), "binary_truth.pt": tensor_sha256(target)},
                "observation_metadata": observation.metadata,
            }
        manifest.update({
            "run_status": "completed",
            "cases": cases,
            "protocol": runtime.asset_record(args.protocol),
            "input_assets": {
                "base_truth": runtime.asset_record(base / "true_model.pt"),
                "base_boreholes": runtime.asset_record(base / "boreholes.pt"),
                "binary_acoustic_config": runtime.asset_record(binary_config_path),
                "seismic_config": runtime.asset_record(seismic_config_path),
            },
            "truth_role": "observation_generation_and_retrospective_evaluation_only",
        })
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
