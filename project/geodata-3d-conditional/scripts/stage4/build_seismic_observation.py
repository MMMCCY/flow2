#!/usr/bin/env python3
"""Build immutable truth-derived Phase-4c convolutional seismic assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from geology_io_utils import write_json
from guidance.seismic import (
    SEISMIC_PROTOCOL_VERSION,
    acoustic_tables_from_config,
    build_seismic_observation,
    hard_labels_to_acoustic,
    seismic_operator_from_config,
    tensor_sha256,
)


def _load_json(path: Path, name: str) -> Mapping[str, object]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {name} JSON: {path}") from exc
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} must contain a JSON object: {path}")
    return value


def _prepare_output(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output directory: {path}")
    path.mkdir(parents=True, exist_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build normal-incidence convolutional synthetic seismic assets."
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--acoustic-config", type=Path, required=True)
    parser.add_argument("--observation-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--num-categories", type=int, default=15)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.num_categories <= 1:
        raise ValueError("num-categories must be greater than one")
    for path in (args.truth_model, args.acoustic_config, args.observation_config):
        if not path.is_file():
            raise FileNotFoundError(f"required asset not found: {path}")
    _prepare_output(args.output_dir)

    dtype = torch.float32 if args.dtype == "float32" else torch.float64
    device = torch.device(args.device)
    truth = runtime.normalize_single_geology(
        runtime.load_tensor(args.truth_model, map_location="cpu"), "truth_model"
    )
    if not torch.isfinite(truth).all() or not torch.equal(truth, truth.round()):
        raise ValueError("truth model must be finite and integer-valued")
    if int(truth.min()) < -1 or int(truth.max()) > args.num_categories - 2:
        raise ValueError(f"truth labels must be in [-1,{args.num_categories - 2}]")

    acoustic_config = _load_json(args.acoustic_config, "acoustic config")
    observation_config = _load_json(args.observation_config, "observation config")
    tables, acoustic_metadata = acoustic_tables_from_config(
        acoustic_config, args.num_categories
    )
    operator, resolved_observation = seismic_operator_from_config(
        observation_config, grid_shape=truth.shape[2:]
    )
    property_table = tables.property_table
    truth_device = truth.to(device)
    truth_acoustic = hard_labels_to_acoustic(
        truth_device,
        property_table.to(device),
    ).to(dtype=dtype)
    subsurface_mask = (truth_device != -1).to(torch.bool)
    noise = resolved_observation["noise"]
    observation = build_seismic_observation(
        truth_acoustic,
        subsurface_mask,
        operator,
        uncertainty_amplitude=float(resolved_observation["uncertainty_amplitude"]),
        noise_std_amplitude=float(noise["std_amplitude"]),
        noise_seed=int(noise["seed"]),
    )
    reflectivity, interface_time_ms, valid_interfaces = operator.interface_response(
        truth_acoustic[:, 0:1],
        truth_acoustic[:, 1:2],
        subsurface_mask,
    )

    tensors = {
        "density_table_kg_m3.pt": tables.density_kg_m3,
        "velocity_table_m_s.pt": tables.velocity_m_s,
        "impedance_table_kg_m2_s.pt": tables.impedance_kg_m2_s,
        "slowness_table_s_m.pt": tables.slowness_s_m,
        "acoustic_property_table.pt": property_table,
        "truth_acoustic.pt": truth_acoustic.detach().cpu(),
        "subsurface_mask.pt": observation.subsurface_mask.detach().cpu(),
        "observed_seismic.pt": observation.values.detach().cpu(),
        "noiseless_seismic.pt": observation.noiseless.detach().cpu(),
        "seismic_noise.pt": observation.noise.detach().cpu(),
        "sample_mask.pt": observation.sample_mask.detach().cpu(),
        "uncertainty_amplitude.pt": observation.uncertainty.detach().cpu(),
        "wavelet.pt": operator.wavelet(torch.device("cpu"), torch.float64),
    }
    for filename, tensor in tensors.items():
        torch.save(tensor, args.output_dir / filename)

    valid_reflectivity = reflectivity[valid_interfaces]
    valid_times = interface_time_ms[valid_interfaces]
    manifest = {
        "protocol_version": SEISMIC_PROTOCOL_VERSION,
        "stage": "phase4c_seismic_observation_builder_v1",
        "status": "complete",
        "description": (
            "Noiseless full-lateral synthetic convolutional seismic inverse-crime "
            "upper bound; not measured geophysics."
        ),
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "source_assets": {
            "truth_model": runtime.asset_record(args.truth_model),
            "acoustic_config": runtime.asset_record(args.acoustic_config),
            "observation_config": runtime.asset_record(args.observation_config),
            "builder_source": runtime.asset_record(Path(__file__)),
            "seismic_source": runtime.asset_record(PROJECT_DIR / "guidance/seismic.py"),
        },
        "acoustic": acoustic_metadata,
        "observation_config_resolved": resolved_observation,
        "observation": observation.metadata,
        "field_statistics": {
            "minimum_amplitude": float(observation.values.min().item()),
            "maximum_amplitude": float(observation.values.max().item()),
            "mean_amplitude": float(observation.values.mean().item()),
            "standard_deviation_amplitude": float(
                observation.values.std(unbiased=False).item()
            ),
            "all_finite": bool(torch.isfinite(observation.values).all().item()),
            "valid_interface_count": int(valid_interfaces.sum().item()),
            "minimum_valid_twt_ms": float(valid_times.min().item()),
            "maximum_valid_twt_ms": float(valid_times.max().item()),
            "minimum_reflectivity": float(valid_reflectivity.min().item()),
            "maximum_reflectivity": float(valid_reflectivity.max().item()),
        },
        "generated_tensors": {
            filename: {
                "shape": list(tensor.shape),
                "dtype": str(tensor.dtype),
                "sha256": tensor_sha256(tensor),
            }
            for filename, tensor in tensors.items()
        },
        "device": str(device),
        "dtype": str(dtype),
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
