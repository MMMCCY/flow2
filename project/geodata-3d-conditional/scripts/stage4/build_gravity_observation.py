#!/usr/bin/env python3
"""Build immutable truth-derived Phase-4a gravity observation assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from geology_io_utils import write_json
from guidance.gravity import (
    GRAVITY_PROTOCOL_VERSION,
    build_gravity_observation,
    density_table_from_config,
    gravity_operator_from_config,
    hard_labels_to_density,
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
        description="Build full-support rectangular-prism synthetic gravity assets."
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--density-config", type=Path, required=True)
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
    for path in (args.truth_model, args.density_config, args.observation_config):
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
        raise ValueError(
            f"truth labels must be in [-1,{args.num_categories - 2}]"
        )

    density_config = _load_json(args.density_config, "density config")
    observation_config = _load_json(args.observation_config, "observation config")
    density_table, density_metadata = density_table_from_config(
        density_config, args.num_categories
    )
    operator, resolved_observation = gravity_operator_from_config(
        observation_config, grid_shape=truth.shape[2:]
    )
    truth_density = hard_labels_to_density(
        truth.to(device), density_table.to(device)
    ).to(dtype=dtype)
    noise = resolved_observation["noise"]
    observation = build_gravity_observation(
        truth_density,
        operator,
        uncertainty_mgal=float(resolved_observation["uncertainty_mgal"]),
        noise_std_mgal=float(noise["std_mgal"]),
        noise_seed=int(noise["seed"]),
    )

    tensors = {
        "density_table_kg_m3.pt": density_table,
        "truth_density_kg_m3.pt": truth_density.detach().cpu(),
        "observed_gravity_mgal.pt": observation.values_mgal.detach().cpu(),
        "noiseless_gravity_mgal.pt": observation.noiseless_mgal.detach().cpu(),
        "gravity_noise_mgal.pt": observation.noise_mgal.detach().cpu(),
        "survey_mask.pt": observation.survey_mask.detach().cpu(),
        "uncertainty_mgal.pt": observation.uncertainty_mgal.detach().cpu(),
    }
    for filename, tensor in tensors.items():
        torch.save(tensor, args.output_dir / filename)

    manifest = {
        "protocol_version": GRAVITY_PROTOCOL_VERSION,
        "stage": "phase4a_gravity_observation_builder_v1",
        "status": "complete",
        "description": (
            "Noiseless regular-grid synthetic inverse-crime gravity upper bound; "
            "not measured geophysics."
        ),
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "source_assets": {
            "truth_model": runtime.asset_record(args.truth_model),
            "density_config": runtime.asset_record(args.density_config),
            "observation_config": runtime.asset_record(args.observation_config),
            "builder_source": runtime.asset_record(Path(__file__)),
            "gravity_source": runtime.asset_record(PROJECT_DIR / "guidance/gravity.py"),
        },
        "density": density_metadata,
        "observation_config_resolved": resolved_observation,
        "observation": observation.metadata,
        "field_statistics_mgal": {
            "minimum": float(observation.values_mgal.min().item()),
            "maximum": float(observation.values_mgal.max().item()),
            "mean": float(observation.values_mgal.mean().item()),
            "standard_deviation": float(
                observation.values_mgal.std(unbiased=False).item()
            ),
            "all_finite": bool(torch.isfinite(observation.values_mgal).all().item()),
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
