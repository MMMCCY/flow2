#!/usr/bin/env python3
"""Build immutable Stage17 synthetic observations from the frozen case cohort."""

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
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
)
from guidance.seismic import build_seismic_observation, seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json
from scripts.stage17.common import require_frozen_config, resolved_path, validate_asset_record


ROOT = PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution"
DEFAULT_CONFIG = ROOT / "configs/observation_generation_v2.json"
DEFAULT_OUTPUT = ROOT / "cases/observations_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def _case_paths(case_record: dict[str, object]) -> tuple[Path, Path, Path, Path]:
    manifest_path = resolved_path(case_record["manifest_path"])
    if runtime.file_sha256(manifest_path) != str(case_record["manifest_sha256"]):
        raise ValueError(f"frozen benchmark manifest changed: {case_record['case_id']}")
    case_root = manifest_path.parent
    return (
        manifest_path,
        case_root / "truth/true_model.pt",
        case_root / "condition/condition_values.pt",
        case_root / "condition/condition_mask.pt",
    )


def _columnwise_contiguous_support(truth: torch.Tensor) -> torch.Tensor:
    """Fill support below the highest non-air cell in each vertical column."""
    nonair = truth != -1
    if nonair.ndim != 5 or nonair.shape[1] != 1 or not bool(nonair.any(dim=-1).all()):
        raise ValueError("every Stage17 column must contain at least one non-air cell")
    z = torch.arange(nonair.shape[-1], device=nonair.device).view(1, 1, 1, 1, -1)
    highest = torch.where(nonair, z, torch.full_like(z, -1)).amax(dim=-1, keepdim=True)
    return z <= highest


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    require_frozen_config(config, "stage17_observation_generation_v2")
    if config.get("subsurface_support_policy") != "columnwise_fill_below_highest_nonair_v1":
        raise ValueError("Stage17 observation support policy changed")
    if config.get("enclosed_air_acoustic_policy") != "binary_background_endpoint":
        raise ValueError("Stage17 enclosed-air acoustic policy changed")
    registry_path = validate_asset_record(config["case_registry"], "case_registry")
    binary_path = validate_asset_record(config["binary_acoustic_config"], "binary_acoustic_config")
    seismic_path = validate_asset_record(config["seismic_config"], "seismic_config")
    registry = read_json(registry_path)
    require_frozen_config(registry, "stage17_case_registry_v1")
    cases = registry.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("Stage17 requires the five-case frozen registry")

    binary_config = read_json(binary_path)
    source_record = binary_config.get("source_acoustic_config")
    if not isinstance(source_record, dict):
        raise TypeError("binary acoustic config lacks source record")
    source_path = validate_asset_record(source_record, "source_acoustic_config")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    seismic_config = read_json(seismic_path)
    device = torch.device(args.device)

    args.output_dir.mkdir(parents=True)
    master = base_manifest("stage17_observation_generation_run_v1", Path(__file__), args.config)
    master.update({
        "run_status": "running",
        "truth_loaded_by_observation_builder": True,
        "truth_loaded_by_inversion_runner": False,
        "truth_role": "synthetic_observation_generation_and_retrospective_evaluation_only",
        "case_order": [str(case["case_id"]) for case in cases],
    })
    write_json(args.output_dir / "manifest.json", master)
    case_records: list[dict[str, object]] = []
    try:
        for case in cases:
            case_id = str(case["case_id"])
            manifest_path, truth_path, condition_values_path, condition_mask_path = _case_paths(case)
            truth = normalize_volume(runtime.load_tensor(truth_path), f"{case_id} truth").long()
            condition_values = normalize_volume(
                runtime.load_tensor(condition_values_path), f"{case_id} condition_values"
            ).long()
            condition_mask = normalize_volume(
                runtime.load_tensor(condition_mask_path), f"{case_id} condition_mask"
            ).bool()
            original_nonair = truth != -1
            subsurface = _columnwise_contiguous_support(truth)
            filled_enclosed_air = subsurface & ~original_nonair
            mismatch = int(((condition_values != truth) & condition_mask).sum())
            if mismatch:
                raise ValueError(f"condition/truth mismatch: {case_id}={mismatch}")
            binary_truth = ((truth == int(config["target_label"])) & subsurface).float()
            binary_fixed_mask = condition_mask & subsurface
            binary_fixed_values = torch.zeros_like(binary_truth)
            binary_fixed_values[binary_fixed_mask] = (
                condition_values[binary_fixed_mask] == int(config["target_label"])
            ).float()
            operator, operator_metadata = seismic_operator_from_config(
                seismic_config, grid_shape=truth.shape[2:]
            )
            impedance, slowness = binary_occupancy_to_acoustic(
                binary_truth.to(device), subsurface.to(device), properties
            )
            observation = build_seismic_observation(
                torch.cat((impedance, slowness), dim=1),
                subsurface.to(device),
                operator,
                uncertainty_amplitude=float(operator_metadata["uncertainty_amplitude"]),
                noise_std_amplitude=float(operator_metadata["noise"]["std_amplitude"]),
                noise_seed=int(operator_metadata["noise"]["seed"]),
            )
            closure = operator(impedance, slowness, subsurface.to(device))
            closure_error = float((closure - observation.values).abs().max().cpu())
            if closure_error > 1e-7:
                raise RuntimeError(f"seismic observation closure failed: {case_id}={closure_error}")

            case_dir = args.output_dir / case_id
            truth_dir = case_dir / "truth_restricted"
            truth_dir.mkdir(parents=True)
            inference_tensors = {
                "observed_seismic.pt": observation.values.cpu(),
                "sample_mask.pt": observation.sample_mask.cpu(),
                "uncertainty.pt": observation.uncertainty.cpu(),
                "subsurface_mask.pt": subsurface.cpu(),
                "condition_values.pt": condition_values.cpu(),
                "condition_mask.pt": condition_mask.cpu(),
                "binary_fixed_values.pt": binary_fixed_values.cpu(),
                "binary_fixed_mask.pt": binary_fixed_mask.cpu(),
            }
            for name, tensor in inference_tensors.items():
                torch.save(tensor, case_dir / name)
            torch.save(binary_truth.cpu(), truth_dir / "binary_truth.pt")
            output_hashes = {name: tensor_sha256(tensor) for name, tensor in inference_tensors.items()}
            case_manifest = base_manifest(
                "stage17_synthetic_observation_case_v1", Path(__file__), args.config
            )
            case_manifest.update({
                "run_status": "completed",
                "case_id": case_id,
                "root_seed": int(case["root_seed"]),
                "truth_loaded_by_observation_builder": True,
                "truth_loaded_by_inversion_runner": False,
                "truth_role": "synthetic_observation_generation_and_retrospective_evaluation_only",
                "source_case_manifest": runtime.asset_record(manifest_path),
                "input_assets": {
                    "truth_model": runtime.asset_record(truth_path),
                    "condition_values": runtime.asset_record(condition_values_path),
                    "condition_mask": runtime.asset_record(condition_mask_path),
                },
                "output_tensor_sha256": output_hashes,
                "truth_tensor_sha256": {"truth_restricted/binary_truth.pt": tensor_sha256(binary_truth)},
                "target_voxels": int(binary_truth.sum()),
                "hidden_target_voxels": int((binary_truth.bool() & ~condition_mask).sum()),
                "conditioned_target_voxels": int((binary_truth.bool() & condition_mask).sum()),
                "condition_voxels": int(condition_mask.sum()),
                "seismic_support_voxels": int(subsurface.sum()),
                "filled_enclosed_air_voxels": int(filled_enclosed_air.sum()),
                "subsurface_support_policy": config["subsurface_support_policy"],
                "enclosed_air_acoustic_policy": config["enclosed_air_acoustic_policy"],
                "condition_mismatches": mismatch,
                "forward_closure_max_abs": closure_error,
                "operator_metadata": operator_metadata,
            })
            write_json(case_dir / "manifest.json", case_manifest)
            case_records.append({
                "case_id": case_id,
                "manifest": runtime.asset_record(case_dir / "manifest.json"),
                "output_tensor_sha256": output_hashes,
            })
            print(f"Stage17 observation {case_id} completed", flush=True)
        master.update({
            "run_status": "completed",
            "case_count": len(case_records),
            "cases": case_records,
            "binary_acoustic_config": runtime.asset_record(binary_path),
            "source_acoustic_config": runtime.asset_record(source_path),
            "seismic_config": runtime.asset_record(seismic_path),
            "parameter_sweep_performed": False,
            "training_performed": False,
        })
        write_json(args.output_dir / "manifest.json", master)
    except Exception as exc:
        master.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", master)
        raise


if __name__ == "__main__":
    main()
