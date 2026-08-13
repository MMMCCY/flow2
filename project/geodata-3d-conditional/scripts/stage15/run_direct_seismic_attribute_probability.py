#!/usr/bin/env python3
"""Calibrate and apply the fixed Stage15-C direct seismic attribute lookup."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for root in (PROJECT_DIR, REPOSITORY_ROOT, STRUCTURALGEO_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from geogen.generation.model_generators import MarkovGeostoryGenerator
from geogen.generation.rng_contract import RNG_CONTRACT_VERSION
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
)
from guidance.full_structuralgeo_benchmark import (
    MODEL_BOUNDS,
    MODEL_RESOLUTION,
    canonical_json_sha256,
)
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from guidance.seismic_attribute_probability import (
    apply_probability_lookup,
    depth_resample_local_energy,
    fit_empirical_probability_lookup,
    local_seismic_energy,
    quantile_bin_edges,
)
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    validate_asset,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/direct_seismic_attribute_probability_v1.json"
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = (
    PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
)
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_CALIBRATION_OUTPUT = EXPERIMENT_ROOT / "direct_attribute/calibration_n128_v1"
DEFAULT_HELDOUT_OUTPUT = EXPERIMENT_ROOT / "direct_attribute/cond_generation_0_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--calibration-output", type=Path, default=DEFAULT_CALIBRATION_OUTPUT)
    parser.add_argument("--heldout-output", type=Path, default=DEFAULT_HELDOUT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def validate_protocol(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("schema") != "stage15_direct_seismic_attribute_probability_v1":
        raise ValueError("unexpected Stage15-C protocol schema")
    if config.get("status") != "frozen_before_calibration":
        raise ValueError("Stage15-C protocol is not frozen")
    seeds = [int(seed) for seed in config["calibration_seeds"]]
    if len(seeds) != 128 or len(set(seeds)) != 128:
        raise ValueError("Stage15-C requires exactly 128 unique calibration seeds")
    if config.get("calibration_case_count") != 128:
        raise ValueError("calibration case count must be frozen at 128")
    if tuple(config.get("grid_shape", ())) != (64, 64, 64):
        raise ValueError("grid shape must be frozen at 64^3")
    if config.get("target_label") != 9:
        raise ValueError("target label must be raw label 9")
    attribute = config["attribute"]
    calibration = config["calibration"]
    time_depth = config["time_to_depth"]
    if attribute.get("window_num_samples") != 17 or attribute.get("window_duration_ms") != 128.0:
        raise ValueError("attribute window must be frozen at 17 samples / 128 ms")
    if calibration.get("quantile_bin_count") != 64:
        raise ValueError("quantile bins must be frozen at 64")
    if calibration.get("class_balancing") is not False:
        raise ValueError("natural label prevalence must be preserved")
    if time_depth.get("background_velocity_m_s") != 2500.0:
        raise ValueError("background velocity must be frozen at 2500 m/s")
    if time_depth.get("vertical_cell_size_m") != 50.0:
        raise ValueError("vertical cell size must be frozen at 50 m")
    if config.get("parameter_sweep") is not False:
        raise ValueError("parameter sweep is forbidden")
    return {"seeds": seeds, "attribute": attribute, "calibration": calibration, "time_depth": time_depth}


def generate_structuralgeo_case(seed: int) -> tuple[torch.Tensor, dict[str, object]]:
    generator = MarkovGeostoryGenerator(
        model_bounds=MODEL_BOUNDS,
        model_resolution=MODEL_RESOLUTION,
        config=None,
        root_seed=int(seed),
    )
    model, metadata = generator.generate_model_with_metadata()
    model.fill_nans()
    geology = torch.from_numpy(np.asarray(model.get_data_grid())).view(1, 1, 64, 64, 64)
    if not torch.isfinite(geology).all() or not torch.equal(geology, geology.round()):
        raise ValueError(f"StructuralGeo seed {seed} produced invalid categorical geology")
    geology = geology.long()
    if int(geology.min()) < -1 or int(geology.max()) > 13:
        raise ValueError(f"StructuralGeo seed {seed} produced labels outside -1..13")
    return geology, metadata


def contiguous_subsurface_from_geology(geology: torch.Tensor) -> torch.Tensor:
    """Fill below each topmost non-air voxel to form a valid acoustic column."""
    if geology.ndim != 5 or geology.shape[1] != 1:
        raise ValueError("geology must have shape [B,1,X,Y,Z]")
    non_air = geology != -1
    if bool((~non_air).all(dim=-1).any()):
        raise ValueError("each StructuralGeo column must contain at least one non-air voxel")
    nz = geology.shape[-1]
    z = torch.arange(nz, device=geology.device).view(1, 1, 1, 1, nz)
    surface_z = torch.where(non_air, z, z.new_full((), -1)).amax(dim=-1, keepdim=True)
    return (z <= surface_z).contiguous()


def build_attribute(
    seismic: torch.Tensor,
    subsurface: torch.Tensor,
    *,
    protocol: Mapping[str, object],
    seismic_parameters: Mapping[str, object],
) -> torch.Tensor:
    window = int(protocol["attribute"]["window_num_samples"])
    energy = local_seismic_energy(seismic, window)
    return depth_resample_local_energy(
        energy,
        subsurface,
        sample_interval_ms=float(seismic_parameters["time_sampling"]["sample_interval_ms"]),
        vertical_cell_size_m=float(protocol["time_to_depth"]["vertical_cell_size_m"]),
        background_velocity_m_s=float(protocol["time_to_depth"]["background_velocity_m_s"]),
    )


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.calibration_output)
    refuse_nonempty(args.heldout_output)
    config = read_json(args.config)
    frozen = validate_protocol(config)

    binary_config = read_json(args.binary_acoustic_config)
    source_record = binary_config.get("source_acoustic_config")
    if not isinstance(source_record, Mapping):
        raise ValueError("binary acoustic config has no source record")
    source_path = REPOSITORY_ROOT / str(source_record["path"])
    validate_asset(source_path, str(source_record["sha256"]))
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    if properties.background_velocity != float(frozen["time_depth"]["background_velocity_m_s"]):
        raise ValueError("time-depth velocity differs from the binary acoustic background velocity")

    seismic_config = read_json(args.seismic_config)
    operator, seismic_parameters = seismic_operator_from_config(
        seismic_config, grid_shape=(64, 64, 64)
    )
    if operator.wavelet_num_samples != int(frozen["attribute"]["window_num_samples"]):
        raise ValueError("attribute window differs from the existing Ricker wavelet support")
    if operator.wavelet_duration_ms != float(frozen["attribute"]["window_duration_ms"]):
        raise ValueError("attribute duration differs from the existing Ricker duration")
    if seismic_parameters.get("lateral_mixing", False):
        raise ValueError("Stage15-C forbids lateral mixing")

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    args.calibration_output.mkdir(parents=True)
    calibration_manifest = base_manifest(
        "stage15_c_direct_attribute_calibration_v1", Path(__file__), args.config
    )
    write_json(args.calibration_output / "calibration_manifest.json", calibration_manifest)
    started = time.perf_counter()
    try:
        attributes: list[torch.Tensor] = []
        labels: list[torch.Tensor] = []
        case_records: list[dict[str, object]] = []
        for case_index, seed in enumerate(frozen["seeds"]):
            geology, metadata = generate_structuralgeo_case(seed)
            subsurface = contiguous_subsurface_from_geology(geology)
            internal_air_as_background = subsurface & (geology == -1)
            binary_label = ((geology == 9) & subsurface).float()
            impedance, slowness = binary_occupancy_to_acoustic(
                binary_label.to(device), subsurface.to(device), properties
            )
            seismic = operator.forward(
                impedance, slowness, subsurface.to(device), require_all_interfaces_in_window=True
            )
            attribute = build_attribute(
                seismic,
                subsurface.to(device),
                protocol=config,
                seismic_parameters=seismic_parameters,
            )
            mask_cpu = subsurface.cpu().bool()
            attributes.append(attribute.detach().cpu()[mask_cpu])
            labels.append(binary_label.cpu().bool()[mask_cpu])
            history_payload = {
                "markov_sequence": metadata["markov_sequence"],
                "events": metadata["events"],
                "packed_history": metadata["packed_history"],
                "unpacked_history": metadata["unpacked_history"],
            }
            values, counts = torch.unique(geology, return_counts=True)
            case_records.append(
                {
                    "case_index": case_index,
                    "root_seed": seed,
                    "geology_tensor_sha256": tensor_sha256(geology),
                    "subsurface_tensor_sha256": tensor_sha256(subsurface),
                    "binary_label9_tensor_sha256": tensor_sha256(binary_label),
                    "synthetic_seismic_tensor_sha256": tensor_sha256(seismic.detach().cpu()),
                    "attribute_tensor_sha256": tensor_sha256(attribute.detach().cpu()),
                    "subsurface_voxels": int(subsurface.sum()),
                    "internal_raw_air_mapped_to_binary_background_voxels": int(
                        internal_air_as_background.sum()
                    ),
                    "label9_voxels": int(binary_label.sum()),
                    "raw_label_counts": {
                        str(int(value)): int(count) for value, count in zip(values, counts)
                    },
                    "markov_sequence": metadata["markov_sequence"],
                    "event_subtypes": [event["subtype"] for event in metadata["events"]],
                    "history_sha256": canonical_json_sha256(history_payload),
                }
            )
            if (case_index + 1) % 8 == 0:
                print(f"calibration cases: {case_index + 1}/128", flush=True)

        pooled_attribute = torch.cat(attributes)
        pooled_label = torch.cat(labels)
        edges = quantile_bin_edges(
            pooled_attribute, int(frozen["calibration"]["quantile_bin_count"])
        )
        lookup, bin_totals, bin_positives = fit_empirical_probability_lookup(
            pooled_attribute, pooled_label, edges
        )
        torch.save(edges, args.calibration_output / "attribute_bin_edges.pt")
        torch.save(lookup, args.calibration_output / "attribute_probability_lookup.pt")
        calibration_manifest.update(
            {
                "run_status": "completed",
                "calibration_case_count": len(case_records),
                "calibration_seeds": frozen["seeds"],
                "cond_generation_0_excluded": True,
                "cond_generation_0_assets_read_during_calibration": [],
                "generator": {
                    "name": "StructuralGeo MarkovGeostoryGenerator",
                    "rng_contract_version": RNG_CONTRACT_VERSION,
                    "model_bounds": [list(row) for row in MODEL_BOUNDS],
                    "model_resolution": list(MODEL_RESOLUTION),
                    "raw_labels_preserved": True,
                    "labels_10_through_13_merged_into_label9": False,
                    "calibration_subsurface_policy": config[
                        "calibration_subsurface_policy"
                    ],
                    "internal_raw_air_below_topmost_non_air_mapping": (
                        "binary background raw-label0 reference"
                    ),
                },
                "binary_acoustic_config": runtime.asset_record(args.binary_acoustic_config),
                "source_acoustic_config": runtime.asset_record(source_path),
                "binary_property_values": properties.__dict__,
                "seismic_config": runtime.asset_record(args.seismic_config),
                "seismic_parameters": seismic_parameters,
                "attribute_definition": config["attribute"],
                "time_to_depth": config["time_to_depth"],
                "case_records": case_records,
                "pooled_subsurface_voxels": int(pooled_label.numel()),
                "pooled_label9_voxels": int(pooled_label.sum()),
                "natural_label9_prevalence": float(pooled_label.float().mean()),
                "bin_total_counts": bin_totals.tolist(),
                "bin_label9_counts": bin_positives.tolist(),
                "empty_quantile_bins_due_to_ties": int((bin_totals == 0).sum()),
                "attribute_bin_edges": runtime.asset_record(
                    args.calibration_output / "attribute_bin_edges.pt"
                ),
                "attribute_probability_lookup": runtime.asset_record(
                    args.calibration_output / "attribute_probability_lookup.pt"
                ),
                "attribute_bin_edges_tensor_sha256": tensor_sha256(edges),
                "attribute_probability_lookup_tensor_sha256": tensor_sha256(lookup),
                "class_balancing_performed": False,
                "parameter_sweep_performed": False,
                "neural_network_training_performed": False,
                "runtime_seconds": time.perf_counter() - started,
            }
        )
        write_json(args.calibration_output / "calibration_manifest.json", calibration_manifest)
    except Exception as exc:
        calibration_manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.calibration_output / "calibration_manifest.json", calibration_manifest)
        raise

    args.heldout_output.mkdir(parents=True)
    heldout_manifest = base_manifest(
        "stage15_c_heldout_direct_attribute_probability_v1", Path(__file__), args.config
    )
    write_json(args.heldout_output / "manifest.json", heldout_manifest)
    try:
        observed_path = args.observation_dir / "observed_seismic.pt"
        subsurface_path = args.observation_dir / "subsurface_mask.pt"
        observed = runtime.load_tensor(observed_path).float().to(device)
        subsurface = normalize_volume(runtime.load_tensor(subsurface_path), "subsurface_mask").bool().to(device)
        expected_inputs = config["held_out_inputs"]
        if tensor_sha256(observed.cpu()) != expected_inputs["observed_seismic_tensor_sha256"]:
            raise ValueError("held-out observed seismic differs from frozen tensor hash")
        if tensor_sha256(subsurface.cpu()) != expected_inputs["subsurface_mask_tensor_sha256"]:
            raise ValueError("held-out subsurface mask differs from frozen tensor hash")
        attribute = build_attribute(
            observed,
            subsurface,
            protocol=config,
            seismic_parameters=seismic_parameters,
        )
        probability = apply_probability_lookup(
            attribute,
            subsurface,
            edges.to(device),
            lookup.to(device),
        )
        if tuple(attribute.shape) != (1, 1, 64, 64, 64):
            raise ValueError("held-out attribute volume must have shape [1,1,64,64,64]")
        if not torch.isfinite(probability).all() or bool(((probability < 0) | (probability > 1)).any()):
            raise ValueError("held-out probability must be continuous finite [0,1]")
        attribute_path = args.heldout_output / "seismic_attribute_volume.pt"
        probability_path = args.heldout_output / "seismic_probability_volume.pt"
        torch.save(attribute.cpu(), attribute_path)
        torch.save(probability.cpu(), probability_path)
        heldout_manifest.update(
            {
                "run_status": "completed",
                "case_id": "cond_generation_0",
                "case_role": "held_out_only",
                "truth_loaded_by_runner": False,
                "observed_seismic": runtime.asset_record(observed_path),
                "subsurface_mask": runtime.asset_record(subsurface_path),
                "calibration_manifest": runtime.asset_record(
                    args.calibration_output / "calibration_manifest.json"
                ),
                "attribute_bin_edges": runtime.asset_record(
                    args.calibration_output / "attribute_bin_edges.pt"
                ),
                "attribute_probability_lookup": runtime.asset_record(
                    args.calibration_output / "attribute_probability_lookup.pt"
                ),
                "seismic_attribute_volume": runtime.asset_record(attribute_path),
                "seismic_probability_volume": runtime.asset_record(probability_path),
                "output_tensor_sha256": {
                    "seismic_attribute_volume.pt": tensor_sha256(attribute.cpu()),
                    "seismic_probability_volume.pt": tensor_sha256(probability.cpu()),
                },
                "attribute_definition": config["attribute"],
                "time_to_depth": config["time_to_depth"],
                "probability_range_subsurface": [
                    float(probability[subsurface].min()),
                    float(probability[subsurface].max()),
                ],
                "thresholding_performed": False,
                "parameter_sweep_performed": False,
                "geological_inversion_performed": False,
                "pcn_performed": False,
                "flow_sampling_or_guidance_performed": False,
            }
        )
        write_json(args.heldout_output / "manifest.json", heldout_manifest)
    except Exception as exc:
        heldout_manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.heldout_output / "manifest.json", heldout_manifest)
        raise


if __name__ == "__main__":
    main()
