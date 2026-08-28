"""Shared frozen-input validation and hard-seismic helpers for Stage18."""

from __future__ import annotations

import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import torch

import inference_runtime as runtime
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
)
from guidance.property_evaluation import size_stratified_component_metrics
from guidance.probability_volume import tensor_sha256
from guidance.seismic import seismic_operator_from_config
from scripts.stage15.common import normalize_volume, read_json
from scripts.stage17.common import read_csv, validate_asset_record


STAGE18_SCHEMA = "stage18_continuous_property_target_v1"
NEW_ARM = "CONTINUOUS_PROPERTY_TARGET"
REFERENCE_ARMS = ("FLOW_ONLY", "TRAJECTORY_EVIDENCE")


def validate_stage18_config(config: Mapping[str, object]) -> None:
    """Enforce the one-variable Stage18 protocol before any output is written."""
    required = {
        "schema": STAGE18_SCHEMA,
        "status": "frozen_before_run",
        "authorized_arm": NEW_ARM,
        "target_semantics": "continuous_normalized_binary_property_target_v1",
        "target_properties_policy": "binary_impedance_score_no_threshold",
        "confidence_policy": "binary_impedance_score_times_free_subsurface_fixed_from_stage17c",
        "controlled_variable": "property_target_semantics_only",
        "spatial_confidence_weighting_held_fixed": True,
        "thresholding": False,
        "score_rescaling": False,
        "case_specific_normalization": False,
        "parameter_sweep": False,
        "training_performed": False,
        "truth_loaded_by_runner": False,
        "primary_statistical_unit": "independent_geology_case",
        "mass_mask_policy": "subsurface_and_not_condition_mask",
        "stop_after_stage18": True,
    }
    for key, expected in required.items():
        if config.get(key) != expected:
            raise ValueError(f"Stage18 frozen field changed: {key}")


def load_stage18_references(
    config: Mapping[str, object],
) -> tuple[dict[str, object], Path, list[dict[str, str]]]:
    """Load and validate immutable Stage17C config/run/sample references."""
    validate_stage18_config(config)
    stage17_config_path = validate_asset_record(config["stage17c_config"], "Stage17C config")
    run_manifest_path = validate_asset_record(
        config["stage17c_run_manifest"], "Stage17C run manifest"
    )
    sample_manifest_path = validate_asset_record(
        config["stage17c_sample_manifest"], "Stage17C sample manifest"
    )
    stage17_config = read_json(stage17_config_path)
    run_manifest = read_json(run_manifest_path)
    if run_manifest.get("run_status") != "completed" or run_manifest.get("smoke_subset") is not False:
        raise ValueError("Stage17C formal reference is incomplete or is smoke")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage17C truth firewall failed")
    if stage17_config.get("authorized_arms") != list(REFERENCE_ARMS):
        raise ValueError("Stage17C reference arms changed")
    if stage17_config.get("source_seeds") != [42, 142, 242]:
        raise ValueError("Stage17C source seeds changed")
    records = read_csv(sample_manifest_path)
    if len(records) != 30:
        raise ValueError("Stage17C formal reference must contain 30 outputs")
    return stage17_config, run_manifest_path.parent, records


def reference_index(records: Sequence[Mapping[str, str]]) -> dict[tuple[str, int, str], Mapping[str, str]]:
    index = {
        (str(row["case_id"]), int(row["source_seed"]), str(row["arm"])): row
        for row in records
    }
    if len(index) != len(records):
        raise ValueError("duplicate Stage17C reference records")
    return index


def load_case_tensors(
    case: Mapping[str, object],
) -> dict[str, torch.Tensor | dict[str, object] | Path]:
    """Load immutable observation/condition/truth tensors for one Stage17 case."""
    case_id = str(case["case_id"])
    assets = case["observation_assets"]
    observation_manifest_path = validate_asset_record(
        assets["observation_manifest"], f"{case_id}/observation_manifest"
    )
    observation_manifest = read_json(observation_manifest_path)
    paths = {
        "observed": validate_asset_record(assets["observed_seismic"], f"{case_id}/observed"),
        "support": validate_asset_record(assets["subsurface_mask"], f"{case_id}/support"),
        "condition_values": validate_asset_record(
            assets["condition_values"], f"{case_id}/condition_values"
        ),
        "condition_mask": validate_asset_record(
            assets["condition_mask"], f"{case_id}/condition_mask"
        ),
        "truth": validate_asset_record(
            observation_manifest["input_assets"]["truth_model"], f"{case_id}/truth"
        ),
    }
    tensors: dict[str, torch.Tensor | dict[str, object] | Path] = {
        "observed": runtime.load_tensor(paths["observed"]).float(),
        "support": normalize_volume(runtime.load_tensor(paths["support"]), "support").bool(),
        "condition_values": normalize_volume(
            runtime.load_tensor(paths["condition_values"]), "condition_values"
        ).long(),
        "condition_mask": normalize_volume(
            runtime.load_tensor(paths["condition_mask"]), "condition_mask"
        ).bool(),
        "truth": normalize_volume(runtime.load_tensor(paths["truth"]), "truth").long(),
        "observation_manifest": observation_manifest,
        "observation_manifest_path": observation_manifest_path,
    }
    for name in ("observed", "support", "condition_values", "condition_mask"):
        expected = observation_manifest["output_tensor_sha256"][
            {
                "observed": "observed_seismic.pt",
                "support": "subsurface_mask.pt",
                "condition_values": "condition_values.pt",
                "condition_mask": "condition_mask.pt",
            }[name]
        ]
        if tensor_sha256(tensors[name]) != expected:
            raise ValueError(f"Stage17 observation tensor changed: {case_id}/{name}")
    return tensors


def binary_physics_from_manifest(
    observation_manifest: Mapping[str, object],
    grid_shape: Sequence[int],
):
    """Reconstruct exactly the frozen Stage17 binary observation model."""
    parameters = observation_manifest["full_parameters"]
    binary_path = validate_asset_record(parameters["binary_acoustic_config"], "binary config")
    seismic_path = validate_asset_record(parameters["seismic_config"], "seismic config")
    binary_config = read_json(binary_path)
    source_path = validate_asset_record(binary_config["source_acoustic_config"], "acoustic source")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    operator, metadata = seismic_operator_from_config(read_json(seismic_path), grid_shape=grid_shape)
    return properties, operator, metadata


@torch.no_grad()
def hard_seismic_metrics(
    decoded: torch.Tensor,
    support: torch.Tensor,
    observed: torch.Tensor,
    properties,
    operator,
    device: torch.device,
) -> dict[str, float]:
    """Evaluate raw-label-9 occupancy with the frozen binary seismic model."""
    geology = normalize_volume(decoded, "decoded_geology").long()
    occupancy = (geology == 9).float().to(device)
    support_device = support.to(device)
    impedance, slowness = binary_occupancy_to_acoustic(
        occupancy, support_device, properties
    )
    predicted = operator(impedance, slowness, support_device)
    residual = predicted - observed.to(device=device, dtype=predicted.dtype)
    mse = float(residual.square().mean().cpu())
    return {"hard_seismic_mse": mse, "hard_seismic_rmse": math.sqrt(mse)}


@torch.no_grad()
def truth_observation_closure(
    truth: torch.Tensor,
    support: torch.Tensor,
    observed: torch.Tensor,
    properties,
    operator,
    device: torch.device,
) -> dict[str, float]:
    """Replay truth binary occupancy and quantify closure to frozen observation."""
    occupancy = (normalize_volume(truth, "truth") == 9).float().to(device)
    support_device = support.to(device)
    impedance, slowness = binary_occupancy_to_acoustic(
        occupancy, support_device, properties
    )
    predicted = operator(impedance, slowness, support_device)
    residual = predicted - observed.to(device=device, dtype=predicted.dtype)
    mse = float(residual.square().mean().cpu())
    return {
        "truth_observation_hard_seismic_mse": mse,
        "truth_observation_hard_seismic_rmse": math.sqrt(mse),
        "truth_observation_hard_seismic_max_abs": float(residual.abs().max().cpu()),
    }


def component_metrics(decoded: torch.Tensor) -> dict[str, object]:
    """Return mandatory six-connected target topology diagnostics."""
    metrics = size_stratified_component_metrics(
        normalize_volume(decoded, "decoded_geology").long() == 9
    )
    total = int((normalize_volume(decoded, "decoded_geology").long() == 9).sum())
    largest = int(metrics["target_component_1_voxels"])
    metrics["target_connected_components"] = sum(
        int(metrics[f"target_component_{rank}_voxels"]) > 0
        for rank in range(1, 9)
    )
    # The full component count and largest fraction already exist in the
    # Stage17 metrics; callers merge those authoritative values.  This fallback
    # is exact for largest fraction and top-k masses.
    metrics["largest_component_fraction"] = largest / total if total else float("nan")
    return metrics


def median(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return float(statistics.median(finite)) if finite else float("nan")
