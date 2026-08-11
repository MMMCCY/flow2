"""Immutable-artifact helpers for Stage 10 inference-visible programs."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments/stage10_geophysical_probability_bridge"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.geophysical_probability_bridge import (
    class_channel,
    validate_grid_alignment,
    validate_probabilities,
)
from guidance.prior_ensemble import file_sha256
from scripts.stage9.common import load_tensor_record, read_json
from scripts.stage9.run_prior_ensemble import load_inference_case


BRIDGE_CASE_SCHEMA = "stage10_truth_blind_bridge_case_v1"
BRIDGE_COLLECTION_SCHEMA = "stage10_truth_blind_bridge_collection_v1"


def resolve_repository_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_frozen_config(path: Path | None = None) -> dict[str, object]:
    config_path = path or EXPERIMENT_DIR / "configs/frozen_experiment_config.json"
    config = read_json(config_path)
    expected = {
        "schema": "stage10_geophysical_probability_bridge_config_v1",
        "status": "frozen_before_stage10_truth_evaluation",
        "model_weight_policy": "ema_trainable_raw_frozen_embedding_v1",
        "integrator": "fixed_euler_midpoint_v1",
        "n_euler_steps": 32,
        "truth_visible_to_bridge_builder": False,
        "truth_visible_to_flow_runner": False,
        "truth_visible_sample_selection": False,
        "neural_weight_optimization_authorized": False,
        "guidance_parameter_sweep_authorized": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Stage10 config {field} must be {value!r}")
    cases = config.get("case_ids")
    if not isinstance(cases, list) or len(cases) != 3 or len(set(cases)) != 3:
        raise ValueError("Stage10 requires exactly three unique registered cases")
    return config


def bridge_case_dir(config: Mapping[str, object], case_id: str) -> Path:
    if case_id not in config["case_ids"]:
        raise ValueError(f"unregistered Stage10 case: {case_id}")
    return EXPERIMENT_DIR / "bridge" / case_id


def inference_case_dir(config: Mapping[str, object], case_id: str) -> Path:
    return resolve_repository_path(config["cases_root"]) / case_id / "inference"


def retrospective_case_dir(config: Mapping[str, object], case_id: str) -> Path:
    return resolve_repository_path(config["cases_root"]) / case_id / "retrospective"


def stage9_pool_dir(config: Mapping[str, object], case_id: str) -> Path:
    return resolve_repository_path(config["stage9_pool_root"]) / case_id / "pool"


def load_completed_bridge_case(
    case_dir: Path,
    *,
    expected_case_id: str,
    expected_config_sha256: str,
    expected_class_model_sha256: str,
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Validate one frozen bridge without accepting or opening truth."""
    manifest_path = Path(case_dir) / "manifest.json"
    manifest = read_json(manifest_path)
    expected = {
        "schema": BRIDGE_CASE_SCHEMA,
        "status": "complete_frozen_before_truth_evaluation",
        "case_id": expected_case_id,
        "truth_tensor_received": False,
        "truth_property_received": False,
        "truth_visible_selection": False,
        "frozen_config_sha256": expected_config_sha256,
        "class_model_sha256": expected_class_model_sha256,
    }
    for field, value in expected.items():
        if manifest.get(field) != value:
            raise ValueError(f"bridge manifest {field} must be {value!r}")
    records = manifest.get("generated_tensors")
    required = {
        "property_samples",
        "property_mean",
        "property_uncertainty",
        "probability_all_classes",
        "probability_label9",
        "entropy",
    }
    if not isinstance(records, Mapping) or set(records) != required:
        raise ValueError("bridge tensor set is incomplete or drifted")
    tensors = {
        name: load_tensor_record(case_dir, record)
        for name, record in records.items()
    }
    validate_grid_alignment(*tensors.values())
    probability = tensors["probability_all_classes"]
    validate_probabilities(probability)
    target_channel = int(manifest["target_probability_channel"])
    if not torch.equal(
        tensors["probability_label9"],
        probability[:, target_channel : target_channel + 1],
    ):
        raise ValueError("saved label9 probability differs from all-class bridge")
    mean = tensors["property_samples"].mean(dim=0, keepdim=True)
    std = tensors["property_samples"].std(dim=0, unbiased=False, keepdim=True)
    if not torch.equal(mean.float(), tensors["property_mean"].float()):
        raise ValueError("saved property mean differs from posterior samples")
    if not torch.equal(std.float(), tensors["property_uncertainty"].float()):
        raise ValueError("saved property uncertainty differs from posterior samples")
    return manifest, tensors


def validate_bridge_collection(
    config: Mapping[str, object],
) -> dict[str, tuple[dict[str, object], dict[str, torch.Tensor]]]:
    """Validate all bridge outputs and their collection manifest before truth."""
    collection_path = EXPERIMENT_DIR / "bridge/manifest.json"
    collection = read_json(collection_path)
    config_path = EXPERIMENT_DIR / "configs/frozen_experiment_config.json"
    class_path = EXPERIMENT_DIR / "configs/petrophysical_class_model.json"
    expected_config_hash = file_sha256(config_path)
    expected_class_hash = file_sha256(class_path)
    if collection.get("schema") != BRIDGE_COLLECTION_SCHEMA:
        raise ValueError("invalid Stage10 bridge collection schema")
    if collection.get("status") != "complete_frozen_before_truth_evaluation":
        raise RuntimeError("Stage10 bridge collection is incomplete")
    if collection.get("truth_tensor_received") is not False:
        raise RuntimeError("truth firewall failed during bridge construction")
    if collection.get("case_ids") != config["case_ids"]:
        raise ValueError("bridge collection case order drifted")
    records = collection.get("case_manifests")
    if not isinstance(records, Mapping) or set(records) != set(config["case_ids"]):
        raise ValueError("bridge collection lacks case manifests")
    result = {}
    for case_id in config["case_ids"]:
        path = bridge_case_dir(config, case_id) / "manifest.json"
        record = records[case_id]
        if not isinstance(record, Mapping) or file_sha256(path) != record.get("sha256"):
            raise ValueError(f"bridge case manifest hash mismatch: {case_id}")
        result[case_id] = load_completed_bridge_case(
            path.parent,
            expected_case_id=case_id,
            expected_config_sha256=expected_config_hash,
            expected_class_model_sha256=expected_class_hash,
        )
    return result


def load_stage10_inference_case(
    config: Mapping[str, object], case_id: str
) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    """Load the registered inference-only case; this API cannot receive truth."""
    return load_inference_case(inference_case_dir(config, case_id), case_id)


def target_probability_channel(class_model: Mapping[str, object], target_label: int) -> int:
    return class_channel(class_model["raw_labels"], target_label)
