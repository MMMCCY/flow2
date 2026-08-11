"""Inference-only immutable-artifact helpers for Stage 12B."""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge"
STAGE12A_DIR = PROJECT_DIR / "experiments/full_structuralgeo_benchmark"
CONFIG_PATH = EXPERIMENT_DIR / "configs/frozen_protocol.json"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.prior_ensemble import file_sha256
from guidance.seismic import tensor_sha256
from scripts.stage9.common import load_tensor_record, read_json


CASE_IDS = tuple(f"fullgeo_case{index:02d}" for index in range(1, 6))


def resolve_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def load_config() -> dict[str, object]:
    config = read_json(CONFIG_PATH)
    expected = {
        "schema": "stage12b_fullgeo_probability_bridge_protocol_v1",
        "status": "frozen_before_stage12b_seismic_generation",
        "case_replacement_authorized": False,
        "target_label": 9,
        "grid_shape": [64, 64, 64],
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Stage12B config {key} must be {value!r}")
    if tuple(config.get("case_ids", ())) != CASE_IDS:
        raise ValueError("Stage12B case order differs from the frozen five-case cohort")
    for section in ("acoustic_config", "seismic_config", "inversion_config", "class_model", "checkpoint"):
        record = config[section]
        if file_sha256(resolve_path(record["path"])) != record["sha256"]:
            raise ValueError(f"frozen {section} hash mismatch")
    return config


def verify_stage12a_files(config: Mapping[str, object]) -> None:
    stage12a = config["stage12a"]
    if file_sha256(STAGE12A_DIR / "benchmark_manifest.json") != stage12a["benchmark_manifest_sha256"]:
        raise ValueError("Stage12A benchmark manifest changed")
    if file_sha256(STAGE12A_DIR / "FULL_STRUCTURALGEO_BENCHMARK_DECISION.json") != stage12a["decision_sha256"]:
        raise ValueError("Stage12A decision changed")
    decision = read_json(STAGE12A_DIR / "FULL_STRUCTURALGEO_BENCHMARK_DECISION.json")
    if decision.get("machine_decision") != stage12a["required_machine_decision"]:
        raise ValueError("Stage12A machine decision is not READY")
    for case_id in CASE_IDS:
        root = STAGE12A_DIR / "cases" / case_id
        frozen = stage12a["case_hashes"][case_id]
        paths = {
            "manifest": root / "manifest.json",
            "true_model": root / "truth/true_model.pt",
            "condition_mask": root / "condition/condition_mask.pt",
        }
        for name, path in paths.items():
            if file_sha256(path) != frozen[name]:
                raise ValueError(f"frozen Stage12A {case_id} {name} hash changed")


def inference_case_dir(case_id: str) -> Path:
    if case_id not in CASE_IDS:
        raise ValueError(f"unregistered Stage12B case: {case_id}")
    return EXPERIMENT_DIR / "observations" / case_id


def bridge_case_dir(case_id: str) -> Path:
    if case_id not in CASE_IDS:
        raise ValueError(f"unregistered Stage12B case: {case_id}")
    return EXPERIMENT_DIR / "bridge" / case_id


def load_inference_case(case_id: str) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    root = inference_case_dir(case_id)
    manifest = read_json(root / "manifest.json")
    expected = {
        "schema": "stage12b_inference_case_v1",
        "status": "complete_truth_generation_process_closed",
        "case_id": case_id,
        "geological_truth_available_to_inference": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"inference manifest {key} must be {value!r}")
    required = {
        "condition_values",
        "condition_mask",
        "surface_mask",
        "borehole_mask",
        "subsurface_mask",
        "acoustic_property_table",
        "observation_correct",
    }
    if set(manifest.get("tensors", {})) != required:
        raise ValueError("Stage12B inference tensor set drifted")
    tensors = {
        name: load_tensor_record(root, record)
        for name, record in manifest["tensors"].items()
    }
    return manifest, tensors


def load_prior_bank() -> tuple[dict[str, object], torch.Tensor]:
    root = EXPERIMENT_DIR / "prior_bank"
    manifest = read_json(root / "manifest.json")
    expected = {
        "schema": "stage12b_common_property_prior_bank_v1",
        "status": "frozen_before_stage12b_seismic_generation",
        "truth_tensor_received": False,
        "candidate_indices": list(range(100, 112)),
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"prior bank {key} must be {value!r}")
    labels = load_tensor_record(root, manifest["hard_prior_members"])
    if labels.shape != (12, 1, 64, 64, 64):
        raise ValueError("common hard prior bank has the wrong shape")
    return manifest, labels.long()


def load_bridge_case(case_id: str) -> tuple[dict[str, object], dict[str, torch.Tensor]]:
    root = bridge_case_dir(case_id)
    manifest = read_json(root / "manifest.json")
    expected = {
        "schema": "stage12b_truth_blind_bridge_case_v1",
        "status": "complete_frozen_before_truth_evaluation",
        "case_id": case_id,
        "truth_tensor_received": False,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"bridge manifest {key} must be {value!r}")
    tensors = {
        name: load_tensor_record(root, record)
        for name, record in manifest["generated_tensors"].items()
    }
    if tensor_sha256(tensors["probability_label9_post"]) != manifest["generated_tensors"]["probability_label9_post"]["tensor_sha256"]:
        raise ValueError("post label9 bridge tensor hash mismatch")
    return manifest, tensors
