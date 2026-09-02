"""Shared frozen-schema, provenance and Stage19 case helpers."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Mapping

import torch

import inference_runtime as runtime
from scripts.stage15.common import normalize_volume, read_json


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROOT = PROJECT_DIR / "experiments/stage19_learned_evidence_adapter"
CONFIG_DIR = ROOT / "configs"

SCHEMAS = {
    "cohort_v1.json": "stage19_cohort_v1",
    "evidence_v1.json": "stage19_evidence_v1",
    "training_v1.json": "stage19_learned_evidence_adapter_training_v1",
    "inference_v1.json": "stage19_inference_v1",
}


def resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_DIR / path


def require_config(path: Path, schema: str) -> dict[str, object]:
    config = read_json(path)
    if config.get("schema") != schema:
        raise ValueError(f"invalid Stage19 schema: expected {schema}")
    if config.get("status") not in {"frozen_before_asset_generation", "frozen_before_cuda_training", "frozen_before_run"}:
        raise ValueError("Stage19 config is not frozen")
    if config.get("parameter_sweep") is not False:
        raise ValueError("Stage19 parameter_sweep must be false")
    return config


def load_all_configs() -> dict[str, dict[str, object]]:
    return {name: require_config(CONFIG_DIR / name, schema) for name, schema in SCHEMAS.items()}


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def asset(path: Path) -> dict[str, object]:
    return runtime.asset_record(path)


def validate_asset(record: Mapping[str, object], name: str) -> Path:
    if set(record) < {"path", "sha256"}:
        raise ValueError(f"incomplete asset record: {name}")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if runtime.file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"asset hash mismatch: {name}")
    return path


def registry_cases(path: Path, expected_split: str | None = None) -> list[dict[str, object]]:
    registry = read_json(path)
    if registry.get("schema") != "stage19_split_registry_v1" or registry.get("run_status") != "completed":
        raise ValueError(f"invalid Stage19 registry: {path}")
    if expected_split is not None and registry.get("split") != expected_split:
        raise ValueError(f"wrong Stage19 split: {path}")
    cases = registry.get("cases")
    if not isinstance(cases, list):
        raise TypeError("registry cases must be a list")
    return cases


def load_case(case: Mapping[str, object], *, include_truth: bool) -> dict[str, torch.Tensor]:
    assets = case.get("assets")
    if not isinstance(assets, Mapping):
        raise TypeError("case assets are absent")
    names = ["condition_values", "condition_mask", "subsurface_mask", "binary_impedance_score"]
    if include_truth:
        names.append("truth")
    loaded: dict[str, torch.Tensor] = {}
    for name in names:
        path = validate_asset(assets[name], f"{case['case_id']}/{name}")
        loaded[name] = normalize_volume(runtime.load_tensor(path), name)
    loaded["condition_values"] = loaded["condition_values"].long()
    loaded["condition_mask"] = loaded["condition_mask"].bool()
    loaded["subsurface_mask"] = loaded["subsurface_mask"].bool()
    loaded["binary_impedance_score"] = loaded["binary_impedance_score"].float()
    if include_truth:
        loaded["truth"] = loaded["truth"].long()
    return loaded


def freeze_base_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None


def project_condition_state(state: torch.Tensor, embedded: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    from guidance.generator_posterior import project_conditions
    return project_conditions(state, embedded, mask)
