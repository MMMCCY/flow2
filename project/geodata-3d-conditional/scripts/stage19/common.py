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
    "cohort_v2.json": "stage19_cohort_v2",
    "evidence_v1.json": "stage19_evidence_v1",
    "evidence_v2.json": "stage19_evidence_v2",
    "training_v1.json": "stage19_learned_evidence_adapter_training_v1",
    "inference_v1.json": "stage19_inference_v1",
}

COHORT_CONTRACT_FIELDS = (
    "recipe",
    "eligibility",
    "fixed_well_xy",
)

STAGE17A_REUSE_FIELDS = {
    "trace_samples": 320,
    "refinement_passes": 2,
    "prior_relative_weight": 0.001,
    "vertical_smoothness_relative_weight": 0.01,
    "full_vertical_trace_used": True,
    "lateral_filter_used": False,
    "thresholding": False,
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
    return runtime.asset_record(path.resolve())


def validate_asset(record: Mapping[str, object], name: str) -> Path:
    if set(record) < {"path", "sha256"}:
        raise ValueError(f"incomplete asset record: {name}")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if runtime.file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"asset hash mismatch: {name}")
    return path


def validate_stage19_cohort_contract(
    config: Mapping[str, object], reference_config: Mapping[str, object]
) -> None:
    """Require exact reuse of the authoritative geology/conditioning contract."""
    for field in COHORT_CONTRACT_FIELDS:
        if config.get(field) != reference_config.get(field):
            raise RuntimeError(f"STOP_GENERATOR_REUSE_MISMATCH: {field}")
    expected_splits = {
        "train": {"accepted": 64, "start": 220260001, "step": 1, "max_candidates": 1024},
        "val": {"accepted": 8, "start": 220270001, "step": 1, "max_candidates": 256},
        "test": {"accepted": 12, "start": 220280001, "step": 1, "max_candidates": 256},
    }
    if config.get("splits") != expected_splits:
        raise ValueError("Stage19R cohort search envelope changed")


def validate_stage17a_reuse(config: Mapping[str, object]) -> dict[str, Path]:
    """Validate v2 paths and hashes against the successful Stage17A config."""
    try:
        authoritative_path = validate_asset(
            config["authoritative_stage17a_config"], "authoritative Stage17A config"
        )
        authoritative = read_json(authoritative_path)
        records = {
            "binary_acoustic_config": config["observation"]["binary_acoustic_config"],
            "seismic_config": config["observation"]["seismic_config"],
            "inversion_config": config["inversion"]["inversion_config"],
        }
        paths = {
            name: validate_asset(record, f"Stage17A {name}")
            for name, record in records.items()
        }
        for name, record in records.items():
            reference = authoritative.get(name)
            if not isinstance(reference, Mapping) or str(record["sha256"]) != str(reference["sha256"]):
                raise ValueError(f"authoritative record mismatch: {name}")
        observed = config["observation"]
        inversion = config["inversion"]
        actual_fields = {
            "trace_samples": observed.get("trace_samples"),
            "refinement_passes": inversion.get("refinement_passes"),
            "prior_relative_weight": inversion.get("prior_relative_weight"),
            "vertical_smoothness_relative_weight": inversion.get("vertical_smoothness_relative_weight"),
            "full_vertical_trace_used": inversion.get("full_vertical_trace_used"),
            "lateral_filter_used": inversion.get("lateral_filter_used"),
            "thresholding": inversion.get("thresholding"),
        }
        if actual_fields != STAGE17A_REUSE_FIELDS:
            raise ValueError("frozen Stage17A scientific fields changed")
        return {"authoritative_stage17a_config": authoritative_path, **paths}
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(f"STOP_STAGE17A_REUSE_MISMATCH: {exc}") from exc


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
