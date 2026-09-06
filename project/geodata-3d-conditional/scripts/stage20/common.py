"""Shared contracts and numerical helpers for frozen Stage20."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Mapping, Sequence

import torch

import inference_runtime as runtime
from guidance.seismic import acoustic_tables_from_config, hard_labels_to_acoustic
from scripts.stage20.acoustic_semantics import stage20_labels_to_acoustic
from scripts.stage15.common import normalize_volume, read_json


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
ROOT = PROJECT_DIR / "experiments/stage20_continuous_impedance_adapter"
CONFIG_DIR = ROOT / "configs"
SPEC_PATH = REPOSITORY_ROOT / "project/FLOW2_STAGE20_CONTINUOUS_IMPEDANCE_ADAPTER_CODEX_SPEC.md"
FROZEN_WELLS = ((8, 46), (9, 5), (10, 24), (27, 17), (35, 26), (39, 59), (44, 60), (48, 6), (57, 32))

SCHEMAS = {
    "cohort_v1.json": "stage20_cohort_v1",
    "evidence_v1.json": "stage20_continuous_impedance_evidence_v1",
    "training_v1.json": "stage20_continuous_impedance_adapter_training_v1",
    "inference_v1.json": "stage20_inference_v1",
}


def resolve_project_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else PROJECT_DIR / path


def require_config(path: Path, schema: str) -> dict[str, object]:
    config = read_json(path)
    if config.get("schema") != schema:
        raise ValueError(f"invalid Stage20 schema: expected {schema}")
    if config.get("status") not in {"frozen_before_asset_generation", "frozen_before_cuda_training", "frozen_before_run"}:
        raise ValueError("Stage20 config is not frozen")
    if config.get("parameter_sweep") is not False:
        raise ValueError("Stage20 parameter_sweep must be false")
    return config


def asset(path: Path) -> dict[str, object]:
    return runtime.asset_record(path.resolve())


def validate_asset(record: Mapping[str, object], name: str) -> Path:
    if not isinstance(record, Mapping) or not {"path", "sha256"} <= set(record):
        raise ValueError(f"incomplete asset record: {name}")
    path = Path(str(record["path"]))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    if runtime.file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"asset hash mismatch: {name}")
    return path


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def load_codebook(path: Path) -> tuple[torch.Tensor, dict[str, object]]:
    tables, metadata = acoustic_tables_from_config(read_json(path), 15)
    return tables.property_table, metadata


def columnwise_support(truth: torch.Tensor) -> torch.Tensor:
    truth = normalize_volume(truth, "truth").long()
    nonair = truth != -1
    if not bool(nonair.any(dim=-1).all()):
        raise ValueError("every column must contain non-air")
    z = torch.arange(truth.shape[-1], device=truth.device).view(1, 1, 1, 1, -1)
    highest = torch.where(nonair, z, torch.full_like(z, -1)).amax(dim=-1, keepdim=True)
    return z <= highest


def hard_labels_to_stage20_acoustic(
    labels: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, int]]:
    """Compatibility alias for the unique Stage20 acoustic mapper."""
    return stage20_labels_to_acoustic(labels, subsurface_mask, property_table)


def validate_wells(well_xy: Sequence[Sequence[int]]) -> tuple[tuple[int, int], ...]:
    wells = tuple((int(point[0]), int(point[1])) for point in well_xy)
    if wells != FROZEN_WELLS:
        raise RuntimeError("STOP_STAGE20_WELL_CONTRACT_MISMATCH")
    return wells


def build_low_frequency_acoustic(
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    subsurface_mask: torch.Tensor,
    well_xy: Sequence[Sequence[int]],
    property_table: torch.Tensor,
    *,
    power: float = 2.0,
) -> tuple[torch.Tensor, dict[str, object]]:
    """Depthwise well IDW with nearest-valid-depth fallback and provenance."""
    for name, tensor in (("condition_values", condition_values), ("condition_mask", condition_mask), ("subsurface_mask", subsurface_mask)):
        if tensor.ndim != 5 or tensor.shape[1] != 1:
            raise ValueError(f"{name} must have shape [B,1,X,Y,Z]")
    values = condition_values.long()
    known = condition_mask.bool()
    support = subsurface_mask.bool()
    if values.shape != known.shape or values.shape != support.shape or values.shape[0] != 1:
        raise ValueError("Stage20 prior tensors must be matching singleton volumes")
    if not math.isclose(float(power), 2.0):
        raise ValueError("Stage20 IDW power must remain 2")
    wells = tuple((int(point[0]), int(point[1])) for point in well_xy)
    if not wells:
        raise ValueError("at least one well is required")
    if any(x < 0 or y < 0 or x >= values.shape[2] or y >= values.shape[3] for x, y in wells):
        raise ValueError("well coordinate is outside the volume")
    table = property_table.to(dtype=torch.float64, device=values.device)
    rock_labels = (values >= 0) & (values <= 13)
    nx, ny, nz = values.shape[2:]
    grid_x = torch.arange(nx, device=values.device, dtype=torch.float64)[:, None]
    grid_y = torch.arange(ny, device=values.device, dtype=torch.float64)[None, :]
    air = table[:, 0]
    result = air.view(1, 2, 1, 1, 1).expand(1, 2, nx, ny, nz).clone()
    active_depths = [depth for depth in range(nz) if bool(support[0, 0, :, :, depth].any())]
    valid_by_depth = {
        depth: [(x, y) for x, y in wells if bool(support[0, 0, x, y, depth] and known[0, 0, x, y, depth] and rock_labels[0, 0, x, y, depth])]
        for depth in active_depths
    }
    valid_depths = sorted(depth for depth, valid in valid_by_depth.items() if valid)
    if active_depths and not valid_depths:
        raise RuntimeError("STOP_LOW_FREQUENCY_PRIOR_UNDEFINED: case has no valid well depth")
    for depth in valid_depths:
        occupied = support[0, 0, :, :, depth]
        valid = valid_by_depth[depth]
        logz_samples = torch.stack([table[0, int(values[0, 0, x, y, depth]) + 1].log() for x, y in valid])
        slow_samples = torch.stack([table[1, int(values[0, 0, x, y, depth]) + 1] for x, y in valid])
        d2 = torch.stack([(grid_x - x).square() + (grid_y - y).square() for x, y in valid])
        exact = d2 == 0
        weights = torch.where(exact, torch.zeros_like(d2), d2.reciprocal())
        weights = weights / weights.sum(dim=0, keepdim=True).clamp_min(torch.finfo(weights.dtype).tiny)
        logz = (weights * logz_samples[:, None, None]).sum(dim=0)
        slow = (weights * slow_samples[:, None, None]).sum(dim=0)
        if bool(exact.any()):
            exact_index = exact.to(torch.int64).argmax(dim=0)
            any_exact = exact.any(dim=0)
            logz = torch.where(any_exact, logz_samples[exact_index], logz)
            slow = torch.where(any_exact, slow_samples[exact_index], slow)
        result[0, 0, :, :, depth] = torch.where(occupied, logz.exp(), air[0])
        result[0, 1, :, :, depth] = torch.where(occupied, slow, air[1])
    fallback_map = []
    for depth in active_depths:
        if valid_by_depth[depth]:
            continue
        reference = min(valid_depths, key=lambda candidate: (abs(candidate - depth), candidate))
        result[..., depth] = result[..., reference]
        current_support = support[..., depth].expand_as(result[..., depth])
        result[..., depth] = torch.where(current_support, result[..., depth], air.view(1, 2, 1, 1))
        fallback_map.append({"z": depth, "z_ref": reference, "distance": abs(reference - depth)})
    exact_acoustic, _ = stage20_labels_to_acoustic(values, support, table)
    result = torch.where(known.expand_as(result), exact_acoustic, result)
    diagnostics = {
        "valid_well_depth_count": len(valid_depths),
        "fallback_depth_count": len(fallback_map),
        "fallback_map": fallback_map,
        "tie_policy": "nearest_then_smaller_z_index_deeper",
        "whole_case_without_valid_well_depth": False,
    }
    return result.to(dtype=property_table.dtype).contiguous(), diagnostics


def normalize_logz(logz: torch.Tensor, property_table: torch.Tensor) -> tuple[torch.Tensor, float, float]:
    rock = property_table[0, 1:].log().to(logz)
    low, high = rock.min(), rock.max()
    normalized = (2.0 * (logz - low) / (high - low) - 1.0).clamp(-1.0, 1.0)
    return normalized, float(low.cpu()), float(high.cpu())


def runtime_evidence(normalized_logz: torch.Tensor, support: torch.Tensor, condition: torch.Tensor) -> torch.Tensor:
    if normalized_logz.ndim != 5 or support.shape != normalized_logz.shape or condition.shape != normalized_logz.shape:
        raise ValueError("evidence/support/condition must be matching [B,1,X,Y,Z] tensors")
    free = support.bool() & ~condition.bool()
    return normalized_logz.float() * free.float()


def wrong_case_evidence(wrong_logz: torch.Tensor, current_support: torch.Tensor, current_condition: torch.Tensor) -> torch.Tensor:
    return runtime_evidence(wrong_logz, current_support, current_condition)


def assert_truth_firewall(payload: object, context: str) -> None:
    serialized = json.dumps(payload, sort_keys=True).lower()
    forbidden = ("truth", "true_model", "true_logz", "binary_truth", "label9_mask", "truth_assets")
    if any(term in serialized for term in forbidden):
        raise RuntimeError(f"{context} contains forbidden truth material")


def freeze_base_model(model) -> None:
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
