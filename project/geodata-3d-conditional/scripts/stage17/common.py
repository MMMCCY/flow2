"""Shared validation and AP-skill helpers for Stage17."""

from __future__ import annotations

import csv
import json
import math
import statistics
from pathlib import Path
from typing import Mapping, Sequence

import torch

import inference_runtime as runtime
from scripts.stage15.common import REPOSITORY_ROOT, read_json


def resolved_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def validate_asset_record(record: Mapping[str, object], name: str = "asset") -> Path:
    if set(record) < {"path", "sha256"}:
        raise ValueError(f"asset record is incomplete: {name}")
    path = resolved_path(record["path"])
    if runtime.file_sha256(path) != str(record["sha256"]):
        raise ValueError(f"frozen asset changed: {name}")
    return path


def require_frozen_config(config: Mapping[str, object], schema: str) -> None:
    if config.get("schema") != schema:
        raise ValueError(f"invalid config schema: expected {schema}")
    if config.get("status") not in {
        "frozen_before_asset_generation",
        "frozen_before_run",
    }:
        raise ValueError("Stage17 config is not frozen")
    if config.get("parameter_sweep") is not False:
        raise ValueError("Stage17 parameter_sweep must be false")
    if config.get("training_performed") is not False:
        raise ValueError("Stage17 training_performed must be false")


def average_precision(scores: torch.Tensor, targets: torch.Tensor) -> float:
    scores = scores.detach().reshape(-1).double().cpu()
    targets = targets.detach().reshape(-1).bool().cpu()
    if scores.numel() != targets.numel() or scores.numel() == 0:
        raise ValueError("scores and targets must be equally sized and non-empty")
    positives = int(targets.sum())
    if positives == 0:
        return 0.0
    order = torch.argsort(scores, descending=True, stable=True)
    ranked = targets[order].double()
    precision = ranked.cumsum(0) / torch.arange(1, ranked.numel() + 1, dtype=torch.double)
    return float(precision[ranked.bool()].mean())


def ap_skill(ap: float, prevalence: float) -> float:
    if not (math.isfinite(ap) and math.isfinite(prevalence)):
        raise ValueError("AP and prevalence must be finite")
    if prevalence < 0 or prevalence >= 1:
        raise ValueError("AP skill requires prevalence in [0,1)")
    return (ap - prevalence) / (1.0 - prevalence)


def median(values: Sequence[float]) -> float:
    if not values:
        raise ValueError("median requires at least one value")
    return float(statistics.median(values))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def canonical_json_sha256(value: Mapping[str, object]) -> str:
    import hashlib

    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()
