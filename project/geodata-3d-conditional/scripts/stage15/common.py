"""Shared Stage15 provenance and immutable-output helpers."""

from __future__ import annotations

import csv
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]


def read_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing to write empty Stage15 trace")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def refuse_nonempty(path: Path) -> None:
    if path.exists() and any(path.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {path}")


def git_value(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=REPOSITORY_ROOT, check=True, capture_output=True, text=True
    )
    return result.stdout.strip()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def base_manifest(schema: str, runner: Path, config: Path | None = None) -> dict[str, object]:
    import inference_runtime as runtime

    result: dict[str, object] = {
        "schema": schema,
        "run_status": "running",
        "created_at_utc": utc_now(),
        "git_head": git_value("rev-parse", "HEAD"),
        "git_branch": git_value("branch", "--show-current"),
        "runner_source": runtime.asset_record(runner),
    }
    if config is not None:
        result["config"] = runtime.asset_record(config)
        result["full_parameters"] = read_json(config)
    return result


def normalize_volume(value: torch.Tensor, name: str, dtype: torch.dtype | None = None) -> torch.Tensor:
    import inference_runtime as runtime

    result = runtime.normalize_single_geology(value, name)
    if tuple(result.shape) != (1, 1, 64, 64, 64):
        raise ValueError(f"{name} must resolve to [1,1,64,64,64]")
    return result.to(dtype=dtype) if dtype is not None else result


def validate_asset(path: Path, expected_sha256: str | None = None) -> dict[str, object]:
    import inference_runtime as runtime

    record = runtime.asset_record(path)
    if expected_sha256 is not None and record["sha256"] != expected_sha256:
        raise ValueError(f"asset hash changed: {path}")
    return record
