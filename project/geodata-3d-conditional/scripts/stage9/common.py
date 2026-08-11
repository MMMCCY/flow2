"""Strict immutable-artifact helpers shared by Stage9A programs."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import tempfile
from typing import Mapping, Sequence

import torch

import inference_runtime as runtime
from guidance.prior_ensemble import file_sha256
from guidance.seismic import tensor_sha256


def read_json(path: Path) -> dict[str, object]:
    with Path(path).open("r", encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise TypeError(f"JSON root must be an object: {path}")
    return value


def write_json_x(path: Path, value: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def write_csv_x(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def save_tensor_x(path: Path, value: torch.Tensor) -> dict[str, object]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = value.detach().cpu().contiguous()
    with path.open("xb") as stream:
        torch.save(tensor, stream)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "tensor_sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def load_tensor_record(root: Path, record: Mapping[str, object]) -> torch.Tensor:
    path = Path(root) / str(record["path"])
    if file_sha256(path) != record.get("file_sha256"):
        raise ValueError(f"tensor file hash mismatch: {path}")
    value = runtime.load_tensor(path, map_location="cpu").contiguous()
    if list(value.shape) != list(record.get("shape", ())):
        raise ValueError(f"tensor shape mismatch: {path}")
    if str(value.dtype) != record.get("dtype"):
        raise ValueError(f"tensor dtype mismatch: {path}")
    if tensor_sha256(value) != record.get("tensor_sha256"):
        raise ValueError(f"tensor content hash mismatch: {path}")
    return value


def file_record(path: Path, *, relative_to: Path | None = None) -> dict[str, object]:
    path = Path(path)
    return {
        "path": str(path.relative_to(relative_to)) if relative_to else str(path),
        "size_bytes": path.stat().st_size,
        "sha256": file_sha256(path),
    }


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def create_staging_directory(final: Path) -> Path:
    final = Path(final)
    if final.exists():
        raise FileExistsError(f"refusing to reuse immutable output: {final}")
    final.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{final.name}.incomplete-", dir=final.parent))


def publish_staging_directory(staging: Path, final: Path) -> None:
    staging = Path(staging)
    final = Path(final)
    if final.exists():
        raise FileExistsError(f"refusing to overwrite immutable output: {final}")
    os.rename(staging, final)


def remove_empty_staging_directory(staging: Path) -> None:
    """Remove only a newly-created empty staging directory after preflight failure."""
    staging = Path(staging)
    if staging.exists() and not any(staging.iterdir()):
        staging.rmdir()


def copy_file_x(source: Path, destination: Path) -> None:
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("xb") as target, Path(source).open("rb") as origin:
        shutil.copyfileobj(origin, target)
