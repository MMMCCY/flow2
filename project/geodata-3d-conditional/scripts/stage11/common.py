"""Shared immutable paths and validators for Stage11."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
EXPERIMENT_DIR = PROJECT_DIR / "experiments/stage11_diverse_geometry_bridge"
CONFIG_DIR = EXPERIMENT_DIR / "configs"
REGISTRY_PATH = CONFIG_DIR / "case_registry.json"
DIVERSITY_SPEC_PATH = CONFIG_DIR / "benchmark_diversity_spec.json"
PROTOCOL_PATH = CONFIG_DIR / "frozen_protocol.json"
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))


def read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def repository_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def file_record(path: Path, *, relative_to: Path = REPOSITORY_ROOT) -> dict[str, object]:
    return {
        "path": str(path.relative_to(relative_to)),
        "sha256": sha256(path),
        "size_bytes": path.stat().st_size,
    }


def load_frozen_inputs() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
    registry = read_json(REGISTRY_PATH)
    diversity = read_json(DIVERSITY_SPEC_PATH)
    protocol = read_json(PROTOCOL_PATH)
    if registry.get("schema") != "stage11_diverse_geometry_case_registry_v1":
        raise ValueError("invalid Stage11 case registry")
    if diversity.get("schema") != "stage11_benchmark_diversity_spec_v1":
        raise ValueError("invalid Stage11 diversity specification")
    if protocol.get("schema") != "stage11_probability_bridge_protocol_v1":
        raise ValueError("invalid Stage11 protocol")
    expected_status = "frozen_before_geometry_audit_and_all_geophysical_computation"
    if any(item.get("status") != expected_status for item in (registry, diversity, protocol)):
        raise ValueError("Stage11 configuration is not frozen at the required boundary")
    cases = registry.get("cases")
    if not isinstance(cases, list) or len(cases) != int(registry["benchmark_size"]):
        raise ValueError("Stage11 registry case count mismatch")
    if len(cases) != int(diversity["benchmark_size"]):
        raise ValueError("Stage11 diversity case count mismatch")
    ids = [str(case["case_id"]) for case in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("Stage11 case IDs are not unique")
    for key in (
        "checkpoint",
        "acoustic_config",
        "seismic_config",
        "inversion_config",
        "inversion_source",
        "class_model",
        "bridge_source",
    ):
        path = repository_path(protocol[key])
        if sha256(path) != protocol[f"{key}_sha256"]:
            raise ValueError(f"frozen Stage11 source hash mismatch: {key}")
    return registry, diversity, protocol


def source_seed(protocol: Mapping[str, object], case_index: int, candidate_index: int) -> int:
    return (
        int(protocol["prior_source_seed_base"])
        + int(case_index) * int(protocol["prior_case_seed_stride"])
        + int(candidate_index)
    )
