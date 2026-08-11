"""Frozen, geology-only builder for same-recipe full StructuralGeo benchmarks.

This module deliberately imports neither Flow nor geophysical code.  Candidate
selection uses only the generated history, final categorical truth, and the
prospectively frozen conditioning layout.
"""

from __future__ import annotations

import copy
import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from typing import Any

import numpy as np
import scipy
from scipy import ndimage
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = Path(__file__).resolve().parents[3]
STRUCTURALGEO_ROOT = REPO_ROOT / "StructuralGeo-main"
STRUCTURALGEO_SRC = STRUCTURALGEO_ROOT / "src"
if str(STRUCTURALGEO_SRC) not in sys.path:
    sys.path.insert(0, str(STRUCTURALGEO_SRC))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from boreholes import make_surface_mask  # noqa: E402
from geogen.generation.model_generators import (  # noqa: E402
    MarkovGeostoryGenerator,
    MarkovMatrixParser,
)
from geogen.generation.rng_contract import RNG_CONTRACT_VERSION  # noqa: E402


SCHEMA_VERSION = "full_structuralgeo_benchmark_v1"
LABEL_SEMANTICS_VERSION = "structuralgeo_raw_labels_v1"
ELIGIBILITY_VERSION = "full_complexity_targeted_v1"
MACHINE_READY = "FULL_STRUCTURALGEO_BENCHMARK_READY"
MACHINE_NOT_REPRODUCIBLE = "STOP_FULL_STRUCTURALGEO_NOT_REPRODUCIBLE"
MACHINE_COHORT_UNAVAILABLE = "STOP_FULL_STRUCTURALGEO_TARGET_COHORT_UNAVAILABLE"

MODEL_BOUNDS = ((-1920, 1920), (-1920, 1920), (-1920, 1920))
MODEL_RESOLUTION = (64, 64, 64)
CANONICAL_NINE_WELL_XY = (
    (8, 46),
    (9, 5),
    (10, 24),
    (27, 17),
    (35, 26),
    (39, 59),
    (44, 60),
    (48, 6),
    (57, 32),
)

DEFAULT_CONFIG_PATH = (
    PROJECT_ROOT
    / "experiments"
    / "full_structuralgeo_benchmark"
    / "configs"
    / "full_complexity_targeted_v1.json"
)
DEFAULT_OUTPUT_DIR = (
    PROJECT_ROOT / "experiments" / "full_structuralgeo_benchmark"
)
DEFAULT_SCRIPT_PATH = (
    PROJECT_ROOT
    / "scripts"
    / "benchmarks"
    / "build_full_structuralgeo_benchmark.py"
)

SOURCE_PATHS = (
    Path("project/geodata-3d-conditional/model_train_sh_inference_cond.py"),
    Path("project/geodata-3d-conditional/boreholes.py"),
    Path("StructuralGeo-main/src/geogen/dataset/dataset.py"),
    Path("StructuralGeo-main/src/geogen/generation/model_generators.py"),
    Path("StructuralGeo-main/src/geogen/generation/categorical_events.py"),
    Path("StructuralGeo-main/src/geogen/generation/geowords.py"),
    Path("StructuralGeo-main/src/geogen/generation/rng_contract.py"),
    Path("StructuralGeo-main/src/geogen/model/geomodel.py"),
    Path("StructuralGeo-main/src/geogen/model/metaballs.py"),
    Path("StructuralGeo-main/src/geogen/probability/random_varibles.py"),
    Path("StructuralGeo-main/src/geogen/probability/sedimentbuilders.py"),
    Path("StructuralGeo-main/src/geogen/probability/wavegenerators.py"),
    Path(
        "StructuralGeo-main/src/geogen/generation/markov_matrix/"
        "default_markov_matrix.csv"
    ),
    Path(
        "project/geodata-3d-conditional/guidance/"
        "full_structuralgeo_benchmark.py"
    ),
    Path(
        "project/geodata-3d-conditional/scripts/benchmarks/"
        "build_full_structuralgeo_benchmark.py"
    ),
    Path(
        "project/geodata-3d-conditional/experiments/"
        "full_structuralgeo_benchmark/configs/"
        "full_complexity_targeted_v1.json"
    ),
)

LABEL_NAMES = {
    -1: "air/unfilled",
    0: "bedrock",
    1: "sediment_1",
    2: "sediment_2",
    3: "sediment_3",
    4: "sediment_4",
    5: "sediment_5",
    6: "dike_6",
    7: "dike_7",
    8: "dike_8",
    9: "intrusion_9_target",
    10: "intrusion_10",
    11: "intrusion_11",
    12: "blob_12",
    13: "blob_13",
}

LITHOLOGY_COLORS = {
    -1: "#f4f4f4",
    0: "#4a4a4a",
    1: "#e6c78d",
    2: "#d89a62",
    3: "#cdbd64",
    4: "#8ebf78",
    5: "#6ca6a8",
    6: "#6d5b9b",
    7: "#8a65ad",
    8: "#aa79bf",
    9: "#d62728",
    10: "#ff6f61",
    11: "#b2182b",
    12: "#3b78c8",
    13: "#62a9e8",
}

ACTUAL_SOURCE_FILES_CHANGED = (
    "StructuralGeo-main/src/geogen/dataset/dataset.py",
    "StructuralGeo-main/src/geogen/generation/categorical_events.py",
    "StructuralGeo-main/src/geogen/generation/geowords.py",
    "StructuralGeo-main/src/geogen/generation/model_generators.py",
    "StructuralGeo-main/src/geogen/generation/rng_contract.py",
    "StructuralGeo-main/src/geogen/model/geomodel.py",
    "StructuralGeo-main/src/geogen/model/metaballs.py",
    "StructuralGeo-main/src/geogen/probability/random_varibles.py",
    "StructuralGeo-main/src/geogen/probability/sedimentbuilders.py",
    "StructuralGeo-main/src/geogen/probability/wavegenerators.py",
    "project/geodata-3d-conditional/guidance/full_structuralgeo_benchmark.py",
    "project/geodata-3d-conditional/scripts/benchmarks/build_full_structuralgeo_benchmark.py",
    "project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/configs/full_complexity_targeted_v1.json",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_tensor_sha256(tensor: torch.Tensor) -> str:
    array = tensor.detach().cpu().contiguous().numpy()
    digest = hashlib.sha256()
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(json.dumps(list(array.shape), separators=(",", ":")).encode("ascii"))
    digest.update(array.tobytes(order="C"))
    return digest.hexdigest()


def canonical_json_sha256(payload: Any) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def source_hashes() -> dict[str, str]:
    result = {}
    for relative_path in SOURCE_PATHS:
        absolute_path = REPO_ROOT / relative_path
        if not absolute_path.is_file():
            raise FileNotFoundError(f"required source is absent: {relative_path}")
        result[relative_path.as_posix()] = sha256_file(absolute_path)
    return result


def default_matrix_path() -> Path:
    return Path(MarkovMatrixParser().path).resolve()


def environment_versions() -> dict[str, str]:
    def distribution_version(name: str) -> str:
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            return "not-installed"

    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "pydtmc": distribution_version("pydtmc"),
        "scipy": scipy.__version__,
        "torch": torch.__version__,
        "platform": platform.platform(),
    }


def git_snapshot() -> dict[str, Any]:
    def run(*args: str) -> str:
        return subprocess.run(
            ["git", *args],
            cwd=REPO_ROOT,
            check=True,
            capture_output=True,
            text=True,
        ).stdout.rstrip("\n")

    status = run("status", "--short")
    return {
        "branch": run("branch", "--show-current"),
        "head": run("rev-parse", "HEAD"),
        "dirty": bool(status),
        "status_short": status.splitlines(),
    }


def load_frozen_config(path: Path = DEFAULT_CONFIG_PATH) -> dict[str, Any]:
    config = json.loads(path.read_text(encoding="utf-8"))
    if not config.get("frozen_before_generation"):
        raise ValueError("benchmark config is not marked frozen_before_generation")
    order = config["root_seed_order"]
    if order["kind"] != "arithmetic_sequence":
        raise ValueError("unsupported root_seed_order")
    if int(order["count"]) != int(config["maximum_seed_budget"]):
        raise ValueError("seed order count differs from maximum_seed_budget")
    if tuple(tuple(row) for row in config["recipe"]["bounds"]) != MODEL_BOUNDS:
        raise ValueError("frozen bounds differ from the training recipe")
    if tuple(config["recipe"]["resolution"]) != MODEL_RESOLUTION:
        raise ValueError("frozen resolution differs from the training recipe")
    if tuple(tuple(row) for row in config["fixed_well_xy"]) != CANONICAL_NINE_WELL_XY:
        raise ValueError("fixed well layout differs from the canonical nine wells")
    return config


def seed_order(config: dict[str, Any]) -> list[int]:
    order = config["root_seed_order"]
    return [
        int(order["start"]) + index * int(order["step"])
        for index in range(int(order["count"]))
    ]


def fixed_borehole_mask(shape: tuple[int, ...]) -> torch.Tensor:
    if shape != (1, 1, *MODEL_RESOLUTION):
        raise ValueError(f"unexpected truth shape for fixed wells: {shape}")
    mask = torch.zeros(shape, dtype=torch.bool)
    for x_index, y_index in CANONICAL_NINE_WELL_XY:
        mask[0, 0, x_index, y_index, :] = True
    return mask


def _render_seed(root_seed: int) -> tuple[torch.Tensor, dict[str, Any]]:
    generator = MarkovGeostoryGenerator(
        model_bounds=MODEL_BOUNDS,
        model_resolution=MODEL_RESOLUTION,
        config=None,
        root_seed=int(root_seed),
    )
    model, metadata = generator.generate_model_with_metadata()
    model.fill_nans()
    data = model.get_data_grid()
    truth = torch.from_numpy(np.asarray(data)).to(torch.float32)[None, None]
    return truth, metadata


def probe_seed(root_seed: int) -> dict[str, Any]:
    """Generate one full 64^3 case without creating benchmark artifacts."""
    truth, metadata = _render_seed(root_seed)
    array = truth.numpy()
    finite = bool(np.isfinite(array).all())
    integer_valued = bool(np.equal(array, np.rint(array)).all())
    labels = sorted(int(value) for value in np.unique(array))
    history_payload = {
        "markov_sequence": metadata["markov_sequence"],
        "events": metadata["events"],
        "packed_history": metadata["packed_history"],
        "unpacked_history": metadata["unpacked_history"],
    }
    hashes = source_hashes()
    return {
        "schema_version": "full_structuralgeo_determinism_probe_v1",
        "root_seed": int(root_seed),
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "markov_sequence": metadata["markov_sequence"],
        "event_subtypes": [event["subtype"] for event in metadata["events"]],
        "history_sha256": canonical_json_sha256(history_payload),
        "tensor_sha256": canonical_tensor_sha256(truth),
        "shape": list(truth.shape),
        "dtype": str(truth.dtype),
        "finite": finite,
        "integer_valued": integer_valued,
        "labels": labels,
        "labels_within_frozen_range": bool(labels and min(labels) >= -1 and max(labels) <= 13),
        "bounds": [list(bound) for bound in MODEL_BOUNDS],
        "resolution": list(MODEL_RESOLUTION),
        "height_tracking": True,
        "normalize": True,
        "fill_nans": True,
        "default_matrix_sha256": sha256_file(default_matrix_path()),
        "source_hashes": hashes,
    }


def _run_fresh_probe(script_path: Path, root_seed: int, output_path: Path) -> dict[str, Any]:
    environment = os.environ.copy()
    python_path_parts = [str(STRUCTURALGEO_SRC), str(PROJECT_ROOT)]
    if environment.get("PYTHONPATH"):
        python_path_parts.append(environment["PYTHONPATH"])
    environment["PYTHONPATH"] = os.pathsep.join(python_path_parts)
    environment.setdefault("MPLCONFIGDIR", "/tmp/stage12a-matplotlib")
    result = subprocess.run(
        [
            sys.executable,
            str(script_path),
            "--probe-seed",
            str(int(root_seed)),
            "--probe-json",
            str(output_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            "fresh-process determinism probe failed: "
            f"returncode={result.returncode}\nstdout={result.stdout}\nstderr={result.stderr}"
        )
    return json.loads(output_path.read_text(encoding="utf-8"))


def run_determinism_audit(
    script_path: Path,
    config: dict[str, Any],
) -> dict[str, Any]:
    seeds = seed_order(config)
    reference_seed = seeds[0]
    different_seed = seeds[1]
    with tempfile.TemporaryDirectory(prefix="stage12a-determinism-", dir="/tmp") as temp_name:
        temp_dir = Path(temp_name)
        fresh_a = _run_fresh_probe(
            script_path, reference_seed, temp_dir / "fresh_a.json"
        )
        fresh_b = _run_fresh_probe(
            script_path, reference_seed, temp_dir / "fresh_b.json"
        )
        fresh_different = _run_fresh_probe(
            script_path, different_seed, temp_dir / "fresh_different.json"
        )
    single_process = probe_seed(reference_seed)

    invariant_keys = (
        "shape",
        "finite",
        "integer_valued",
        "labels_within_frozen_range",
        "bounds",
        "resolution",
        "height_tracking",
        "normalize",
        "fill_nans",
    )
    invariant_expectations = {
        "shape": [1, 1, 64, 64, 64],
        "finite": True,
        "integer_valued": True,
        "labels_within_frozen_range": True,
        "bounds": [list(bound) for bound in MODEL_BOUNDS],
        "resolution": list(MODEL_RESOLUTION),
        "height_tracking": True,
        "normalize": True,
        "fill_nans": True,
    }
    invariant_checks = {
        key: all(
            probe[key] == expected
            for probe in (fresh_a, fresh_b, fresh_different, single_process)
        )
        for key, expected in invariant_expectations.items()
    }
    same_seed_fresh_processes = (
        fresh_a["history_sha256"] == fresh_b["history_sha256"]
        and fresh_a["tensor_sha256"] == fresh_b["tensor_sha256"]
    )
    single_vs_fresh = (
        single_process["history_sha256"] == fresh_a["history_sha256"]
        and single_process["tensor_sha256"] == fresh_a["tensor_sha256"]
    )
    different_seed_changes_output = (
        fresh_different["history_sha256"] != fresh_a["history_sha256"]
        or fresh_different["tensor_sha256"] != fresh_a["tensor_sha256"]
    )
    provenance_identical = all(
        probe["source_hashes"] == fresh_a["source_hashes"]
        and probe["default_matrix_sha256"] == fresh_a["default_matrix_sha256"]
        for probe in (fresh_b, fresh_different, single_process)
    )
    config_recipe_matches = (
        config["recipe"]["markov_matrix"] == "packaged_default"
        and tuple(tuple(bound) for bound in config["recipe"]["bounds"])
        == MODEL_BOUNDS
        and tuple(config["recipe"]["resolution"]) == MODEL_RESOLUTION
    )
    passed = all(
        (
            same_seed_fresh_processes,
            single_vs_fresh,
            different_seed_changes_output,
            provenance_identical,
            config_recipe_matches,
            *invariant_checks.values(),
        )
    )
    return {
        "schema_version": "full_structuralgeo_determinism_audit_v1",
        "passed": bool(passed),
        "reference_seed": reference_seed,
        "different_seed": different_seed,
        "checks": {
            "same_seed_two_fresh_processes": same_seed_fresh_processes,
            "single_process_matches_fresh_process": single_vs_fresh,
            "different_seed_changes_history_or_tensor": different_seed_changes_output,
            "source_and_matrix_hashes_identical_across_processes": provenance_identical,
            "config_matches_training_recipe": config_recipe_matches,
            **invariant_checks,
        },
        "probe_summaries": {
            "fresh_a": fresh_a,
            "fresh_b": fresh_b,
            "fresh_different": fresh_different,
            "single_process": single_process,
        },
        "invariant_keys": list(invariant_keys),
    }


def _history_only_candidate(root_seed: int):
    """Sample the frozen history before paying the 64^3 materialization cost."""
    generator = MarkovGeostoryGenerator(
        model_bounds=MODEL_BOUNDS,
        model_resolution=MODEL_RESOLUTION,
        config=None,
        root_seed=int(root_seed),
    )
    model_contract = generator.rng_contract.child("model_000000")
    history = generator.build_geostory(rng_contract=model_contract)
    event_records = copy.deepcopy(generator._last_event_records)
    markov_sequence = list(generator._last_sequence)
    return generator, model_contract, history, markov_sequence, event_records


def history_eligibility(
    markov_sequence: list[str], event_records: list[dict[str, Any]]
) -> tuple[bool, list[str]]:
    reasons = []
    if "BaseStrata" not in markov_sequence:
        reasons.append("missing_base_strata_history")
    if not ({"Fold", "Fault"} & set(markov_sequence)):
        reasons.append("missing_fold_or_fault_event")
    label9_event = any(
        event["state"] in {"Dike", "Sills", "Pluton"}
        and 9 in [value for value in event["deposition_values"] if value is not None]
        for event in event_records
    )
    if not label9_event:
        reasons.append("missing_label9_producing_intrusion_event")
    return not reasons, reasons


def _materialize_history(generator, model_contract, history):
    normalization_rng = model_contract.generator("height_normalization")
    model = generator._history_to_model(
        history, normalization_rng=normalization_rng
    )
    metadata = {
        "model_index": 0,
        "rng_contract": model_contract.describe(),
        "markov_sequence": list(generator._last_sequence),
        "events": copy.deepcopy(generator._last_event_records),
        "packed_history": [str(process) for process in model.history],
        "unpacked_history": [str(process) for process in model.history_unpacked],
        "normalization_stream": "height_normalization",
    }
    model.fill_nans()
    data = model.get_data_grid()
    truth = torch.from_numpy(np.asarray(data)).to(torch.float32)[None, None]
    return truth, metadata


def condition_tensors(truth: torch.Tensor) -> dict[str, torch.Tensor]:
    surface_mask = make_surface_mask(truth).cpu().bool()
    borehole_mask = fixed_borehole_mask(tuple(truth.shape))
    condition_mask = surface_mask | borehole_mask
    condition_values = torch.full_like(truth, fill_value=-1)
    condition_values[condition_mask] = truth[condition_mask]
    return {
        "surface_mask": surface_mask,
        "borehole_mask": borehole_mask,
        "condition_mask": condition_mask,
        "condition_values": condition_values,
    }


def _index_to_physical(index_xyz: list[float]) -> list[float]:
    physical = []
    for axis, coordinate in enumerate(index_xyz):
        low, high = MODEL_BOUNDS[axis]
        physical.append(
            float(low + float(coordinate) * (high - low) / (MODEL_RESOLUTION[axis] - 1))
        )
    return physical


def _component_records(
    mask: np.ndarray,
    connectivity: int,
    hidden_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    structure = ndimage.generate_binary_structure(rank=3, connectivity=connectivity)
    labeled, count = ndimage.label(mask, structure=structure)
    records = []
    voxel_spacing = [
        (high - low) / (resolution - 1)
        for (low, high), resolution in zip(MODEL_BOUNDS, MODEL_RESOLUTION)
    ]
    voxel_volume = float(np.prod(voxel_spacing))
    for component_id in range(1, count + 1):
        indices = np.argwhere(labeled == component_id)
        centroid = indices.mean(axis=0).tolist()
        bbox_min = indices.min(axis=0).astype(int).tolist()
        bbox_max = indices.max(axis=0).astype(int).tolist()
        hidden_count = (
            int(hidden_mask[labeled == component_id].sum())
            if hidden_mask is not None
            else None
        )
        records.append(
            {
                "component_id": component_id,
                "voxel_count": int(indices.shape[0]),
                "hidden_voxel_count": hidden_count,
                "centroid_index_xyz": [float(value) for value in centroid],
                "centroid_physical_xyz": _index_to_physical(centroid),
                "bbox_index_inclusive": {"min_xyz": bbox_min, "max_xyz": bbox_max},
                "bbox_physical_inclusive": {
                    "min_xyz": _index_to_physical(bbox_min),
                    "max_xyz": _index_to_physical(bbox_max),
                },
                "physical_volume": float(indices.shape[0] * voxel_volume),
            }
        )
    return labeled, records


def geometry_statistics(
    label9_mask: torch.Tensor, hidden_label9_mask: torch.Tensor
) -> dict[str, Any]:
    raw = label9_mask[0, 0].cpu().numpy().astype(bool)
    hidden = hidden_label9_mask[0, 0].cpu().numpy().astype(bool)
    _, records_6 = _component_records(raw, connectivity=1, hidden_mask=hidden)
    _, records_26 = _component_records(raw, connectivity=3, hidden_mask=hidden)
    primary = None
    if records_26:
        primary = max(
            records_26,
            key=lambda item: (item["hidden_voxel_count"], item["voxel_count"]),
        )
    indices = np.argwhere(raw)
    overall = None
    if indices.size:
        centroid = indices.mean(axis=0).tolist()
        bbox_min = indices.min(axis=0).astype(int).tolist()
        bbox_max = indices.max(axis=0).astype(int).tolist()
        spacing = [
            (high - low) / (resolution - 1)
            for (low, high), resolution in zip(MODEL_BOUNDS, MODEL_RESOLUTION)
        ]
        overall = {
            "centroid_index_xyz": [float(value) for value in centroid],
            "centroid_physical_xyz": _index_to_physical(centroid),
            "bbox_index_inclusive": {"min_xyz": bbox_min, "max_xyz": bbox_max},
            "bbox_physical_inclusive": {
                "min_xyz": _index_to_physical(bbox_min),
                "max_xyz": _index_to_physical(bbox_max),
            },
            "voxel_count": int(indices.shape[0]),
            "physical_volume": float(indices.shape[0] * np.prod(spacing)),
        }
    return {
        "connectivity_definition": {
            "6_connected": "face neighbors",
            "26_connected": "face, edge, or corner neighbors",
        },
        "component_count_6": len(records_6),
        "component_count_26": len(records_26),
        "components_6": records_6,
        "components_26": records_26,
        "overall_label9_geometry": overall,
        "primary_body_diagnostic": {
            "definition": "largest final raw-label9 26-connected component by hidden voxel count; retrospective metadata only",
            "component": primary,
        },
    }


def _validate_truth_tensor(truth: torch.Tensor) -> None:
    if list(truth.shape) != [1, 1, 64, 64, 64]:
        raise ValueError(f"unexpected truth shape: {list(truth.shape)}")
    array = truth.detach().cpu().numpy()
    if not np.isfinite(array).all():
        raise ValueError("truth contains non-finite values after fill_nans")
    if not np.equal(array, np.rint(array)).all():
        raise ValueError("truth is not integer-valued")
    if array.min() < -1 or array.max() > 13:
        raise ValueError(f"truth labels outside -1..13: {np.unique(array)}")


def _label_counts(truth: torch.Tensor) -> dict[str, int]:
    labels, counts = torch.unique(truth.to(torch.int64), return_counts=True)
    return {
        str(int(label)): int(count)
        for label, count in zip(labels.tolist(), counts.tolist())
    }


def prepare_candidate(root_seed: int) -> dict[str, Any]:
    (
        generator,
        model_contract,
        history,
        markov_sequence,
        event_records,
    ) = _history_only_candidate(root_seed)
    history_ok, history_reasons = history_eligibility(
        markov_sequence, event_records
    )
    candidate = {
        "root_seed": int(root_seed),
        "rng_contract": model_contract.describe(),
        "markov_sequence": markov_sequence,
        "event_subtypes": [event["subtype"] for event in event_records],
        "events": event_records,
        "history_eligible": history_ok,
        "history_rejection_reasons": history_reasons,
    }
    if not history_ok:
        candidate["eligible"] = False
        candidate["rejection_reasons"] = history_reasons
        return candidate

    truth, metadata = _materialize_history(generator, model_contract, history)
    _validate_truth_tensor(truth)
    conditions = condition_tensors(truth)
    label9_mask = truth == 9
    observed_label9_mask = label9_mask & conditions["condition_mask"]
    hidden_label9_mask = label9_mask & ~conditions["condition_mask"]
    total_count = int(label9_mask.sum())
    observed_count = int(observed_label9_mask.sum())
    hidden_count = int(hidden_label9_mask.sum())
    truth_reasons = []
    if total_count <= 0:
        truth_reasons.append("final_raw_label9_absent")
    if hidden_count <= 0:
        truth_reasons.append("no_hidden_raw_label9_under_fixed_condition")
    candidate.update(
        {
            "truth": truth,
            "metadata": metadata,
            "conditions": conditions,
            "label9_mask": label9_mask,
            "observed_label9_mask": observed_label9_mask,
            "hidden_label9_mask": hidden_label9_mask,
            "label9_counts": {
                "total": total_count,
                "observed": observed_count,
                "hidden": hidden_count,
                "hidden_fraction": (
                    float(hidden_count / total_count) if total_count else None
                ),
            },
            "eligible": not truth_reasons,
            "rejection_reasons": truth_reasons,
        }
    )
    if not truth_reasons:
        candidate["geometry"] = geometry_statistics(
            label9_mask, hidden_label9_mask
        )
    return candidate


def save_case(
    candidate: dict[str, Any],
    case_id: str,
    output_dir: Path,
    frozen_source_hashes: dict[str, str],
    versions: dict[str, str],
    matrix_hash: str,
) -> dict[str, Any]:
    case_dir = output_dir / "cases" / case_id
    if case_dir.exists():
        raise FileExistsError(f"refusing to overwrite existing case: {case_dir}")
    truth_dir = case_dir / "truth"
    condition_dir = case_dir / "condition"
    geometry_dir = case_dir / "geometry"
    truth_dir.mkdir(parents=True)
    condition_dir.mkdir(parents=True)
    geometry_dir.mkdir(parents=True)

    truth = candidate["truth"]
    label9_mask = candidate["label9_mask"]
    hidden_label9_mask = candidate["hidden_label9_mask"]
    conditions = candidate["conditions"]
    artifact_tensors = {
        truth_dir / "true_model.pt": truth,
        truth_dir / "label9_mask.pt": label9_mask,
        truth_dir / "hidden_label9_mask.pt": hidden_label9_mask,
        condition_dir / "boreholes.pt": conditions["condition_values"],
        condition_dir / "condition_values.pt": conditions["condition_values"],
        condition_dir / "condition_mask.pt": conditions["condition_mask"],
        condition_dir / "surface_mask.pt": conditions["surface_mask"],
        condition_dir / "borehole_mask.pt": conditions["borehole_mask"],
    }
    for path, tensor in artifact_tensors.items():
        torch.save(tensor.cpu(), path)

    history_payload = {
        "schema_version": "full_structuralgeo_history_v1",
        "root_seed": candidate["root_seed"],
        "rng_contract": candidate["rng_contract"],
        "markov_sequence": candidate["metadata"]["markov_sequence"],
        "event_subtypes": candidate["event_subtypes"],
        "events": candidate["metadata"]["events"],
        "packed_history": candidate["metadata"]["packed_history"],
        "unpacked_history": candidate["metadata"]["unpacked_history"],
    }
    write_json(truth_dir / "history.json", history_payload)
    write_json(geometry_dir / "connected_components.json", candidate["geometry"])
    write_json(
        condition_dir / "well_xy.json",
        {
            "layout_id": "canonical_nine_well_xy_v1",
            "truth_independent": True,
            "well_xy": [list(point) for point in CANONICAL_NINE_WELL_XY],
            "vertical_extent": [0, 63],
        },
    )

    condition_stats = {
        "surface_voxels": int(conditions["surface_mask"].sum()),
        "borehole_voxels": int(conditions["borehole_mask"].sum()),
        "surface_borehole_overlap_voxels": int(
            (conditions["surface_mask"] & conditions["borehole_mask"]).sum()
        ),
        "condition_union_voxels": int(conditions["condition_mask"].sum()),
        "condition_fraction": float(conditions["condition_mask"].float().mean()),
        "well_count": len(CANONICAL_NINE_WELL_XY),
    }
    tensor_content_hashes = {
        path.relative_to(case_dir).as_posix(): canonical_tensor_sha256(tensor)
        for path, tensor in artifact_tensors.items()
    }
    saved_file_hashes = {
        path.relative_to(case_dir).as_posix(): sha256_file(path)
        for path in (*artifact_tensors.keys(), truth_dir / "history.json", geometry_dir / "connected_components.json", condition_dir / "well_xy.json")
    }
    manifest = {
        "schema_version": SCHEMA_VERSION,
        "case_id": case_id,
        "cohort_id": ELIGIBILITY_VERSION,
        "root_seed": candidate["root_seed"],
        "child_seed_derivation": candidate["rng_contract"],
        "markov_state_sequence": candidate["metadata"]["markov_sequence"],
        "event_subtypes": candidate["event_subtypes"],
        "packed_history": candidate["metadata"]["packed_history"],
        "unpacked_history": candidate["metadata"]["unpacked_history"],
        "events": candidate["metadata"]["events"],
        "source_hashes": frozen_source_hashes,
        "default_matrix_sha256": matrix_hash,
        "environment_versions": versions,
        "bounds": [list(bound) for bound in MODEL_BOUNDS],
        "resolution": list(MODEL_RESOLUTION),
        "normalize": True,
        "height_tracking": True,
        "fill_nans": True,
        "raw_label_counts": _label_counts(truth),
        "raw_label_semantics": {
            "version": LABEL_SEMANTICS_VERSION,
            "mapping": {str(key): value for key, value in LABEL_NAMES.items()},
            "no_truth_voxel_modified": True,
            "labels_10_to_13_not_merged_into_9": True,
        },
        "well_coordinates_xy": [list(point) for point in CANONICAL_NINE_WELL_XY],
        "well_layout_truth_independent": True,
        "condition_statistics": condition_stats,
        "condition_content_hashes": {
            key: value
            for key, value in tensor_content_hashes.items()
            if key.startswith("condition/")
        },
        "label9": candidate["label9_counts"],
        "component_statistics": candidate["geometry"],
        "roles": {
            "true_model": "truth_only",
            "label9_mask": "truth_only_retrospective_target_metadata",
            "hidden_label9_mask": "truth_only_retrospective_target_metadata",
            "component_statistics": "truth_only_retrospective_metadata",
            "condition_values": "inference_visible",
            "condition_mask": "inference_visible",
            "surface_mask": "inference_visible",
            "borehole_mask": "inference_visible",
            "well_xy": "inference_visible_prospectively_fixed_layout",
        },
        "selection_firewall": {
            "no_seismic_in_selection": True,
            "no_flow_in_selection": True,
            "no_downstream_metric_in_selection": True,
            "visual_qc_not_used_for_selection": True,
        },
        "tensor_content_hashes": tensor_content_hashes,
        "saved_file_sha256": saved_file_hashes,
    }
    write_json(case_dir / "manifest.json", manifest)
    manifest["saved_file_sha256"]["manifest.json"] = sha256_file(
        case_dir / "manifest.json"
    )
    return manifest


def _subsample_points(points: np.ndarray, maximum: int = 18000) -> np.ndarray:
    if len(points) <= maximum:
        return points
    indices = np.linspace(0, len(points) - 1, maximum, dtype=int)
    return points[indices]


def create_qc_figure(
    case_id: str,
    truth: torch.Tensor,
    hidden_label9_mask: torch.Tensor,
    conditions: dict[str, torch.Tensor],
    output_dir: Path,
) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import BoundaryNorm, ListedColormap
    from matplotlib.lines import Line2D

    output_dir.mkdir(parents=True, exist_ok=True)
    volume = truth[0, 0].cpu().numpy().astype(int)
    hidden = hidden_label9_mask[0, 0].cpu().numpy().astype(bool)
    cmap = ListedColormap([LITHOLOGY_COLORS[label] for label in range(-1, 14)])
    norm = BoundaryNorm(np.arange(-1.5, 14.5, 1), cmap.N)

    figure = plt.figure(figsize=(15.8, 8.2), constrained_layout=True)
    grid = figure.add_gridspec(2, 4)
    ax_cut = figure.add_subplot(grid[0, 0], projection="3d")
    ax_xy = figure.add_subplot(grid[0, 1])
    ax_xz = figure.add_subplot(grid[0, 2])
    ax_yz = figure.add_subplot(grid[0, 3])
    ax_label9 = figure.add_subplot(grid[1, 0], projection="3d")
    ax_hidden = figure.add_subplot(grid[1, 1], projection="3d")
    ax_condition = figure.add_subplot(grid[1, 2], projection="3d")
    ax_legend = figure.add_subplot(grid[1, 3])

    coarse = volume[::2, ::2, ::2]
    coarse_points = np.argwhere(
        (coarse != -1)
        & (
            (np.indices(coarse.shape)[0] <= coarse.shape[0] // 2)
            | (np.indices(coarse.shape)[1] <= coarse.shape[1] // 2)
        )
    )
    coarse_points = _subsample_points(coarse_points)
    coarse_values = coarse[
        coarse_points[:, 0], coarse_points[:, 1], coarse_points[:, 2]
    ]
    ax_cut.scatter(
        coarse_points[:, 0] * 2,
        coarse_points[:, 1] * 2,
        coarse_points[:, 2] * 2,
        c=[LITHOLOGY_COLORS[int(value)] for value in coarse_values],
        s=2.0,
        marker="s",
        linewidths=0,
        alpha=0.88,
        rasterized=True,
    )
    ax_cut.set_title("Full geology cutaway")

    slice_specs = (
        (ax_xy, volume[:, :, 32].T, "central z: XY"),
        (ax_xz, volume[:, 32, :].T, "central y: XZ"),
        (ax_yz, volume[32, :, :].T, "central x: YZ"),
    )
    for axis, image, title in slice_specs:
        axis.imshow(
            image,
            origin="lower",
            interpolation="nearest",
            cmap=cmap,
            norm=norm,
            aspect="equal",
        )
        axis.set_title(title)
        axis.set_xlabel("index")
        axis.set_ylabel("index")

    label9_points = _subsample_points(np.argwhere(volume == 9))
    if len(label9_points):
        ax_label9.scatter(
            label9_points[:, 0],
            label9_points[:, 1],
            label9_points[:, 2],
            color=LITHOLOGY_COLORS[9],
            s=3,
            linewidths=0,
            alpha=0.78,
            rasterized=True,
        )
    ax_label9.set_title("Raw label9 only")

    hidden_points = _subsample_points(np.argwhere(hidden))
    if len(hidden_points):
        ax_hidden.scatter(
            hidden_points[:, 0],
            hidden_points[:, 1],
            hidden_points[:, 2],
            color="#d62728",
            s=3,
            linewidths=0,
            alpha=0.78,
            rasterized=True,
        )
    ax_hidden.set_title("Hidden raw label9")

    surface_points = np.argwhere(conditions["surface_mask"][0, 0].numpy())
    surface_points = _subsample_points(surface_points, maximum=9000)
    if len(surface_points):
        ax_condition.scatter(
            surface_points[:, 0],
            surface_points[:, 1],
            surface_points[:, 2],
            color="#6baed6",
            s=1.5,
            linewidths=0,
            alpha=0.16,
            rasterized=True,
        )
    for x_index, y_index in CANONICAL_NINE_WELL_XY:
        ax_condition.plot(
            [x_index, x_index],
            [y_index, y_index],
            [0, 63],
            color="#111111",
            linewidth=1.1,
        )
    ax_condition.set_title("Fixed wells + surface condition")

    for axis in (ax_cut, ax_label9, ax_hidden, ax_condition):
        axis.set_xlim(0, 63)
        axis.set_ylim(0, 63)
        axis.set_zlim(0, 63)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=24, azim=-55)
        axis.set_xlabel("x", labelpad=-6)
        axis.set_ylabel("y", labelpad=-6)
        axis.set_zlabel("z", labelpad=-6)
        axis.tick_params(labelsize=6, pad=-2)

    ax_legend.axis("off")
    handles = [
        Line2D(
            [0],
            [0],
            marker="s",
            linestyle="none",
            markerfacecolor=LITHOLOGY_COLORS[label],
            markeredgecolor="none",
            label=f"{label}: {LABEL_NAMES[label]}",
            markersize=7,
        )
        for label in range(-1, 14)
    ]
    ax_legend.legend(
        handles=handles,
        loc="center left",
        frameon=False,
        fontsize=8,
        ncol=2,
        title="Frozen raw labels",
        title_fontsize=9,
    )
    ax_legend.text(
        0.0,
        0.04,
        "QC only; never used for case selection.\nCommon colors and camera across all cases.",
        transform=ax_legend.transAxes,
        fontsize=8,
        color="#444444",
    )
    figure.suptitle(
        f"{case_id} — full StructuralGeo same-recipe benchmark QC",
        fontsize=13,
    )
    png_path = output_dir / f"{case_id}_qc.png"
    pdf_path = output_dir / f"{case_id}_qc.pdf"
    figure.savefig(png_path, dpi=300, bbox_inches="tight")
    figure.savefig(pdf_path, bbox_inches="tight")
    plt.close(figure)
    return [png_path, pdf_path]


def _case_summary_row(manifest: dict[str, Any]) -> str:
    sequence = " → ".join(manifest["markov_state_sequence"])
    label9 = manifest["label9"]
    return (
        f"| {manifest['case_id']} | {manifest['root_seed']} | "
        f"{label9['total']} | {label9['observed']} | {label9['hidden']} | "
        f"{label9['hidden_fraction']:.6f} | {sequence} |"
    )


def write_build_report(
    output_dir: Path,
    decision: str,
    git: dict[str, Any],
    config: dict[str, Any],
    determinism: dict[str, Any],
    accepted: list[dict[str, Any]],
    rejected: list[dict[str, Any]],
    frozen_source_hashes: dict[str, str],
    matrix_hash: str,
    qc_paths: list[Path],
) -> Path:
    checks = determinism.get("checks", {})
    lines = [
        "# Full StructuralGeo benchmark build report",
        "",
        "Date: 2026-08-11",
        f"Machine decision: `{decision}`",
        "",
        "## Scope and frozen stop rule",
        "",
        "This build is geology-only. It did not generate seismic, load a Flow checkpoint, run inference, run property inversion, build a probability bridge, or execute Stage11/Stage12B.",
        "",
        "The benchmark uses the training generator path `MarkovGeostoryGenerator` with the packaged default Markov matrix, `(-1920,1920)^3` bounds, `64^3` resolution, height tracking, height normalization, and final NaN-to-`-1` fill. Truth voxels were not edited, implanted, moved, deleted, relabeled, or merged.",
        "",
        "## Repository snapshot",
        "",
        f"- Branch: `{git['branch']}`",
        f"- HEAD: `{git['head']}`",
        f"- Dirty at build start: `{git['dirty']}`",
        "- Pre-build `git status --short`:",
        "",
        "```text",
        *(git["status_short"] or ["(clean)"]),
        "```",
        "",
        "## Deterministic RNG gate",
        "",
        f"RNG contract: `{RNG_CONTRACT_VERSION}`. Gate passed: `{determinism.get('passed', False)}`.",
        "",
        "The contract starts from one root `numpy.random.SeedSequence` and derives order-independent named streams for the Markov sequence, categorical event subtype, each event/GeoWord and nested GeoWord, legacy probability helpers, Fourier helpers, sediment helpers, metaballs, and height normalization. The fixed condition layout requires no random stream.",
        "",
    ]
    for name, result in checks.items():
        lines.append(f"- `{name}`: `{result}`")
    lines.extend(
        [
            "",
            "The full probe output, including two fresh-process replicates, a single-process replicate, a different-seed probe, categorical tensor SHA-256 values, histories, shape/range checks, matrix hash, and frozen source hashes, is in `audit/determinism_audit.json`.",
            "",
            "## Actual source files changed",
            "",
            *[f"- `{path}`" for path in ACTUAL_SOURCE_FILES_CHANGED],
            "",
            "## Prospectively frozen cohort rule",
            "",
            f"- Eligibility: `{ELIGIBILITY_VERSION}`",
            f"- Target accepted cases: `{config['accepted_case_target']}`",
            f"- Maximum seed budget: `{config['maximum_seed_budget']}`",
            f"- Seeds evaluated in the arithmetic order starting at `{config['root_seed_order']['start']}` with step `{config['root_seed_order']['step']}`.",
            "- Required before acceptance: BaseStrata history; at least one Fold or Fault event; at least one raw-label9-producing Dike/Sills/Pluton event; final raw label9 > 0; hidden raw label9 under the fixed condition > 0.",
            "- Forbidden selection variables: target centroid, visual attractiveness, similarity to cond_generation_0, seismic response, future bridge performance, and Flow performance.",
            "",
            "## Accepted cases",
            "",
            "| Case | Root seed | label9 total | observed | hidden | hidden fraction | Markov history |",
            "|---|---:|---:|---:|---:|---:|---|",
        ]
    )
    lines.extend(_case_summary_row(manifest) for manifest in accepted)
    lines.extend(
        [
            "",
            "Each case manifest records event subtypes, packed/unpacked histories, raw label counts, fixed well coordinates, condition hashes/statistics, 6/26-connected label9 geometry, and the retrospective primary-body diagnostic. `seed_registry.json` and `rejected_seed_registry.json` preserve the complete prospective search trace.",
            "",
            "## Rejected seeds",
            "",
            f"Rejected count: `{len(rejected)}`.",
            "",
            "| Root seed | Rejection reason(s) | Markov sequence |",
            "|---:|---|---|",
        ]
    )
    for item in rejected:
        lines.append(
            f"| {item['root_seed']} | {', '.join(item['rejection_reasons'])} | "
            f"{' → '.join(item['markov_sequence'])} |"
        )
    lines.extend(
        [
            "",
            "## Fixed condition and hidden-target semantics",
            "",
            "All cases use the same nine full-depth wells: `(8,46), (9,5), (10,24), (27,17), (35,26), (39,59), (44,60), (48,6), (57,32)`. The condition mask is exactly the fixed borehole mask OR the existing `make_surface_mask(truth)`. Explicit surface, borehole, and union masks are saved; no inference relies on sentinel `-1` alone.",
            "",
            "The target is exactly `raw_label9 = truth == 9`; hidden target is exactly `raw_label9 & ~condition_mask`. Labels 10–13 remain unchanged and are not merged into label9.",
            "",
            "## Provenance hashes",
            "",
            f"Default matrix SHA-256: `{matrix_hash}`",
            "",
            "| Source | SHA-256 |",
            "|---|---|",
        ]
    )
    for source, digest in sorted(frozen_source_hashes.items()):
        lines.append(f"| `{source}` | `{digest}` |")
    lines.extend(
        [
            "",
            "## QC figures",
            "",
            "QC figures were generated only after acceptance. They use identical lithology colors and camera settings and were not used to accept, reject, or replace any seed.",
            "",
        ]
    )
    for path in qc_paths:
        lines.append(f"- `{path.relative_to(output_dir).as_posix()}`")
    lines.extend(
        [
            "",
            "## Training-overlap statement",
            "",
            "independently generated, prospectively registered same-recipe test cohort; no case was used to choose inference parameters or update the frozen checkpoint. Historical sample-level overlap with the original streaming training run cannot be certified because no seed/sample manifest was retained.",
            "",
            "## Machine decision",
            "",
            f"`{decision}`",
            "",
            "Stage 12A stops here and awaits manual approval. No downstream experiment was started.",
        ]
    )
    report_path = output_dir / "FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md"
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return report_path


def _ensure_no_prior_generated_output(output_dir: Path) -> None:
    generated_targets = (
        output_dir / "seed_registry.json",
        output_dir / "rejected_seed_registry.json",
        output_dir / "benchmark_manifest.json",
        output_dir / "FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md",
        output_dir / "FULL_STRUCTURALGEO_BENCHMARK_DECISION.json",
        output_dir / "audit",
        output_dir / "cases",
        output_dir / "qc",
    )
    existing = [path for path in generated_targets if path.exists()]
    if existing:
        joined = ", ".join(str(path) for path in existing)
        raise FileExistsError(f"refusing to overwrite frozen benchmark outputs: {joined}")


def _rejected_record(candidate: dict[str, Any]) -> dict[str, Any]:
    return {
        "root_seed": candidate["root_seed"],
        "rng_contract": candidate["rng_contract"],
        "markov_sequence": candidate["markov_sequence"],
        "event_subtypes": candidate["event_subtypes"],
        "events": candidate["events"],
        "history_eligible": candidate["history_eligible"],
        "rejection_reasons": candidate["rejection_reasons"],
        "label9_counts": candidate.get("label9_counts"),
        "selection_stage": (
            "history_only"
            if not candidate["history_eligible"]
            else "final_truth_and_fixed_condition"
        ),
    }


def build_benchmark(
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    config_path: Path = DEFAULT_CONFIG_PATH,
    script_path: Path = DEFAULT_SCRIPT_PATH,
) -> dict[str, Any]:
    output_dir = output_dir.resolve()
    config_path = config_path.resolve()
    script_path = script_path.resolve()
    _ensure_no_prior_generated_output(output_dir)
    config = load_frozen_config(config_path)
    git = git_snapshot()
    versions = environment_versions()
    frozen_source_hashes = source_hashes()
    matrix_path = default_matrix_path()
    matrix_hash = sha256_file(matrix_path)
    audit_dir = output_dir / "audit"
    audit_dir.mkdir(parents=True, exist_ok=False)

    try:
        determinism = run_determinism_audit(script_path, config)
    except Exception as error:
        determinism = {
            "schema_version": "full_structuralgeo_determinism_audit_v1",
            "passed": False,
            "error_type": error.__class__.__name__,
            "error": str(error),
            "checks": {},
        }
    write_json(audit_dir / "determinism_audit.json", determinism)

    accepted_manifests: list[dict[str, Any]] = []
    accepted_registry: list[dict[str, Any]] = []
    rejected_registry: list[dict[str, Any]] = []
    qc_paths: list[Path] = []
    if determinism.get("passed", False):
        for root_seed in seed_order(config):
            if len(accepted_manifests) >= int(config["accepted_case_target"]):
                break
            candidate = prepare_candidate(root_seed)
            if not candidate["eligible"]:
                rejected_registry.append(_rejected_record(candidate))
                continue
            case_id = f"fullgeo_case{len(accepted_manifests) + 1:02d}"
            manifest = save_case(
                candidate=candidate,
                case_id=case_id,
                output_dir=output_dir,
                frozen_source_hashes=frozen_source_hashes,
                versions=versions,
                matrix_hash=matrix_hash,
            )
            accepted_manifests.append(manifest)
            accepted_registry.append(
                {
                    "case_id": case_id,
                    "root_seed": candidate["root_seed"],
                    "rng_contract": candidate["rng_contract"],
                    "markov_sequence": candidate["markov_sequence"],
                    "event_subtypes": candidate["event_subtypes"],
                    "label9_counts": candidate["label9_counts"],
                    "case_manifest": f"cases/{case_id}/manifest.json",
                }
            )
            qc_paths.extend(
                create_qc_figure(
                    case_id=case_id,
                    truth=candidate["truth"],
                    hidden_label9_mask=candidate["hidden_label9_mask"],
                    conditions=candidate["conditions"],
                    output_dir=output_dir / "qc",
                )
            )

    target_count = int(config["accepted_case_target"])
    if not determinism.get("passed", False):
        decision = MACHINE_NOT_REPRODUCIBLE
    elif len(accepted_manifests) < target_count:
        decision = MACHINE_COHORT_UNAVAILABLE
    else:
        decision = MACHINE_READY

    write_json(
        output_dir / "seed_registry.json",
        {
            "schema_version": "full_structuralgeo_accepted_seed_registry_v1",
            "cohort_id": ELIGIBILITY_VERSION,
            "frozen_config": config_path.relative_to(REPO_ROOT).as_posix(),
            "accepted_case_target": target_count,
            "accepted_count": len(accepted_registry),
            "accepted": accepted_registry,
        },
    )
    write_json(
        output_dir / "rejected_seed_registry.json",
        {
            "schema_version": "full_structuralgeo_rejected_seed_registry_v1",
            "cohort_id": ELIGIBILITY_VERSION,
            "maximum_seed_budget": int(config["maximum_seed_budget"]),
            "rejected_count": len(rejected_registry),
            "rejected": rejected_registry,
        },
    )
    decision_payload = {
        "schema_version": "full_structuralgeo_benchmark_decision_v1",
        "machine_decision": decision,
        "determinism_passed": bool(determinism.get("passed", False)),
        "cohort_id": ELIGIBILITY_VERSION,
        "accepted_case_target": target_count,
        "accepted_count": len(accepted_manifests),
        "rejected_count": len(rejected_registry),
        "seeds_examined": len(accepted_registry) + len(rejected_registry),
        "maximum_seed_budget": int(config["maximum_seed_budget"]),
        "stop_rule_honored": True,
        "downstream_actions_executed": [],
    }
    write_json(
        output_dir / "FULL_STRUCTURALGEO_BENCHMARK_DECISION.json",
        decision_payload,
    )
    report_path = write_build_report(
        output_dir=output_dir,
        decision=decision,
        git=git,
        config=config,
        determinism=determinism,
        accepted=accepted_manifests,
        rejected=rejected_registry,
        frozen_source_hashes=frozen_source_hashes,
        matrix_hash=matrix_hash,
        qc_paths=qc_paths,
    )
    benchmark_manifest = {
        "schema_version": SCHEMA_VERSION,
        "machine_decision": decision,
        "cohort_id": ELIGIBILITY_VERSION,
        "config_path": config_path.relative_to(REPO_ROOT).as_posix(),
        "config_sha256": sha256_file(config_path),
        "git": git,
        "rng_contract_version": RNG_CONTRACT_VERSION,
        "determinism_audit": "audit/determinism_audit.json",
        "determinism_audit_sha256": sha256_file(
            audit_dir / "determinism_audit.json"
        ),
        "accepted_case_count": len(accepted_manifests),
        "rejected_seed_count": len(rejected_registry),
        "cases": [
            {
                "case_id": manifest["case_id"],
                "root_seed": manifest["root_seed"],
                "manifest_path": f"cases/{manifest['case_id']}/manifest.json",
                "manifest_sha256": sha256_file(
                    output_dir / "cases" / manifest["case_id"] / "manifest.json"
                ),
            }
            for manifest in accepted_manifests
        ],
        "source_hashes": frozen_source_hashes,
        "default_matrix_path": matrix_path.relative_to(REPO_ROOT).as_posix(),
        "default_matrix_sha256": matrix_hash,
        "environment_versions": versions,
        "report_path": report_path.relative_to(output_dir).as_posix(),
        "report_sha256": sha256_file(report_path),
        "qc_files": {
            path.relative_to(output_dir).as_posix(): sha256_file(path)
            for path in qc_paths
        },
        "selection_firewall": {
            "no_seismic": True,
            "no_flow": True,
            "no_downstream_metric": True,
            "qc_generated_after_acceptance": True,
        },
        "training_overlap_wording": (
            "independently generated, prospectively registered same-recipe test cohort; "
            "no case was used to choose inference parameters or update the frozen checkpoint. "
            "Historical sample-level overlap with the original streaming training run cannot "
            "be certified because no seed/sample manifest was retained."
        ),
    }
    write_json(output_dir / "benchmark_manifest.json", benchmark_manifest)
    return decision_payload
