"""Truth-blind Stage20 formal-output inventory and freeze contract."""

from __future__ import annotations

from pathlib import Path
from typing import Mapping, Sequence

import torch

import inference_runtime as runtime
from guidance.seismic import tensor_sha256
from scripts.stage15.common import read_json, write_json


FORMAL_OUTPUT_INDEX_SCHEMA = "stage20_formal_output_index_v1"
FORMAL_OUTPUT_INDEX_NAME = "formal_output_index.json"
REQUIRED_OUTPUT_FIELDS = {
    "case_id", "current_case_id", "source_seed", "arm", "initial_noise_sha256",
    "base_checkpoint_sha256", "adapter_checkpoint_sha256", "inference_config_sha256",
    "runner_source_sha256", "condition_asset_sha256", "evidence_asset_sha256",
    "evidence_source_case_id", "mask_source_case_id", "n_steps", "adapter_scale",
    "residual_cap", "hard_condition_violation_count", "output_tensor_sha256",
    "output_file_sha256", "output_path",
}


def _ordered_records(
    records: Sequence[Mapping[str, object]],
    case_ids: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[str],
) -> list[dict[str, object]]:
    order = {
        (str(case_id), int(seed), str(arm)): index
        for index, (case_id, seed, arm) in enumerate(
            (case_id, seed, arm) for case_id in case_ids for seed in seeds for arm in arms
        )
    }
    actual: dict[tuple[str, int, str], dict[str, object]] = {}
    for source in records:
        record = dict(source)
        missing = REQUIRED_OUTPUT_FIELDS - set(record)
        if missing:
            raise RuntimeError(f"formal output provenance fields missing: {sorted(missing)}")
        key = (str(record["case_id"]), int(record["source_seed"]), str(record["arm"]))
        if key in actual:
            raise RuntimeError(f"duplicate formal output: {key}")
        if key not in order:
            raise RuntimeError(f"unexpected formal output: {key}")
        actual[key] = record
    if set(actual) != set(order):
        missing = sorted(set(order) - set(actual), key=order.get)
        raise RuntimeError(f"missing formal outputs: {missing[:3]} (count={len(missing)})")
    return [actual[key] for key in sorted(actual, key=order.get)]


def validate_formal_output_records(
    records: Sequence[Mapping[str, object]],
    *,
    run_dir: Path,
    case_ids: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[str],
    expected_base_checkpoint_sha256: str,
    expected_adapter_checkpoint_sha256: str,
    expected_inference_config_sha256: str,
    expected_runner_source_sha256: str,
    n_steps: int,
    residual_cap: float,
) -> list[dict[str, object]]:
    """Validate exact inventory and every output before truth can be accessed."""
    ordered = _ordered_records(records, case_ids, seeds, arms)
    root = run_dir.resolve()
    for record in ordered:
        case_id = str(record["case_id"]); arm = str(record["arm"])
        if str(record["current_case_id"]) != case_id or str(record["mask_source_case_id"]) != case_id:
            raise RuntimeError("formal output current-case/mask provenance mismatch")
        source_case = str(record["evidence_source_case_id"])
        if (arm == "ADAPTER_WRONG_CASE") != (source_case != case_id):
            raise RuntimeError("formal wrong-case evidence provenance mismatch")
        expected_scale = 0.0 if arm == "FLOW_ONLY" else 1.0
        if float(record["adapter_scale"]) != expected_scale:
            raise RuntimeError("formal adapter scale mismatch")
        if int(record["n_steps"]) != int(n_steps) or float(record["residual_cap"]) != float(residual_cap):
            raise RuntimeError("formal sampler provenance mismatch")
        for field, expected in (
            ("base_checkpoint_sha256", expected_base_checkpoint_sha256),
            ("adapter_checkpoint_sha256", expected_adapter_checkpoint_sha256),
            ("inference_config_sha256", expected_inference_config_sha256),
            ("runner_source_sha256", expected_runner_source_sha256),
        ):
            if str(record[field]) != expected:
                raise RuntimeError(f"formal output {field} mismatch")
        if int(record["hard_condition_violation_count"]) != 0:
            raise RuntimeError("STOP_STAGE20_HARD_CONDITION_VIOLATION")
        output = (run_dir / str(record["output_path"])).resolve()
        if root not in output.parents or not output.is_file():
            raise RuntimeError("formal output path missing or outside run directory")
        if runtime.file_sha256(output) != str(record["output_file_sha256"]):
            raise RuntimeError("formal output file SHA256 mismatch")
        tensor = runtime.load_tensor(output)
        if not torch.is_tensor(tensor) or tensor_sha256(tensor) != str(record["output_tensor_sha256"]):
            raise RuntimeError("formal output tensor SHA256 mismatch")
    return ordered


def freeze_formal_output_index(
    records: Sequence[Mapping[str, object]],
    *,
    run_dir: Path,
    case_ids: Sequence[str],
    seeds: Sequence[int],
    arms: Sequence[str],
    validation: Mapping[str, object],
) -> dict[str, object]:
    ordered = validate_formal_output_records(
        records, run_dir=run_dir, case_ids=case_ids, seeds=seeds, arms=arms, **validation
    )
    index = {
        "schema": FORMAL_OUTPUT_INDEX_SCHEMA,
        "formal_output_count": len(ordered),
        "case_ids": list(case_ids),
        "source_seeds": [int(seed) for seed in seeds],
        "arms": list(arms),
        "sort_order": ["case_id", "source_seed", "frozen_arm_order"],
        "truth_loaded_by_runner": False,
        "records": ordered,
    }
    path = run_dir / FORMAL_OUTPUT_INDEX_NAME
    write_json(path, index)
    return runtime.asset_record(path.resolve())


def audit_frozen_formal_inference(
    *,
    run_dir: Path,
    test_registry_path: Path,
    expected_test_registry_sha256: str,
    validation: Mapping[str, object],
) -> tuple[dict[str, object], list[dict[str, object]]]:
    """Complete truth-blind firewall; caller may load truth only after return."""
    try:
        manifest = read_json(run_dir / "run_manifest.json")
        if manifest.get("run_status") != "completed" or manifest.get("formal_inference_complete") is not True:
            raise RuntimeError("formal inference is not frozen complete")
        if manifest.get("formal_output_count") != 144 or manifest.get("truth_loaded_by_runner") is not False:
            raise RuntimeError("formal output count or truth firewall mismatch")
        if runtime.file_sha256(test_registry_path) != expected_test_registry_sha256:
            raise RuntimeError("stripped TEST registry SHA256 mismatch")
        registry = read_json(test_registry_path)
        if registry.get("schema") != "stage20_test_inference_registry_v1" or registry.get("case_count") != 12:
            raise RuntimeError("stripped TEST registry schema/count mismatch")
        case_ids = [str(case["case_id"]) for case in registry["cases"]]
        index_record = manifest.get("formal_output_index")
        if not isinstance(index_record, dict) or not {"path", "sha256"} <= set(index_record):
            raise RuntimeError("formal output index asset is absent")
        index_path = Path(str(index_record["path"]))
        if runtime.file_sha256(index_path) != str(index_record["sha256"]):
            raise RuntimeError("formal output index SHA256 mismatch")
        if manifest.get("formal_output_index_sha256") != index_record["sha256"]:
            raise RuntimeError("formal output index manifest SHA256 mismatch")
        for manifest_field, expected_field in (
            ("base_checkpoint", "expected_base_checkpoint_sha256"),
            ("adapter_checkpoint", "expected_adapter_checkpoint_sha256"),
            ("inference_config", "expected_inference_config_sha256"),
        ):
            record = manifest.get(manifest_field)
            if not isinstance(record, dict) or record.get("sha256") != validation[expected_field]:
                raise RuntimeError(f"formal run {manifest_field} SHA256 mismatch")
            path = Path(str(record.get("path")))
            if not path.is_file() or runtime.file_sha256(path) != record["sha256"]:
                raise RuntimeError(f"formal run {manifest_field} asset mismatch")
        runner = manifest.get("runner_source")
        if not isinstance(runner, dict) or runner.get("sha256") != validation["expected_runner_source_sha256"]:
            raise RuntimeError("formal run runner source SHA256 mismatch")
        if runtime.file_sha256(Path(str(runner.get("path")))) != runner["sha256"]:
            raise RuntimeError("formal run runner source asset mismatch")
        index = read_json(index_path)
        if index.get("schema") != FORMAL_OUTPUT_INDEX_SCHEMA:
            raise RuntimeError("formal output index schema mismatch")
        records = index.get("records")
        if not isinstance(records, list):
            raise RuntimeError("formal output index records missing")
        ordered = validate_formal_output_records(
            records, run_dir=run_dir, case_ids=case_ids,
            seeds=[42, 142, 242],
            arms=["FLOW_ONLY", "ADAPTER_PRIOR_ONLY", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE"],
            **validation,
        )
        if records != ordered:
            raise RuntimeError("formal output index ordering mismatch")
        case_records = {str(case["case_id"]): case for case in registry["cases"]}
        cyclic = {case_id: case_ids[(index + 1) % len(case_ids)] for index, case_id in enumerate(case_ids)}
        for record in ordered:
            case_id = str(record["case_id"]); arm = str(record["arm"])
            source_id = str(record["evidence_source_case_id"])
            expected_source = cyclic[case_id] if arm == "ADAPTER_WRONG_CASE" else case_id
            if source_id != expected_source:
                raise RuntimeError("formal cyclic wrong-case mapping mismatch")
            current = case_records[case_id]; source = case_records[source_id]
            expected_evidence = source["normalized_low_frequency_logz" if arm == "ADAPTER_PRIOR_ONLY" else "normalized_inverted_logz"]
            if record["condition_asset_sha256"] != current["condition_values"]["sha256"] or record["evidence_asset_sha256"] != expected_evidence["sha256"]:
                raise RuntimeError("formal condition/evidence asset provenance mismatch")
        return manifest, ordered
    except Exception as exc:
        raise RuntimeError(f"STOP_STAGE20_RETROSPECTIVE_EVALUATION_FIREWALL: {exc}") from exc
