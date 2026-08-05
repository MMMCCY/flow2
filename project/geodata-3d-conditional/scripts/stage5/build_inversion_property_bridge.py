#!/usr/bin/env python3
"""Build truth-blind Phase-5b property target/confidence assets."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.inversion_bridge import (
    PHASE5B_BRIDGE_MANIFEST_SCHEMA,
    log_impedance_property_table,
    posterior_spread_confidence,
    property_config_from_table,
    validate_bridge_config,
)
from guidance.seismic import tensor_sha256
from scripts.stage4.audit_seismic_identifiability import validate_output_directory
from scripts.stage4.run_seismic_guidance import read_json, write_json


OUTPUTS = (
    "property_table.pt",
    "target_properties.pt",
    "property_confidence.pt",
    "subsurface_mask.pt",
    "condition_mask.pt",
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage5_acoustic_inversion"
    parser = argparse.ArgumentParser(
        description="Convert the completed Phase-5a posterior to Phase-5b assets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--phase5a-dir",
        type=Path,
        default=experiment
        / "outputs/cond_generation_0/model_based_fixed12_v1",
    )
    parser.add_argument(
        "--bridge-config",
        type=Path,
        default=experiment / "configs/inversion_property_bridge_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment
        / "bridge_observations/cond_generation_0/fixed12_log_impedance_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _load_recorded_tensor(
    directory: Path,
    filename: str,
    records: Mapping[str, object],
) -> torch.Tensor:
    record = records.get(filename)
    if not isinstance(record, Mapping):
        raise ValueError(f"Phase-5a manifest lacks tensor record: {filename}")
    path = directory / filename
    if runtime.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"Phase-5a tensor file hash mismatch: {filename}")
    value = runtime.load_tensor(path)
    if tensor_sha256(value) != record.get("tensor_sha256"):
        raise ValueError(f"Phase-5a tensor content mismatch: {filename}")
    return value


def _tensor_record(path: Path, value: torch.Tensor) -> dict[str, object]:
    return {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": runtime.file_sha256(path),
        "tensor_sha256": tensor_sha256(value),
        "shape": list(value.shape),
        "dtype": str(value.dtype),
    }


def main() -> None:
    args = parse_args()
    validate_output_directory(args.output_dir, overwrite=args.overwrite)
    phase5a_manifest = read_json(args.phase5a_dir / "manifest.json")
    if phase5a_manifest.get("status") != "complete":
        raise ValueError("Phase-5a posterior is not complete")
    anti_leakage = phase5a_manifest.get("anti_leakage")
    if not isinstance(anti_leakage, Mapping) or any(
        anti_leakage.get(field) is not False
        for field in ("truth_geology_loaded", "truth_acoustic_loaded", "unconstrained_truth_used")
    ):
        raise ValueError("Phase-5a anti-leakage declaration is invalid")
    audit_summary_path = args.phase5a_dir / "audit/summary.json"
    audit_manifest_path = args.phase5a_dir / "audit/manifest.json"
    audit_summary = read_json(audit_summary_path)
    audit_manifest = read_json(audit_manifest_path)
    if audit_manifest.get("status") != "complete":
        raise ValueError("Phase-5a truth audit is not complete")
    if audit_summary.get("promoted_to_property_guidance_bridge_test") is not True:
        raise ValueError("Phase-5a did not authorize the property-guidance bridge")
    # The pass bit is a stop gate only. No truth-derived metric is consumed by
    # target/confidence construction below.
    bridge_config = read_json(args.bridge_config)
    validate_bridge_config(bridge_config)
    records = phase5a_manifest.get("generated_tensors")
    if not isinstance(records, Mapping):
        raise ValueError("Phase-5a manifest lacks generated tensors")
    log_mean = _load_recorded_tensor(
        args.phase5a_dir, "posterior_log_impedance_mean.pt", records
    )
    log_std = _load_recorded_tensor(
        args.phase5a_dir, "posterior_log_impedance_std.pt", records
    )
    condition_mask = _load_recorded_tensor(
        args.phase5a_dir, "condition_mask.pt", records
    ).bool()

    sources = phase5a_manifest.get("source_assets")
    if not isinstance(sources, Mapping) or not isinstance(
        sources.get("observation_manifest"), Mapping
    ):
        raise ValueError("Phase-5a manifest lacks observation source")
    observation_manifest_path = Path(str(sources["observation_manifest"]["path"]))
    if not observation_manifest_path.is_absolute():
        observation_manifest_path = PROJECT_DIR.parents[1] / observation_manifest_path
    if runtime.file_sha256(observation_manifest_path) != sources["observation_manifest"].get(
        "sha256"
    ):
        raise ValueError("Phase-5a observation manifest hash mismatch")
    observation_manifest = read_json(observation_manifest_path)
    observation_records = observation_manifest.get("generated_tensors")
    if not isinstance(observation_records, Mapping):
        raise ValueError("observation manifest lacks generated tensors")
    observation_dir = observation_manifest_path.parent

    def load_observation(filename: str) -> torch.Tensor:
        record = observation_records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"observation tensor record missing: {filename}")
        value = runtime.load_tensor(observation_dir / filename)
        if tensor_sha256(value) != record.get("sha256"):
            raise ValueError(f"observation tensor mismatch: {filename}")
        return value

    acoustic_table = load_observation("acoustic_property_table.pt")
    subsurface = load_observation("subsurface_mask.pt").bool()
    if log_mean.shape != log_std.shape or log_mean.shape != subsurface.shape:
        raise ValueError("posterior moments and subsurface mask must match")
    if condition_mask.shape != subsurface.shape:
        raise ValueError("condition mask must match posterior spatial shape")
    property_table = log_impedance_property_table(acoustic_table).float()
    target = log_mean.float()
    confidence, confidence_metadata = posterior_spread_confidence(
        log_std, subsurface, condition_mask
    )
    confidence = confidence.float()
    property_config = property_config_from_table(
        property_table,
        description=(
            "Phase-5b log-impedance codebook derived from the immutable Phase-4c "
            "acoustic table; target is the truth-blind Phase-5a posterior mean."
        ),
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    tensors = {
        "property_table.pt": property_table,
        "target_properties.pt": target,
        "property_confidence.pt": confidence,
        "subsurface_mask.pt": subsurface,
        "condition_mask.pt": condition_mask,
    }
    generated: dict[str, object] = {}
    for filename, value in tensors.items():
        path = args.output_dir / filename
        torch.save(value.contiguous(), path)
        generated[filename] = _tensor_record(path, value.contiguous())
    if set(generated) != set(OUTPUTS):
        raise AssertionError("internal Phase-5b output inventory mismatch")
    write_json(args.output_dir / "property_config_resolved.json", property_config)
    manifest = {
        "schema": PHASE5B_BRIDGE_MANIFEST_SCHEMA,
        "status": "complete",
        "description": "Truth-blind log-impedance posterior target for one Phase-5b flow gate.",
        "source_assets": {
            "phase5a_manifest": runtime.asset_record(args.phase5a_dir / "manifest.json"),
            "phase5a_audit_manifest": runtime.asset_record(audit_manifest_path),
            "phase5a_audit_summary_gate_only": runtime.asset_record(audit_summary_path),
            "observation_manifest": runtime.asset_record(observation_manifest_path),
            "bridge_config": runtime.asset_record(args.bridge_config),
            "builder_source": runtime.asset_record(Path(__file__)),
            "bridge_source": runtime.asset_record(PROJECT_DIR / "guidance/inversion_bridge.py"),
        },
        "generated_tensors": generated,
        "property_config": runtime.asset_record(
            args.output_dir / "property_config_resolved.json"
        ),
        "confidence_metadata": confidence_metadata,
        "truth_geology_loaded": False,
        "truth_acoustic_loaded": False,
        "truth_metrics_used_for_construction": False,
        "phase5a_pass_bit_used_as_stop_gate": True,
        "measured_geophysics": False,
        "inverse_crime_source": True,
    }
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Phase-5b property bridge complete: {args.output_dir}")
    print(f"Active confidence voxels: {confidence_metadata['active_voxels']}")


if __name__ == "__main__":
    main()
