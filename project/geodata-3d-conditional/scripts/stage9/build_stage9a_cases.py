#!/usr/bin/env python3
"""Build frozen Stage9A inference and retrospective case assets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import socket
import subprocess
import sys

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import inference_runtime as runtime
from guidance.native_geology_audit import build_structuralgeo_native_case
from guidance.prior_ensemble import hard_seismic_response, validate_protocol_config
from guidance.seismic import acoustic_tables_from_config, seismic_operator_from_config
from guidance.structured_posterior import controlled_observations
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_json,
    save_tensor_x,
    utc_now,
    write_json_x,
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage9_flow_prior_posterior"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs/stage9a_prior_support_v1.json",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment / "cases/stage9a_native_v1",
    )
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _resolve(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def build_case_assets(
    *,
    case_config: dict[str, object],
    config: dict[str, object],
    staging_root: Path,
    property_table: torch.Tensor,
    forward_operator,
    device: torch.device,
) -> dict[str, object]:
    case_id = str(case_config["case_id"])
    case, native_metadata = build_structuralgeo_native_case(
        seed=int(case_config["native_seed"])
    )
    wrong_case, wrong_metadata = build_structuralgeo_native_case(
        seed=int(case_config["wrong_case_seed"])
    )
    condition_values = torch.full_like(case.truth_labels, case.background_label)
    condition_values[~case.subsurface_mask] = case.air_label
    condition_values[case.condition_mask] = case.truth_labels[case.condition_mask]
    if not torch.equal(
        condition_values[case.condition_mask], case.truth_labels[case.condition_mask]
    ):
        raise RuntimeError("case condition extraction failed")

    correct = hard_seismic_response(
        case.truth_labels.to(device),
        property_table=property_table,
        subsurface_mask=case.subsurface_mask.to(device),
        forward_operator=forward_operator,
    ).cpu()
    wrong = hard_seismic_response(
        wrong_case.truth_labels.to(device),
        property_table=property_table,
        subsurface_mask=wrong_case.subsurface_mask.to(device),
        forward_operator=forward_operator,
    ).cpu()
    observations = controlled_observations(
        correct,
        wrong_case=wrong,
        shuffle_seed=int(case_config["shuffle_seed"]),
    )

    case_root = staging_root / case_id
    inference_root = case_root / "inference"
    retrospective_root = case_root / "retrospective"
    inference_root.mkdir(parents=True)
    retrospective_root.mkdir(parents=True)
    tensor_records = {
        "condition_values": save_tensor_x(
            inference_root / "condition_values.pt", condition_values.long()
        ),
        "condition_mask": save_tensor_x(
            inference_root / "condition_mask.pt", case.condition_mask.bool()
        ),
        "subsurface_mask": save_tensor_x(
            inference_root / "subsurface_mask.pt", case.subsurface_mask.bool()
        ),
        "acoustic_property_table": save_tensor_x(
            inference_root / "acoustic_property_table.pt", property_table.cpu().float()
        ),
    }
    for name, observation in observations.items():
        output_name = "wrong_case" if name == "wrong_case_observation" else name
        tensor_records[f"observation_{output_name}"] = save_tensor_x(
            inference_root / f"observation_{output_name}.pt", observation.float()
        )
    inference_manifest = {
        "schema": "stage9a_inference_case_assets_v1",
        "status": "complete",
        "case_id": case_id,
        "case_index": int(case_config["case_index"]),
        "native_seed": int(case_config["native_seed"]),
        "wrong_case_seed": int(case_config["wrong_case_seed"]),
        "shuffle_seed": int(case_config["shuffle_seed"]),
        "grid_shape": [64, 64, 64],
        "air_start_z": 56,
        "condition_policy": "surface_plus_three_fixed_native_boreholes_v1",
        "observation_role": "synthetic_observation_generation_only",
        "geological_labels_available_to_inference": False,
        "tensors": tensor_records,
    }
    write_json_x(inference_root / "manifest.json", inference_manifest)

    retrospective_records = {
        "truth_labels": save_tensor_x(
            retrospective_root / "truth_labels.pt", case.truth_labels.long()
        ),
        "native_body_masks": save_tensor_x(
            retrospective_root / "native_body_masks.pt", case.body_masks.bool()
        ),
    }
    retrospective_manifest = {
        "schema": "stage9a_retrospective_case_assets_v1",
        "status": "complete",
        "case_id": case_id,
        "truth_role": "retrospective_metrics_only_after_pool_and_ranking_freeze",
        "target_label": int(config["target_label"]),
        "native_metadata": native_metadata,
        "wrong_case_metadata": wrong_metadata,
        "tensors": retrospective_records,
        "inference_manifest_sha256": runtime.file_sha256(
            inference_root / "manifest.json"
        ),
    }
    write_json_x(retrospective_root / "manifest.json", retrospective_manifest)
    return {
        "case_id": case_id,
        "inference_manifest": file_record(
            inference_root / "manifest.json", relative_to=staging_root
        ),
        "retrospective_manifest": file_record(
            retrospective_root / "manifest.json", relative_to=staging_root
        ),
    }


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    validate_protocol_config(config)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    acoustic_config_path = _resolve(config["acoustic_config"])
    seismic_config_path = _resolve(config["seismic_config"])
    acoustic_config = read_json(acoustic_config_path)
    seismic_config = read_json(seismic_config_path)
    tables, acoustic_metadata = acoustic_tables_from_config(acoustic_config, 15)
    property_table = tables.property_table.to(device=device, dtype=torch.float32)
    operator, seismic_metadata = seismic_operator_from_config(
        seismic_config, grid_shape=(64, 64, 64)
    )
    staging = create_staging_directory(args.output_dir)
    case_records = []
    for case_config in config["primary_cases"]:
        case_records.append(
            build_case_assets(
                case_config=dict(case_config),
                config=config,
                staging_root=staging,
                property_table=property_table,
                forward_operator=operator,
                device=device,
            )
        )
    manifest = {
        "schema": "stage9a_case_cohort_v1",
        "status": "complete",
        "completed_at_utc": utc_now(),
        "config": file_record(args.config),
        "spec": file_record(PROJECT_DIR / "docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md"),
        "acoustic_config": file_record(acoustic_config_path),
        "seismic_config": file_record(seismic_config_path),
        "native_builder": file_record(PROJECT_DIR / "guidance/native_geology_audit.py"),
        "runner": file_record(Path(__file__)),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "runtime": {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "device": str(device),
            "gpu_name": torch.cuda.get_device_name(device)
            if device.type == "cuda"
            else None,
        },
        "acoustic_metadata": acoustic_metadata,
        "seismic_metadata": seismic_metadata,
        "cases": case_records,
    }
    write_json_x(staging / "manifest.json", manifest)
    publish_staging_directory(staging, args.output_dir)
    print(
        json.dumps(
            {
                "status": "complete",
                "case_count": len(case_records),
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
