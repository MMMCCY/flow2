#!/usr/bin/env python3
"""Truth-blind Stage17A full-trace evidence inversion."""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_seismic_inversion import binary_acoustic_properties_from_configs
from guidance.binary_trace_boundary import refine_binary_trace_volume, vertical_boundary_strength
from guidance.seismic import seismic_operator_from_config, tensor_sha256
from guidance.seismic_inversion import ModelBasedInversionConfig
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage17.common import require_frozen_config, validate_asset_record


ROOT = PROJECT_DIR / "experiments/stage17_evidence_coupling_attribution"
DEFAULT_CONFIG = ROOT / "configs/evidence_multicase_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument(
        "--smoke", action="store_true",
        help="select the first frozen case only; never changes scientific parameters",
    )
    return parser.parse_args()


def _validated_case_assets(case: dict[str, object]) -> dict[str, Path]:
    required = {
        "observation_manifest", "observed_seismic", "subsurface_mask",
        "binary_fixed_values", "binary_fixed_mask", "condition_values", "condition_mask",
    }
    assets = case.get("assets")
    if not isinstance(assets, dict) or set(assets) != required:
        raise ValueError(f"invalid evidence asset API: {case.get('case_id')}")
    return {name: validate_asset_record(record, f"{case['case_id']}/{name}") for name, record in assets.items()}


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    require_frozen_config(config, "stage17_evidence_multicase_v1")
    if config.get("truth_loaded_by_runner") is not False:
        raise ValueError("Stage17A inversion must be truth-blind")
    if config.get("full_vertical_trace_used") is not True or int(config.get("trace_samples", 0)) != 320:
        raise ValueError("Stage17A must use complete 320-sample traces")
    cases = config.get("cases")
    if not isinstance(cases, list) or len(cases) != 5:
        raise ValueError("formal evidence config must contain five cases")
    executed = cases[:1] if args.smoke else cases
    binary_path = validate_asset_record(config["binary_acoustic_config"], "binary_acoustic_config")
    seismic_path = validate_asset_record(config["seismic_config"], "seismic_config")
    inversion_path = validate_asset_record(config["inversion_config"], "inversion_config")
    binary_config = read_json(binary_path)
    source_path = validate_asset_record(binary_config["source_acoustic_config"], "source_acoustic_config")
    properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
    inversion = read_json(inversion_path)
    frozen_parameters = config["inversion_parameters"]
    if (
        int(frozen_parameters["refinement_passes"]) != int(inversion["refinement_passes"])
        or float(frozen_parameters["prior_relative_weight"]) != float(inversion["prior_relative_weight"])
        or float(frozen_parameters["vertical_smoothness_relative_weight"])
        != float(inversion["vertical_smoothness_relative_weight"])
    ):
        raise ValueError("evidence config silently changed Stage15-H inversion parameters")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "config.json")
    manifest = base_manifest("stage17_multicase_trace_evidence_run_v1", Path(__file__), args.config)
    manifest.update({
        "run_status": "running",
        "run_class": "engineering_smoke" if args.smoke else "formal_evidence_confirmation",
        "smoke_subset": bool(args.smoke),
        "truth_loaded_by_runner": False,
        "truth_role": "not_available_to_runner",
        "configured_case_order": [str(case["case_id"]) for case in cases],
        "executed_case_order": [str(case["case_id"]) for case in executed],
    })
    write_json(args.output_dir / "run_manifest.json", manifest)
    records: list[dict[str, object]] = []
    try:
        for case in executed:
            case_id = str(case["case_id"])
            assets = _validated_case_assets(case)
            observation_manifest = read_json(assets["observation_manifest"])
            if observation_manifest.get("run_status") != "completed":
                raise ValueError(f"observation incomplete: {case_id}")
            if observation_manifest.get("truth_loaded_by_inversion_runner") is not False:
                raise ValueError(f"observation manifest violates inversion firewall: {case_id}")
            observed = runtime.load_tensor(assets["observed_seismic"]).to(device=device, dtype=torch.float32)
            if observed.shape[-1] != 320:
                raise ValueError(f"not a complete 320-sample observation: {case_id}")
            support = normalize_volume(runtime.load_tensor(assets["subsurface_mask"]), "support").bool()
            fixed_values = normalize_volume(runtime.load_tensor(assets["binary_fixed_values"]), "fixed_values", torch.float32)
            fixed_mask = normalize_volume(runtime.load_tensor(assets["binary_fixed_mask"]), "fixed_mask").bool()
            operator, operator_metadata = seismic_operator_from_config(
                read_json(seismic_path), grid_shape=support.shape[2:]
            )
            inversion_config = ModelBasedInversionConfig(
                "stage17_binary_trace_boundary_v1",
                float(frozen_parameters["prior_relative_weight"]),
                float(frozen_parameters["vertical_smoothness_relative_weight"]),
            )
            score, acoustic, predicted, trace = refine_binary_trace_volume(
                observed,
                support.to(device),
                operator,
                properties,
                inversion_config,
                int(frozen_parameters["refinement_passes"]),
                fixed_values,
                fixed_mask,
            )
            boundary = vertical_boundary_strength(score)
            case_dir = args.output_dir / case_id
            case_dir.mkdir()
            outputs = {
                "binary_impedance_score.pt": score,
                "binary_acoustic_volume.pt": acoustic,
                "predicted_seismic.pt": predicted,
                "vertical_boundary_strength.pt": boundary,
            }
            for name, tensor in outputs.items():
                torch.save(tensor, case_dir / name)
            write_csv(case_dir / "refinement_trace.csv", trace)
            case_manifest = base_manifest(
                "stage17_multicase_trace_evidence_case_v1", Path(__file__), args.config
            )
            case_manifest.update({
                "run_status": "completed",
                "run_class": manifest["run_class"],
                "case_id": case_id,
                "truth_loaded_by_runner": False,
                "truth_role": "not_available_to_runner",
                "full_vertical_trace_used": True,
                "trace_sample_count": int(observed.shape[-1]),
                "lateral_mixing": False,
                "input_assets": {name: runtime.asset_record(path) for name, path in assets.items()},
                "output_tensor_sha256": {name: tensor_sha256(tensor) for name, tensor in outputs.items()},
                "operator_metadata": operator_metadata,
                "refinement_trace": trace,
            })
            write_json(case_dir / "manifest.json", case_manifest)
            records.append({
                "case_id": case_id,
                "manifest": runtime.asset_record(case_dir / "manifest.json"),
                "evidence_tensor_sha256": tensor_sha256(score),
                "boundary_tensor_sha256": tensor_sha256(boundary),
            })
            print(f"Stage17A evidence {case_id} completed", flush=True)
        manifest.update({
            "run_status": "completed",
            "executed_case_count": len(records),
            "cases": records,
            "scientific_conclusion_allowed": not args.smoke,
            "parameter_sweep_performed": False,
            "training_performed": False,
        })
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
