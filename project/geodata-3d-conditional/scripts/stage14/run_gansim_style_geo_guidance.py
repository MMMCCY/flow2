#!/usr/bin/env python3
"""Run the truth-blind Stage14 BASELINE/GEO_PROB_GUIDED Flow pairs."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for import_root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

import inference_runtime as runtime
from guidance.probability_sampling import fixed_euler_probability_sample
from guidance.probability_volume import tensor_sha256


DEFAULT_PROTOCOL = (
    PROJECT_DIR
    / "experiments/stage14_gansim_style_geo_guidance/configs/frozen_protocol.json"
)
STAGE12B_ROOT = PROJECT_DIR / "experiments/stage12b_fullgeo_probability_bridge"
HISTORICAL_DECISIONS = {
    "stage10": PROJECT_DIR
    / "experiments/stage10_geophysical_probability_bridge/reports/STAGE10_MACHINE_DECISION.json",
    "stage12b": STAGE12B_ROOT / "reports/STAGE12B_A_MACHINE_DECISION.json",
    "stage13": PROJECT_DIR
    / "experiments/stage13_binary_label9_bridge/reports/STAGE13A_MACHINE_DECISION.json",
}


def _load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise TypeError(f"expected JSON object: {path}")
    return value


def _write_json(path: Path, value: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)


def _write_rows(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    if not rows:
        return
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


def _git(*command: str) -> str:
    result = subprocess.run(
        ["git", *command],
        cwd=REPOSITORY_ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def _validate_protocol(protocol: Mapping[str, object], protocol_path: Path) -> None:
    if protocol.get("status") != "frozen_before_flow_execution":
        raise ValueError("Stage14 protocol is not frozen before Flow execution")
    if protocol.get("arms") != ["BASELINE", "GEO_PROB_GUIDED"]:
        raise ValueError("Stage14 must contain exactly the two registered arms")
    source_protocol = protocol.get("source_protocol")
    if not isinstance(source_protocol, Mapping):
        raise TypeError("source_protocol record is missing")
    source_path = REPOSITORY_ROOT / str(source_protocol["path"])
    if runtime.file_sha256(source_path) != source_protocol.get("sha256"):
        raise ValueError("frozen Stage12B source protocol hash changed")
    if runtime.file_sha256(protocol_path) == "":
        raise RuntimeError("unreachable protocol hash failure")
    expected_decisions = protocol.get("historical_machine_decision_sha256")
    if not isinstance(expected_decisions, Mapping):
        raise TypeError("historical decision hashes are missing")
    for name, path in HISTORICAL_DECISIONS.items():
        if runtime.file_sha256(path) != expected_decisions.get(name):
            raise ValueError(f"historical {name} machine decision changed")


def _normalize_tensor(value: torch.Tensor, name: str) -> torch.Tensor:
    value = runtime.normalize_single_geology(value, name)
    if tuple(value.shape) != (1, 1, 64, 64, 64):
        raise ValueError(f"{name} has unexpected shape {tuple(value.shape)}")
    return value


def _decode(model, state: torch.Tensor) -> torch.Tensor:
    return (model.decode(state).detach().cpu() - 1)[0].long()


def _sample_one(
    *,
    model,
    initial_cpu: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    target_probability: torch.Tensor,
    target_core: torch.Tensor,
    guidance_roi: torch.Tensor,
    settings: Mapping[str, object],
    alpha: float,
    sample_id: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    final_state, trace = fixed_euler_probability_sample(
        model=model,
        initial_state=initial_cpu.to(embedded_conditions.device),
        conditioning=conditioning,
        embedded_truth=embedded_conditions,
        truth_model=condition_values,
        condition_mask=condition_mask,
        target_probability=target_probability,
        target_mask=target_core,
        roi_mask=guidance_roi,
        target_label=9,
        n_steps=int(settings["n_euler_steps"]),
        alpha=float(alpha),
        max_guidance_ratio=float(settings["max_guidance_ratio"]),
        tau_start=float(settings["tau_start"]),
        tau_end=float(settings["tau_end"]),
        tau_schedule=str(settings["tau_schedule"]),
        guidance_start=float(settings["guidance_start"]),
        guidance_schedule=str(settings["guidance_schedule"]),
        grad_clip_norm=float(settings["grad_clip_norm"]),
        bce_weight=float(settings["bce_weight"]),
        dice_weight=float(settings["dice_weight"]),
        spatial_gradient_weight=float(settings["spatial_gradient_weight"]),
        probability_loss_mode=str(settings["probability_loss_mode"]),
        guidance_scaling_mode=str(settings["guidance_scaling_mode"]),
        sample_id=sample_id,
    )
    if not torch.isfinite(final_state).all():
        raise FloatingPointError(f"non-finite final state for sample {sample_id}")
    return _decode(model, final_state), trace


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--case-id", action="append", default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.output_root.exists() and any(args.output_root.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {args.output_root}")
    protocol = _load_json(args.protocol)
    _validate_protocol(protocol, args.protocol)
    all_case_ids = [str(value) for value in protocol["case_ids"]]
    case_ids = list(args.case_id or all_case_ids)
    if not case_ids or any(case_id not in all_case_ids for case_id in case_ids):
        raise ValueError("requested case IDs must be a non-empty subset of the protocol")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    checkpoint_path = REPOSITORY_ROOT / str(protocol["checkpoint"]["path"])
    if runtime.file_sha256(checkpoint_path) != protocol["checkpoint"]["sha256"]:
        raise ValueError("checkpoint hash differs from frozen protocol")
    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=checkpoint_path,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device)
    if model_report["ema_missing_trainable"] or model_report["ema_shape_mismatches"]:
        raise RuntimeError("EMA validation failed")

    args.output_root.mkdir(parents=True, exist_ok=True)
    settings = {
        **protocol["flow"],
        **protocol["guidance"],
    }
    seed_map = protocol["source_seeds"]
    case_manifests: dict[str, object] = {}

    for case_id in case_ids:
        case_dir = args.output_root / "cases" / case_id
        observations = STAGE12B_ROOT / "observations" / case_id
        probability_path = STAGE12B_ROOT / "bridge" / case_id / "probability_label9_post.pt"
        condition_values_path = observations / "condition_values.pt"
        condition_mask_path = observations / "condition_mask.pt"
        subsurface_mask_path = observations / "subsurface_mask.pt"

        condition_values_cpu = _normalize_tensor(
            runtime.load_tensor(condition_values_path), "condition_values"
        ).long()
        condition_mask_cpu = _normalize_tensor(
            runtime.load_tensor(condition_mask_path), "condition_mask"
        ).bool()
        subsurface_mask_cpu = _normalize_tensor(
            runtime.load_tensor(subsurface_mask_path), "subsurface_mask"
        ).bool()
        probability_cpu = _normalize_tensor(
            runtime.load_tensor(probability_path), "probability_label9_post"
        ).float()
        if not torch.isfinite(probability_cpu).all() or bool(
            ((probability_cpu < 0) | (probability_cpu > 1)).any()
        ):
            raise ValueError(f"invalid probability volume for {case_id}")
        if bool((condition_mask_cpu & ~((condition_values_cpu >= -1) & (condition_values_cpu <= 13))).any()):
            raise ValueError(f"invalid condition labels for {case_id}")
        guidance_roi_cpu = subsurface_mask_cpu & ~condition_mask_cpu
        if not bool(guidance_roi_cpu.any()):
            raise ValueError(f"empty unconditioned-subsurface ROI for {case_id}")
        target_core_cpu = probability_cpu >= 0.5

        condition_values = condition_values_cpu.to(device)
        condition_mask = condition_mask_cpu.to(device)
        embedded_conditions = model.embed(condition_values)
        expanded_mask = condition_mask.expand(
            -1, embedded_conditions.shape[1], -1, -1, -1
        )
        conditioning = embedded_conditions * expanded_mask
        probability = probability_cpu.to(device)
        target_core = target_core_cpu.to(device)
        guidance_roi = guidance_roi_cpu.to(device)
        traces: dict[str, list[dict[str, object]]] = {
            "BASELINE": [],
            "GEO_PROB_GUIDED": [],
        }
        sample_records: list[dict[str, object]] = []

        for sample_id, source_seed in enumerate(seed_map[case_id]):
            generator = torch.Generator(device="cpu").manual_seed(int(source_seed))
            initial_cpu = torch.randn(
                1,
                model.embedding_dim,
                *model.data_shape,
                generator=generator,
                dtype=embedded_conditions.dtype,
            )
            initial_hash = tensor_sha256(initial_cpu)
            outputs: dict[str, torch.Tensor] = {}
            for arm, alpha in (("BASELINE", 0.0), ("GEO_PROB_GUIDED", settings["alpha"])):
                decoded, arm_trace = _sample_one(
                    model=model,
                    initial_cpu=initial_cpu,
                    conditioning=conditioning,
                    embedded_conditions=embedded_conditions,
                    condition_values=condition_values,
                    condition_mask=condition_mask,
                    target_probability=probability,
                    target_core=target_core,
                    guidance_roi=guidance_roi,
                    settings=settings,
                    alpha=float(alpha),
                    sample_id=sample_id,
                )
                output_path = case_dir / arm / f"source_seed_{source_seed}.pt"
                output_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(decoded, output_path)
                for row in arm_trace:
                    row["case_id"] = case_id
                    row["source_seed"] = int(source_seed)
                    row["arm"] = arm
                traces[arm].extend(arm_trace)
                outputs[arm] = decoded
            condition_expand = condition_mask_cpu[0]
            baseline_violations = int(
                ((outputs["BASELINE"] != condition_values_cpu[0]) & condition_expand).sum()
            )
            guided_violations = int(
                ((outputs["GEO_PROB_GUIDED"] != condition_values_cpu[0]) & condition_expand).sum()
            )
            if baseline_violations or guided_violations:
                raise RuntimeError(f"final hard-condition violation for {case_id}/{source_seed}")
            sample_records.append(
                {
                    "case_id": case_id,
                    "sample_id": sample_id,
                    "source_seed": int(source_seed),
                    "initial_noise_sha256": initial_hash,
                    "baseline_sample_sha256": tensor_sha256(outputs["BASELINE"]),
                    "guided_sample_sha256": tensor_sha256(outputs["GEO_PROB_GUIDED"]),
                    "baseline_condition_violations": baseline_violations,
                    "guided_condition_violations": guided_violations,
                }
            )

        for arm, rows in traces.items():
            _write_rows(case_dir / arm / "guidance_trace.csv", rows)
        _write_rows(case_dir / "pair_manifest.csv", sample_records)
        case_manifest = {
            "schema": "stage14_truth_blind_flow_case_v1",
            "status": "completed",
            "case_id": case_id,
            "source_seeds": [int(value) for value in seed_map[case_id]],
            "probability_asset": runtime.asset_record(probability_path),
            "probability_tensor_sha256": tensor_sha256(probability_cpu),
            "condition_values_asset": runtime.asset_record(condition_values_path),
            "condition_mask_asset": runtime.asset_record(condition_mask_path),
            "subsurface_mask_asset": runtime.asset_record(subsurface_mask_path),
            "condition_mask_tensor_sha256": tensor_sha256(condition_mask_cpu),
            "guidance_roi_tensor_sha256": tensor_sha256(guidance_roi_cpu),
            "guidance_roi_voxels": int(guidance_roi_cpu.sum()),
            "fixed_internal_dice_core_voxels": int((target_core_cpu & guidance_roi_cpu).sum()),
            "probability_preprocessing": None,
            "truth_loaded_by_flow_runner": False,
            "samples": sample_records,
        }
        _write_json(case_dir / "manifest.json", case_manifest)
        case_manifests[case_id] = case_manifest
        del condition_values, condition_mask, embedded_conditions, conditioning
        del probability, target_core, guidance_roi
        if device.type == "cuda":
            torch.cuda.empty_cache()

    run_manifest = {
        "schema": "stage14_truth_blind_flow_run_v1",
        "status": "completed",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "experiment_role": protocol["experiment_role"],
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "protocol": runtime.asset_record(args.protocol),
        "runner": runtime.asset_record(Path(__file__)),
        "checkpoint": runtime.asset_record(checkpoint_path),
        "model_load_report": model_report,
        "device": str(device),
        "case_ids": case_ids,
        "n_pairs": sum(len(seed_map[case_id]) for case_id in case_ids),
        "case_manifests": {
            case_id: f"cases/{case_id}/manifest.json" for case_id in case_ids
        },
        "truth_loaded_by_flow_runner": False,
        "training_performed": False,
        "parameter_sweep_performed": False,
        "sample_selection_performed": False,
    }
    _write_json(args.output_root / "run_manifest.json", run_manifest)
    _write_json(args.output_root / "model_load_report.json", model_report)


if __name__ == "__main__":
    main()
