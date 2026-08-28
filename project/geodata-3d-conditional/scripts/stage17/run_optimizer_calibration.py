#!/usr/bin/env python3
"""Run frozen Stage17B reference and D-Flow optimizer calibration arms."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.dflow import (
    differentiable_flow_solve,
    freeze_model_for_dflow,
    model_state_dict_hashes,
    optimize_source_noise,
)
from guidance.probability_sampling import fixed_euler_probability_sample
from guidance.probability_volume import tensor_sha256
from guidance.property_sampling import fixed_euler_property_sample
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage16.run_dflow import (
    _conditioning_from_assets,
    _decode,
    _objective,
    _validate_config as validate_stage16_config,
    _validated_assets,
)
from scripts.stage17.common import resolved_path, validate_asset_record


DEFAULT_CONFIG = (
    PROJECT_DIR
    / "experiments/stage17_evidence_coupling_attribution/configs/optimizer_calibration_v1.json"
)
RUN_CLASSES = ("engineering_smoke", "optimizer_calibration_confirmation")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--run-class", choices=RUN_CLASSES, required=True)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _validate_config(config: Mapping[str, object]) -> list[int]:
    if config.get("schema") != "stage17_optimizer_calibration_v1":
        raise ValueError("invalid Stage17B config schema")
    if config.get("status") != "frozen_before_run":
        raise ValueError("Stage17B config must be frozen_before_run")
    expected_arms = [
        "FLOW_ONLY",
        "REFERENCE_TRAJECTORY_GUIDANCE",
        "DFLOW_LEGACY_ADAM",
        "DFLOW_PAPER_ALIGNED_LBFGS",
    ]
    if config.get("arms") != expected_arms:
        raise ValueError("Stage17B arm set/order changed")
    if config.get("solver") != "fixed_euler" or int(config.get("n_steps", 0)) != 32:
        raise ValueError("Stage17B requires the authoritative 32-step fixed Euler solver")
    if config.get("smoke_may_tune_scientific_parameters") is not False:
        raise ValueError("engineering smoke must not tune scientific parameters")
    if config.get("parameter_sweep") is not False or config.get("training_performed") is not False:
        raise ValueError("Stage17B forbids parameter sweeps and training")
    if float(config.get("source_regularization_weight", -1.0)) != 0.0:
        raise ValueError("Stage17B source regularization must remain zero")
    seeds = [int(value) for value in config["source_seeds"]]
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise ValueError("Stage17B requires three unique preregistered seeds")
    if set(config["tasks"]) != {"oracle_probability", "property"}:
        raise ValueError("Stage17B requires separate oracle_probability and property tasks")
    return seeds


def _measured_call(device: torch.device, function: Callable[[], object]) -> tuple[object, float, int]:
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    result = function()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        peak = int(torch.cuda.max_memory_allocated(device))
    else:
        peak = 0
    return result, time.perf_counter() - started, peak


def _reference_sample(
    objective: str,
    reference: Mapping[str, object],
    model,
    initial: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
    assets: Mapping[str, Path],
    device: torch.device,
    sample_id: int,
) -> tuple[torch.Tensor, list[dict[str, object]]]:
    common = {
        "model": model,
        "initial_state": initial,
        "conditioning": conditioning,
        "embedded_truth": embedded_conditions,
        "truth_model": condition_values,
        "condition_mask": condition_mask,
        "n_steps": int(reference["n_steps"]),
        "alpha": float(reference["alpha"]),
        "max_guidance_ratio": float(reference["max_guidance_ratio"]),
        "tau_start": float(reference["tau_start"]),
        "tau_end": float(reference["tau_end"]),
        "tau_schedule": str(reference["tau_schedule"]),
        "guidance_start": float(reference["guidance_start"]),
        "guidance_schedule": str(reference["guidance_schedule"]),
        "grad_clip_norm": float(reference["grad_clip_norm"]),
        "guidance_scaling_mode": str(reference["guidance_scaling_mode"]),
        "sample_id": sample_id,
    }
    if objective == "oracle_probability":
        return fixed_euler_probability_sample(
            **common,
            target_probability=runtime.load_tensor(assets["target_probability"]).to(device),
            target_mask=runtime.load_tensor(assets["target_mask"]).to(device).bool(),
            roi_mask=runtime.load_tensor(assets["target_roi_mask"]).to(device).bool(),
            target_label=int(reference["target_label"]),
            bce_weight=float(reference["bce_weight"]),
            dice_weight=float(reference["dice_weight"]),
            spatial_gradient_weight=float(reference["spatial_gradient_weight"]),
            probability_loss_mode=str(reference["probability_loss_mode"]),
        )
    return fixed_euler_property_sample(
        **common,
        target_properties=runtime.load_tensor(assets["target_properties"]).to(device),
        property_table=runtime.load_tensor(assets["property_table"]).to(device),
        confidence=runtime.load_tensor(assets["property_confidence"]).to(device),
        property_sigmas=[float(v) for v in reference["property_sigmas"]],
        property_scale_weights=[float(v) for v in reference["property_scale_weights"]],
        property_channel_weights=torch.tensor(
            reference["property_channel_weights"], device=device, dtype=initial.dtype
        ),
        loss_mode=str(reference["property_loss_mode"]),
    )


def _save_arm(
    output_dir: Path,
    task: str,
    source_seed: int,
    arm: str,
    state: torch.Tensor,
    decoded: torch.Tensor,
) -> tuple[str, str]:
    arm_dir = output_dir / task / f"seed_{source_seed}" / arm
    arm_dir.mkdir(parents=True)
    state_path = arm_dir / "final_state.pt"
    decoded_path = arm_dir / "decoded_geology.pt"
    torch.save(state.detach().cpu(), state_path)
    torch.save(decoded, decoded_path)
    return (
        str(state_path.relative_to(output_dir)),
        str(decoded_path.relative_to(output_dir)),
    )


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    source_seeds = _validate_config(config)
    if args.run_class == "engineering_smoke":
        source_seeds = source_seeds[:1]
    device = torch.device(args.device or str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "config.json")
    manifest = base_manifest("stage17_optimizer_calibration_run_v1", Path(__file__), args.config)
    manifest.update(
        {
            "run_status": "running",
            "run_class": args.run_class,
            "scientific_evidence_eligible": args.run_class == "optimizer_calibration_confirmation",
            "smoke_results_may_tune_scientific_parameters": False,
            "executed_source_seeds": source_seeds,
            "task_aggregation_for_progression": "separate_no_pooled_compensation",
            "training_performed": False,
            "parameter_sweep_performed": False,
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)
    rows: list[dict[str, object]] = []
    traces: list[dict[str, object]] = []

    try:
        from model_train_sh_inference_cond import Geo3DStochInterp

        task_assets: dict[str, dict[str, object]] = {}
        for task, task_record in config["tasks"].items():
            source_path = validate_asset_record(task_record["stage16_source_config"])
            reference_path = validate_asset_record(task_record["authoritative_reference_config"])
            source_config = read_json(source_path)
            objective, frozen_seeds = validate_stage16_config(source_config)
            if objective != task or frozen_seeds != [int(v) for v in config["source_seeds"]]:
                raise ValueError(f"Stage16 source config mismatch for {task}")
            reference = read_json(reference_path)
            if int(reference["n_steps"]) != int(config["n_steps"]):
                raise ValueError(f"authoritative reference solver mismatch for {task}")
            assets = _validated_assets(source_config, objective)
            task_assets[task] = {
                "stage16_source_config": runtime.asset_record(source_path),
                "authoritative_reference_config": runtime.asset_record(reference_path),
                "nested_exact_assets": {
                    name: runtime.asset_record(path) for name, path in assets.items()
                },
            }
            model, model_report = runtime.load_model_with_weight_policy(
                model_class=Geo3DStochInterp,
                checkpoint_path=assets["checkpoint"],
                map_location=device,
                weight_source="ema",
            )
            model = model.to(device)
            freeze_model_for_dflow(model)
            model_hash_before, _ = model_state_dict_hashes(model)
            condition_values, condition_mask, embedded_conditions, conditioning, condition_report = (
                _conditioning_from_assets(objective, assets, model, device)
            )
            endpoint_loss, trace_diagnostics, objective_report = _objective(
                objective, source_config, assets, model, device
            )

            for sample_id, source_seed in enumerate(source_seeds):
                generator = torch.Generator(device="cpu").manual_seed(source_seed)
                initial_cpu = torch.randn(
                    1,
                    model.embedding_dim,
                    *model.data_shape,
                    generator=generator,
                    dtype=embedded_conditions.dtype,
                )
                initial = initial_cpu.to(device)
                seed_dir = args.output_dir / task / f"seed_{source_seed}"
                seed_dir.mkdir(parents=True)
                torch.save(initial_cpu, seed_dir / "initial_source.pt")

                (baseline_result, baseline_runtime, baseline_peak) = _measured_call(
                    device,
                    lambda: differentiable_flow_solve(
                        model=model,
                        initial_state=initial,
                        conditioning=conditioning,
                        embedded_conditions=embedded_conditions,
                        condition_mask=condition_mask,
                        n_steps=int(config["n_steps"]),
                        solver=str(config["solver"]),
                        use_gradient_checkpointing=bool(config["use_gradient_checkpointing"]),
                        condition_values=condition_values,
                    ),
                )
                baseline_state, _ = baseline_result
                baseline_decoded = _decode(model, baseline_state)
                baseline_loss, _ = endpoint_loss(baseline_state)
                state_path, decoded_path = _save_arm(
                    args.output_dir, task, source_seed, "FLOW_ONLY", baseline_state, baseline_decoded
                )
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "source_seed": source_seed,
                        "arm": "FLOW_ONLY",
                        "initial_noise_sha256": tensor_sha256(initial_cpu),
                        "final_state_sha256": tensor_sha256(baseline_state),
                        "decoded_geology_sha256": tensor_sha256(baseline_decoded),
                        "endpoint_primary_loss": float(baseline_loss.detach().cpu()),
                        "source_update_norm": 0.0,
                        "closure_evaluations": 0,
                        "endpoint_solves": 1,
                        "runtime_seconds": baseline_runtime,
                        "peak_cuda_memory_bytes": baseline_peak,
                        "condition_violations": int(((baseline_decoded != condition_values.cpu()) & condition_mask.cpu()).sum()),
                        "final_state_path": state_path,
                        "decoded_path": decoded_path,
                    }
                )
                del baseline_result, baseline_state

                (reference_result, reference_runtime, reference_peak) = _measured_call(
                    device,
                    lambda: _reference_sample(
                        objective,
                        reference,
                        model,
                        initial,
                        conditioning,
                        embedded_conditions,
                        condition_values,
                        condition_mask,
                        assets,
                        device,
                        sample_id,
                    ),
                )
                reference_state, reference_trace = reference_result
                reference_decoded = _decode(model, reference_state)
                reference_loss, _ = endpoint_loss(reference_state)
                state_path, decoded_path = _save_arm(
                    args.output_dir,
                    task,
                    source_seed,
                    "REFERENCE_TRAJECTORY_GUIDANCE",
                    reference_state,
                    reference_decoded,
                )
                rows.append(
                    {
                        "task": task,
                        "sample_id": sample_id,
                        "source_seed": source_seed,
                        "arm": "REFERENCE_TRAJECTORY_GUIDANCE",
                        "initial_noise_sha256": tensor_sha256(initial_cpu),
                        "final_state_sha256": tensor_sha256(reference_state),
                        "decoded_geology_sha256": tensor_sha256(reference_decoded),
                        "endpoint_primary_loss": float(reference_loss.detach().cpu()),
                        "source_update_norm": 0.0,
                        "closure_evaluations": 0,
                        "endpoint_solves": 1,
                        "runtime_seconds": reference_runtime,
                        "peak_cuda_memory_bytes": reference_peak,
                        "condition_violations": int(((reference_decoded != condition_values.cpu()) & condition_mask.cpu()).sum()),
                        "final_state_path": state_path,
                        "decoded_path": decoded_path,
                    }
                )
                for trace_row in reference_trace:
                    traces.append({"task": task, "source_seed": source_seed, "arm": "REFERENCE_TRAJECTORY_GUIDANCE", **trace_row})
                del reference_result, reference_state

                for arm in ("DFLOW_LEGACY_ADAM", "DFLOW_PAPER_ALIGNED_LBFGS"):
                    optimizer = config["optimizers"][arm]
                    (result, arm_runtime, arm_peak) = _measured_call(
                        device,
                        lambda optimizer=optimizer: optimize_source_noise(
                            model=model,
                            initial_source=initial,
                            conditioning=conditioning,
                            embedded_conditions=embedded_conditions,
                            condition_mask=condition_mask,
                            endpoint_loss=endpoint_loss,
                            n_steps=int(config["n_steps"]),
                            solver=str(config["solver"]),
                            optimizer_name=str(optimizer["optimizer"]),
                            learning_rate=float(optimizer["learning_rate"]),
                            optimization_iterations=int(optimizer["optimization_iterations"]),
                            lbfgs_max_iter=int(optimizer["lbfgs_max_iter"]),
                            source_regularization_weight=float(config["source_regularization_weight"]),
                            use_gradient_checkpointing=bool(config["use_gradient_checkpointing"]),
                            condition_values=condition_values,
                            trace_diagnostics=trace_diagnostics,
                        ),
                    )
                    decoded = _decode(model, result.final_state)
                    state_path, decoded_path = _save_arm(
                        args.output_dir, task, source_seed, arm, result.final_state, decoded
                    )
                    arm_dir = args.output_dir / task / f"seed_{source_seed}" / arm
                    torch.save(result.optimized_source, arm_dir / "optimized_source.pt")
                    write_json(arm_dir / "optimization_diagnostics.json", result.diagnostics)
                    arm_trace = [
                        {"task": task, "source_seed": source_seed, "arm": arm, **row}
                        for row in result.trace
                    ]
                    write_csv(arm_dir / "optimization_trace.csv", arm_trace)
                    traces.extend(arm_trace)
                    closure_count = int(result.diagnostics["closure_evaluation_count"])
                    iteration_count = int(result.diagnostics["optimization_iterations"])
                    rows.append(
                        {
                            "task": task,
                            "sample_id": sample_id,
                            "source_seed": source_seed,
                            "arm": arm,
                            "initial_noise_sha256": tensor_sha256(initial_cpu),
                            "final_state_sha256": tensor_sha256(result.final_state),
                            "decoded_geology_sha256": tensor_sha256(decoded),
                            "endpoint_primary_loss": result.trace[-1]["endpoint_primary_loss"] if result.trace else float("nan"),
                            "source_update_norm": float((result.optimized_source - initial).norm().cpu()),
                            "closure_evaluations": closure_count,
                            "endpoint_solves": closure_count + iteration_count + 1,
                            "runtime_seconds": arm_runtime,
                            "peak_cuda_memory_bytes": arm_peak,
                            "condition_violations": int(((decoded != condition_values.cpu()) & condition_mask.cpu()).sum()),
                            "final_state_path": state_path,
                            "decoded_path": decoded_path,
                        }
                    )
                    del result
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                print(f"Stage17B {task} seed={source_seed} completed", flush=True)
                del initial

            model_hash_after, _ = model_state_dict_hashes(model)
            if model_hash_before != model_hash_after:
                raise RuntimeError(f"model state changed during Stage17B task {task}")
            task_assets[task].update(
                {
                    "model_load_report": model_report,
                    "conditioning_report": condition_report,
                    "objective_report": objective_report,
                    "model_state_dict_sha256_before": model_hash_before,
                    "model_state_dict_sha256_after": model_hash_after,
                    "model_state_dict_unchanged": True,
                }
            )
            del model
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_csv(args.output_dir / "arm_manifest.csv", rows)
        write_csv(args.output_dir / "all_traces.csv", traces)
        if any(int(row["condition_violations"]) != 0 for row in rows):
            raise RuntimeError("Stage17B produced hard-condition violations")
        for task in config["tasks"]:
            for seed in source_seeds:
                hashes = {row["initial_noise_sha256"] for row in rows if row["task"] == task and int(row["source_seed"]) == seed}
                if len(hashes) != 1:
                    raise RuntimeError(f"arms do not share initial noise: {task}/{seed}")
        manifest.update(
            {
                "run_status": "completed",
                "task_assets": task_assets,
                "arm_count": len(rows),
                "arm_manifest": runtime.asset_record(args.output_dir / "arm_manifest.csv"),
                "all_traces": runtime.asset_record(args.output_dir / "all_traces.csv"),
                "all_model_state_dicts_unchanged": True,
                "all_condition_violations_zero": True,
                "results_selected_or_ranked": False,
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
