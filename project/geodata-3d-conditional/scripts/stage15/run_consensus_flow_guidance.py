#!/usr/bin/env python3
"""Run strict paired frozen-Flow arms from Stage15 binary consensus."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.probability_sampling import fixed_euler_probability_sample
from guidance.seismic import tensor_sha256
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_PROTOCOL = EXPERIMENT_ROOT / "configs/flow_guidance_protocol_v1.json"
DEFAULT_PHASE1_BASELINE = (
    PROJECT_DIR
    / "experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4"
    / "calibrated_reference_windowed/seed42_n4_s32/baseline"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--consensus-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--protocol", type=Path, default=DEFAULT_PROTOCOL)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n-samples", type=int, default=None)
    parser.add_argument("--phase1-baseline-dir", type=Path, default=DEFAULT_PHASE1_BASELINE)
    return parser.parse_args()


def _decode(model, state: torch.Tensor) -> torch.Tensor:
    return (model.decode(state).detach().cpu() - 1).unsqueeze(1).long()


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    protocol = read_json(args.protocol)
    required = {
        "schema": "stage15_consensus_flow_protocol_v1",
        "status": "frozen_before_flow_execution",
        "arms": ["FLOW_ONLY", "FLOW_PLUS_BINARY_CONSENSUS"],
        "model_weights": "ema",
        "n_euler_steps": 32,
        "integrator": "fixed_euler_midpoint_v1",
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "tau_start": 0.5,
        "tau_end": 0.1,
        "tau_schedule": "cosine",
        "guidance_start": 0.25,
        "guidance_schedule": "windowed_sine",
        "guidance_scaling_mode": "reference_norm_relative_v2",
        "grad_clip_norm": 1.0,
        "probability_loss_mode": "calibrated_soft_bce_hard_dice_v2",
        "bce_weight": 1.0,
        "dice_weight": 1.0,
        "spatial_gradient_weight": 0.0,
        "target_label": 9,
    }
    for key, expected in required.items():
        if protocol.get(key) != expected:
            raise ValueError(f"frozen Flow protocol mismatch for {key}")
    seed = int(protocol["default_seed"] if args.seed is None else args.seed)
    n_samples = int(protocol["default_n_samples"] if args.n_samples is None else args.n_samples)
    if n_samples <= 0:
        raise ValueError("n-samples must be positive")
    if seed != 42 and args.phase1_baseline_dir == DEFAULT_PHASE1_BASELINE:
        raise ValueError("default historical regression assets apply only to seed 42")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    manifest = base_manifest("stage15_consensus_flow_run_v1", Path(__file__), args.protocol)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_manifest.json", manifest)
    try:
        observation_manifest = read_json(args.observation_dir / "manifest.json")
        consensus_manifest = read_json(args.consensus_dir / "consensus_manifest.json")
        if observation_manifest.get("run_status") != "completed" or consensus_manifest.get("run_status") != "completed":
            raise ValueError("observation and consensus must both be completed")
        condition_values_cpu = normalize_volume(runtime.load_tensor(args.observation_dir / "flow_condition_values.pt"), "flow_condition_values").long()
        condition_mask_cpu = normalize_volume(runtime.load_tensor(args.observation_dir / "flow_condition_mask.pt"), "flow_condition_mask").bool()
        subsurface_cpu = normalize_volume(runtime.load_tensor(args.observation_dir / "subsurface_mask.pt"), "subsurface_mask").bool()
        target_probability_cpu = normalize_volume(runtime.load_tensor(args.consensus_dir / "consensus_target.pt"), "consensus_target", torch.float32)
        positive_cpu = normalize_volume(runtime.load_tensor(args.consensus_dir / "positive_mask.pt"), "positive_mask").bool()
        guidance_roi_cpu = normalize_volume(runtime.load_tensor(args.consensus_dir / "guidance_roi.pt"), "guidance_roi").bool()
        if bool((guidance_roi_cpu & ~(subsurface_cpu & ~condition_mask_cpu)).any()):
            raise ValueError("guidance ROI leaks outside confident unconditioned subsurface")
        confidence_cpu = normalize_volume(runtime.load_tensor(args.consensus_dir / "negative_mask.pt"), "negative_mask").bool() | positive_cpu
        if not torch.equal(guidance_roi_cpu, confidence_cpu & subsurface_cpu & ~condition_mask_cpu):
            raise ValueError("guidance ROI differs from frozen consensus definition")

        checkpoint_path = REPOSITORY_ROOT / str(protocol["checkpoint"]["path"])
        if runtime.file_sha256(checkpoint_path) != protocol["checkpoint"]["sha256"]:
            raise ValueError("checkpoint hash changed")
        from model_train_sh_inference_cond import Geo3DStochInterp

        model, model_report = runtime.load_model_with_weight_policy(
            Geo3DStochInterp, checkpoint_path, device, weight_source="ema"
        )
        model = model.to(device)
        condition_values = condition_values_cpu.to(device)
        condition_mask = condition_mask_cpu.to(device)
        embedded = model.embed(condition_values)
        expanded_mask = condition_mask.expand(-1, embedded.shape[1], -1, -1, -1)
        conditioning = embedded * expanded_mask
        target_probability = target_probability_cpu.to(device)
        positive = positive_cpu.to(device)
        guidance_roi = guidance_roi_cpu.to(device)
        generator = torch.Generator(device="cpu").manual_seed(seed)
        traces = {"FLOW_ONLY": [], "FLOW_PLUS_BINARY_CONSENSUS": []}
        records: list[dict[str, object]] = []

        for sample_id in range(n_samples):
            initial_cpu = torch.randn(
                1, model.embedding_dim, *model.data_shape,
                generator=generator, dtype=embedded.dtype, device="cpu"
            )
            outputs: dict[str, torch.Tensor] = {}
            for arm, alpha in (("FLOW_ONLY", 0.0), ("FLOW_PLUS_BINARY_CONSENSUS", 0.25)):
                final_state, arm_trace = fixed_euler_probability_sample(
                    model=model,
                    initial_state=initial_cpu.to(device),
                    conditioning=conditioning,
                    embedded_truth=embedded,
                    truth_model=condition_values,
                    condition_mask=condition_mask,
                    target_probability=target_probability,
                    target_mask=positive,
                    roi_mask=guidance_roi,
                    target_label=9,
                    n_steps=32,
                    alpha=alpha,
                    max_guidance_ratio=0.25,
                    tau_start=0.5,
                    tau_end=0.1,
                    tau_schedule="cosine",
                    guidance_start=0.25,
                    guidance_schedule="windowed_sine",
                    grad_clip_norm=1.0,
                    bce_weight=1.0,
                    dice_weight=1.0,
                    spatial_gradient_weight=0.0,
                    probability_loss_mode="calibrated_soft_bce_hard_dice_v2",
                    guidance_scaling_mode="reference_norm_relative_v2",
                    sample_id=sample_id,
                )
                decoded = _decode(model, final_state)
                violations = int(((decoded != condition_values_cpu) & condition_mask_cpu).sum())
                if violations:
                    raise RuntimeError(f"condition projection failed for {arm}/{sample_id}")
                arm_dir = args.output_dir / arm
                arm_dir.mkdir(exist_ok=True)
                torch.save(decoded, arm_dir / f"sample_{sample_id}.pt")
                for row in arm_trace:
                    row["arm"] = arm
                    row["source_seed"] = seed
                traces[arm].extend(arm_trace)
                outputs[arm] = decoded

            historical_path = args.phase1_baseline_dir / f"sample_{sample_id}.pt"
            historical = normalize_volume(
                runtime.load_tensor(historical_path), f"historical_sample_{sample_id}"
            ).long()
            regression_equal = torch.equal(outputs["FLOW_ONLY"], historical)
            if not regression_equal:
                mismatch = int((outputs["FLOW_ONLY"] != historical).sum())
                raise RuntimeError(f"Phase1 alpha=0 regression failed at sample {sample_id}: {mismatch} voxels")
            records.append(
                {
                    "sample_id": sample_id,
                    "source_seed": seed,
                    "initial_noise_sha256": tensor_sha256(initial_cpu),
                    "flow_only_sha256": tensor_sha256(outputs["FLOW_ONLY"]),
                    "guided_sha256": tensor_sha256(outputs["FLOW_PLUS_BINARY_CONSENSUS"]),
                    "phase1_alpha0_regression_equal": regression_equal,
                    "flow_only_condition_violations": 0,
                    "guided_condition_violations": 0,
                }
            )
        for arm, rows in traces.items():
            write_csv(args.output_dir / arm / "guidance_trace.csv", rows)
        write_csv(args.output_dir / "pair_manifest.csv", records)
        manifest.update(
            {
                "run_status": "completed",
                "seed": seed,
                "n_samples": n_samples,
                "samples": records,
                "model_load_report": model_report,
                "checkpoint": runtime.asset_record(checkpoint_path),
                "observation_manifest": runtime.asset_record(args.observation_dir / "manifest.json"),
                "consensus_manifest": runtime.asset_record(args.consensus_dir / "consensus_manifest.json"),
                "input_tensor_sha256": {
                    "flow_condition_values": tensor_sha256(condition_values_cpu),
                    "flow_condition_mask": tensor_sha256(condition_mask_cpu),
                    "subsurface_mask": tensor_sha256(subsurface_cpu),
                    "consensus_target": tensor_sha256(target_probability_cpu),
                    "positive_mask": tensor_sha256(positive_cpu),
                    "guidance_roi": tensor_sha256(guidance_roi_cpu),
                },
                "truth_loaded_by_runner": False,
                "training_performed": False,
                "parameter_sweep_performed": False,
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        write_json(args.output_dir / "model_load_report.json", model_report)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
