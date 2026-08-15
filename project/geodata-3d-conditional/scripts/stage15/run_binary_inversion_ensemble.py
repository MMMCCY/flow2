#!/usr/bin/env python3
"""Run truth-blind Stage15 hard-binary inversion ensemble members."""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
    inversion_tau,
    masked_first_difference_tv,
    smooth_initial_logits,
    straight_through_binary,
)
from guidance.seismic import seismic_field_loss, seismic_operator_from_config, tensor_sha256
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    validate_asset,
    write_csv,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_OBSERVATION = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/binary_inversion_pilot_v1.json"
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--n-members", type=int, default=4)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _load_observation_inputs(directory: Path) -> dict[str, torch.Tensor]:
    names = (
        "observed_seismic.pt",
        "sample_mask.pt",
        "uncertainty.pt",
        "subsurface_mask.pt",
        "binary_well_values.pt",
        "binary_well_mask.pt",
    )
    return {name: runtime.load_tensor(directory / name) for name in names}


def _rmse(predicted: torch.Tensor, observed: torch.Tensor, mask: torch.Tensor) -> float:
    weights = mask.to(predicted)
    return float(torch.sqrt((weights * (predicted - observed.to(predicted)).square()).sum() / weights.sum()).detach().cpu())


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    if args.n_members <= 0 or args.n_members > 100:
        raise ValueError("n-members must be in [1,100]; Stage15 does not auto-run N=100")
    config = read_json(args.config)
    if config.get("schema") != "stage15_binary_inversion_config_v1":
        raise ValueError("invalid Stage15 inversion config")
    if config.get("optimizer") != "Adam" or config.get("parameter_sweep") is not False:
        raise ValueError("Stage15 v1 requires fixed Adam with no sweep")
    seeds = [int(value) for value in config["member_seeds"]]
    if args.n_members > len(seeds) or len(set(seeds[: args.n_members])) != args.n_members:
        raise ValueError("config does not provide enough unique member seeds")
    device = torch.device(args.device or str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(str(config["dtype"]))
    if dtype is None:
        raise ValueError("dtype must be float32 or float64")

    manifest = base_manifest("stage15_binary_inversion_ensemble_v1", Path(__file__), args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "manifest.json", manifest)
    try:
        observation_manifest = read_json(args.observation_dir / "manifest.json")
        if observation_manifest.get("run_status") != "completed":
            raise ValueError("observation is not completed")
        inputs = _load_observation_inputs(args.observation_dir)
        expected_hashes = observation_manifest["output_tensor_sha256"]
        for name, tensor in inputs.items():
            if tensor_sha256(tensor) != expected_hashes[name]:
                raise ValueError(f"observation tensor hash changed: {name}")
        observed = inputs["observed_seismic.pt"].to(device=device, dtype=dtype)
        sample_mask = inputs["sample_mask.pt"].to(device=device, dtype=dtype)
        uncertainty = inputs["uncertainty.pt"].to(device=device, dtype=dtype)
        subsurface = normalize_volume(inputs["subsurface_mask.pt"], "subsurface_mask").bool().to(device)
        well_values = normalize_volume(inputs["binary_well_values.pt"], "binary_well_values", dtype).to(device)
        well_mask = normalize_volume(inputs["binary_well_mask.pt"], "binary_well_mask").bool().to(device)

        binary_config = read_json(args.binary_acoustic_config)
        source_record = binary_config["source_acoustic_config"]
        source_path = REPOSITORY_ROOT / str(source_record["path"])
        validate_asset(source_path, str(source_record["sha256"]))
        properties = binary_acoustic_properties_from_configs(binary_config, read_json(source_path))
        operator, _ = seismic_operator_from_config(read_json(args.seismic_config), grid_shape=subsurface.shape[2:])

        n_steps = int(config["number_of_optimization_steps"])
        lr = float(config["learning_rate"])
        lambda_tv = float(config["lambda_tv"])
        trace_interval = int(config["trace_interval"])
        smoothing_passes = int(config["initialization_smoothing"]["passes"])
        member_summaries: list[dict[str, object]] = []
        total_start = time.perf_counter()
        for member_index, seed in enumerate(seeds[: args.n_members]):
            member_dir = args.output_dir / f"member_{member_index:03d}"
            member_dir.mkdir(parents=True)
            member_manifest = {
                "schema": "stage15_binary_inversion_member_v1",
                "run_status": "running",
                "member_index": member_index,
                "seed": seed,
                "full_parameters": config,
                "input_tensor_sha256": {name: tensor_sha256(value) for name, value in inputs.items()},
            }
            write_json(member_dir / "manifest.json", member_manifest)
            member_start = time.perf_counter()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            logits = torch.randn(tuple(subsurface.shape), generator=generator, dtype=dtype)
            logits = float(config["initialization_mean"]) + float(config["initialization_scale"]) * logits
            logits = smooth_initial_logits(logits, smoothing_passes).to(device).requires_grad_(True)
            optimizer = torch.optim.Adam([logits], lr=lr)
            trace: list[dict[str, object]] = []
            with torch.no_grad():
                tau0 = inversion_tau(0, n_steps, float(config["tau_start"]), float(config["tau_end"]), str(config["tau_schedule"]))
                _, initial_hard, _ = straight_through_binary(logits, tau0, subsurface, well_values, well_mask)
                initial_impedance, initial_slowness = binary_occupancy_to_acoustic(initial_hard, subsurface, properties)
                initial_field = operator(initial_impedance, initial_slowness, subsurface)
                initial_rmse = _rmse(initial_field, observed, sample_mask)

            for step in range(n_steps):
                tau = inversion_tau(step, n_steps, float(config["tau_start"]), float(config["tau_end"]), str(config["tau_schedule"]))
                optimizer.zero_grad(set_to_none=True)
                probability, hard, ste = straight_through_binary(logits, tau, subsurface, well_values, well_mask)
                projected_probability = torch.where(well_mask, well_values, probability)
                projected_probability = torch.where(subsurface, projected_probability, torch.zeros_like(projected_probability))
                impedance, slowness = binary_occupancy_to_acoustic(ste, subsurface, properties)
                predicted = operator(impedance, slowness, subsurface)
                seismic_loss, diagnostics = seismic_field_loss(predicted, observed, sample_mask, uncertainty)
                tv_loss = masked_first_difference_tv(projected_probability, subsurface)
                total_loss = seismic_loss + lambda_tv * tv_loss
                if not torch.isfinite(total_loss):
                    raise FloatingPointError(f"non-finite loss at member {member_index} step {step}")
                total_loss.backward()
                gradient_norm = float(logits.grad.norm().detach().cpu())
                if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise FloatingPointError(f"invalid gradient at member {member_index} step {step}: {gradient_norm}")
                logits.grad.masked_fill_(well_mask | ~subsurface, 0)
                optimizer.step()
                if step % trace_interval == 0 or step == n_steps - 1:
                    condition_violations = int(((hard != well_values) & well_mask).sum().item())
                    trace.append(
                        {
                            "step": step,
                            "tau": tau,
                            "total_loss": float(total_loss.detach().cpu()),
                            "seismic_loss": float(seismic_loss.detach().cpu()),
                            "seismic_rmse_amplitude": float(diagnostics["seismic_rmse_amplitude"].detach().cpu()),
                            "tv_loss": float(tv_loss.detach().cpu()),
                            "gradient_norm": gradient_norm,
                            "target_voxels": int((hard.bool() & subsurface).sum().item()),
                            "condition_violations": condition_violations,
                        }
                    )

            with torch.no_grad():
                tau_final = float(config["tau_end"])
                probability, hard, ste = straight_through_binary(logits, tau_final, subsurface, well_values, well_mask)
                projected_probability = torch.where(well_mask, well_values, probability)
                projected_probability = torch.where(subsurface, projected_probability, torch.zeros_like(projected_probability))
                hard_impedance, hard_slowness = binary_occupancy_to_acoustic(hard, subsurface, properties)
                ste_impedance, ste_slowness = binary_occupancy_to_acoustic(ste, subsurface, properties)
                hard_field = operator(hard_impedance, hard_slowness, subsurface)
                ste_field = operator(ste_impedance, ste_slowness, subsurface)
                ste_hard_max_abs = float((hard_field - ste_field).abs().max().cpu())
                if ste_hard_max_abs > 1e-7:
                    raise RuntimeError(f"hard/STE forward mismatch: {ste_hard_max_abs}")
                final_loss, final_diag = seismic_field_loss(hard_field, observed, sample_mask, uncertainty)
                final_tv = masked_first_difference_tv(projected_probability, subsurface)
                violations = int(((hard != well_values) & well_mask).sum().item())
                if violations:
                    raise RuntimeError(f"binary well condition violation: {violations}")
            outputs = {
                "hard_binary.pt": hard.cpu(),
                "soft_probability.pt": projected_probability.cpu(),
                "final_logits.pt": logits.detach().cpu(),
                "predicted_seismic_hard.pt": hard_field.cpu(),
            }
            for filename, tensor in outputs.items():
                torch.save(tensor, member_dir / filename)
            runtime_seconds = time.perf_counter() - member_start
            rock_count = int(subsurface.sum().item())
            target_count = int((hard.bool() & subsurface).sum().item())
            metrics = {
                "initial_seismic_rmse": initial_rmse,
                "final_hard_seismic_rmse": float(final_diag["seismic_rmse_amplitude"].cpu()),
                "final_seismic_loss": float(final_loss.cpu()),
                "final_tv_loss": float(final_tv.cpu()),
                "condition_violations": violations,
                "target_voxel_count": target_count,
                "background_voxel_count": rock_count - target_count,
                "target_voxel_fraction": target_count / rock_count,
                "background_voxel_fraction": 1.0 - target_count / rock_count,
                "unique_model_hash": tensor_sha256(hard),
                "seed": seed,
                "runtime_seconds": runtime_seconds,
                "hard_ste_forward_max_abs": ste_hard_max_abs,
                "completion_status": "completed",
            }
            write_json(member_dir / "metrics.json", metrics)
            write_csv(member_dir / "trace.csv", trace)
            member_manifest.update(
                {
                    "run_status": "completed",
                    "output_tensor_sha256": {name: tensor_sha256(value) for name, value in outputs.items()},
                    "metrics": metrics,
                    "metrics_asset": runtime.asset_record(member_dir / "metrics.json"),
                    "trace_asset": runtime.asset_record(member_dir / "trace.csv"),
                    "truth_loaded_by_runner": False,
                }
            )
            write_json(member_dir / "manifest.json", member_manifest)
            member_summaries.append(metrics)

        manifest.update(
            {
                "run_status": "completed",
                "n_members": args.n_members,
                "member_seeds": seeds[: args.n_members],
                "runtime_seconds": time.perf_counter() - total_start,
                "observation_manifest": runtime.asset_record(args.observation_dir / "manifest.json"),
                "input_assets": {name: runtime.asset_record(args.observation_dir / name) for name in inputs},
                "binary_acoustic_config": runtime.asset_record(args.binary_acoustic_config),
                "source_acoustic_config": runtime.asset_record(source_path),
                "seismic_config": runtime.asset_record(args.seismic_config),
                "binary_inversion_source": runtime.asset_record(PROJECT_DIR / "guidance/binary_seismic_inversion.py"),
                "members": member_summaries,
                "truth_loaded_by_runner": False,
                "ensemble_interpretation": "optimization_endpoint_binary_inversion_ensemble_realizations_not_calibrated_posterior_samples",
            }
        )
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
