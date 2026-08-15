#!/usr/bin/env python3
"""Truth-blind Stage15-E 4^3 coarse binary seismic inversion."""

from __future__ import annotations

import argparse
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
)
from guidance.coarse_binary_seismic import upsample_coarse_occupancy
from guidance.seismic import seismic_operator_from_config, tensor_sha256
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
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/coarse_binary_seismic_inversion_4x4x4_v1.json"
DEFAULT_OBSERVATION_DIR = EXPERIMENT_ROOT / "observations/cond_generation_0"
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = (
    PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"
)
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "coarse_inversion/coarse_4x4x4_n8_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION_DIR)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def validate_config(config: dict[str, object]) -> None:
    expected = {
        "schema": "stage15_coarse_binary_seismic_inversion_v1",
        "status": "frozen_before_truth_evaluation",
        "grid_shape": [64, 64, 64],
        "coarse_grid_shape": [4, 4, 4],
        "fine_voxels_per_coarse_cell": [16, 16, 16],
        "optimizer": "Adam",
        "learning_rate": 0.1,
        "number_of_iterations": 300,
        "loss": "raw_mean_squared_seismic_misfit",
        "regularization": None,
        "parameter_sweep": False,
        "flow_used": False,
        "observation_regenerated": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"frozen Stage15-E config mismatch: {key}")
    seeds = config.get("inversion_seeds")
    if not isinstance(seeds, list) or len(seeds) != 8 or len(set(seeds)) != 8:
        raise ValueError("Stage15-E requires exactly eight unique inversion seeds")


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    validate_config(config)
    paths = {
        "observed_seismic": args.observation_dir / "observed_seismic.pt",
        "subsurface_mask": args.observation_dir / "subsurface_mask.pt",
        "binary_acoustic_config": args.binary_acoustic_config,
        "seismic_config": args.seismic_config,
    }
    expected_files = {
        "observed_seismic": config["observed_seismic_file_sha256"],
        "subsurface_mask": config["subsurface_mask_file_sha256"],
        "binary_acoustic_config": config["binary_acoustic_config_sha256"],
        "seismic_config": config["seismic_config_sha256"],
    }
    for name, path in paths.items():
        if runtime.file_sha256(path) != expected_files[name]:
            raise ValueError(f"frozen Stage15-E input file changed: {name}")

    device = torch.device(args.device or str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    dtype = torch.float32
    observed_cpu = runtime.load_tensor(paths["observed_seismic"]).float()
    subsurface_cpu = normalize_volume(
        runtime.load_tensor(paths["subsurface_mask"]), "subsurface_mask"
    ).bool()
    if tensor_sha256(observed_cpu) != config["observed_seismic_tensor_sha256"]:
        raise ValueError("frozen observed seismic tensor hash changed")
    if tensor_sha256(subsurface_cpu) != config["subsurface_mask_tensor_sha256"]:
        raise ValueError("frozen subsurface tensor hash changed")
    observed = observed_cpu.to(device=device, dtype=dtype)
    subsurface = subsurface_cpu.to(device)

    binary_config = read_json(args.binary_acoustic_config)
    source_record = binary_config["source_acoustic_config"]
    source_path = REPOSITORY_ROOT / str(source_record["path"])
    validate_asset(source_path, str(source_record["sha256"]))
    properties = binary_acoustic_properties_from_configs(
        binary_config, read_json(source_path)
    )
    operator, seismic_parameters = seismic_operator_from_config(
        read_json(args.seismic_config), grid_shape=(64, 64, 64)
    )

    manifest = base_manifest(
        "stage15_e_coarse_binary_seismic_inversion_run_v1", Path(__file__), args.config
    )
    args.output_dir.mkdir(parents=True)
    write_json(args.output_dir / "manifest.json", manifest)
    all_q: list[torch.Tensor] = []
    summaries: list[dict[str, object]] = []
    started = time.perf_counter()
    try:
        for run_index, seed_value in enumerate(config["inversion_seeds"]):
            seed = int(seed_value)
            run_dir = args.output_dir / f"run_{run_index:02d}"
            run_dir.mkdir()
            generator = torch.Generator(device="cpu").manual_seed(seed)
            u = torch.randn((1, 1, 4, 4, 4), generator=generator, dtype=dtype)
            u = u.to(device).requires_grad_(True)
            optimizer = torch.optim.Adam([u], lr=float(config["learning_rate"]))
            trace: list[dict[str, object]] = []
            run_started = time.perf_counter()
            with torch.no_grad():
                q_initial = torch.sigmoid(u)
                initial_fine = upsample_coarse_occupancy(q_initial)
                initial_impedance, initial_slowness = binary_occupancy_to_acoustic(
                    initial_fine, subsurface, properties
                )
                initial_prediction = operator(
                    initial_impedance, initial_slowness, subsurface
                )
                initial_loss = float((initial_prediction - observed).square().mean().cpu())

            for iteration in range(int(config["number_of_iterations"])):
                optimizer.zero_grad(set_to_none=True)
                q_coarse = torch.sigmoid(u)
                q_fine = upsample_coarse_occupancy(q_coarse)
                impedance, slowness = binary_occupancy_to_acoustic(
                    q_fine, subsurface, properties
                )
                predicted = operator(impedance, slowness, subsurface)
                loss = (predicted - observed).square().mean()
                if not torch.isfinite(loss):
                    raise FloatingPointError(
                        f"non-finite loss at run {run_index}, iteration {iteration}"
                    )
                loss.backward()
                gradient_norm = float(u.grad.norm().detach().cpu())
                if not math.isfinite(gradient_norm) or gradient_norm <= 0:
                    raise FloatingPointError(
                        f"invalid gradient at run {run_index}, iteration {iteration}"
                    )
                optimizer.step()
                if iteration % int(config["trace_interval"]) == 0 or iteration == int(config["number_of_iterations"]) - 1:
                    trace.append(
                        {
                            "iteration": iteration,
                            "seismic_mse": float(loss.detach().cpu()),
                            "seismic_rmse": float(torch.sqrt(loss.detach()).cpu()),
                            "gradient_norm": gradient_norm,
                            "q_min": float(q_coarse.min().detach().cpu()),
                            "q_mean": float(q_coarse.mean().detach().cpu()),
                            "q_max": float(q_coarse.max().detach().cpu()),
                        }
                    )

            with torch.no_grad():
                q_final = torch.sigmoid(u).cpu().contiguous()
                q_fine_final = upsample_coarse_occupancy(q_final.to(device))
                final_impedance, final_slowness = binary_occupancy_to_acoustic(
                    q_fine_final, subsurface, properties
                )
                final_prediction = operator(
                    final_impedance, final_slowness, subsurface
                )
                final_loss = float((final_prediction - observed).square().mean().cpu())
            torch.save(q_final, run_dir / "coarse_occupancy.pt")
            write_csv(run_dir / "trace.csv", trace)
            metrics = {
                "run_index": run_index,
                "seed": seed,
                "initial_seismic_mse": initial_loss,
                "final_seismic_mse": final_loss,
                "initial_seismic_rmse": math.sqrt(initial_loss),
                "final_seismic_rmse": math.sqrt(final_loss),
                "loss_reduction_fraction": (initial_loss - final_loss) / initial_loss,
                "final_q_min": float(q_final.min()),
                "final_q_mean": float(q_final.mean()),
                "final_q_max": float(q_final.max()),
                "runtime_seconds": time.perf_counter() - run_started,
                "coarse_occupancy_tensor_sha256": tensor_sha256(q_final),
            }
            write_json(run_dir / "metrics.json", metrics)
            summaries.append(metrics)
            all_q.append(q_final)
            print(
                f"coarse inversion {run_index + 1}/8: final MSE={final_loss:.8g}",
                flush=True,
            )

        mean_q = torch.stack(all_q).mean(dim=0)
        torch.save(mean_q, args.output_dir / "mean_coarse_occupancy.pt")
        write_csv(args.output_dir / "run_summary.csv", summaries)
        input_hashes = {
            str(path.resolve()): runtime.file_sha256(path) for path in paths.values()
        }
        manifest.update(
            {
                "run_status": "completed",
                "run_count": len(summaries),
                "run_summaries": summaries,
                "input_file_sha256_before_and_after": input_hashes,
                "inputs_unchanged": True,
                "binary_acoustic_source_config": runtime.asset_record(source_path),
                "seismic_parameters": seismic_parameters,
                "mean_coarse_occupancy": runtime.asset_record(
                    args.output_dir / "mean_coarse_occupancy.pt"
                ),
                "mean_coarse_occupancy_tensor_sha256": tensor_sha256(mean_q),
                "runtime_seconds": time.perf_counter() - started,
                "truth_loaded_by_runner": False,
                "observation_manifest_opened_by_runner": False,
                "flow_used": False,
                "checkpoint_loaded": False,
                "regularization_used": False,
                "parameter_sweep_performed": False,
                "observation_regenerated": False,
            }
        )
        for path, expected in input_hashes.items():
            if runtime.file_sha256(Path(path)) != expected:
                raise RuntimeError(f"Stage15-E input changed during inversion: {path}")
        write_json(args.output_dir / "manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.output_dir / "manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
