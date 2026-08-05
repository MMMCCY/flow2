#!/usr/bin/env python3
"""Run the frozen Phase-5c pCN search in conditional-flow noise space."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import sys
import time
from typing import Dict, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance import generator_posterior as posterior
from guidance import seismic as seismic_module
from guidance.seismic import tensor_sha256
from scripts.stage4 import run_seismic_guidance as phase4_runner
from scripts.stage4.run_seismic_guidance import (
    add_hard_seismic_metrics,
    load_observation_assets,
    read_json,
    write_json,
    write_rows,
)


PHASE5C_RUN_SCHEMA = "phase5c_generator_posterior_run_v1"
PHASE5C_CONFIG_SCHEMA = "phase5c_generator_posterior_config_v1"
PHASE5C_STAGE = "phase5c_direct_generator_posterior_v1"


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage5_generator_posterior"
    case_dir = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0"
    parser = argparse.ArgumentParser(
        description="Run the frozen no-training Phase-5c hard-likelihood pCN screen.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--protocol-config",
        type=Path,
        default=experiment / "configs/cond0_hard_pcn_mechanism_v1.json",
    )
    parser.add_argument(
        "--mode", choices=("performance_smoke", "primary_pilot"), required=True
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_DIR / "demo_model/conditional-weights.ckpt",
    )
    parser.add_argument("--samples-dir", type=Path, default=case_dir)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/observations/cond_generation_0"
        / "distinct_upper_bound_v1_fix2",
    )
    parser.add_argument(
        "--baseline-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/baseline",
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_protocol_config(
    config: Mapping[str, object], mode: str
) -> Dict[str, object]:
    if config.get("schema") != PHASE5C_CONFIG_SCHEMA:
        raise ValueError("invalid Phase-5c protocol config schema")
    if config.get("status") != "frozen_before_cuda_output":
        raise ValueError("Phase-5c protocol must be frozen before execution")
    if config.get("sampler") != posterior.GENERATOR_POSTERIOR_VERSION:
        raise ValueError("protocol sampler version does not match implementation")
    if config.get("condition_policy") != posterior.CONDITION_PROJECTION_POLICY:
        raise ValueError("protocol condition policy does not match implementation")
    if config.get("truth_used_by_sampler") is not False:
        raise ValueError("Phase-5c sampler may not use truth metrics")
    if config.get("allow_parameter_sweep_on_case") is not False:
        raise ValueError("legacy mechanism case must prohibit parameter sweeps")

    initial_seed = int(config["initial_seed"])
    proposal_seed = int(config["proposal_seed"])
    n_steps = int(config["n_euler_steps"])
    beta = float(config["pcn_beta"])
    likelihood_weight = float(config["likelihood_weight"])
    proposal_field = (
        "performance_smoke_proposals"
        if mode == "performance_smoke"
        else "primary_pilot_proposals"
    )
    chain_proposals = int(config[proposal_field])
    if n_steps <= 0 or chain_proposals <= 0:
        raise ValueError("Euler steps and chain proposals must be positive")
    if not math.isfinite(beta) or not 0.0 < beta <= 1.0:
        raise ValueError("frozen pCN beta must lie in (0,1]")
    if not math.isfinite(likelihood_weight) or likelihood_weight <= 0:
        raise ValueError("likelihood weight must be finite and positive")
    return {
        "initial_seed": initial_seed,
        "proposal_seed": proposal_seed,
        "n_steps": n_steps,
        "beta": beta,
        "likelihood_weight": likelihood_weight,
        "chain_proposals": chain_proposals,
    }


def _condition_violations(
    decoded: torch.Tensor,
    truth: torch.Tensor,
    condition_mask: torch.Tensor,
) -> int:
    prediction = runtime.normalize_single_geology(decoded, "decoded").long()
    target = runtime.normalize_single_geology(truth, "truth").long()
    mask = condition_mask.bool()
    return int(((prediction != target) & mask).sum().item())


def _changed_voxels(left: torch.Tensor, right: torch.Tensor) -> tuple[int, float]:
    if left.shape != right.shape:
        raise ValueError("paired hard samples must have matching shapes")
    changed = left != right
    return int(changed.sum().item()), float(changed.float().mean().item())


def _evaluate_latent(
    *,
    model,
    latent_cpu: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_truth: torch.Tensor,
    truth_cpu: torch.Tensor,
    condition_mask_cpu: torch.Tensor,
    property_table: torch.Tensor,
    target_acoustic: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, Dict[str, object]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_state = posterior.projected_fixed_euler_prior_sample(
        model,
        latent_cpu.to(device),
        conditioning,
        embedded_truth,
        condition_mask_cpu,
        n_steps=n_steps,
    )
    if not torch.isfinite(final_state).all():
        raise FloatingPointError("conditional-flow proposal contains NaN or Inf")
    decoded = (model.decode(final_state).detach().cpu() - 1).long()
    row: Dict[str, object] = {}
    field = add_hard_seismic_metrics(
        row,
        prediction=decoded,
        target_acoustic=target_acoustic,
        condition_mask=condition_mask_cpu,
        property_table=property_table,
        subsurface_mask=subsurface_mask,
        forward_operator=forward_operator,
        observed=observed,
        sample_mask=sample_mask,
        uncertainty=uncertainty,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    row.update(
        {
            "condition_violations": _condition_violations(
                decoded, truth_cpu, condition_mask_cpu
            ),
            "sample_sha256": tensor_sha256(decoded),
            "evaluation_seconds": time.perf_counter() - started,
        }
    )
    del final_state
    return decoded, field, row


def _validate_historical_baseline(
    *,
    baseline_dir: Path,
    baseline_config: Mapping[str, object],
    initial_decoded: torch.Tensor,
    initial_noise_sha256: str,
    ckpt_path: Path,
    truth_path: Path,
    boreholes_path: Path,
    observation_dir: Path,
    initial_seed: int,
    n_steps: int,
) -> dict[str, object]:
    expected = {
        "run_status": "completed",
        "model_weight_source": "ema",
        "ema_applied": True,
        "integrator": runtime.PAIRED_INTEGRATOR,
        "seed": int(initial_seed),
        "n_steps": int(n_steps),
        "alpha": 0.0,
    }
    for field, value in expected.items():
        if baseline_config.get(field) != value:
            raise ValueError(f"historical baseline {field} must be {value!r}")
    hash_checks = {
        "checkpoint_sha256": runtime.file_sha256(ckpt_path),
        "truth_model_sha256": runtime.file_sha256(truth_path),
        "boreholes_sha256": runtime.file_sha256(boreholes_path),
        "observation_manifest_sha256": runtime.file_sha256(
            observation_dir / "manifest.json"
        ),
    }
    for field, value in hash_checks.items():
        if baseline_config.get(field) != value:
            raise ValueError(f"historical baseline asset differs: {field}")
    recorded_noise = baseline_config.get("initial_noise_sha256")
    if not isinstance(recorded_noise, list) or not recorded_noise:
        raise ValueError("historical baseline lacks initial-noise hash")
    if recorded_noise[0] != initial_noise_sha256:
        raise ValueError("initial noise does not match historical paired baseline")
    baseline_sample_path = baseline_dir / "sample_0.pt"
    baseline_sample = runtime.load_tensor(baseline_sample_path, map_location="cpu")
    initial_normalized = runtime.normalize_single_geology(
        initial_decoded, "initial_decoded"
    ).long()
    baseline_normalized = runtime.normalize_single_geology(
        baseline_sample, "historical_baseline_sample"
    ).long()
    if not torch.equal(initial_normalized, baseline_normalized):
        changed, _ = _changed_voxels(initial_normalized, baseline_normalized)
        raise ValueError(
            f"initial pCN state differs from historical alpha-zero sample at {changed} voxels"
        )
    return {
        "exact_initial_hard_regression": True,
        "initial_changed_voxels": 0,
        "baseline_sample": runtime.asset_record(baseline_sample_path),
        "baseline_config": runtime.asset_record(baseline_dir / "config.json"),
    }


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("the frozen Phase-5c mechanism run requires --device cuda")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable in this environment")

    protocol_config = read_json(args.protocol_config)
    resolved = validate_protocol_config(protocol_config, args.mode)
    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_load_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.ckpt_path,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device)
    conditioning_report = runtime.validate_conditioning_pair(
        truth_cpu, boreholes_cpu, model.num_categories, target_label=9
    )
    tensors, observation_manifest, forward_operator, resolved_observation = (
        load_observation_assets(
            args.observation_dir,
            truth_cpu,
            truth_path=truth_path,
            num_categories=model.num_categories,
        )
    )
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    truth_device = truth_cpu.to(device)
    embedded_truth = model.embed(truth_device)
    conditioning = embedded_truth * condition_mask_cpu.to(device).expand_as(
        embedded_truth
    )

    source_assets = runtime.experiment_asset_records(
        protocol_config=args.protocol_config,
        protocol_spec=PROJECT_DIR / "docs/PHASE5C_SPEC.md",
        truth_model=truth_path,
        boreholes=boreholes_path,
        observation_manifest=args.observation_dir / "manifest.json",
        historical_baseline_config=args.baseline_dir / "config.json",
        runner_source=Path(__file__),
        posterior_source=Path(posterior.__file__),
        runtime_source=Path(runtime.__file__),
        seismic_source=Path(seismic_module.__file__),
        phase4_loader_source=Path(phase4_runner.__file__),
    )
    source_assets["checkpoint"] = model_load_report["checkpoint"]
    run_config: Dict[str, object] = {
        "schema": PHASE5C_RUN_SCHEMA,
        "stage": PHASE5C_STAGE,
        "mode": args.mode,
        "run_status": "running",
        "protocol_config": protocol_config,
        "resolved_protocol": resolved,
        "generator_posterior_version": posterior.GENERATOR_POSTERIOR_VERSION,
        "integrator": runtime.PAIRED_INTEGRATOR,
        "condition_projection": posterior.CONDITION_PROJECTION_POLICY,
        "likelihood": protocol_config["likelihood"],
        "truth_metrics_computed_by_sampler": False,
        "truth_used_for": "condition_validation_and_immutable_observation_validation_only",
        "model_weight_source": "ema",
        "ema_applied": bool(model_load_report["ema_applied"]),
        "device": str(device),
        "torch_version": torch.__version__,
        "cuda_device_name": torch.cuda.get_device_name(device),
        "output_dir": str(args.output_dir),
        "asset_records": source_assets,
        "model_load_report": model_load_report,
        "conditioning_report": conditioning_report,
        "observation_config_resolved": resolved_observation,
    }
    run_config.update(runtime.flatten_asset_hashes(source_assets))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "config.json", run_config)
    write_json(args.output_dir / "model_load_report.json", model_load_report)
    write_json(args.output_dir / "input_validation.json", conditioning_report)
    write_json(args.output_dir / "observation_manifest.json", observation_manifest)

    initial_generator = torch.Generator(device="cpu").manual_seed(
        int(resolved["initial_seed"])
    )
    initial_latent = torch.randn(
        1,
        model.embedding_dim,
        *model.data_shape,
        generator=initial_generator,
        dtype=embedded_truth.dtype,
    ).contiguous()
    initial_noise_hash = tensor_sha256(initial_latent)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    initial_decoded, initial_field, initial_eval = _evaluate_latent(
        model=model,
        latent_cpu=initial_latent,
        conditioning=conditioning,
        embedded_truth=embedded_truth,
        truth_cpu=truth_cpu,
        condition_mask_cpu=condition_mask_cpu,
        property_table=tensors["acoustic_property_table.pt"],
        target_acoustic=tensors["truth_acoustic.pt"],
        subsurface_mask=tensors["subsurface_mask.pt"],
        forward_operator=forward_operator,
        observed=tensors["observed_seismic.pt"],
        sample_mask=tensors["sample_mask.pt"],
        uncertainty=tensors["uncertainty_amplitude.pt"],
        n_steps=int(resolved["n_steps"]),
        device=device,
    )
    baseline_config = read_json(args.baseline_dir / "config.json")
    baseline_validation = _validate_historical_baseline(
        baseline_dir=args.baseline_dir,
        baseline_config=baseline_config,
        initial_decoded=initial_decoded,
        initial_noise_sha256=initial_noise_hash,
        ckpt_path=args.ckpt_path,
        truth_path=truth_path,
        boreholes_path=boreholes_path,
        observation_dir=args.observation_dir,
        initial_seed=int(resolved["initial_seed"]),
        n_steps=int(resolved["n_steps"]),
    )
    if int(initial_eval["condition_violations"]):
        raise RuntimeError("initial state violates hard conditions")

    current_latent = initial_latent
    current_decoded = initial_decoded
    current_field = initial_field
    current_loss = float(initial_eval["hard_seismic_loss"])
    current_energy = posterior.posterior_energy(
        current_loss, float(resolved["likelihood_weight"])
    )
    retained_samples = [current_decoded.to(torch.int8)]
    trace_rows: list[Dict[str, object]] = []
    accepted_count = 0
    proposal_generator = torch.Generator(device="cpu").manual_seed(
        int(resolved["proposal_seed"])
    )

    for iteration in range(1, int(resolved["chain_proposals"]) + 1):
        proposal_latent, innovation = posterior.pcn_proposal(
            current_latent,
            beta=float(resolved["beta"]),
            generator=proposal_generator,
        )
        proposed_decoded, proposed_field, proposed_eval = _evaluate_latent(
            model=model,
            latent_cpu=proposal_latent,
            conditioning=conditioning,
            embedded_truth=embedded_truth,
            truth_cpu=truth_cpu,
            condition_mask_cpu=condition_mask_cpu,
            property_table=tensors["acoustic_property_table.pt"],
            target_acoustic=tensors["truth_acoustic.pt"],
            subsurface_mask=tensors["subsurface_mask.pt"],
            forward_operator=forward_operator,
            observed=tensors["observed_seismic.pt"],
            sample_mask=tensors["sample_mask.pt"],
            uncertainty=tensors["uncertainty_amplitude.pt"],
            n_steps=int(resolved["n_steps"]),
            device=device,
        )
        proposed_loss = float(proposed_eval["hard_seismic_loss"])
        proposed_energy = posterior.posterior_energy(
            proposed_loss, float(resolved["likelihood_weight"])
        )
        uniform = float(
            torch.rand((), generator=proposal_generator, dtype=torch.float64).item()
        )
        decision = posterior.metropolis_decision(
            current_energy, proposed_energy, uniform
        )
        changed_from_current, changed_fraction = _changed_voxels(
            proposed_decoded, current_decoded
        )
        previous_loss = current_loss
        if bool(decision["accepted"]):
            current_latent = proposal_latent
            current_decoded = proposed_decoded
            current_field = proposed_field
            current_loss = proposed_loss
            current_energy = proposed_energy
            accepted_count += 1
        retained_samples.append(current_decoded.to(torch.int8))
        changed_from_initial, changed_initial_fraction = _changed_voxels(
            current_decoded, initial_decoded
        )
        trace_rows.append(
            {
                "iteration": iteration,
                "beta": float(resolved["beta"]),
                "likelihood_weight": float(resolved["likelihood_weight"]),
                "accepted": bool(decision["accepted"]),
                "acceptance_probability": decision["acceptance_probability"],
                "log_acceptance_ratio": decision["log_acceptance_ratio"],
                "log_uniform": decision["log_uniform"],
                "previous_hard_seismic_loss": previous_loss,
                "proposed_hard_seismic_loss": proposed_loss,
                "current_hard_seismic_loss": current_loss,
                "proposed_hard_seismic_rmse_amplitude": proposed_eval[
                    "hard_seismic_rmse_amplitude"
                ],
                "proposed_hard_seismic_mae_amplitude": proposed_eval[
                    "hard_seismic_mae_amplitude"
                ],
                "proposed_condition_violations": proposed_eval[
                    "condition_violations"
                ],
                "proposed_sample_sha256": proposed_eval["sample_sha256"],
                "current_sample_sha256": tensor_sha256(current_decoded),
                "proposal_latent_sha256": tensor_sha256(proposal_latent),
                "innovation_sha256": tensor_sha256(innovation),
                "proposal_changed_from_current_voxels": changed_from_current,
                "proposal_changed_from_current_fraction": changed_fraction,
                "retained_changed_from_initial_voxels": changed_from_initial,
                "retained_changed_from_initial_fraction": changed_initial_fraction,
                "proposal_evaluation_seconds": proposed_eval["evaluation_seconds"],
                **{
                    f"proposal_{name}": value
                    for name, value in posterior.latent_diagnostics(
                        proposal_latent
                    ).items()
                },
            }
        )

    retained_tensor = torch.stack(retained_samples).contiguous()
    torch.save(retained_tensor, args.output_dir / "retained_samples.pt")
    torch.save(current_latent.contiguous(), args.output_dir / "final_latent.pt")
    torch.save(current_field, args.output_dir / "final_hard_seismic.pt")
    torch.save(initial_field, args.output_dir / "initial_hard_seismic.pt")
    write_rows(args.output_dir / "chain_trace.csv", trace_rows)
    write_json(args.output_dir / "historical_baseline_validation.json", baseline_validation)

    unique_sample_hashes = sorted(
        {tensor_sha256(retained_tensor[index]) for index in range(retained_tensor.shape[0])}
    )
    max_violations = max(
        [int(initial_eval["condition_violations"])]
        + [int(row["proposed_condition_violations"]) for row in trace_rows]
    )
    run_config.update(
        {
            "run_status": "completed",
            "initial_noise_sha256": initial_noise_hash,
            "initial_sample_sha256": tensor_sha256(initial_decoded),
            "initial_hard_seismic_loss": current_loss
            if not trace_rows
            else float(initial_eval["hard_seismic_loss"]),
            "initial_hard_seismic_rmse_amplitude": initial_eval[
                "hard_seismic_rmse_amplitude"
            ],
            "initial_evaluation_seconds": initial_eval["evaluation_seconds"],
            "accepted_proposals": accepted_count,
            "acceptance_fraction": accepted_count / int(resolved["chain_proposals"]),
            "unique_retained_hard_samples": len(unique_sample_hashes),
            "unique_retained_sample_sha256": unique_sample_hashes,
            "final_hard_seismic_loss": current_loss,
            "final_sample_sha256": tensor_sha256(current_decoded),
            "final_latent_sha256": tensor_sha256(current_latent),
            "retained_samples_sha256": tensor_sha256(retained_tensor),
            "max_condition_violations": max_violations,
            "historical_baseline_validation": baseline_validation,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        }
    )
    write_json(args.output_dir / "config.json", run_config)
    print(
        "Phase-5c chain complete: "
        f"mode={args.mode}, accepted={accepted_count}/{resolved['chain_proposals']}, "
        f"unique={len(unique_sample_hashes)}, final_loss={current_loss:.8g}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()
