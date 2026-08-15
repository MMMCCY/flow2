#!/usr/bin/env python3
"""Truth-blind frozen-Flow pCN ensemble with a hard binary-label9 likelihood."""

from __future__ import annotations

import argparse
import math
import statistics
import sys
import time
from pathlib import Path
from typing import Mapping

import numpy as np
from scipy import ndimage
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance import generator_posterior as posterior
from guidance.binary_seismic_inversion import (
    binary_acoustic_properties_from_configs,
    binary_occupancy_to_acoustic,
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
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/binary_flow_pcn_pilot_v1.json"
DEFAULT_BINARY_CONFIG = EXPERIMENT_ROOT / "configs/binary_acoustic_upper_bound_v1.json"
DEFAULT_SEISMIC_CONFIG = PROJECT_DIR / "experiments/stage4_seismic/configs/full_cube_noiseless_inverse_crime_v1.json"


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    required = {
        "schema": "stage15_binary_flow_pcn_config_v1",
        "status": "frozen_before_cuda_execution",
        "sampler": posterior.GENERATOR_POSTERIOR_VERSION,
        "parameterization": "frozen_conditional_flow_initial_gaussian_noise_only",
        "model_weights": "ema",
        "n_euler_steps": 32,
        "pcn_beta": 0.1,
        "likelihood_weight": 1.0,
        "likelihood": "binary_label9_hard_seismic_likelihood",
        "n_chains": 4,
        "proposals_per_chain": 32,
        "burn_in": 8,
        "thinning": 1,
        "positive_threshold": 0.8,
        "negative_threshold": 0.2,
        "condition_policy": posterior.CONDITION_PROJECTION_POLICY,
        "state_accounting": "post_burnin_complete_current_chain_states_including_rejection_duplicates_v1",
        "parameter_sweep": False,
        "training": False,
        "truth_used_by_runner": False,
    }
    for field, expected in required.items():
        if config.get(field) != expected:
            raise ValueError(f"frozen B2 config mismatch for {field}")
    initial_seeds = [int(value) for value in config.get("initial_seeds", [])]
    proposal_seeds = [int(value) for value in config.get("proposal_seeds", [])]
    if len(initial_seeds) != 4 or len(set(initial_seeds)) != 4:
        raise ValueError("B2 requires four distinct initial seeds")
    if len(proposal_seeds) != 4 or len(set(proposal_seeds)) != 4:
        raise ValueError("B2 requires four distinct proposal seeds")
    return {
        "n_chains": 4,
        "proposals_per_chain": 32,
        "burn_in": 8,
        "thinning": 1,
        "expected_retained_states": 96,
        "initial_seeds": initial_seeds,
        "proposal_seeds": proposal_seeds,
    }


def categorical_to_binary_label9(
    decoded: torch.Tensor,
    subsurface_mask: torch.Tensor,
) -> torch.Tensor:
    """Collapse label 9 only inside support; outside support remains separate air."""
    categorical = _as_single_volume(decoded, "decoded").long()
    subsurface = _as_single_volume(subsurface_mask, "subsurface_mask").bool()
    if categorical.shape != subsurface.shape:
        raise ValueError("decoded and subsurface_mask must match")
    return ((categorical == 9) & subsurface).to(torch.float32)


def condition_violation_count(
    decoded: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
) -> int:
    prediction = _as_single_volume(decoded, "decoded").long()
    values = _as_single_volume(condition_values, "condition_values").long()
    mask = _as_single_volume(condition_mask, "condition_mask").bool()
    if not (prediction.shape == values.shape == mask.shape):
        raise ValueError("decoded, condition_values, and condition_mask must match")
    return int(((prediction != values) & mask).sum().item())


def _as_single_volume(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3:
        value = value.unsqueeze(0).unsqueeze(0)
    elif value.ndim == 4 and value.shape[0] == 1:
        value = value.unsqueeze(1)
    elif value.ndim != 5 or value.shape[:2] != (1, 1):
        raise ValueError(f"{name} must contain one single-channel 3-D volume")
    return value


def should_retain_iteration(iteration: int, burn_in: int, thinning: int) -> bool:
    if iteration <= 0 or burn_in < 0 or thinning <= 0:
        raise ValueError("invalid MCMC accounting arguments")
    return iteration > burn_in and (iteration - burn_in - 1) % thinning == 0


def record_current_state(
    categorical_states: list[torch.Tensor],
    binary_states: list[torch.Tensor],
    *,
    iteration: int,
    burn_in: int,
    thinning: int,
    current_categorical: torch.Tensor,
    current_binary: torch.Tensor,
) -> bool:
    """Record the complete current state, irrespective of proposal acceptance."""
    if not should_retain_iteration(iteration, burn_in, thinning):
        return False
    categorical_states.append(current_categorical.detach().cpu().to(torch.int8).clone())
    binary_states.append(current_binary.detach().cpu().to(torch.uint8).clone())
    return True


def _decode(model, final_state: torch.Tensor) -> torch.Tensor:
    decoded = (model.decode(final_state).detach().cpu() - 1).long()
    return _as_single_volume(decoded, "decoded")


def binary_label9_hard_seismic_likelihood(
    *,
    decoded: torch.Tensor,
    subsurface_mask: torch.Tensor,
    properties,
    forward_operator,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, dict[str, float]]:
    """Evaluate numerically hard label9-vs-background normalized seismic MSE."""
    binary_cpu = categorical_to_binary_label9(decoded, subsurface_mask)
    binary = binary_cpu.to(device=device, dtype=observed.dtype)
    support = subsurface_mask.to(device=device, dtype=torch.bool)
    impedance, slowness = binary_occupancy_to_acoustic(binary, support, properties)
    predicted = forward_operator(impedance, slowness, support)
    loss, diagnostics = seismic_field_loss(
        predicted, observed, sample_mask, uncertainty
    )
    return binary_cpu, predicted, {
        "hard_seismic_loss": float(loss.detach().cpu()),
        "hard_seismic_rmse_amplitude": float(
            diagnostics["seismic_rmse_amplitude"].detach().cpu()
        ),
        "hard_seismic_mae_amplitude": float(
            diagnostics["seismic_mae_amplitude"].detach().cpu()
        ),
    }


def _evaluate_latent(
    *,
    model,
    latent_cpu: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_values_cpu: torch.Tensor,
    condition_mask_cpu: torch.Tensor,
    subsurface_cpu: torch.Tensor,
    properties,
    forward_operator,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    n_steps: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    started = time.perf_counter()
    final_state = posterior.projected_fixed_euler_prior_sample(
        model,
        latent_cpu.to(device),
        conditioning,
        embedded_conditions,
        condition_mask_cpu,
        n_steps=n_steps,
    )
    if not torch.isfinite(final_state).all():
        raise FloatingPointError("frozen Flow evaluation contains NaN or Inf")
    decoded = _decode(model, final_state)
    violations = condition_violation_count(
        decoded, condition_values_cpu, condition_mask_cpu
    )
    if violations:
        raise RuntimeError(f"categorical condition projection failed: {violations}")
    binary, predicted, likelihood = binary_label9_hard_seismic_likelihood(
        decoded=decoded,
        subsurface_mask=subsurface_cpu,
        properties=properties,
        forward_operator=forward_operator,
        observed=observed,
        sample_mask=sample_mask,
        uncertainty=uncertainty,
        device=device,
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    target_voxels = int(binary.sum().item())
    support_voxels = int(subsurface_cpu.sum().item())
    diagnostics: dict[str, object] = {
        **likelihood,
        "condition_violations": violations,
        "sample_sha256": tensor_sha256(decoded),
        "binary_sha256": tensor_sha256(binary),
        "target_voxel_count": target_voxels,
        "target_fraction": target_voxels / support_voxels,
        "evaluation_seconds": time.perf_counter() - started,
    }
    del final_state
    return decoded, binary, predicted.detach().cpu(), diagnostics


def _positive_geometry(positive: torch.Tensor) -> dict[str, object]:
    array = positive[0, 0].detach().cpu().numpy().astype(bool)
    labels, count = ndimage.label(
        array, structure=ndimage.generate_binary_structure(3, 1)
    )
    sizes = np.bincount(labels.ravel())[1:]
    coordinates = np.argwhere(array)
    total = int(array.sum())
    largest = int(sizes.max()) if sizes.size else 0
    return {
        "six_connected_component_count": int(count),
        "components_at_least_20_voxels": int((sizes >= 20).sum()),
        "largest_component_size": largest,
        "largest_component_fraction": largest / total if total else 0.0,
        "bounding_box_min": coordinates.min(axis=0).tolist() if coordinates.size else None,
        "bounding_box_max": coordinates.max(axis=0).tolist() if coordinates.size else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--observation-dir", type=Path, default=DEFAULT_OBSERVATION)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--binary-acoustic-config", type=Path, default=DEFAULT_BINARY_CONFIG)
    parser.add_argument("--seismic-config", type=Path, default=DEFAULT_SEISMIC_CONFIG)
    parser.add_argument("--device", default=None)
    return parser.parse_args()


def _summary_markdown(summary: Mapping[str, object]) -> str:
    sampler = summary["sampler"]
    seismic = summary["seismic"]
    geology = summary["geology"]
    occupancy = summary["occupancy"]
    geometry = summary["positive_geometry"]
    return f"""# Stage15-B2 frozen-Flow binary-seismic pCN pilot

Status: truth-blind fixed 4×32 pilot complete. No retrospective evaluation or Flow guidance was run.

## Sampler

- Per-chain acceptance: {sampler['acceptance_rate_per_chain']}
- Overall acceptance: {sampler['overall_acceptance_rate']:.6f}
- Post-burn-in states: {sampler['retained_state_count']} (includes rejection duplicates)
- Unique categorical / binary states: {sampler['unique_hard_models']} / {sampler['unique_binary_models']}

## Seismic and geology

- Initial hard seismic losses: {seismic['initial_hard_seismic_loss_per_chain']}
- Minimum hard seismic loss: {seismic['minimum_post_burnin_hard_seismic_loss']:.8g}
- Median post-burn-in loss: {seismic['median_post_burnin_hard_seismic_loss']:.8g}
- Initial target fractions: {geology['initial_target_fraction_per_chain']}
- Post-burn-in target-fraction range: {geology['post_burnin_target_fraction_min']:.6f}–{geology['post_burnin_target_fraction_max']:.6f}
- Condition violations: {geology['maximum_condition_violations']}

## Occupancy consensus

- P9 min / max / mean: {occupancy['p9_min']:.6f} / {occupancy['p9_max']:.6f} / {occupancy['p9_mean_subsurface']:.6f}
- Positive / negative / unknown: {occupancy['positive_voxels']} / {occupancy['negative_voxels']} / {occupancy['unknown_voxels']}
- Confidence coverage: {occupancy['confidence_coverage']:.6f}
- Guidance ROI voxels: {occupancy['guidance_roi_voxels']}
- Positive components / >=20 / largest: {geometry['six_connected_component_count']} / {geometry['components_at_least_20_voxels']} / {geometry['largest_component_size']}

This is an ensemble occupancy frequency from frozen-Flow pCN chain states, not a calibrated Bayesian posterior probability. Phase5c used a 15-class petrophysical likelihood; Stage15-B2 uses a target-specific hard binary label9-vs-all-other-subsurface likelihood.
"""


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    resolved = validate_config(config)
    device = torch.device(args.device or str(config["device"]))
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen B2 pilot requires an available CUDA device")
    dtype = {"float32": torch.float32, "float64": torch.float64}.get(
        str(config["dtype"])
    )
    if dtype is None:
        raise ValueError("dtype must be float32 or float64")

    manifest = base_manifest(
        "stage15_binary_flow_pcn_run_v1", Path(__file__), args.config
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "run_manifest.json", manifest)
    total_started = time.perf_counter()
    try:
        observation_manifest = read_json(args.observation_dir / "manifest.json")
        if observation_manifest.get("run_status") != "completed":
            raise ValueError("Stage15 observation is not completed")
        names = (
            "observed_seismic.pt",
            "sample_mask.pt",
            "uncertainty.pt",
            "subsurface_mask.pt",
            "flow_condition_values.pt",
            "flow_condition_mask.pt",
        )
        inputs = {name: runtime.load_tensor(args.observation_dir / name) for name in names}
        expected_hashes = observation_manifest["output_tensor_sha256"]
        for name, tensor in inputs.items():
            if tensor_sha256(tensor) != expected_hashes[name]:
                raise ValueError(f"observation tensor hash changed: {name}")
        observed = inputs["observed_seismic.pt"].to(device=device, dtype=dtype)
        sample_mask = inputs["sample_mask.pt"].to(device=device, dtype=dtype)
        uncertainty = inputs["uncertainty.pt"].to(device=device, dtype=dtype)
        subsurface_cpu = normalize_volume(inputs["subsurface_mask.pt"], "subsurface_mask").bool()
        condition_values_cpu = normalize_volume(inputs["flow_condition_values.pt"], "flow_condition_values").long()
        condition_mask_cpu = normalize_volume(inputs["flow_condition_mask.pt"], "flow_condition_mask").bool()
        if bool((condition_mask_cpu & ~((condition_values_cpu >= -1) & (condition_values_cpu <= 13))).any()):
            raise ValueError("invalid categorical condition values")

        binary_config = read_json(args.binary_acoustic_config)
        source_record = binary_config["source_acoustic_config"]
        source_path = REPOSITORY_ROOT / str(source_record["path"])
        validate_asset(source_path, str(source_record["sha256"]))
        properties = binary_acoustic_properties_from_configs(
            binary_config, read_json(source_path)
        )
        forward_operator, resolved_seismic = seismic_operator_from_config(
            read_json(args.seismic_config), grid_shape=subsurface_cpu.shape[2:]
        )

        checkpoint_path = REPOSITORY_ROOT / str(config["checkpoint"]["path"])
        validate_asset(checkpoint_path, str(config["checkpoint"]["sha256"]))
        from model_train_sh_inference_cond import Geo3DStochInterp

        model, model_report = runtime.load_model_with_weight_policy(
            Geo3DStochInterp,
            checkpoint_path,
            map_location=device,
            weight_source="ema",
        )
        model = model.to(device)
        embedded_conditions = model.embed(condition_values_cpu.to(device))
        conditioning = embedded_conditions * condition_mask_cpu.to(device).expand_as(
            embedded_conditions
        )

        retained_categorical: list[torch.Tensor] = []
        retained_binary: list[torch.Tensor] = []
        trace_rows: list[dict[str, object]] = []
        chain_summaries: list[dict[str, object]] = []
        initial_binary_models: list[torch.Tensor] = []
        all_proposal_violations: list[int] = []

        for chain_id in range(int(resolved["n_chains"])):
            chain_dir = args.output_dir / f"chain_{chain_id:02d}"
            chain_dir.mkdir()
            initial_seed = int(resolved["initial_seeds"][chain_id])
            proposal_seed = int(resolved["proposal_seeds"][chain_id])
            initial_generator = torch.Generator(device="cpu").manual_seed(initial_seed)
            initial_latent = torch.randn(
                1,
                model.embedding_dim,
                *model.data_shape,
                generator=initial_generator,
                device="cpu",
                dtype=embedded_conditions.dtype,
            ).contiguous()
            current_decoded, current_binary, current_field, initial_eval = _evaluate_latent(
                model=model,
                latent_cpu=initial_latent,
                conditioning=conditioning,
                embedded_conditions=embedded_conditions,
                condition_values_cpu=condition_values_cpu,
                condition_mask_cpu=condition_mask_cpu,
                subsurface_cpu=subsurface_cpu,
                properties=properties,
                forward_operator=forward_operator,
                observed=observed,
                sample_mask=sample_mask,
                uncertainty=uncertainty,
                n_steps=int(config["n_euler_steps"]),
                device=device,
            )
            current_latent = initial_latent
            current_loss = float(initial_eval["hard_seismic_loss"])
            current_energy = posterior.posterior_energy(
                current_loss, float(config["likelihood_weight"])
            )
            initial_binary_models.append(current_binary.to(torch.uint8))
            torch.save(initial_latent, chain_dir / "initial_latent.pt")
            torch.save(current_decoded.to(torch.int8), chain_dir / "initial_categorical.pt")
            torch.save(current_binary.to(torch.uint8), chain_dir / "initial_binary.pt")
            torch.save(current_field, chain_dir / "initial_predicted_seismic.pt")

            proposal_generator = torch.Generator(device="cpu").manual_seed(proposal_seed)
            accepted_count = 0
            chain_retained_start = len(retained_binary)
            chain_trace: list[dict[str, object]] = []
            for iteration in range(1, int(resolved["proposals_per_chain"]) + 1):
                proposed_latent, innovation = posterior.pcn_proposal(
                    current_latent,
                    beta=float(config["pcn_beta"]),
                    generator=proposal_generator,
                )
                proposed_decoded, proposed_binary, proposed_field, proposed_eval = _evaluate_latent(
                    model=model,
                    latent_cpu=proposed_latent,
                    conditioning=conditioning,
                    embedded_conditions=embedded_conditions,
                    condition_values_cpu=condition_values_cpu,
                    condition_mask_cpu=condition_mask_cpu,
                    subsurface_cpu=subsurface_cpu,
                    properties=properties,
                    forward_operator=forward_operator,
                    observed=observed,
                    sample_mask=sample_mask,
                    uncertainty=uncertainty,
                    n_steps=int(config["n_euler_steps"]),
                    device=device,
                )
                proposed_loss = float(proposed_eval["hard_seismic_loss"])
                proposed_energy = posterior.posterior_energy(
                    proposed_loss, float(config["likelihood_weight"])
                )
                uniform = float(
                    torch.rand(
                        (), generator=proposal_generator, dtype=torch.float64
                    ).item()
                )
                decision = posterior.metropolis_decision(
                    current_energy, proposed_energy, uniform
                )
                if bool(decision["accepted"]):
                    current_latent = proposed_latent
                    current_decoded = proposed_decoded
                    current_binary = proposed_binary
                    current_field = proposed_field
                    current_loss = proposed_loss
                    current_energy = proposed_energy
                    accepted_count += 1
                current_violations = condition_violation_count(
                    current_decoded, condition_values_cpu, condition_mask_cpu
                )
                if current_violations:
                    raise RuntimeError("retained pCN state violates categorical conditions")
                recorded = record_current_state(
                    retained_categorical,
                    retained_binary,
                    iteration=iteration,
                    burn_in=int(resolved["burn_in"]),
                    thinning=int(resolved["thinning"]),
                    current_categorical=current_decoded,
                    current_binary=current_binary,
                )
                row = {
                    "chain_id": chain_id,
                    "iteration": iteration,
                    "post_burnin_recorded": recorded,
                    "accepted": bool(decision["accepted"]),
                    "current_energy": current_energy,
                    "proposed_energy": proposed_energy,
                    "current_hard_seismic_loss": current_loss,
                    "proposed_hard_seismic_loss": proposed_loss,
                    "acceptance_probability": decision["acceptance_probability"],
                    "log_acceptance_ratio": decision["log_acceptance_ratio"],
                    "log_uniform": decision["log_uniform"],
                    "current_sample_sha256": tensor_sha256(current_decoded),
                    "current_binary_sha256": tensor_sha256(current_binary),
                    "proposed_sample_sha256": proposed_eval["sample_sha256"],
                    "proposed_binary_sha256": proposed_eval["binary_sha256"],
                    "target_voxel_count": int(current_binary.sum().item()),
                    "target_fraction": float(current_binary.sum() / subsurface_cpu.sum()),
                    "condition_violations": current_violations,
                    "proposed_condition_violations": int(proposed_eval["condition_violations"]),
                    "proposal_latent_sha256": tensor_sha256(proposed_latent),
                    "innovation_sha256": tensor_sha256(innovation),
                    "proposal_evaluation_seconds": proposed_eval["evaluation_seconds"],
                }
                chain_trace.append(row)
                trace_rows.append(row)
                all_proposal_violations.append(int(proposed_eval["condition_violations"]))

            retained_in_chain = len(retained_binary) - chain_retained_start
            expected_per_chain = (
                int(resolved["proposals_per_chain"]) - int(resolved["burn_in"])
            ) // int(resolved["thinning"])
            if retained_in_chain != expected_per_chain:
                raise RuntimeError(
                    f"chain {chain_id} retained {retained_in_chain}, expected {expected_per_chain}"
                )
            torch.save(current_latent, chain_dir / "final_latent.pt")
            torch.save(current_decoded.to(torch.int8), chain_dir / "final_categorical.pt")
            torch.save(current_binary.to(torch.uint8), chain_dir / "final_binary.pt")
            torch.save(current_field, chain_dir / "final_predicted_seismic.pt")
            write_csv(chain_dir / "trace.csv", chain_trace)
            chain_summary = {
                "run_status": "completed",
                "chain_id": chain_id,
                "initial_seed": initial_seed,
                "proposal_seed": proposal_seed,
                "initial_noise_sha256": tensor_sha256(initial_latent),
                "initial_sample_sha256": initial_eval["sample_sha256"],
                "initial_binary_sha256": initial_eval["binary_sha256"],
                "initial_hard_seismic_loss": initial_eval["hard_seismic_loss"],
                "initial_target_voxel_count": initial_eval["target_voxel_count"],
                "initial_target_fraction": initial_eval["target_fraction"],
                "accepted_proposals": accepted_count,
                "acceptance_rate": accepted_count
                / int(resolved["proposals_per_chain"]),
                "retained_state_count": retained_in_chain,
                "final_hard_seismic_loss": current_loss,
                "final_sample_sha256": tensor_sha256(current_decoded),
                "final_binary_sha256": tensor_sha256(current_binary),
                "condition_violations": 0,
            }
            write_json(chain_dir / "manifest.json", chain_summary)
            chain_summaries.append(chain_summary)

        if len(retained_binary) != int(resolved["expected_retained_states"]):
            raise RuntimeError(
                f"retained {len(retained_binary)} states; expected exactly 96"
            )
        categorical_tensor = torch.cat(retained_categorical, dim=0).contiguous()
        binary_tensor = torch.cat(retained_binary, dim=0).contiguous()
        initial_binary_tensor = torch.cat(initial_binary_models, dim=0).contiguous()
        occupancy = binary_tensor.float().mean(dim=0, keepdim=True)
        occupancy = torch.where(subsurface_cpu, occupancy, torch.zeros_like(occupancy))
        prior_frequency = initial_binary_tensor.float().mean(dim=0, keepdim=True)
        prior_frequency = torch.where(
            subsurface_cpu, prior_frequency, torch.zeros_like(prior_frequency)
        )
        positive = (occupancy >= float(config["positive_threshold"])) & subsurface_cpu
        negative = (occupancy <= float(config["negative_threshold"])) & subsurface_cpu
        unknown = subsurface_cpu & ~(positive | negative)
        guidance_roi = (positive | negative) & subsurface_cpu & ~condition_mask_cpu
        consensus_target = positive.float()
        outputs = {
            "retained_categorical_states.pt": categorical_tensor,
            "retained_binary_states.pt": binary_tensor,
            "initial_binary_models.pt": initial_binary_tensor,
            "prior_frequency.pt": prior_frequency,
            "occupancy_frequency.pt": occupancy,
            "consensus_target.pt": consensus_target,
            "positive_mask.pt": positive,
            "negative_mask.pt": negative,
            "unknown_mask.pt": unknown,
            "guidance_roi.pt": guidance_roi,
        }
        for filename, tensor in outputs.items():
            torch.save(tensor, args.output_dir / filename)
        write_csv(args.output_dir / "chain_trace.csv", trace_rows)

        postburn_rows = [row for row in trace_rows if row["post_burnin_recorded"]]
        postburn_losses = [float(row["current_hard_seismic_loss"]) for row in postburn_rows]
        target_counts = [int(row["target_voxel_count"]) for row in postburn_rows]
        target_fractions = [float(row["target_fraction"]) for row in postburn_rows]
        categorical_hashes = [tensor_sha256(value) for value in retained_categorical]
        binary_hashes = [tensor_sha256(value) for value in retained_binary]
        support_count = int(subsurface_cpu.sum())
        positive_count = int(positive.sum())
        negative_count = int(negative.sum())
        unknown_count = int(unknown.sum())
        summary = {
            "sampler": {
                "acceptance_rate_per_chain": [
                    float(value["acceptance_rate"]) for value in chain_summaries
                ],
                "overall_acceptance_rate": sum(
                    int(value["accepted_proposals"]) for value in chain_summaries
                )
                / (int(resolved["n_chains"]) * int(resolved["proposals_per_chain"])),
                "retained_state_count": len(retained_binary),
                "unique_hard_models": len(set(categorical_hashes)),
                "unique_binary_models": len(set(binary_hashes)),
                "repeated_retained_categorical_states": len(categorical_hashes)
                - len(set(categorical_hashes)),
                "repeated_retained_binary_states": len(binary_hashes)
                - len(set(binary_hashes)),
            },
            "seismic": {
                "initial_hard_seismic_loss_per_chain": [
                    float(value["initial_hard_seismic_loss"])
                    for value in chain_summaries
                ],
                "minimum_post_burnin_hard_seismic_loss": min(postburn_losses),
                "median_post_burnin_hard_seismic_loss": statistics.median(
                    postburn_losses
                ),
                "maximum_post_burnin_hard_seismic_loss": max(postburn_losses),
                "post_burnin_loss_per_state": postburn_losses,
            },
            "geology": {
                "initial_target_voxel_count_per_chain": [
                    int(value["initial_target_voxel_count"])
                    for value in chain_summaries
                ],
                "initial_target_fraction_per_chain": [
                    float(value["initial_target_fraction"])
                    for value in chain_summaries
                ],
                "post_burnin_target_voxel_count_min": min(target_counts),
                "post_burnin_target_voxel_count_median": statistics.median(
                    target_counts
                ),
                "post_burnin_target_voxel_count_max": max(target_counts),
                "post_burnin_target_fraction_min": min(target_fractions),
                "post_burnin_target_fraction_median": statistics.median(
                    target_fractions
                ),
                "post_burnin_target_fraction_max": max(target_fractions),
                "maximum_condition_violations": max(
                    [0] + all_proposal_violations
                ),
            },
            "occupancy": {
                "p9_min": float(occupancy[subsurface_cpu].min()),
                "p9_max": float(occupancy[subsurface_cpu].max()),
                "p9_mean_subsurface": float(occupancy[subsurface_cpu].mean()),
                "prior_frequency_mean_subsurface_four_sample_diagnostic": float(
                    prior_frequency[subsurface_cpu].mean()
                ),
                "positive_voxels": positive_count,
                "negative_voxels": negative_count,
                "unknown_voxels": unknown_count,
                "confidence_coverage": (positive_count + negative_count)
                / support_count,
                "guidance_roi_voxels": int(guidance_roi.sum()),
            },
            "positive_geometry": _positive_geometry(positive),
            "scientific_distinction": {
                "phase5c": "15-class petrophysical hard seismic likelihood",
                "stage15_b2": "target-specific binary label9-vs-all-non-label9-subsurface hard seismic likelihood",
            },
            "occupancy_interpretation": config["occupancy_interpretation"],
            "truth_loaded_by_runner": False,
        }
        write_json(args.output_dir / "pilot_summary.json", summary)
        (args.output_dir / "PILOT_REPORT.md").write_text(
            _summary_markdown(summary), encoding="utf-8"
        )

        manifest.update(
            {
                "run_status": "completed",
                "runtime_seconds": time.perf_counter() - total_started,
                "resolved_protocol": resolved,
                "device": str(device),
                "cuda_device_name": torch.cuda.get_device_name(device),
                "model_load_report": model_report,
                "chains": chain_summaries,
                "summary": summary,
                "observation_manifest": runtime.asset_record(
                    args.observation_dir / "manifest.json"
                ),
                "input_assets": {
                    name: runtime.asset_record(args.observation_dir / name)
                    for name in inputs
                },
                "input_tensor_sha256": {
                    name: tensor_sha256(value) for name, value in inputs.items()
                },
                "checkpoint": runtime.asset_record(checkpoint_path),
                "binary_acoustic_config": runtime.asset_record(
                    args.binary_acoustic_config
                ),
                "source_acoustic_config": runtime.asset_record(source_path),
                "seismic_config": runtime.asset_record(args.seismic_config),
                "seismic_parameters": resolved_seismic,
                "generator_posterior_source": runtime.asset_record(
                    Path(posterior.__file__)
                ),
                "binary_inversion_source": runtime.asset_record(
                    PROJECT_DIR / "guidance/binary_seismic_inversion.py"
                ),
                "output_tensor_sha256": {
                    name: tensor_sha256(value) for name, value in outputs.items()
                },
                "pilot_summary": runtime.asset_record(
                    args.output_dir / "pilot_summary.json"
                ),
                "truth_loaded_by_runner": False,
                "training_performed": False,
                "parameter_sweep_performed": False,
                "stage15_b1_results_modified": False,
            }
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update(
            {"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}
        )
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
