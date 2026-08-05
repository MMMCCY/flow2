#!/usr/bin/env python3
"""Train and sample the frozen Phase-6 acoustic-oracle adapter smoke."""

from __future__ import annotations

import argparse
import hashlib
import math
from pathlib import Path
import statistics
import sys
import time
from typing import Dict, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.generator_posterior import project_conditions
from guidance.probability_evaluation import class_transition_records
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.property_evaluation import (
    sample_property_hard_metrics,
    truth_component_recovery_rows,
)
from guidance.residual_velocity_adapter import (
    RESIDUAL_ADAPTER_VERSION,
    ResidualVelocityAdapter,
    cap_residual_velocity,
    class_balancing_weights,
    fixed_euler_adapter_sample,
    residual_adapter_losses,
)
from guidance.seismic import tensor_sha256
from scripts.stage4.run_seismic_guidance import (
    add_hard_seismic_metrics,
    load_observation_assets,
    read_json,
    write_json,
    write_rows,
)
from scripts.stage5.run_generator_posterior import _validate_historical_baseline


PHASE6A_CONFIG_SCHEMA = "phase6a_adapter_smoke_config_v1"
PHASE6A_RUN_SCHEMA = "phase6a_oracle_adapter_smoke_run_v1"


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage6_geo_adapter"
    case_dir = PROJECT_DIR / "samples/jupyter-demo/cond_generation_0"
    parser = argparse.ArgumentParser(
        description="Train the frozen Phase-6 acoustic-oracle adapter engineering smoke.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs/oracle_acoustic_tiny_overfit_v1.json",
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
        "--historical-baseline-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/baseline",
    )
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def validate_config(config: Mapping[str, object]) -> dict[str, object]:
    if config.get("schema") != PHASE6A_CONFIG_SCHEMA:
        raise ValueError("invalid Phase-6 adapter config schema")
    if config.get("status") != "frozen_before_cuda_output":
        raise ValueError("Phase-6 smoke config must be frozen before CUDA execution")
    if config.get("truth_derived_oracle_input") is not True:
        raise ValueError("first Phase-6 smoke must be explicitly marked oracle")
    if config.get("publication_evidence") is not False:
        raise ValueError("legacy oracle smoke cannot be publication evidence")
    if config.get("allow_hyperparameter_sweep_on_case") is not False:
        raise ValueError("legacy oracle case must prohibit hyperparameter sweeps")
    adapter = config.get("adapter")
    training = config.get("training")
    sampling = config.get("sampling")
    if not all(isinstance(value, Mapping) for value in (adapter, training, sampling)):
        raise ValueError("Phase-6 config sections are incomplete")
    state_seeds = [int(value) for value in training["state_seeds"]]
    times = [float(value) for value in training["times"]]
    dilations = [int(value) for value in adapter["dilations"]]
    if not state_seeds or len(state_seeds) != len(times):
        raise ValueError("state seeds and interpolation times must be nonempty and paired")
    if not all(0.0 < value < 1.0 for value in times):
        raise ValueError("interpolation times must lie in (0,1)")
    resolved = {
        "base_width": int(adapter["base_width"]),
        "dilations": dilations,
        "geophysics_channels": int(adapter["geophysics_channels"]),
        "max_parameters": int(adapter["max_parameters"]),
        "max_residual_ratio": float(adapter["max_residual_ratio"]),
        "training_seed": int(training["seed"]),
        "state_seeds": state_seeds,
        "times": times,
        "training_steps": int(training["steps"]),
        "learning_rate": float(training["learning_rate"]),
        "weight_decay": float(training["weight_decay"]),
        "gradient_clip_norm": float(training["gradient_clip_norm"]),
        "flow_weight": float(training["flow_weight"]),
        "cross_entropy_weight": float(training["cross_entropy_weight"]),
        "dice_weight": float(training["dice_weight"]),
        "residual_regularizer_weight": float(
            training["residual_regularizer_weight"]
        ),
        "logit_temperature": float(training["logit_temperature"]),
        "sampling_seed": int(sampling["seed"]),
        "sampling_steps": int(sampling["n_steps"]),
        "adapter_scale": float(sampling["adapter_scale"]),
    }
    positive = (
        "base_width",
        "geophysics_channels",
        "max_parameters",
        "training_steps",
        "learning_rate",
        "gradient_clip_norm",
        "logit_temperature",
        "sampling_steps",
        "adapter_scale",
    )
    for field in positive:
        value = float(resolved[field])
        if not math.isfinite(value) or value <= 0:
            raise ValueError(f"resolved Phase-6 field must be positive: {field}")
    if not 0 < resolved["max_residual_ratio"] <= 1:
        raise ValueError("max residual ratio must lie in (0,1]")
    return resolved


def _model_tensor_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode("utf-8"))
        digest.update(tensor_sha256(parameter.detach().cpu()).encode("ascii"))
    return digest.hexdigest()


def _oracle_feature(
    truth_acoustic: torch.Tensor, subsurface_mask: torch.Tensor
) -> tuple[torch.Tensor, list[dict[str, float]]]:
    value = truth_acoustic.float()
    mask = subsurface_mask.bool().expand_as(value)
    normalized = torch.zeros_like(value)
    records: list[dict[str, float]] = []
    for channel in range(value.shape[1]):
        active = value[:, channel : channel + 1][mask[:, channel : channel + 1]]
        mean = active.mean()
        std = active.std(unbiased=False).clamp_min(1e-8)
        normalized[:, channel : channel + 1] = torch.where(
            mask[:, channel : channel + 1],
            (value[:, channel : channel + 1] - mean) / std,
            torch.zeros_like(value[:, channel : channel + 1]),
        )
        records.append(
            {"channel": channel, "mean": float(mean), "std": float(std)}
        )
    return normalized.contiguous(), records


def _cached_training_states(
    *,
    model,
    embedded_truth: torch.Tensor,
    conditioning: torch.Tensor,
    condition_mask: torch.Tensor,
    seeds: list[int],
    times: list[float],
) -> list[dict[str, torch.Tensor | int | float | str]]:
    records: list[dict[str, torch.Tensor | int | float | str]] = []
    for seed, time_value in zip(seeds, times):
        generator = torch.Generator(device="cpu").manual_seed(seed)
        initial_cpu = torch.randn(
            embedded_truth.shape,
            generator=generator,
            dtype=embedded_truth.dtype,
        ).contiguous()
        initial = initial_cpu.to(embedded_truth.device)
        time_tensor = torch.tensor(
            [time_value], device=embedded_truth.device, dtype=embedded_truth.dtype
        )
        with torch.no_grad():
            state, target_velocity = model.interpolator.flow_objective(
                time_tensor, initial, embedded_truth
            )
            state = project_conditions(state, embedded_truth, condition_mask)
            base_velocity = model.net(state, conditioning, time_tensor)
        records.append(
            {
                "seed": seed,
                "time_value": time_value,
                "time": time_tensor,
                "state": state.detach(),
                "target_velocity": target_velocity.detach(),
                "base_velocity": base_velocity.detach(),
                "initial_noise_sha256": tensor_sha256(initial_cpu),
            }
        )
    return records


def _mean_endpoint_accuracy(
    *,
    adapter: ResidualVelocityAdapter,
    cached: list[dict[str, torch.Tensor | int | float | str]],
    conditioning: torch.Tensor,
    condition_mask: torch.Tensor,
    geophysics: torch.Tensor,
    truth: torch.Tensor,
    embedding_weight: torch.Tensor,
    class_weights: torch.Tensor,
    resolved: Mapping[str, object],
) -> tuple[float, float]:
    losses: list[float] = []
    accuracies: list[float] = []
    adapter.eval()
    with torch.no_grad():
        for item in cached:
            raw = adapter(
                item["state"],
                item["base_velocity"],
                conditioning,
                condition_mask,
                geophysics,
                item["time"],
            )
            correction, _ = cap_residual_velocity(
                raw,
                item["base_velocity"],
                condition_mask,
                max_ratio=float(resolved["max_residual_ratio"]),
            )
            loss, diagnostics = residual_adapter_losses(
                state=item["state"],
                target_velocity=item["target_velocity"],
                base_velocity=item["base_velocity"],
                correction=correction,
                truth=truth,
                condition_mask=condition_mask,
                embedding_weight=embedding_weight,
                time=item["time"],
                class_weights=class_weights,
                logit_temperature=float(resolved["logit_temperature"]),
                flow_weight=float(resolved["flow_weight"]),
                cross_entropy_weight=float(resolved["cross_entropy_weight"]),
                dice_weight=float(resolved["dice_weight"]),
                residual_regularizer_weight=float(
                    resolved["residual_regularizer_weight"]
                ),
            )
            losses.append(float(loss))
            accuracies.append(float(diagnostics["endpoint_accuracy"]))
    adapter.train()
    return float(statistics.mean(losses)), float(statistics.mean(accuracies))


def _major_four_mean(rows: list[Mapping[str, object]], sample_id: int) -> float:
    values = [
        float(row["recall"])
        for row in rows
        if int(row["sample_id"]) == sample_id
        and int(row["truth_component_rank"]) <= 4
    ]
    return float(statistics.mean(values)) if values else float("nan")


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("the frozen Phase-6 adapter smoke requires CUDA")
    config = read_json(args.config)
    resolved = validate_config(config)
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
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    base_hash_before = _model_tensor_hash(model)
    conditioning_report = runtime.validate_conditioning_pair(
        truth_cpu,
        boreholes_cpu,
        model.num_categories,
        target_label=args.target_label,
    )
    tensors, observation_manifest, forward_operator, resolved_observation = (
        load_observation_assets(
            args.observation_dir,
            truth_cpu,
            truth_path=truth_path,
            num_categories=model.num_categories,
        )
    )
    truth = truth_cpu.to(device)
    condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
    condition_mask = condition_mask_cpu.to(device)
    embedded_truth = model.embed(truth)
    conditioning = embedded_truth * condition_mask.expand_as(embedded_truth)
    geophysics_cpu, geophysics_normalization = _oracle_feature(
        tensors["truth_acoustic.pt"], tensors["subsurface_mask.pt"]
    )
    geophysics = geophysics_cpu.to(device)

    torch.manual_seed(int(resolved["training_seed"]))
    torch.cuda.manual_seed_all(int(resolved["training_seed"]))
    adapter = ResidualVelocityAdapter(
        model.embedding_dim,
        geophysics_channels=int(resolved["geophysics_channels"]),
        base_width=int(resolved["base_width"]),
        dilations=resolved["dilations"],
    ).to(device)
    adapter_parameters = [parameter for parameter in adapter.parameters()]
    parameter_count = adapter.parameter_count()
    if parameter_count > int(resolved["max_parameters"]):
        raise ValueError("adapter exceeds frozen parameter budget")
    optimizer = torch.optim.AdamW(
        adapter_parameters,
        lr=float(resolved["learning_rate"]),
        weight_decay=float(resolved["weight_decay"]),
    )
    optimizer_ids = {
        id(parameter)
        for group in optimizer.param_groups
        for parameter in group["params"]
    }
    if optimizer_ids != {id(parameter) for parameter in adapter_parameters}:
        raise RuntimeError("optimizer contains parameters outside the adapter")

    cached = _cached_training_states(
        model=model,
        embedded_truth=embedded_truth,
        conditioning=conditioning,
        condition_mask=condition_mask,
        seeds=resolved["state_seeds"],
        times=resolved["times"],
    )
    active_mask = (~condition_mask) & (truth != -1)
    class_weights = class_balancing_weights(
        truth, active_mask, model.num_categories
    ).to(device)
    initial_cached_loss, initial_endpoint_accuracy = _mean_endpoint_accuracy(
        adapter=adapter,
        cached=cached,
        conditioning=conditioning,
        condition_mask=condition_mask,
        geophysics=geophysics,
        truth=truth,
        embedding_weight=model.embedding.weight,
        class_weights=class_weights,
        resolved=resolved,
    )

    sources = runtime.experiment_asset_records(
        smoke_config=args.config,
        phase6_spec=PROJECT_DIR / "docs/PHASE6_ADAPTER_SPEC.md",
        truth_model=truth_path,
        boreholes=boreholes_path,
        observation_manifest=args.observation_dir / "manifest.json",
        historical_baseline_config=args.historical_baseline_dir / "config.json",
        runner_source=Path(__file__),
        adapter_source=PROJECT_DIR / "guidance/residual_velocity_adapter.py",
        runtime_source=Path(runtime.__file__),
    )
    sources["checkpoint"] = model_load_report["checkpoint"]
    run_config: Dict[str, object] = {
        "schema": PHASE6A_RUN_SCHEMA,
        "run_status": "running",
        "config": config,
        "resolved_config": resolved,
        "adapter_version": RESIDUAL_ADAPTER_VERSION,
        "adapter_parameter_count": parameter_count,
        "base_model_frozen": True,
        "base_model_tensor_sha256_before": base_hash_before,
        "optimizer_contains_adapter_only": True,
        "model_weight_source": "ema",
        "ema_applied": bool(model_load_report["ema_applied"]),
        "truth_derived_oracle_input": True,
        "publication_evidence": False,
        "device": str(device),
        "cuda_device_name": torch.cuda.get_device_name(device),
        "torch_version": torch.__version__,
        "asset_records": sources,
        "model_load_report": model_load_report,
        "conditioning_report": conditioning_report,
        "observation_config_resolved": resolved_observation,
        "geophysics_normalization": geophysics_normalization,
        "cached_training_states": [
            {
                "seed": int(item["seed"]),
                "time": float(item["time_value"]),
                "initial_noise_sha256": item["initial_noise_sha256"],
            }
            for item in cached
        ],
        "initial_cached_loss": initial_cached_loss,
        "initial_endpoint_accuracy": initial_endpoint_accuracy,
    }
    run_config.update(runtime.flatten_asset_hashes(sources))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(geophysics_cpu, args.output_dir / "oracle_geophysics.pt")
    write_json(args.output_dir / "config.json", run_config)
    write_json(args.output_dir / "model_load_report.json", model_load_report)
    write_json(args.output_dir / "input_validation.json", conditioning_report)
    write_json(args.output_dir / "observation_manifest.json", observation_manifest)

    adapter.train()
    torch.cuda.reset_peak_memory_stats(device)
    started = time.perf_counter()
    training_rows: list[Dict[str, object]] = []
    any_nonzero_gradient = False
    for step in range(int(resolved["training_steps"])):
        item = cached[step % len(cached)]
        optimizer.zero_grad(set_to_none=True)
        raw_correction = adapter(
            item["state"],
            item["base_velocity"],
            conditioning,
            condition_mask,
            geophysics,
            item["time"],
        )
        correction, used_ratio = cap_residual_velocity(
            raw_correction,
            item["base_velocity"],
            condition_mask,
            max_ratio=float(resolved["max_residual_ratio"]),
        )
        loss, diagnostics = residual_adapter_losses(
            state=item["state"],
            target_velocity=item["target_velocity"],
            base_velocity=item["base_velocity"],
            correction=correction,
            truth=truth,
            condition_mask=condition_mask,
            embedding_weight=model.embedding.weight,
            time=item["time"],
            class_weights=class_weights,
            logit_temperature=float(resolved["logit_temperature"]),
            flow_weight=float(resolved["flow_weight"]),
            cross_entropy_weight=float(resolved["cross_entropy_weight"]),
            dice_weight=float(resolved["dice_weight"]),
            residual_regularizer_weight=float(
                resolved["residual_regularizer_weight"]
            ),
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("adapter training loss became non-finite")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            adapter_parameters, float(resolved["gradient_clip_norm"])
        )
        if math.isfinite(float(gradient_norm)) and float(gradient_norm) > 0:
            any_nonzero_gradient = True
        if any(parameter.grad is not None for parameter in model.parameters()):
            raise RuntimeError("frozen base model unexpectedly received a gradient")
        optimizer.step()
        training_rows.append(
            {
                "step": step,
                "cached_state_index": step % len(cached),
                "time": float(item["time_value"]),
                "total_loss": float(diagnostics["total_loss"].detach()),
                "flow_loss": float(diagnostics["flow_loss"].detach()),
                "cross_entropy_loss": float(
                    diagnostics["cross_entropy_loss"].detach()
                ),
                "dice_loss": float(diagnostics["dice_loss"].detach()),
                "residual_regularizer": float(
                    diagnostics["residual_regularizer"].detach()
                ),
                "endpoint_accuracy": float(
                    diagnostics["endpoint_accuracy"].detach()
                ),
                "used_residual_ratio": float(used_ratio.mean().detach()),
                "gradient_norm_before_clip": float(gradient_norm),
            }
        )
    torch.cuda.synchronize(device)
    training_seconds = time.perf_counter() - started
    final_cached_loss, final_endpoint_accuracy = _mean_endpoint_accuracy(
        adapter=adapter,
        cached=cached,
        conditioning=conditioning,
        condition_mask=condition_mask,
        geophysics=geophysics,
        truth=truth,
        embedding_weight=model.embedding.weight,
        class_weights=class_weights,
        resolved=resolved,
    )
    base_hash_after = _model_tensor_hash(model)
    base_hash_unchanged = base_hash_after == base_hash_before
    base_gradients_absent = all(
        parameter.grad is None for parameter in model.parameters()
    )

    adapter.eval()
    sampling_generator = torch.Generator(device="cpu").manual_seed(
        int(resolved["sampling_seed"])
    )
    sampling_initial_cpu = torch.randn(
        embedded_truth.shape,
        generator=sampling_generator,
        dtype=embedded_truth.dtype,
    ).contiguous()
    baseline_state, baseline_trace = fixed_euler_adapter_sample(
        model=model,
        adapter=adapter,
        initial_state=sampling_initial_cpu.to(device),
        conditioning=conditioning,
        embedded_conditions=embedded_truth,
        condition_mask=condition_mask,
        geophysics=geophysics,
        n_steps=int(resolved["sampling_steps"]),
        adapter_scale=0.0,
        max_residual_ratio=float(resolved["max_residual_ratio"]),
    )
    adapted_state, adapted_trace = fixed_euler_adapter_sample(
        model=model,
        adapter=adapter,
        initial_state=sampling_initial_cpu.to(device),
        conditioning=conditioning,
        embedded_conditions=embedded_truth,
        condition_mask=condition_mask,
        geophysics=geophysics,
        n_steps=int(resolved["sampling_steps"]),
        adapter_scale=float(resolved["adapter_scale"]),
        max_residual_ratio=float(resolved["max_residual_ratio"]),
    )
    baseline_decoded = (model.decode(baseline_state).detach().cpu() - 1).long()
    adapted_decoded = (model.decode(adapted_state).detach().cpu() - 1).long()
    historical_config = read_json(args.historical_baseline_dir / "config.json")
    historical_validation = _validate_historical_baseline(
        baseline_dir=args.historical_baseline_dir,
        baseline_config=historical_config,
        initial_decoded=baseline_decoded,
        initial_noise_sha256=tensor_sha256(sampling_initial_cpu),
        ckpt_path=args.ckpt_path,
        truth_path=truth_path,
        boreholes_path=boreholes_path,
        observation_dir=args.observation_dir,
        initial_seed=int(resolved["sampling_seed"]),
        n_steps=int(resolved["sampling_steps"]),
    )
    target_mask, target_metadata = build_target_mask(
        truth_cpu, target_label=args.target_label, component_mode="all"
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)
    property_confidence = ((truth_cpu != -1) & ~condition_mask_cpu).float()
    property_table = tensors["acoustic_property_table.pt"]
    metrics_rows: list[Dict[str, object]] = []
    component_rows: list[Dict[str, object]] = []
    for sample_id, (role, prediction) in enumerate(
        (("adapter_scale_zero", baseline_decoded), ("adapter_scale_one", adapted_decoded))
    ):
        metrics = sample_property_hard_metrics(
            prediction=prediction,
            truth_model=truth_cpu,
            condition_mask=condition_mask_cpu,
            target_mask=target_mask,
            target_roi_mask=target_roi,
            target_label=args.target_label,
            property_table=property_table,
            property_confidence=property_confidence,
            property_sigmas=(0.0,),
            property_scale_weights=(1.0,),
            property_channel_weights=torch.ones(property_table.shape[0]),
            sample_id=sample_id,
            baseline_prediction=baseline_decoded if sample_id else None,
        )
        add_hard_seismic_metrics(
            metrics,
            prediction=prediction,
            target_acoustic=tensors["truth_acoustic.pt"],
            condition_mask=condition_mask_cpu,
            property_table=property_table,
            subsurface_mask=tensors["subsurface_mask.pt"],
            forward_operator=forward_operator,
            observed=tensors["observed_seismic.pt"],
            sample_mask=tensors["sample_mask.pt"],
            uncertainty=tensors["uncertainty_amplitude.pt"],
            device=device,
        )
        metrics["role"] = role
        metrics_rows.append(metrics)
        component_rows.extend(
            truth_component_recovery_rows(
                prediction, truth_cpu, args.target_label, sample_id
            )
        )
    baseline_metrics, adapted_metrics = metrics_rows
    baseline_major = _major_four_mean(component_rows, 0)
    adapted_major = _major_four_mean(component_rows, 1)

    first_window = statistics.mean(
        float(row["total_loss"]) for row in training_rows[:10]
    )
    last_window = statistics.mean(
        float(row["total_loss"]) for row in training_rows[-10:]
    )
    condition_zero = all(
        float(row["used_residual_ratio"]) == 0.0 for row in baseline_trace
    ) and int(adapted_metrics["condition_violation_count"]) == 0
    engineering_gates = {
        "base_tensor_hash_unchanged": base_hash_unchanged,
        "base_gradients_absent": base_gradients_absent,
        "optimizer_adapter_only": True,
        "adapter_nonzero_gradient": any_nonzero_gradient,
        "adapter_parameter_budget": parameter_count <= int(
            resolved["max_parameters"]
        ),
        "historical_scale_zero_exact": bool(
            historical_validation["exact_initial_hard_regression"]
        ),
        "conditions_exact": condition_zero,
        "loss_window_decreased": last_window < first_window,
        "cached_endpoint_accuracy_increased": final_endpoint_accuracy
        > initial_endpoint_accuracy,
    }
    geology_directions = {
        "truth_present_mean_iou_improved": float(
            adapted_metrics["truth_present_mean_iou"]
        )
        > float(baseline_metrics["truth_present_mean_iou"]),
        "target_iou_nonregression": float(adapted_metrics["target_iou"])
        >= float(baseline_metrics["target_iou"]),
        "target_recall_nonregression": float(adapted_metrics["target_recall"])
        >= float(baseline_metrics["target_recall"]),
        "major_four_recall_nonregression": adapted_major >= baseline_major,
    }
    engineering_pass = all(engineering_gates.values())
    oracle_mechanism_pass = engineering_pass and all(geology_directions.values())

    checkpoint_payload = {
        "schema": "phase6_adapter_checkpoint_v1",
        "adapter_version": RESIDUAL_ADAPTER_VERSION,
        "adapter_state_dict": {
            name: value.detach().cpu()
            for name, value in adapter.state_dict().items()
        },
        "base_checkpoint_sha256": model_load_report["checkpoint"]["sha256"],
        "smoke_config_sha256": runtime.file_sha256(args.config),
        "base_model_tensor_sha256": base_hash_after,
        "optimizer_state_dict": optimizer.state_dict(),
        "training_steps": int(resolved["training_steps"]),
    }
    torch.save(checkpoint_payload, args.output_dir / "adapter_checkpoint.pt")
    torch.save(baseline_decoded.to(torch.int8), args.output_dir / "baseline_sample.pt")
    torch.save(adapted_decoded.to(torch.int8), args.output_dir / "adapted_sample.pt")
    write_rows(args.output_dir / "training_trace.csv", training_rows)
    write_rows(args.output_dir / "sample_metrics.csv", metrics_rows)
    write_rows(args.output_dir / "truth_component_recovery.csv", component_rows)
    write_rows(
        args.output_dir / "paired_class_transitions.csv",
        class_transition_records(baseline_decoded, adapted_decoded, 1),
    )
    write_rows(args.output_dir / "baseline_sampling_trace.csv", baseline_trace)
    write_rows(args.output_dir / "adapted_sampling_trace.csv", adapted_trace)

    run_config.update(
        {
            "run_status": "completed",
            "training_seconds": training_seconds,
            "cuda_peak_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
            "base_model_tensor_sha256_after": base_hash_after,
            "base_model_tensor_hash_unchanged": base_hash_unchanged,
            "base_gradients_absent": base_gradients_absent,
            "adapter_nonzero_gradient": any_nonzero_gradient,
            "first_ten_step_mean_loss": first_window,
            "last_ten_step_mean_loss": last_window,
            "final_cached_loss": final_cached_loss,
            "final_endpoint_accuracy": final_endpoint_accuracy,
            "sampling_initial_noise_sha256": tensor_sha256(sampling_initial_cpu),
            "baseline_sample_sha256": tensor_sha256(baseline_decoded),
            "adapted_sample_sha256": tensor_sha256(adapted_decoded),
            "adapter_checkpoint_sha256": runtime.file_sha256(
                args.output_dir / "adapter_checkpoint.pt"
            ),
            "oracle_geophysics_sha256": tensor_sha256(geophysics_cpu),
            "historical_baseline_validation": historical_validation,
            "engineering_gates": engineering_gates,
            "engineering_pass": engineering_pass,
            "geology_directions": geology_directions,
            "oracle_mechanism_pass": oracle_mechanism_pass,
            "decision": (
                "PASS: authorize deterministic grouped data and a held-out oracle pilot"
                if oracle_mechanism_pass
                else "FAIL: do not start formal Phase-6 training; diagnose adapter mechanism"
            ),
            "baseline_metrics": {
                "global_voxel_accuracy": baseline_metrics[
                    "global_voxel_accuracy"
                ],
                "truth_present_mean_iou": baseline_metrics[
                    "truth_present_mean_iou"
                ],
                "target_iou": baseline_metrics["target_iou"],
                "target_precision": baseline_metrics["target_precision"],
                "target_recall": baseline_metrics["target_recall"],
                "major_four_mean_recall": baseline_major,
                "hard_seismic_loss": baseline_metrics["hard_seismic_loss"],
            },
            "adapted_metrics": {
                "global_voxel_accuracy": adapted_metrics[
                    "global_voxel_accuracy"
                ],
                "truth_present_mean_iou": adapted_metrics[
                    "truth_present_mean_iou"
                ],
                "target_iou": adapted_metrics["target_iou"],
                "target_precision": adapted_metrics["target_precision"],
                "target_recall": adapted_metrics["target_recall"],
                "major_four_mean_recall": adapted_major,
                "hard_seismic_loss": adapted_metrics["hard_seismic_loss"],
            },
            "target_metadata": target_metadata,
        }
    )
    write_json(args.output_dir / "config.json", run_config)
    print(
        "Phase-6 oracle adapter smoke complete: "
        f"engineering_pass={engineering_pass}, oracle_mechanism_pass={oracle_mechanism_pass}, "
        f"loss={initial_cached_loss:.6g}->{final_cached_loss:.6g}, "
        f"endpoint_acc={initial_endpoint_accuracy:.6g}->{final_endpoint_accuracy:.6g}, "
        f"output={args.output_dir}"
    )


if __name__ == "__main__":
    main()

