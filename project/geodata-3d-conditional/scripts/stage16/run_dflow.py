#!/usr/bin/env python3
"""Run paired FLOW_ONLY/DFLOW source optimization for a frozen objective."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Callable, Mapping

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
from guidance.dflow import (
    DFLOW_ENGINE_VERSION,
    differentiable_flow_solve,
    freeze_model_for_dflow,
    model_state_dict_hashes,
    optimize_source_noise,
)
from guidance.probability_volume import probability_volume_loss, tensor_sha256
from guidance.property_volume import property_volume_loss
from guidance.seismic import seismic_operator_from_config
from guided_geophysical_sampling import soft_decode_to_probs
from scripts.stage15.common import (
    base_manifest,
    normalize_volume,
    read_json,
    refuse_nonempty,
    write_csv,
    write_json,
)


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage16_dflow_source_optimization"
DEFAULT_CONFIG = EXPERIMENT_ROOT / "configs/oracle_probability_v1.json"
OBJECTIVES = ("oracle_probability", "property", "seismic")
ASSET_KEYS = {
    "oracle_probability": {
        "checkpoint",
        "truth_model",
        "boreholes",
        "target_probability",
        "target_mask",
        "target_roi_mask",
    },
    "property": {
        "checkpoint",
        "truth_model",
        "boreholes",
        "property_table",
        "target_properties",
        "property_confidence",
        "target_mask",
        "target_roi_mask",
    },
    "seismic": {
        "checkpoint",
        "condition_values",
        "condition_mask",
        "subsurface_mask",
        "binary_well_values",
        "binary_well_mask",
        "observed_seismic",
        "binary_acoustic_config",
        "seismic_config",
    },
}


def objective_required_asset_keys(objective: str) -> set[str]:
    """Return the exact inference-visible asset API for one objective."""
    if objective not in ASSET_KEYS:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    return set(ASSET_KEYS[objective])


def seismic_runner_requires_truth() -> bool:
    """Program-level firewall assertion used by the CPU API regression test."""
    keys = objective_required_asset_keys("seismic")
    return any("truth" in key for key in keys)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="run only the first pre-registered source seed without changing config",
    )
    return parser.parse_args()


def _resolved_path(value: object) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_config(config: Mapping[str, object]) -> tuple[str, list[int]]:
    if config.get("schema") != "stage16_dflow_source_optimization_v1":
        raise ValueError("invalid Stage16 D-Flow config schema")
    if config.get("status") != "frozen_before_run":
        raise ValueError("formal D-Flow config must be frozen_before_run")
    objective = str(config.get("objective"))
    if objective not in OBJECTIVES:
        raise ValueError(f"objective must be one of {OBJECTIVES}")
    if config.get("arms") != ["FLOW_ONLY", "DFLOW"]:
        raise ValueError("Stage16 requires exactly FLOW_ONLY and DFLOW arms")
    if config.get("solver") != "fixed_euler" or int(config.get("n_steps", 0)) != 32:
        raise ValueError("Stage16 formal configs require the authoritative 32-step fixed Euler")
    if config.get("initialization") != "standard_normal_cpu_generator_v1":
        raise ValueError("Stage16 source initialization must be standard Gaussian")
    if float(config.get("source_regularization_weight", -1.0)) != 0.0:
        raise ValueError("Stage16 v1 formal configs keep source regularization disabled")
    if config.get("parameter_sweep") is not False or config.get("training_performed") is not False:
        raise ValueError("Stage16 forbids parameter sweeps and training")
    seeds = config.get("source_seeds")
    if not isinstance(seeds, list) or not seeds or len(set(int(v) for v in seeds)) != len(seeds):
        raise ValueError("source_seeds must be a non-empty unique list")
    assets = config.get("assets")
    if not isinstance(assets, Mapping) or set(assets) != objective_required_asset_keys(objective):
        raise ValueError("config assets do not match the objective inference API")
    if objective == "seismic" and seismic_runner_requires_truth():
        raise RuntimeError("seismic inference API unexpectedly exposes truth")
    return objective, [int(value) for value in seeds]


def _validated_assets(
    config: Mapping[str, object], objective: str
) -> dict[str, Path]:
    records = config["assets"]
    paths: dict[str, Path] = {}
    for name in objective_required_asset_keys(objective):
        record = records[name]
        if not isinstance(record, Mapping):
            raise TypeError(f"asset record must be an object: {name}")
        path = _resolved_path(record["path"])
        actual = runtime.file_sha256(path)
        if actual != str(record["sha256"]):
            raise ValueError(f"frozen input asset changed: {name}")
        paths[name] = path
    return paths


def _decode(model, state: torch.Tensor) -> torch.Tensor:
    return (model.decode(state) - 1).unsqueeze(1).detach().cpu().long().contiguous()


def _conditioning_from_assets(
    objective: str,
    assets: Mapping[str, Path],
    model,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict[str, object]]:
    if objective == "seismic":
        condition_values_cpu = normalize_volume(
            runtime.load_tensor(assets["condition_values"]), "condition_values"
        ).long()
        condition_mask_cpu = normalize_volume(
            runtime.load_tensor(assets["condition_mask"]), "condition_mask"
        ).bool()
        report = {
            "source": "stage15_binary_seismic_condition_assets",
            "condition_voxels": int(condition_mask_cpu.sum()),
            "truth_loaded": False,
        }
    else:
        truth_cpu = normalize_volume(
            runtime.load_tensor(assets["truth_model"]), "truth_model"
        ).long()
        boreholes_cpu = normalize_volume(
            runtime.load_tensor(assets["boreholes"]), "boreholes"
        ).long()
        report = runtime.validate_conditioning_pair(
            truth_cpu,
            boreholes_cpu,
            num_categories=model.num_categories,
            target_label=9,
        )
        condition_values_cpu = truth_cpu
        condition_mask_cpu = (boreholes_cpu != -1) | (truth_cpu == -1)
        report = {**report, "source": "phase1_phase2_authoritative_condition_semantics", "truth_loaded": True}

    condition_values = condition_values_cpu.to(device)
    condition_mask = condition_mask_cpu.to(device)
    embedded_conditions = model.embed(condition_values)
    expanded_mask = condition_mask.expand(
        -1, embedded_conditions.shape[1], -1, -1, -1
    )
    conditioning = embedded_conditions * expanded_mask
    return (
        condition_values,
        condition_mask,
        embedded_conditions,
        conditioning,
        report,
    )


def _oracle_objective(
    config: Mapping[str, object],
    assets: Mapping[str, Path],
    model,
    device: torch.device,
) -> tuple[Callable, Callable | None, dict[str, object]]:
    target_probability = runtime.load_tensor(assets["target_probability"]).to(device)
    target_mask = runtime.load_tensor(assets["target_mask"]).to(device).bool()
    roi = runtime.load_tensor(assets["target_roi_mask"]).to(device).bool()

    def endpoint(state: torch.Tensor):
        loss, diagnostics = probability_volume_loss(
            state,
            model.embedding.weight,
            target_probability,
            roi,
            target_label=int(config["target_label"]),
            tau=float(config["endpoint_tau"]),
            bce_weight=float(config["bce_weight"]),
            dice_weight=float(config["dice_weight"]),
            spatial_gradient_weight=0.0,
            loss_mode=str(config["probability_loss_mode"]),
            target_mask=target_mask,
        )
        diagnostics = dict(diagnostics)
        diagnostics["soft_target_probability_mean"] = diagnostics[
            "roi_target_probability_mean"
        ]
        return loss, diagnostics

    return endpoint, None, {
        "target_probability_tensor_sha256": tensor_sha256(target_probability),
        "target_mask_tensor_sha256": tensor_sha256(target_mask),
        "target_roi_mask_tensor_sha256": tensor_sha256(roi),
        "endpoint_tau": float(config["endpoint_tau"]),
    }


def _property_objective(
    config: Mapping[str, object],
    assets: Mapping[str, Path],
    model,
    device: torch.device,
) -> tuple[Callable, Callable | None, dict[str, object]]:
    table = runtime.load_tensor(assets["property_table"]).to(device)
    target = runtime.load_tensor(assets["target_properties"]).to(device)
    confidence = runtime.load_tensor(assets["property_confidence"]).to(device)
    channel_weights = torch.tensor(
        config["property_channel_weights"], device=device, dtype=table.dtype
    )

    def endpoint(state: torch.Tensor):
        return property_volume_loss(
            state,
            model.embedding.weight,
            target,
            table,
            confidence,
            tau=float(config["endpoint_tau"]),
            sigmas=[float(v) for v in config["property_sigmas"]],
            scale_weights=[float(v) for v in config["property_scale_weights"]],
            channel_weights=channel_weights,
        )

    return endpoint, None, {
        "property_table_tensor_sha256": tensor_sha256(table),
        "target_properties_tensor_sha256": tensor_sha256(target),
        "property_confidence_tensor_sha256": tensor_sha256(confidence),
        "endpoint_tau": float(config["endpoint_tau"]),
    }


def _seismic_objective(
    config: Mapping[str, object],
    assets: Mapping[str, Path],
    model,
    device: torch.device,
) -> tuple[Callable, Callable, dict[str, object]]:
    observed = runtime.load_tensor(assets["observed_seismic"]).to(
        device=device, dtype=torch.float32
    )
    subsurface = normalize_volume(
        runtime.load_tensor(assets["subsurface_mask"]), "subsurface_mask"
    ).to(device).bool()
    well_values = normalize_volume(
        runtime.load_tensor(assets["binary_well_values"]), "binary_well_values"
    ).to(device=device, dtype=torch.float32)
    well_mask = normalize_volume(
        runtime.load_tensor(assets["binary_well_mask"]), "binary_well_mask"
    ).to(device).bool()
    binary_config = read_json(assets["binary_acoustic_config"])
    source_record = binary_config.get("source_acoustic_config")
    if not isinstance(source_record, Mapping):
        raise TypeError("binary acoustic config is missing its frozen source record")
    source_path = _resolved_path(source_record["path"])
    if runtime.file_sha256(source_path) != str(source_record["sha256"]):
        raise ValueError("Stage15 source acoustic config changed")
    properties = binary_acoustic_properties_from_configs(
        binary_config, read_json(source_path)
    )
    operator, operator_metadata = seismic_operator_from_config(
        read_json(assets["seismic_config"]), grid_shape=(64, 64, 64)
    )
    target_category = int(config["target_label"]) + 1

    def seismic_from_occupancy(occupancy: torch.Tensor) -> torch.Tensor:
        exact = torch.where(well_mask, well_values.to(occupancy), occupancy)
        impedance, slowness = binary_occupancy_to_acoustic(
            exact,
            subsurface,
            properties,
        )
        return operator(impedance, slowness, subsurface)

    def endpoint(state: torch.Tensor):
        probabilities = soft_decode_to_probs(
            state,
            model.embedding.weight,
            tau=float(config["endpoint_tau"]),
        )
        occupancy = probabilities[:, target_category : target_category + 1]
        synthetic = seismic_from_occupancy(occupancy)
        loss = (synthetic - observed).square().mean()
        free = subsurface & ~well_mask
        return loss, {
            "soft_seismic_loss": loss,
            "soft_target_probability_mean": occupancy[free].mean(),
        }

    def diagnostics(state: torch.Tensor):
        decoded = (model.decode(state.detach()) - 1).unsqueeze(1)
        hard_occupancy = (decoded == int(config["target_label"])).to(state.dtype)
        hard_synthetic = seismic_from_occupancy(hard_occupancy)
        hard_loss = (hard_synthetic - observed).square().mean()
        return {"hard_seismic_loss": hard_loss}

    return endpoint, diagnostics, {
        "observed_seismic_tensor_sha256": tensor_sha256(observed),
        "subsurface_mask_tensor_sha256": tensor_sha256(subsurface),
        "binary_well_values_tensor_sha256": tensor_sha256(well_values),
        "binary_well_mask_tensor_sha256": tensor_sha256(well_mask),
        "source_acoustic_config": runtime.asset_record(source_path),
        "seismic_operator": operator_metadata,
        "seismic_loss": "raw_mean_squared_seismic_misfit",
        "soft_mapping": "soft_categorical_label9_probability_to_stage15_binary_acoustic_v1",
        "straight_through_used": False,
        "truth_loaded_by_runner": False,
    }


def _objective(
    objective: str,
    config: Mapping[str, object],
    assets: Mapping[str, Path],
    model,
    device: torch.device,
):
    builders = {
        "oracle_probability": _oracle_objective,
        "property": _property_objective,
        "seismic": _seismic_objective,
    }
    return builders[objective](config, assets, model, device)


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = read_json(args.config)
    objective, source_seeds = _validate_config(config)
    if args.smoke:
        source_seeds = source_seeds[:1]
    assets = _validated_assets(config, objective)
    device = torch.device(args.device or str(config["device"]))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=assets["checkpoint"],
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device)
    freeze_model_for_dflow(model)
    model_hash_before, tensor_hashes_before = model_state_dict_hashes(model)
    model_report = {
        **model_report,
        "trainable_model_parameter_count_after_dflow_freeze": 0,
        "all_model_parameters_require_grad_false": True,
    }
    (
        condition_values,
        condition_mask,
        embedded_conditions,
        conditioning,
        conditioning_report,
    ) = _conditioning_from_assets(objective, assets, model, device)
    endpoint_loss, trace_diagnostics, objective_report = _objective(
        objective, config, assets, model, device
    )

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "config.json")
    write_json(args.output_dir / "model_load_report.json", model_report)
    write_json(
        args.output_dir / "model_state_tensor_hashes_before.json",
        tensor_hashes_before,
    )
    manifest = base_manifest(
        "stage16_dflow_source_optimization_run_v1", Path(__file__), args.config
    )
    manifest.update(
        {
            "run_status": "running",
            "objective": objective,
            "engine_version": DFLOW_ENGINE_VERSION,
            "smoke_subset": bool(args.smoke),
            "executed_source_seeds": source_seeds,
            "input_assets": {
                name: runtime.asset_record(path) for name, path in assets.items()
            },
            "conditioning_report": conditioning_report,
            "objective_report": objective_report,
            "model_load_report": model_report,
            "model_state_dict_sha256_before": model_hash_before,
            "truth_loaded_by_runner": objective != "seismic",
            "training_performed": False,
            "parameter_sweep_performed": False,
        }
    )
    write_json(args.output_dir / "run_manifest.json", manifest)

    pair_rows: list[dict[str, object]] = []
    all_trace: list[dict[str, object]] = []
    try:
        for sample_id, source_seed in enumerate(source_seeds):
            generator = torch.Generator(device="cpu").manual_seed(int(source_seed))
            initial_cpu = torch.randn(
                1,
                model.embedding_dim,
                *model.data_shape,
                generator=generator,
                dtype=embedded_conditions.dtype,
            )
            initial = initial_cpu.to(device)
            sample_dir = args.output_dir / f"sample_{sample_id:03d}_seed_{source_seed}"
            baseline_dir = sample_dir / "FLOW_ONLY"
            dflow_dir = sample_dir / "DFLOW"
            baseline_dir.mkdir(parents=True)
            dflow_dir.mkdir(parents=True)
            torch.save(initial_cpu, sample_dir / "initial_source.pt")

            baseline_state, baseline_flow_report = differentiable_flow_solve(
                model=model,
                initial_state=initial,
                conditioning=conditioning,
                embedded_conditions=embedded_conditions,
                condition_mask=condition_mask,
                n_steps=int(config["n_steps"]),
                solver=str(config["solver"]),
                use_gradient_checkpointing=bool(config["use_gradient_checkpointing"]),
                condition_values=condition_values,
            )
            baseline_decoded = _decode(model, baseline_state)
            baseline_loss, baseline_objective_diagnostics = endpoint_loss(baseline_state)
            baseline_extra = (
                trace_diagnostics(baseline_state)
                if trace_diagnostics is not None
                else {}
            )

            result = optimize_source_noise(
                model=model,
                initial_source=initial,
                conditioning=conditioning,
                embedded_conditions=embedded_conditions,
                condition_mask=condition_mask,
                endpoint_loss=endpoint_loss,
                n_steps=int(config["n_steps"]),
                solver=str(config["solver"]),
                optimizer_name=str(config["optimizer"]),
                learning_rate=float(config["learning_rate"]),
                optimization_iterations=int(config["optimization_iterations"]),
                lbfgs_max_iter=int(config.get("lbfgs_max_iter", 5)),
                source_regularization_weight=float(config["source_regularization_weight"]),
                use_gradient_checkpointing=bool(config["use_gradient_checkpointing"]),
                condition_values=condition_values,
                trace_diagnostics=trace_diagnostics,
            )
            dflow_decoded = _decode(model, result.final_state)
            condition_cpu = condition_values.detach().cpu().long()
            mask_cpu = condition_mask.detach().cpu().bool()
            baseline_violations = int(((baseline_decoded != condition_cpu) & mask_cpu).sum())
            dflow_violations = int(((dflow_decoded != condition_cpu) & mask_cpu).sum())
            if baseline_violations or dflow_violations:
                raise RuntimeError("paired output violates hard conditions")

            torch.save(baseline_state.detach().cpu(), baseline_dir / "final_state.pt")
            torch.save(baseline_decoded, baseline_dir / "decoded_geology.pt")
            torch.save(result.optimized_source.cpu(), dflow_dir / "optimized_source.pt")
            torch.save(result.final_state.cpu(), dflow_dir / "final_state.pt")
            torch.save(dflow_decoded, dflow_dir / "decoded_geology.pt")
            write_json(dflow_dir / "optimization_diagnostics.json", result.diagnostics)
            sample_trace = []
            for row in result.trace:
                sample_trace.append(
                    {
                        "sample_id": sample_id,
                        "source_seed": int(source_seed),
                        **row,
                    }
                )
            write_csv(dflow_dir / "optimization_trace.csv", sample_trace)
            all_trace.extend(sample_trace)

            baseline_diag_scalars = {}
            for key, value in {**baseline_objective_diagnostics, **baseline_extra}.items():
                if torch.is_tensor(value) and value.numel() == 1:
                    baseline_diag_scalars[key] = float(value.detach().cpu())
            pair_rows.append(
                {
                    "sample_id": sample_id,
                    "source_seed": int(source_seed),
                    "initial_noise_sha256": tensor_sha256(initial_cpu),
                    "dflow_initial_noise_sha256": result.diagnostics["initial_noise_sha256"],
                    "paired_initial_noise_equal": tensor_sha256(initial_cpu)
                    == result.diagnostics["initial_noise_sha256"],
                    "optimized_noise_sha256": result.diagnostics["optimized_noise_sha256"],
                    "baseline_final_state_sha256": tensor_sha256(baseline_state),
                    "dflow_final_state_sha256": result.diagnostics["final_state_sha256"],
                    "baseline_decoded_model_sha256": tensor_sha256(baseline_decoded),
                    "dflow_decoded_model_sha256": tensor_sha256(dflow_decoded),
                    "baseline_endpoint_primary_loss": float(baseline_loss.detach().cpu()),
                    "dflow_endpoint_primary_loss": (
                        result.trace[-1]["endpoint_primary_loss"]
                        if result.trace
                        else float(baseline_loss.detach().cpu())
                    ),
                    "baseline_hard_condition_violations": baseline_violations,
                    "dflow_hard_condition_violations": dflow_violations,
                    "baseline_final_state_path": str(
                        (baseline_dir / "final_state.pt").relative_to(args.output_dir)
                    ),
                    "baseline_decoded_path": str(
                        (baseline_dir / "decoded_geology.pt").relative_to(args.output_dir)
                    ),
                    "dflow_optimized_source_path": str(
                        (dflow_dir / "optimized_source.pt").relative_to(args.output_dir)
                    ),
                    "dflow_final_state_path": str(
                        (dflow_dir / "final_state.pt").relative_to(args.output_dir)
                    ),
                    "dflow_decoded_path": str(
                        (dflow_dir / "decoded_geology.pt").relative_to(args.output_dir)
                    ),
                    "baseline_flow_report": json.dumps(baseline_flow_report, sort_keys=True),
                    **{f"baseline_{key}": value for key, value in baseline_diag_scalars.items()},
                }
            )
            if not pair_rows[-1]["paired_initial_noise_equal"]:
                raise RuntimeError("FLOW_ONLY and DFLOW did not share initial source noise")
            print(
                f"Stage16 {objective} pair {sample_id + 1}/{len(source_seeds)} "
                f"seed={source_seed} completed",
                flush=True,
            )
            del baseline_state, result, initial
            if device.type == "cuda":
                torch.cuda.empty_cache()

        write_csv(args.output_dir / "paired_manifest.csv", pair_rows)
        write_csv(args.output_dir / "optimization_trace.csv", all_trace)
        model_hash_after, tensor_hashes_after = model_state_dict_hashes(model)
        write_json(
            args.output_dir / "model_state_tensor_hashes_after.json",
            tensor_hashes_after,
        )
        if model_hash_before != model_hash_after or tensor_hashes_before != tensor_hashes_after:
            raise RuntimeError("model weights changed across Stage16 run")
        manifest.update(
            {
                "run_status": "completed",
                "pair_count": len(pair_rows),
                "paired_manifest": runtime.asset_record(
                    args.output_dir / "paired_manifest.csv"
                ),
                "optimization_trace": runtime.asset_record(
                    args.output_dir / "optimization_trace.csv"
                ),
                "model_state_dict_sha256_after": model_hash_after,
                "model_state_dict_unchanged": True,
                "trainable_model_parameter_count": 0,
                "optimizer_parameter_scope": "source_state_only",
                "truth_loaded_by_runner": objective != "seismic",
                "seismic_truth_loaded_by_runner": False,
                "results_selected_or_ranked": False,
                "scientific_failure_is_preserved": True,
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
