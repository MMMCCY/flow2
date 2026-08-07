#!/usr/bin/env python3
"""Run Stage6Q D2 gradient, decoder and applied-controller audits."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import shlex
import socket
import subprocess
import sys

import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.causality_gradient_audit import (
    GRADIENT_AUDIT_VERSION,
    cosine_decode_categories,
    finite_difference_directional_audit,
    update_semantics_row,
)
from guidance.generator_posterior import project_conditions
from guidance.gravity import (
    hard_labels_to_density,
    overwrite_exact_condition_density,
    probabilities_to_density,
    density_table_from_config,
    gravity_operator_from_config,
)
from guidance.probability_sampling import build_probability_guidance_velocity
from guidance.property_volume import (
    gaussian_blur_property_channels,
    hard_labels_to_properties,
    probabilities_to_expected_properties,
)
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import (
    AnalyticObservationSuite,
    build_simple_causal_case,
    build_voxel_search_mask,
)
from guided_geophysical_sampling import clip_gradient_by_norm, soft_decode_to_probs
from scripts.stage6.run_embedding_endpoint_causality import _checkpoint_embedding
from scripts.stage6.run_simple_causality import (
    _file_sha256,
    _read_json,
    _resolve_repo_path,
    _tensor_sha256,
    _write_json,
    _write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage6Q D2 gradient audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY_ROOT, text=True).strip()


def _mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right.to(device=left.device, dtype=left.dtype)).square().mean()


def _cosine(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-12) -> float:
    a = left.detach().double().reshape(-1)
    b = right.detach().double().reshape(-1)
    denominator = torch.linalg.vector_norm(a) * torch.linalg.vector_norm(b)
    return float(torch.dot(a, b) / denominator.clamp_min(eps))


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 6Q D2 gradient and controller semantics report",
        "",
        f"Gradient verdict: **{'PASS' if summary['gradient_correctness_pass'] else 'FAIL'}**",
        f"Decoder/mapping verdict: **{'PASS' if summary['decoder_mapping_pass'] else 'FAIL'}**",
        f"Applied-controller verdict: **{summary['controller_verdict']}**",
        "",
        "| Chain | Best eps | Best relative error | -grad descends |",
        "|---|---:|---:|---:|",
    ]
    for row in summary["chain_summaries"]:
        lines.append(
            f"| {row['chain']} | {row['best_epsilon']:.1e} | "
            f"{row['best_relative_error']:.6g} | {row['negative_gradient_local_descent']} |"
        )
    lines.extend(
        [
            "",
            "## Applied updates",
            "",
            "| Update | Norm | Soft delta | Hard delta | Truth-direction fraction |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary["applied_update_rows"]:
        lines.append(
            f"| {row['update']} | {row['update_norm']:.6g} | "
            f"{row['soft_loss_delta']:.6g} | "
            f"{row['hard_loss_after'] - row['hard_loss_before']:.6g} | "
            f"{row['truth_direction_response_fraction']:.6g} |"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    config_path = args.config.resolve()
    config = _read_json(config_path)
    if config.get("schema") != "phase6q_gradient_audit_config_v1":
        raise ValueError("unexpected gradient audit schema")
    if bool(config.get("formal_training_authorized")):
        raise ValueError("D2 forbids training")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    base_path = _resolve_repo_path(config.get("base_config"), "base_config")
    voxel_path = _resolve_repo_path(config.get("voxel_config"), "voxel_config")
    checkpoint_path = _resolve_repo_path(config.get("checkpoint"), "checkpoint")
    if _file_sha256(checkpoint_path) != str(config.get("checkpoint_sha256")):
        raise ValueError("checkpoint hash mismatch")
    base_config = _read_json(base_path)
    voxel_config = _read_json(voxel_path)
    case = build_simple_causal_case(base_config)
    search_mask, search_report = build_voxel_search_mask(case, voxel_config)

    embedding64 = _checkpoint_embedding(
        checkpoint_path, str(config["checkpoint_embedding_key"])
    ).to(device=device, dtype=torch.float64)
    acoustic_path = _resolve_repo_path(base_config.get("acoustic_config"), "acoustic_config")
    seismic_path = _resolve_repo_path(base_config.get("seismic_config"), "seismic_config")
    density_path = _resolve_repo_path(base_config.get("density_config"), "density_config")
    gravity_path = _resolve_repo_path(base_config.get("gravity_config"), "gravity_config")
    acoustic_tables, acoustic_metadata = acoustic_tables_from_config(
        _read_json(acoustic_path), embedding64.shape[0]
    )
    density_table, density_metadata = density_table_from_config(
        _read_json(density_path), embedding64.shape[0]
    )
    seismic_operator, seismic_metadata = seismic_operator_from_config(
        _read_json(seismic_path), grid_shape=case.truth_labels.shape[2:]
    )
    gravity_operator, gravity_metadata = gravity_operator_from_config(
        _read_json(gravity_path), grid_shape=case.truth_labels.shape[2:]
    )
    acoustic64 = acoustic_tables.property_table.to(device=device, dtype=torch.float64)
    density64 = density_table.to(device=device, dtype=torch.float64)
    suite = AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic64,
        density_table=density64,
        seismic_operator=seismic_operator,
        gravity_operator=gravity_operator,
        blur_sigma_voxels=float(base_config["blur_sigma_voxels"]),
    )
    truth_labels = case.truth_labels.to(device=device)
    baseline_labels = case.baseline_labels.to(device=device)
    condition_mask = case.condition_mask.to(device=device)
    subsurface_mask = case.subsurface_mask.to(device=device)
    truth_categories = truth_labels.long()[:, 0] + 1
    baseline_categories = baseline_labels.long()[:, 0] + 1
    truth_state64 = embedding64[truth_categories].permute(0, 4, 1, 2, 3).contiguous()
    baseline_state64 = embedding64[baseline_categories].permute(0, 4, 1, 2, 3).contiguous()
    target_acoustic64 = hard_labels_to_acoustic(truth_labels, acoustic64).to(torch.float64)
    target_density64 = hard_labels_to_density(truth_labels, density64).to(torch.float64)
    truth_occupancy64 = (truth_labels == case.target_label).to(torch.float64)
    truth_blurred64 = gaussian_blur_property_channels(
        truth_occupancy64, float(base_config["blur_sigma_voxels"])
    )
    truth_reflectivity64 = seismic_operator.reflectivity_spikes(
        target_acoustic64[:, 0:1], target_acoustic64[:, 1:2], subsurface_mask
    ).detach()
    truth_seismic64 = seismic_operator(
        target_acoustic64[:, 0:1], target_acoustic64[:, 1:2], subsurface_mask
    ).detach()
    truth_gravity64 = gravity_operator(target_density64).detach()
    tau = float(config["temperature"])

    def probabilities(state: torch.Tensor) -> torch.Tensor:
        return soft_decode_to_probs(state, embedding64, tau=tau)

    def predicted_acoustic(state: torch.Tensor) -> torch.Tensor:
        predicted = probabilities_to_subsurface_acoustic(
            probabilities(state), acoustic64, subsurface_mask
        )
        return overwrite_exact_condition_acoustic(predicted, target_acoustic64, condition_mask)

    def acoustic_objective(state: torch.Tensor) -> torch.Tensor:
        return _mse(predicted_acoustic(state), target_acoustic64)

    def blurred_objective(state: torch.Tensor) -> torch.Tensor:
        occupancy = probabilities(state)[:, case.target_label + 1 : case.target_label + 2]
        return _mse(
            gaussian_blur_property_channels(occupancy, float(base_config["blur_sigma_voxels"])),
            truth_blurred64,
        )

    def reflectivity_response(state: torch.Tensor) -> torch.Tensor:
        acoustic = predicted_acoustic(state)
        return seismic_operator.reflectivity_spikes(
            acoustic[:, 0:1], acoustic[:, 1:2], subsurface_mask
        )

    def reflectivity_objective(state: torch.Tensor) -> torch.Tensor:
        return _mse(reflectivity_response(state), truth_reflectivity64)

    def seismic_response(state: torch.Tensor) -> torch.Tensor:
        acoustic = predicted_acoustic(state)
        return seismic_operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface_mask)

    def seismic_objective(state: torch.Tensor) -> torch.Tensor:
        return _mse(seismic_response(state), truth_seismic64)

    def gravity_response(state: torch.Tensor) -> torch.Tensor:
        density = probabilities_to_density(probabilities(state), density64)
        known = overwrite_exact_condition_density(density, target_density64, condition_mask)
        return gravity_operator(known)

    def gravity_objective(state: torch.Tensor) -> torch.Tensor:
        return _mse(gravity_response(state), truth_gravity64)

    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    direction_cpu = torch.randn(
        baseline_state64.shape, generator=generator, dtype=torch.float64
    )
    direction = direction_cpu.to(device)
    direction *= search_mask.to(device=device, dtype=torch.float64).expand_as(direction)
    chain_objectives = {
        "embedding_to_acoustic_property": acoustic_objective,
        "explicit_blurred_property": blurred_objective,
        "property_to_reflectivity_and_twt_deposition": reflectivity_objective,
        "wavelet_convolved_seismic_loss": seismic_objective,
        "gravity_forward_and_loss": gravity_objective,
    }
    chain_summaries: list[dict[str, object]] = []
    finite_rows: list[dict[str, object]] = []
    gradients: dict[str, torch.Tensor] = {}
    for name, objective in chain_objectives.items():
        chain_summary, rows, gradient = finite_difference_directional_audit(
            name=name,
            objective=objective,
            state=baseline_state64,
            direction=direction,
            epsilons=[float(value) for value in config["epsilons"]],
            negative_gradient_step_norm=float(config["negative_gradient_step_norm"]),
        )
        chain_summaries.append(chain_summary)
        finite_rows.extend(rows)
        gradients[name] = gradient

    # Decoder equivalence across temperatures and representative continuous states.
    from model_train_sh_inference_cond import Geo3DStochInterp

    model, model_load_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=checkpoint_path,
        map_location=device,
        weight_source="ema",
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    embedding32 = model.embedding.weight.detach()
    baseline_state32 = model.embed(baseline_labels.to(device)).detach()
    continuous_generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]) + 1)
    continuous_noise = torch.randn(
        baseline_state32.shape, generator=continuous_generator, dtype=torch.float32
    ).to(device)
    representative_states = {
        "exact_baseline": baseline_state32,
        "continuous_perturbed": baseline_state32 + 0.05 * continuous_noise,
    }
    decoder_rows: list[dict[str, object]] = []
    for state_name, state_value in representative_states.items():
        model_categories = model.decode(state_value)
        for temperature in config["decoder_temperatures"]:
            soft_categories = soft_decode_to_probs(
                state_value, embedding32, tau=float(temperature)
            ).argmax(dim=1)
            decoder_rows.append(
                {
                    "state": state_name,
                    "temperature": float(temperature),
                    "mismatch_count": int((model_categories != soft_categories).sum()),
                }
            )
    tie_state = (embedding32[1] + embedding32[2]).reshape(1, -1, 1, 1, 1)
    tie_logits = F.normalize(tie_state, dim=1).flatten() @ F.normalize(embedding32, dim=1).T
    tie_max = tie_logits.max()
    tie_categories = torch.nonzero(
        torch.isclose(tie_logits, tie_max, rtol=1e-6, atol=1e-7), as_tuple=False
    ).flatten().tolist()
    tie_report = {
        "categories_at_max_with_tolerance": tie_categories,
        "model_argmax_category": int(model.decode(tie_state).item()),
        "exact_tie_reported_separately": len(tie_categories) > 1,
    }

    all_labels = torch.arange(-1, embedding32.shape[0] - 1, device=device).reshape(1, 1, -1, 1, 1)
    all_one_hot = F.one_hot(
        (all_labels[:, 0] + 1).long(), num_classes=embedding32.shape[0]
    ).permute(0, 4, 1, 2, 3).float()
    density_property_table = density_table.reshape(1, -1)
    hard_property = hard_labels_to_properties(all_labels, density_property_table)
    soft_property = probabilities_to_expected_properties(all_one_hot, density_property_table)
    hard_acoustic = hard_labels_to_acoustic(all_labels, acoustic_tables.property_table)
    soft_acoustic = probabilities_to_acoustic(all_one_hot, acoustic_tables.property_table)
    nonair_labels = torch.arange(0, embedding32.shape[0] - 1, device=device).reshape(1, 1, -1, 1, 1)
    nonair_one_hot = F.one_hot(
        (nonair_labels[:, 0] + 1).long(), num_classes=embedding32.shape[0]
    ).permute(0, 4, 1, 2, 3).float()
    nonair_mask = torch.ones_like(nonair_labels, dtype=torch.bool)
    subsurface_acoustic = probabilities_to_subsurface_acoustic(
        nonair_one_hot, acoustic_tables.property_table, nonair_mask
    )
    nonair_hard_acoustic = hard_labels_to_acoustic(
        nonair_labels, acoustic_tables.property_table
    )
    mixture = torch.zeros((1, embedding32.shape[0], 1, 1, 1), device=device)
    mixture[:, 0] = 0.75
    mixture[:, 1] = 0.25
    rock_conditional = probabilities_to_subsurface_acoustic(
        mixture, acoustic_tables.property_table, torch.ones((1, 1, 1, 1, 1), device=device, dtype=torch.bool)
    )
    expected_background = acoustic_tables.property_table[:, 1].to(device).reshape(1, 2, 1, 1, 1)
    mapping_report = {
        "decoder_rows": decoder_rows,
        "decoder_total_mismatches": sum(int(row["mismatch_count"]) for row in decoder_rows),
        "tie": tie_report,
        "one_hot_property_max_abs_difference": float((hard_property - soft_property).abs().max()),
        "one_hot_acoustic_max_abs_difference": float((hard_acoustic - soft_acoustic).abs().max()),
        "subsurface_nonair_one_hot_max_abs_difference": float(
            (subsurface_acoustic - nonair_hard_acoustic).abs().max()
        ),
        "subsurface_air_exclusion_background_max_abs_difference": float(
            (rock_conditional - expected_background).abs().max()
        ),
    }
    decoder_mapping_pass = (
        mapping_report["decoder_total_mismatches"] == 0
        and mapping_report["one_hot_property_max_abs_difference"] == 0.0
        and mapping_report["one_hot_acoustic_max_abs_difference"] == 0.0
        and mapping_report["subsurface_nonair_one_hot_max_abs_difference"] == 0.0
        and mapping_report["subsurface_air_exclusion_background_max_abs_difference"] == 0.0
    )

    # Production-dtype raw gradient versus the actual controller transform.
    acoustic32 = acoustic_tables.property_table.to(device=device, dtype=torch.float32)
    target_acoustic32 = hard_labels_to_acoustic(truth_labels, acoustic32)
    truth_seismic32 = seismic_operator(
        target_acoustic32[:, 0:1], target_acoustic32[:, 1:2], subsurface_mask
    ).detach()

    def probabilities32(state: torch.Tensor) -> torch.Tensor:
        return soft_decode_to_probs(state, embedding32, tau=tau)

    def response32(state: torch.Tensor) -> torch.Tensor:
        predicted = probabilities_to_subsurface_acoustic(
            probabilities32(state), acoustic32, subsurface_mask
        )
        known = overwrite_exact_condition_acoustic(
            predicted, target_acoustic32, condition_mask
        )
        return seismic_operator(known[:, 0:1], known[:, 1:2], subsurface_mask)

    def objective32(state: torch.Tensor) -> torch.Tensor:
        return _mse(response32(state), truth_seismic32)

    def hard_loss32(state: torch.Tensor) -> float:
        categories = cosine_decode_categories(state, embedding32)
        labels = (categories - 1).unsqueeze(1)
        field = suite.field_from_labels(labels, "seismic")
        return float(_mse(field, truth_seismic64))

    differentiable32 = baseline_state32.detach().requires_grad_(True)
    production_loss = objective32(differentiable32)
    production_gradient = torch.autograd.grad(production_loss, differentiable32)[0]
    used_gradient = clip_gradient_by_norm(
        production_gradient, float(config["controller"]["gradient_clip_norm"])
    )
    embedded_truth32 = model.embed(truth_labels).detach()
    conditioning32 = embedded_truth32 * condition_mask.expand_as(embedded_truth32)
    t_value = float(config["controller"]["t"])
    time_tensor = torch.full((1,), t_value, device=device, dtype=torch.float32)
    with torch.no_grad():
        base_velocity = model.net(baseline_state32, conditioning32, time_tensor)
    guidance_velocity, controller_diagnostics, _ = build_probability_guidance_velocity(
        used_gradient,
        base_velocity,
        requested_ratio=float(config["controller"]["requested_ratio"]),
        max_ratio=float(config["controller"]["max_ratio"]),
        scaling_mode=str(config["controller"]["scaling_mode"]),
        reference_gradient_norm=None,
    )
    applied_physics_velocity = -guidance_velocity
    dt = 1.0 / int(config["controller"]["n_steps"])
    gradient_norm32 = torch.linalg.vector_norm(production_gradient)
    updates = {
        "raw_negative_gradient_small_step": -float(config["raw_gradient_learning_rate"]) * production_gradient,
        "normalized_negative_gradient_small_step": -float(config["normalized_gradient_step_norm"]) * production_gradient / gradient_norm32.clamp_min(1e-12),
        "actual_controller_velocity_unit_time": applied_physics_velocity,
        "actual_euler_applied_physics_update": dt * applied_physics_velocity,
    }
    baseline_soft_response32 = response32(baseline_state32).detach()
    applied_rows = [
        update_semantics_row(
            name=name,
            state=baseline_state32,
            update=update,
            objective=objective32,
            soft_response=response32,
            hard_loss=hard_loss32,
            baseline_response=baseline_soft_response32,
            truth_response=truth_seismic32,
        )
        for name, update in updates.items()
    ]
    projected_update = project_conditions(
        baseline_state32 + updates["actual_euler_applied_physics_update"],
        embedded_truth32,
        condition_mask,
    ) - baseline_state32
    applied_rows.append(
        update_semantics_row(
            name="actual_euler_update_after_condition_projection",
            state=baseline_state32,
            update=projected_update,
            objective=objective32,
            soft_response=response32,
            hard_loss=hard_loss32,
            baseline_response=baseline_soft_response32,
            truth_response=truth_seismic32,
        )
    )
    raw_negative = -production_gradient
    update_cosine = _cosine(raw_negative, applied_physics_velocity)
    raw_improves = bool(applied_rows[0]["soft_loss_improved"])
    actual_euler_improves = bool(applied_rows[3]["soft_loss_improved"])
    if raw_improves and actual_euler_improves and update_cosine > 0.999:
        controller_verdict = "PASS_SIGN_AND_LOCAL_DESCENT"
    elif raw_improves and not actual_euler_improves:
        controller_verdict = "FAIL_CONTROLLER_TRANSFORM_OR_STEP_SCALE"
    else:
        controller_verdict = "FAIL_RAW_GRADIENT_PRODUCTION_DTYPE"

    gradient_correctness_pass = all(
        bool(row["all_sign_match"])
        and bool(row["negative_gradient_local_descent"])
        and float(row["best_relative_error"]) <= 5e-4
        for row in chain_summaries
    )
    base_gradients_absent = all(parameter.grad is None for parameter in model.parameters())
    git_status = _git("status", "--short")
    summary = {
        "stage": "phase6q_d2_gradient_controller_audit",
        "version": GRADIENT_AUDIT_VERSION,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "gradient_correctness_pass": gradient_correctness_pass,
        "decoder_mapping_pass": decoder_mapping_pass,
        "controller_verdict": controller_verdict,
        "controller_pass": controller_verdict == "PASS_SIGN_AND_LOCAL_DESCENT",
        "chain_summaries": chain_summaries,
        "decoder_mapping": mapping_report,
        "applied_update_rows": applied_rows,
        "production_raw_gradient_norm": float(gradient_norm32),
        "production_normalized_gradient_norm": float(torch.linalg.vector_norm(production_gradient / gradient_norm32.clamp_min(1e-12))),
        "controller_velocity_norm": float(torch.linalg.vector_norm(guidance_velocity)),
        "dt_times_controller_norm": float(torch.linalg.vector_norm(dt * guidance_velocity)),
        "cosine_raw_negative_gradient_actual_applied_update": update_cosine,
        "controller_diagnostics": controller_diagnostics,
        "base_model_gradients_absent": base_gradients_absent,
        "search_report": search_report,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": "clean" if not git_status else "dirty",
        "git_status_short": git_status.splitlines(),
        "seed": int(config["seed"]),
        "device": str(device),
        "finite_difference_dtype": "torch.float64",
        "production_dtype": "torch.float32",
        "dtype": "torch.float64 finite-difference; torch.float32 production controller",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_used_for_flow": True,
        "flow_unet_loaded": True,
        "flow_unet_evaluated_only_for_controller_reference_velocity": True,
        "ema_raw_policy": model_load_report,
        "acoustic_table_sha256": _tensor_sha256(acoustic_tables.property_table),
        "property_table_sha256": _tensor_sha256(density_table),
        "wavelet_sha256": seismic_operator.metadata()["wavelet"]["sha256"],
        "observation_sha256": _tensor_sha256(truth_seismic64),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "audit": _file_sha256(PROJECT_DIR / "guidance/causality_gradient_audit.py"),
            "decoder": _file_sha256(PROJECT_DIR / "guided_geophysical_sampling.py"),
            "controller": _file_sha256(PROJECT_DIR / "guidance/probability_sampling.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
            "gravity": _file_sha256(PROJECT_DIR / "guidance/gravity.py"),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "platform": platform.platform(),
            "python": sys.version,
            "torch": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "acoustic_metadata": acoustic_metadata,
        "density_metadata": density_metadata,
        "seismic_metadata": seismic_metadata,
        "gravity_metadata": gravity_metadata,
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config_input.json", config)
    _write_json(
        output_dir / "config_resolved.json",
        {
            **config,
            "device": str(device),
            "git_sha": summary["git_sha"],
            "base_config_sha256": _file_sha256(base_path),
            "voxel_config_sha256": _file_sha256(voxel_path),
        },
    )
    _write_rows(output_dir / "finite_difference.csv", finite_rows)
    _write_rows(output_dir / "applied_updates.csv", applied_rows)
    _write_rows(output_dir / "decoder_equivalence.csv", decoder_rows)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(
        json.dumps(
            {
                "gradient_correctness_pass": gradient_correctness_pass,
                "decoder_mapping_pass": decoder_mapping_pass,
                "controller_verdict": controller_verdict,
                "output_dir": str(output_dir),
            },
            indent=2,
        )
    )
    if not (gradient_correctness_pass and decoder_mapping_pass and summary["controller_pass"]):
        raise SystemExit(2)


if __name__ == "__main__":
    main()
