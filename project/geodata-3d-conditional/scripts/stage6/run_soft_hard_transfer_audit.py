#!/usr/bin/env python3
"""Run Stage6Q D3 generator-free soft/hard transfer localization."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
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

from guidance.causality_gradient_audit import cosine_decode_categories
from guidance.generator_posterior import project_conditions
from guidance.gravity import (
    density_table_from_config,
    hard_labels_to_density,
    gravity_operator_from_config,
    overwrite_exact_condition_density,
    probabilities_to_density,
)
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import (
    AnalyticObservationSuite,
    build_simple_causal_case,
    build_voxel_search_mask,
    controlled_observation,
)
from guidance.soft_hard_transfer_audit import (
    SOFT_HARD_TRANSFER_VERSION,
    decision_boundary_statistics,
    paired_attainment,
    projection_erasure_fraction,
    response_distance_geometry,
    spatial_energy_fractions,
)
from guided_geophysical_sampling import soft_decode_to_probs
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
    parser = argparse.ArgumentParser(description="Run Stage6Q D3 soft/hard transfer audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY_ROOT, text=True).strip()


def _mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right.to(device=left.device, dtype=left.dtype)).square().mean()


def _temperature_values(config: dict[str, object]) -> list[float]:
    values: list[float] = []
    for segment in config["temperature_schedule"]:
        values.extend([float(segment["temperature"])] * int(segment["steps"]))
    if len(values) != int(config["updates"]):
        raise ValueError("temperature schedule must match updates")
    return values


def _report(summary: dict[str, object]) -> str:
    lines = [
        "# Phase 6Q D3 soft/hard transfer report",
        "",
        "U-Net was not loaded. Truth geology was used only for post-update mechanism diagnostics.",
        "",
        "| Level | Control | Max/final soft attainment | Max/final hard attainment | Final soft/hard closer | Best hard step |",
        "|---|---|---:|---:|---|---:|",
    ]
    for run in summary["runs"]:
        lines.append(
            f"| {run['level']} | {run['control']} | "
            f"{run['maximum_soft_attainment']:.4f}/{run['final_soft_attainment']:.4f} | "
            f"{run['maximum_hard_attainment']:.4f}/{run['final_hard_attainment']:.4f} | "
            f"{run['final_soft_closer_to']}/{run['final_hard_closer_to']} | {run['best_hard_step']} |"
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
    if config.get("schema") != "phase6q_soft_hard_transfer_config_v1":
        raise ValueError("unexpected D3 config schema")
    if bool(config.get("flow_unet_loaded")) or bool(config.get("checkpoint_used_for_flow")):
        raise ValueError("D3 forbids the flow U-Net")
    if bool(config.get("formal_training_authorized")):
        raise ValueError("D3 forbids training")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    temperatures = _temperature_values(config)

    base_path = _resolve_repo_path(config.get("base_config"), "base_config")
    voxel_path = _resolve_repo_path(config.get("voxel_config"), "voxel_config")
    checkpoint_path = _resolve_repo_path(config.get("checkpoint"), "checkpoint")
    if _file_sha256(checkpoint_path) != str(config["checkpoint_sha256"]):
        raise ValueError("checkpoint hash mismatch")
    base_config = _read_json(base_path)
    voxel_config = _read_json(voxel_path)
    case = build_simple_causal_case(base_config)
    search_mask_cpu, search_report = build_voxel_search_mask(case, voxel_config)
    search = search_mask_cpu.to(device=device, dtype=torch.bool)
    embedding = _checkpoint_embedding(
        checkpoint_path, str(config["checkpoint_embedding_key"])
    ).to(device)
    acoustic_path = _resolve_repo_path(base_config.get("acoustic_config"), "acoustic_config")
    seismic_path = _resolve_repo_path(base_config.get("seismic_config"), "seismic_config")
    density_path = _resolve_repo_path(base_config.get("density_config"), "density_config")
    gravity_path = _resolve_repo_path(base_config.get("gravity_config"), "gravity_config")
    acoustic_tables, acoustic_metadata = acoustic_tables_from_config(
        _read_json(acoustic_path), embedding.shape[0]
    )
    density_table, density_metadata = density_table_from_config(
        _read_json(density_path), embedding.shape[0]
    )
    acoustic_table = acoustic_tables.property_table.to(device)
    density_table_device = density_table.to(device)
    seismic_operator, seismic_metadata = seismic_operator_from_config(
        _read_json(seismic_path), grid_shape=case.truth_labels.shape[2:]
    )
    gravity_operator, gravity_metadata = gravity_operator_from_config(
        _read_json(gravity_path), grid_shape=case.truth_labels.shape[2:]
    )
    suite = AnalyticObservationSuite(
        case,
        acoustic_property_table=acoustic_table,
        density_table=density_table_device,
        seismic_operator=seismic_operator,
        gravity_operator=gravity_operator,
        blur_sigma_voxels=float(base_config["blur_sigma_voxels"]),
    )
    truth_labels = case.truth_labels.to(device)
    baseline_labels = case.baseline_labels.to(device)
    truth_categories = truth_labels.long()[:, 0] + 1
    baseline_categories = baseline_labels.long()[:, 0] + 1
    truth_state = embedding[truth_categories].permute(0, 4, 1, 2, 3).contiguous()
    baseline_state = embedding[baseline_categories].permute(0, 4, 1, 2, 3).contiguous()
    initial_state_sha256 = _tensor_sha256(baseline_state)
    condition = case.condition_mask.to(device)
    subsurface = case.subsurface_mask.to(device)
    target_acoustic = hard_labels_to_acoustic(truth_labels, acoustic_table)
    target_density = hard_labels_to_density(truth_labels, density_table_device)
    target_category = case.target_label + 1
    target_occupancy = (truth_labels == case.target_label).float()
    fixed_target = case.fixed_target_mask.to(device)
    hidden = (truth_labels == case.target_label) & ~fixed_target
    all_target = truth_labels == case.target_label
    regions = {
        "hidden": hidden,
        "fixed_drilled": fixed_target,
        "condition": condition,
        "unconstrained_background": search & (truth_labels == case.background_label),
        "outside_all_target_rois": search & ~all_target,
    }

    def probabilities(state: torch.Tensor, tau: float) -> torch.Tensor:
        return soft_decode_to_probs(state, embedding, tau=tau)

    def acoustic_from_probabilities(values: torch.Tensor) -> torch.Tensor:
        predicted = probabilities_to_subsurface_acoustic(values, acoustic_table, subsurface)
        return overwrite_exact_condition_acoustic(predicted, target_acoustic, condition)

    def soft_response(level: str, state: torch.Tensor, tau: float) -> torch.Tensor:
        probs = probabilities(state, tau)
        if level == "probability":
            return probs[:, target_category : target_category + 1]
        if level == "expected_property":
            return acoustic_from_probabilities(probs)
        if level == "blurred_property":
            return suite.field_from_probabilities(probs, "blurred_property")
        if level in {"reflectivity_spikes", "seismic"}:
            acoustic = acoustic_from_probabilities(probs)
            if level == "reflectivity_spikes":
                return seismic_operator.reflectivity_spikes(
                    acoustic[:, 0:1], acoustic[:, 1:2], subsurface
                )
            return seismic_operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)
        if level == "gravity":
            density = probabilities_to_density(probs, density_table_device)
            known = overwrite_exact_condition_density(density, target_density, condition)
            return gravity_operator(known)
        raise ValueError(level)

    def hard_response(level: str, labels: torch.Tensor) -> torch.Tensor:
        if level == "probability":
            return (labels == case.target_label).float()
        if level == "expected_property":
            return hard_labels_to_acoustic(labels, acoustic_table)
        return suite.field_from_labels(labels, level)

    truth_responses = {
        level: (
            target_occupancy
            if level == "probability"
            else hard_response(level, truth_labels)
        ).detach()
        for level in config["operator_ladder"]
    }
    baseline_hard_responses = {
        level: hard_response(level, baseline_labels).detach()
        for level in config["operator_ladder"]
    }

    def seismic_diagnostics(state: torch.Tensor, tau: float) -> dict[str, object]:
        acoustic = acoustic_from_probabilities(probabilities(state, tau))
        predicted_r, predicted_t, valid = seismic_operator.interface_response(
            acoustic[:, 0:1], acoustic[:, 1:2], subsurface
        )
        truth_r, truth_t, truth_valid = seismic_operator.interface_response(
            target_acoustic[:, 0:1], target_acoustic[:, 1:2], subsurface
        )
        position = predicted_t / seismic_operator.sample_interval_ms
        out = valid & ((position < 0) | (position > seismic_operator.num_time_samples - 1))
        inside = valid & ~out
        fraction = position - torch.floor(position)
        boundary_distance = torch.minimum(fraction, 1.0 - fraction)
        spikes = seismic_operator.deposit_reflectivity(predicted_r, predicted_t, valid)
        truth_spikes = seismic_operator.deposit_reflectivity(truth_r, truth_t, truth_valid)
        amplitudes = seismic_operator.convolve_reflectivity_spikes(spikes)
        truth_amplitudes = seismic_operator.convolve_reflectivity_spikes(truth_spikes)
        return {
            "valid_interface_count": int(valid.sum()),
            "in_recording_window_interface_count": int(inside.sum()),
            "out_of_window_interface_count": int(out.sum()),
            "cropped_interface_fraction": float(out.sum() / valid.sum().clamp_min(1)),
            "time_sample_boundary_distance_mean": float(boundary_distance[valid].mean()),
            "reflectivity_rmse": float(_mse(predicted_r[valid], truth_r[truth_valid]).sqrt()),
            "arrival_twt_rmse_ms": float(_mse(predicted_t[valid], truth_t[truth_valid]).sqrt()),
            "deposited_spike_rmse": float(_mse(spikes, truth_spikes).sqrt()),
            "wavelet_amplitude_rmse": float(_mse(amplitudes, truth_amplitudes).sqrt()),
        }

    run_summaries: list[dict[str, object]] = []
    all_trace: list[dict[str, object]] = []
    pairing_hashes: dict[str, str] = {}
    for level in config["operator_ladder"]:
        controls = config["controls"] if level in config["controlled_levels"] else ["correct"]
        for control_name in controls:
            pairing_hashes[f"{level}/{control_name}"] = initial_state_sha256
            observation = controlled_observation(
                truth_responses[level], control_name, shuffle_seed=int(base_config["shuffle_seed"])
            ).detach()
            state = torch.nn.Parameter(baseline_state.clone())
            optimizer = torch.optim.Adam(
                [state], lr=float(config["learning_rate"]), weight_decay=float(config["weight_decay"])
            )
            best_hard_loss = float(_mse(baseline_hard_responses[level], observation))
            best_hard_step = 0
            best_state = baseline_state.detach().cpu().clone()
            rows: list[dict[str, object]] = []
            for step, tau in enumerate(temperatures, start=1):
                optimizer.zero_grad(set_to_none=True)
                before_state = state.detach().clone()
                before_probs = probabilities(before_state, tau).detach()
                before_categories = cosine_decode_categories(before_state, embedding)
                predicted = soft_response(level, state, tau)
                loss = _mse(predicted, observation)
                loss.backward()
                if state.grad is None or not torch.isfinite(state.grad).all():
                    raise FloatingPointError("D3 gradient invalid")
                state.grad.masked_fill_(~search.expand_as(state.grad), 0.0)
                raw_gradient = state.grad.detach().clone()
                gradient_norm = torch.nn.utils.clip_grad_norm_([state], float(config["gradient_clip_norm"]))
                optimizer.step()
                pre_projection_state = state.detach().clone()
                projected_state = project_conditions(pre_projection_state, truth_state, condition)
                with torch.no_grad():
                    state.copy_(projected_state)
                update = projected_state - before_state
                after_probs = probabilities(projected_state, tau).detach()
                similarities_before = torch.einsum(
                    "bexyz,ce->bcxyz",
                    F.normalize(before_state, dim=1),
                    F.normalize(embedding, dim=1),
                )
                after_categories = cosine_decode_categories(projected_state, embedding)
                after_labels = (after_categories - 1).unsqueeze(1)
                soft_after = soft_response(level, projected_state, tau).detach()
                hard_after = hard_response(level, after_labels).detach()
                soft_baseline = soft_response(level, baseline_state, tau).detach()
                hard_baseline = baseline_hard_responses[level]
                truth_response = truth_responses[level]
                soft_baseline_loss = float(_mse(soft_baseline, observation))
                hard_baseline_loss = float(_mse(hard_baseline, observation))
                soft_truth_loss = float(_mse(truth_response, observation))
                hard_truth_loss = soft_truth_loss
                soft_loss = float(_mse(soft_after, observation))
                hard_loss = float(_mse(hard_after, observation))
                attainment = paired_attainment(
                    soft_baseline_loss=soft_baseline_loss,
                    guided_soft_loss=soft_loss,
                    soft_truth_loss=soft_truth_loss,
                    hard_baseline_loss=hard_baseline_loss,
                    guided_hard_loss=hard_loss,
                    hard_truth_loss=hard_truth_loss,
                )
                soft_geometry = response_distance_geometry(
                    baseline=soft_baseline, guided=soft_after, truth=truth_response
                )
                hard_geometry = response_distance_geometry(
                    baseline=hard_baseline, guided=hard_after, truth=truth_response
                )
                pre_projection_soft_loss = float(
                    _mse(soft_response(level, pre_projection_state, tau), observation)
                )
                pre_projection_categories = cosine_decode_categories(pre_projection_state, embedding)
                pre_projection_labels = (pre_projection_categories - 1).unsqueeze(1)
                pre_projection_hard_loss = float(
                    _mse(hard_response(level, pre_projection_labels), observation)
                )
                oracle_direction = (truth_state - before_state) * search.expand_as(before_state)
                oracle_cosine_denominator = (
                    torch.linalg.vector_norm(raw_gradient)
                    * torch.linalg.vector_norm(oracle_direction)
                )
                oracle_cosine = float(
                    torch.sum(-raw_gradient * oracle_direction)
                    / oracle_cosine_denominator.clamp_min(1e-12)
                )
                boundary = decision_boundary_statistics(
                    probabilities_before=before_probs,
                    probabilities_after=after_probs,
                    similarities_before=similarities_before,
                    categories_before=before_categories,
                    categories_after=after_categories,
                    target_category=target_category,
                    search_mask=search,
                    margin_edges=config["similarity_margin_edges"],
                )
                changed = (after_categories != before_categories) & search[:, 0]
                row: dict[str, object] = {
                    "level": level,
                    "control": control_name,
                    "step": step,
                    "temperature": tau,
                    "soft_loss": soft_loss,
                    "hard_loss": hard_loss,
                    "soft_baseline_loss": soft_baseline_loss,
                    "hard_baseline_loss": hard_baseline_loss,
                    "truth_loss": hard_truth_loss,
                    **attainment,
                    **{f"soft_{key}": value for key, value in soft_geometry.items()},
                    **{f"hard_{key}": value for key, value in hard_geometry.items()},
                    "raw_gradient_norm": float(torch.linalg.vector_norm(raw_gradient)),
                    "gradient_norm_before_clip": float(gradient_norm),
                    "actual_update_norm": float(torch.linalg.vector_norm(update)),
                    "cosine_negative_gradient_oracle_state_direction": oracle_cosine,
                    "directional_derivative_along_oracle": float(torch.sum(raw_gradient * oracle_direction)),
                    "pre_projection_soft_loss": pre_projection_soft_loss,
                    "post_projection_soft_loss": soft_loss,
                    "pre_projection_hard_loss": pre_projection_hard_loss,
                    "post_projection_hard_loss": hard_loss,
                    "condition_projection_change_norm": float(torch.linalg.vector_norm(projected_state - pre_projection_state)),
                    "projection_erasure_fraction": projection_erasure_fraction(
                        loss_before=float(loss.detach()),
                        loss_pre_projection=pre_projection_soft_loss,
                        loss_post_projection=soft_loss,
                    ),
                    "hard_label_flip_count": int(changed.sum()),
                    **spatial_energy_fractions(raw_gradient, {f"gradient_{key}": value for key, value in regions.items()}),
                    **spatial_energy_fractions(update, {f"applied_update_{key}": value for key, value in regions.items()}),
                    **boundary,
                }
                counts = torch.bincount(after_categories[changed], minlength=embedding.shape[0])
                for category, count in enumerate(counts.tolist()):
                    row[f"hard_flip_into_raw_label_{category - 1}"] = int(count)
                if level in {"reflectivity_spikes", "seismic"}:
                    row.update(seismic_diagnostics(projected_state, tau))
                if hard_loss < best_hard_loss:
                    best_hard_loss = hard_loss
                    best_hard_step = step
                    best_state = projected_state.detach().cpu().clone()
                rows.append(row)
                all_trace.append(row)
            soft_values = [float(row["soft_attainment"]) for row in rows if math.isfinite(float(row["soft_attainment"]))]
            hard_values = [float(row["hard_attainment"]) for row in rows if math.isfinite(float(row["hard_attainment"]))]
            run_summary = {
                "level": level,
                "control": control_name,
                "initial_state_sha256": initial_state_sha256,
                "maximum_soft_attainment": max(soft_values) if soft_values else float("nan"),
                "final_soft_attainment": float(rows[-1]["soft_attainment"]),
                "maximum_hard_attainment": max(hard_values) if hard_values else float("nan"),
                "final_hard_attainment": float(rows[-1]["hard_attainment"]),
                "step_of_best_soft_attainment": int(max(rows, key=lambda row: float(row["soft_attainment"]) if math.isfinite(float(row["soft_attainment"])) else -math.inf)["step"]),
                "step_of_best_hard_attainment": int(max(rows, key=lambda row: float(row["hard_attainment"]) if math.isfinite(float(row["hard_attainment"])) else -math.inf)["step"]),
                "best_hard_step": best_hard_step,
                "best_hard_loss": best_hard_loss,
                "final_soft_closer_to": rows[-1]["soft_closer_to"],
                "final_hard_closer_to": rows[-1]["hard_closer_to"],
                "final_projection_erasure_fraction": rows[-1]["projection_erasure_fraction"],
                "trace_path": f"traces/{level}/{control_name}.csv",
            }
            run_summaries.append(run_summary)
            trace_path = output_dir / "traces" / level / f"{control_name}.csv"
            trace_path.parent.mkdir(parents=True, exist_ok=True)
            _write_rows(trace_path, rows)
            state_dir = output_dir / "states" / level / control_name
            state_dir.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, state_dir / "best_hard_state.pt")
            torch.save(state.detach().cpu(), state_dir / "final_state.pt")

    target_vector_acoustic = acoustic_table[:, target_category]
    target_density = density_table_device[target_category]
    codebook = []
    for category in range(embedding.shape[0]):
        codebook.append(
            {
                "raw_label": category - 1,
                "acoustic_l2_distance_to_target": float(torch.linalg.vector_norm(acoustic_table[:, category] - target_vector_acoustic)),
                "density_absolute_distance_to_target": float(torch.abs(density_table_device[category] - target_density)),
            }
        )
    git_status = _git("status", "--short")
    summary = {
        "stage": "phase6q_d3_soft_hard_transfer",
        "version": SOFT_HARD_TRANSFER_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "runs": run_summaries,
        "codebook_distances": codebook,
        "pairing_initial_state_hashes": pairing_hashes,
        "all_controls_share_initial_state": len(set(pairing_hashes.values())) == 1,
        "search_report": search_report,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": "clean" if not git_status else "dirty",
        "git_status_short": git_status.splitlines(),
        "seed": int(config["seed"]),
        "device": str(device),
        "dtype": "torch.float32",
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": _file_sha256(checkpoint_path),
        "checkpoint_used_for_flow": False,
        "flow_unet_loaded": False,
        "ema_raw_policy": "raw frozen checkpoint embedding only; U-Net not loaded",
        "property_table_sha256": _tensor_sha256(density_table),
        "acoustic_table_sha256": _tensor_sha256(acoustic_tables.property_table),
        "wavelet_sha256": seismic_operator.metadata()["wavelet"]["sha256"],
        "observation_hashes": {level: _tensor_sha256(value) for level, value in truth_responses.items()},
        "observation_sha256": _tensor_sha256(truth_responses["seismic"]),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "audit": _file_sha256(PROJECT_DIR / "guidance/soft_hard_transfer_audit.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
            "gravity": _file_sha256(PROJECT_DIR / "guidance/gravity.py"),
        },
        "runtime": {
            "hostname": socket.gethostname(),
            "torch": torch.__version__,
            "gpu_name": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        },
        "acoustic_metadata": acoustic_metadata,
        "density_metadata": density_metadata,
        "seismic_metadata": seismic_metadata,
        "gravity_metadata": gravity_metadata,
    }
    _write_json(output_dir / "config_input.json", config)
    _write_json(
        output_dir / "config_resolved.json",
        {**config, "device": str(device), "git_sha": summary["git_sha"]},
    )
    _write_rows(output_dir / "trace_all.csv", all_trace)
    _write_json(output_dir / "summary.json", summary)
    (output_dir / "REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"runs": len(run_summaries), "output_dir": str(output_dir)}, indent=2))


if __name__ == "__main__":
    main()
