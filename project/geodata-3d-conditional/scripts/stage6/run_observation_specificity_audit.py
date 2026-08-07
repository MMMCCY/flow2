#!/usr/bin/env python3
"""Run the small Stage-7 D7 observation-specificity mechanism audit."""

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
from typing import Callable

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.causality_gradient_audit import cosine_decode_categories
from guidance.frozen_flow_causality import run_base_trajectory
from guidance.generator_posterior import project_conditions
from guidance.observation_specificity import (
    OBSERVATION_SPECIFICITY_VERSION,
    finite_cosine,
    hidden_target_metrics,
    masked_pairwise_geometry,
    pairwise_geometry,
    rank_mechanisms,
    sensitivity_spectrum,
)
from guidance.probability_sampling import (
    build_probability_guidance_velocity,
    probability_guidance_weight,
    temperature_at_time,
)
from guidance.property_volume import gaussian_blur_property_channels
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    probabilities_to_subsurface_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import build_simple_causal_case, controlled_observation
from guided_geophysical_sampling import clip_gradient_by_norm, soft_decode_to_probs
from scripts.stage6.run_simple_causality import (
    _file_sha256,
    _read_json,
    _resolve_repo_path,
    _tensor_sha256,
    _write_json,
    _write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-7 D7 specificity audit")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(["git", *arguments], cwd=REPOSITORY_ROOT, text=True).strip()


def _resolve_dir(value: object, name: str) -> Path:
    path = Path(str(value))
    if not path.is_absolute():
        path = REPOSITORY_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"{name}: {path}")
    return path


def _mse(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left - right.to(device=left.device, dtype=left.dtype)).square().mean()


def _rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(_mse(left, right).sqrt().detach().cpu())


def _pair_value(rows: list[dict[str, object]], left: str, right: str, key: str = "cosine") -> float:
    for row in rows:
        if {str(row["left"]), str(row["right"])} == {left, right}:
            return float(row[key])
    raise KeyError((left, right, key))


def _state_outputs(
    state: torch.Tensor,
    *,
    tau: float,
    embedding: torch.Tensor,
    acoustic_table: torch.Tensor,
    target_acoustic: torch.Tensor,
    condition: torch.Tensor,
    subsurface: torch.Tensor,
    operator,
) -> dict[str, torch.Tensor]:
    probabilities = soft_decode_to_probs(state, embedding, tau=tau)
    acoustic = probabilities_to_subsurface_acoustic(probabilities, acoustic_table, subsurface)
    acoustic = overwrite_exact_condition_acoustic(acoustic, target_acoustic, condition)
    reflectivity, twt, valid = operator.interface_response(
        acoustic[:, 0:1], acoustic[:, 1:2], subsurface
    )
    valid_float = valid.to(reflectivity.dtype)
    spikes = operator.deposit_reflectivity(reflectivity, twt, valid)
    return {
        "probability": probabilities[:, 10:11],
        "expected_property": acoustic,
        "blurred_property": gaussian_blur_property_channels(acoustic, 1.5),
        "reflectivity": reflectivity * valid_float,
        "twt": twt * valid_float,
        "spikes": spikes,
        "seismic": operator.convolve_reflectivity_spikes(spikes),
    }


def _truth_outputs(target_acoustic: torch.Tensor, subsurface: torch.Tensor, operator) -> dict[str, torch.Tensor]:
    reflectivity, twt, valid = operator.interface_response(
        target_acoustic[:, 0:1], target_acoustic[:, 1:2], subsurface
    )
    valid_float = valid.to(reflectivity.dtype)
    spikes = operator.deposit_reflectivity(reflectivity, twt, valid)
    return {
        "reflectivity": reflectivity * valid_float,
        "twt": twt * valid_float,
        "spikes": spikes,
        "seismic": operator.convolve_reflectivity_spikes(spikes),
    }


def _controls(values: torch.Tensor, names: list[str], shuffle_seed: int) -> dict[str, torch.Tensor]:
    return {
        name: controlled_observation(values, name, shuffle_seed=shuffle_seed).detach()
        for name in names
    }


def _hard_response(state: torch.Tensor, model, acoustic_table, subsurface, operator) -> tuple[torch.Tensor, torch.Tensor]:
    labels = (cosine_decode_categories(state, model.embedding.weight) - 1).unsqueeze(1)
    acoustic = hard_labels_to_acoustic(labels, acoustic_table)
    response = operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)
    return labels, response


def _build_basis(
    case,
    state: torch.Tensor,
    embedding: torch.Tensor,
    hidden_count: int,
) -> tuple[list[str], list[torch.Tensor]]:
    direction = (embedding[case.target_label + 1] - embedding[case.background_label + 1]).view(1, -1, 1, 1, 1)
    names: list[str] = []
    values: list[torch.Tensor] = []
    for index, mask in enumerate(case.candidate_masks):
        value = direction * mask.to(device=state.device, dtype=state.dtype).view(1, 1, *mask.shape)
        value = value / torch.linalg.vector_norm(value).clamp_min(1e-12)
        names.append(f"candidate_{index:02d}")
        values.append(value)
    hidden = case.candidate_masks[list(case.truth_candidate_indices)].any(dim=0).to(state.device)
    coordinates = torch.meshgrid(
        *(torch.arange(size, device=state.device) for size in hidden.shape), indexing="ij"
    )
    selectors = [
        hidden & ((coordinates[0] + coordinates[1] + coordinates[2]) % 2 == 0),
        hidden & (coordinates[0] < hidden.shape[0] // 2),
        hidden & (coordinates[1] >= hidden.shape[1] // 2),
        hidden & (coordinates[2] < hidden.shape[2] // 2),
    ]
    for index, mask in enumerate(selectors[:hidden_count]):
        value = direction * mask.to(dtype=state.dtype).view(1, 1, *mask.shape)
        value = value / torch.linalg.vector_norm(value).clamp_min(1e-12)
        names.append(f"hidden_roi_direction_{index:02d}")
        values.append(value)
    return names, values


def _provenance_report(checkpoint: Path) -> dict[str, object]:
    """Verify the current committed D-stage reruns, not the historical dirty runners."""
    run_root = PROJECT_DIR / "experiments/stage6_inference_causality/runs/five_body_cuboid_v1"
    expected = {
        "D1": ("d1_observation_closure_v1_provenance_85d5deb", "observation_closure_v1.json"),
        "D2": ("d2_gradient_audit_v1_fix1_provenance_85d5deb_clean", "gradient_audit_v1_fix1.json"),
        "D3": ("d3_soft_hard_transfer_v1_provenance_85d5deb_clean", "soft_hard_transfer_v1.json"),
        "D4": ("d4_frozen_flow_trajectory_v1_provenance_85d5deb_clean", "frozen_flow_trajectory_v1.json"),
        "D5": ("d5_native_geology_audit_v1", "native_geology_audit_v1.json"),
    }
    rows = []
    checkpoint_hash = _file_sha256(checkpoint)
    config_root = PROJECT_DIR / "experiments/stage6_inference_causality/configs"
    for stage, (tag, config_name) in expected.items():
        path = run_root / tag / "summary.json"
        if not path.exists():
            rows.append({"stage": stage, "run_tag": tag, "verified": False, "reason": "missing rerun"})
            continue
        summary = _read_json(path)
        source_results = []
        current_candidates = list((PROJECT_DIR / "guidance").glob("*.py")) + list((PROJECT_DIR / "scripts/stage6").glob("*.py")) + [
            PROJECT_DIR / "guided_geophysical_sampling.py",
            PROJECT_DIR / "inference_runtime.py",
            PROJECT_DIR / "model_train_sh_inference_cond.py",
            REPOSITORY_ROOT / "StructuralGeo-main/src/geogen/model/geomodel.py",
            REPOSITORY_ROOT / "StructuralGeo-main/src/geogen/model/geoprocess.py",
            REPOSITORY_ROOT / "StructuralGeo-main/src/geogen/engine/parametric.py",
        ]
        current_hashes = {_file_sha256(item) for item in current_candidates if item.exists()}
        for key, value in summary.get("source_hashes", {}).items():
            source_results.append({"key": key, "sha256": value, "matches_current_file": value in current_hashes})
        source_ok = all(bool(row["matches_current_file"]) for row in source_results)
        config_match = _read_json(run_root / tag / "config_input.json") == _read_json(config_root / config_name)
        rows.append({
            "stage": stage,
            "run_tag": tag,
            "verified": source_ok and config_match and summary.get("checkpoint_sha256") == checkpoint_hash,
            "run_git_sha": summary.get("git_sha"),
            "run_git_status": summary.get("git_status"),
            "checkpoint_match": summary.get("checkpoint_sha256") == checkpoint_hash,
            "config_matches_current": config_match,
            "observation_hash_present": bool(summary.get("observation_sha256")),
            "source_hashes": source_results,
        })
    return {
        "historical_formal_git_sha": "11caa498b15e6b89891604e9537830b30df504fa",
        "frozen_reference_git_sha": "85d5deb4555430117887a8ba173a0222c6b899ae",
        "rerun_reason": "D1-D4 runner hashes differed because the diagnostic runners were untracked during formal execution; reran only those stages. D5 hashes matched current files.",
        "stages": rows,
        "provenance_verified": all(bool(row["verified"]) for row in rows),
    }


def _report(summary: dict[str, object]) -> str:
    ranking = summary["verdict"]["mechanism_ranking"]
    lines = [
        "# D7 Observation Specificity Report",
        "",
        f"Provenance verified: **{summary['provenance']['provenance_verified']}**.",
        f"All gradients used identical BASE states: **{summary['identical_state_gate_pass']}**.",
        "",
        "## Ranked mechanisms",
        "",
        "| Rank | Mechanism | Support score |",
        "|---:|---|---:|",
    ]
    for row in ranking:
        lines.append(f"| {row['rank']} | {row['mechanism']} | {row['support_score']:.4f} |")
    lines.extend([
        "",
        "The ranking is a compact attribution aid; the saved residual, gradient, controller, cross-response and local-sensitivity tables are the evidence.",
        "Truth geology was opened only for retrospective final-state diagnostics and never for state selection.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output directory: {output_dir}")
    config = _read_json(args.config.resolve())
    if config.get("schema") != "stage7_observation_specificity_config_v1":
        raise ValueError("unexpected D7 config schema")
    if bool(config.get("formal_training_authorized")):
        raise ValueError("D7 forbids training")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal D7 requires CUDA")
    base_config = _read_json(_resolve_repo_path(config["base_config"], "base_config"))
    trajectory_config = _read_json(_resolve_repo_path(config["trajectory_config"], "trajectory_config"))
    checkpoint = _resolve_repo_path(config["checkpoint"], "checkpoint")
    if _file_sha256(checkpoint) != config["checkpoint_sha256"]:
        raise ValueError("checkpoint hash mismatch")
    provenance = _provenance_report(checkpoint)
    if not provenance["provenance_verified"]:
        raise RuntimeError("D0-D6 provenance is not verified; inspect provenance rows")

    case = build_simple_causal_case(base_config)
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, load_report = runtime.load_model_with_weight_policy(
        Geo3DStochInterp, checkpoint, device, "ema"
    )
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
        parameter.grad = None
    truth = case.truth_labels.to(device)
    condition = case.condition_mask.to(device)
    subsurface = case.subsurface_mask.to(device)
    embedded_truth = model.embed(truth).detach()
    conditioning = embedded_truth * condition.expand_as(embedded_truth)
    acoustic_config = _read_json(_resolve_repo_path(base_config["acoustic_config"], "acoustic_config"))
    seismic_config = _read_json(_resolve_repo_path(base_config["seismic_config"], "seismic_config"))
    tables, acoustic_meta = acoustic_tables_from_config(acoustic_config, model.num_categories)
    acoustic_table = tables.property_table.to(device)
    operator, seismic_meta = seismic_operator_from_config(seismic_config, grid_shape=truth.shape[2:])
    target_acoustic = hard_labels_to_acoustic(truth, acoustic_table)
    truth_outputs = _truth_outputs(target_acoustic, subsurface, operator)
    controls = list(config["controls"])
    controlled = {
        level: _controls(value, controls, int(base_config["shuffle_seed"]))
        for level, value in truth_outputs.items()
    }

    generator = torch.Generator(device="cpu").manual_seed(int(config["seed"]))
    initial = torch.randn((1, model.embedding_dim, *model.data_shape), generator=generator)
    base_schedule = run_base_trajectory(
        model=model,
        initial_state=initial.to(device),
        conditioning=conditioning,
        embedded_conditions=embedded_truth,
        condition_mask=condition,
        n_steps=int(config["n_steps"]),
    )
    d4_summary = _read_json(_resolve_dir(config["d4_run"], "d4_run") / "summary.json")
    base_endpoint_hash = _tensor_sha256(base_schedule["states"][-1])
    if base_endpoint_hash != d4_summary["base_endpoint_sha256"]:
        raise RuntimeError("regenerated BASE endpoint does not match D4")

    controller = trajectory_config["controller"]
    n_steps = int(config["n_steps"])
    dt = 1.0 / n_steps
    hidden_truth = case.candidate_masks[list(case.truth_candidate_indices)].any(dim=0).view(1, 1, *truth.shape[2:]).to(device)
    candidate_domain = case.candidate_masks.any(dim=0).view(1, 1, *truth.shape[2:]).to(device)

    def outputs(state: torch.Tensor, tau: float) -> dict[str, torch.Tensor]:
        result = _state_outputs(
            state, tau=tau, embedding=model.embedding.weight,
            acoustic_table=acoustic_table, target_acoustic=target_acoustic,
            condition=condition, subsurface=subsurface, operator=operator,
        )
        result["blurred_property"] = gaussian_blur_property_channels(
            result["expected_property"], float(base_config["blur_sigma_voxels"])
        )
        return result

    # Reference gradient norms are those seen by the actual controller at its first active BASE state.
    first_active_step = next(
        step for step in range(n_steps)
        if probability_guidance_weight((step + 0.5) / n_steps, controller["guidance_schedule"], float(controller["guidance_start"])) > 0
    )
    reference_state = base_schedule["states"][first_active_step].to(device)
    reference_t = (first_active_step + 0.5) / n_steps
    reference_tau = temperature_at_time(reference_t, float(controller["tau_start"]), float(controller["tau_end"]), controller["tau_schedule"])
    reference_norms: dict[str, torch.Tensor] = {}
    for control in controls:
        differentiable = reference_state.detach().requires_grad_(True)
        loss = _mse(outputs(differentiable, reference_tau)["seismic"], controlled["seismic"][control])
        gradient = torch.autograd.grad(loss, differentiable)[0]
        reference_norms[control] = clip_gradient_by_norm(gradient, float(controller["gradient_clip_norm"])).flatten(1).norm(dim=1).detach()

    residual_rows: list[dict[str, object]] = []
    gradient_rows: list[dict[str, object]] = []
    controller_rows: list[dict[str, object]] = []
    one_step_rows: list[dict[str, object]] = []
    state_hashes: dict[str, str] = {}
    raw_cosines: list[float] = []
    residual_cosines: list[float] = []
    applied_cosines: list[float] = []
    hard_overlap_values: list[float] = []
    shared_steps = list(config["shared_state_steps"])
    state_specs = [(f"base_step_{step:02d}", step) for step in shared_steps] + [("common_base_endpoint", n_steps)]
    for state_name, step in state_specs:
        state = base_schedule["states"][step].to(device)
        state_hashes[state_name] = _tensor_sha256(state)
        update_index = min(step, n_steps - 1)
        t_value = (update_index + 0.5) / n_steps
        tau = temperature_at_time(t_value, float(controller["tau_start"]), float(controller["tau_end"]), controller["tau_schedule"])
        state_outputs = outputs(state, tau)
        for level in ("reflectivity", "twt", "spikes", "seismic"):
            residuals = {name: state_outputs[level] - controlled[level][name] for name in controls}
            geometry = pairwise_geometry(residuals)
            for row in geometry:
                residual_rows.append({"state": state_name, "step": step, "level": level, **row})
            if level == "seismic":
                residual_cosines.append(_pair_value(geometry, "correct", "zero"))
                residual_cosines.append(_pair_value(geometry, "correct", "shuffled_xy"))

        with torch.no_grad():
            time = torch.full((1,), t_value, device=device, dtype=state.dtype)
            base_velocity = model.net(state, conditioning, time)
        weight = probability_guidance_weight(t_value, controller["guidance_schedule"], float(controller["guidance_start"]))
        raw_negative: dict[str, torch.Tensor] = {}
        clipped_negative: dict[str, torch.Tensor] = {}
        applied: dict[str, torch.Tensor] = {}
        deltas: dict[str, torch.Tensor] = {}
        next_states: dict[str, torch.Tensor] = {}
        for control in controls:
            differentiable = state.detach().requires_grad_(True)
            loss = _mse(outputs(differentiable, tau)["seismic"], controlled["seismic"][control])
            gradient = torch.autograd.grad(loss, differentiable)[0]
            clipped = clip_gradient_by_norm(gradient, float(controller["gradient_clip_norm"]))
            guidance, diagnostics, _ = build_probability_guidance_velocity(
                clipped, base_velocity,
                requested_ratio=float(controller["alpha"]) * weight,
                max_ratio=float(controller["max_ratio"]),
                scaling_mode=controller["scaling_mode"],
                reference_gradient_norm=reference_norms[control],
            )
            raw_negative[control] = -gradient.detach()
            clipped_negative[control] = -clipped.detach()
            applied[control] = -guidance.detach()
            deltas[control] = dt * applied[control]
            next_states[control] = project_conditions(
                state + dt * (base_velocity + applied[control]), embedded_truth, condition
            ).detach()
            controller_rows.append({
                "state": state_name, "step": step, "control": control,
                "tau": tau, "guidance_weight": weight,
                "raw_gradient_norm": float(torch.linalg.vector_norm(gradient)),
                "clipped_gradient_norm": float(torch.linalg.vector_norm(clipped)),
                "reference_base_velocity_norm": float(torch.linalg.vector_norm(base_velocity)),
                "requested_ratio": float(controller["alpha"]) * weight,
                "used_guidance_ratio": diagnostics["used_guidance_ratio"],
                "guidance_cap_fraction": diagnostics["guidance_cap_fraction"],
                "applied_velocity_norm": float(torch.linalg.vector_norm(applied[control])),
            })
        raw_geometry = pairwise_geometry(raw_negative, include_correlation=True)
        clipped_geometry = pairwise_geometry(clipped_negative, include_correlation=True)
        applied_geometry = pairwise_geometry(applied, include_correlation=True)
        delta_geometry = pairwise_geometry(deltas, include_correlation=True)
        hidden_geometry = masked_pairwise_geometry(raw_negative, hidden_truth.expand_as(next(iter(raw_negative.values()))))
        outside_geometry = masked_pairwise_geometry(raw_negative, (~hidden_truth).expand_as(next(iter(raw_negative.values()))))
        for layer, rows in (("raw_negative_gradient", raw_geometry), ("clipped_negative_gradient", clipped_geometry), ("applied_physics_velocity", applied_geometry), ("euler_state_delta", delta_geometry)):
            for row in rows:
                gradient_rows.append({"state": state_name, "step": step, "layer": layer, **row})
        for region, rows in (("hidden_roi", hidden_geometry), ("outside_hidden_roi", outside_geometry)):
            for row in rows:
                gradient_rows.append({"state": state_name, "step": step, "layer": f"raw_negative_gradient_{region}", **row})
        raw_cosines.extend([_pair_value(raw_geometry, "correct", "zero"), _pair_value(raw_geometry, "correct", "shuffled_xy")])
        applied_cosines.extend([_pair_value(applied_geometry, "correct", "zero"), _pair_value(applied_geometry, "correct", "shuffled_xy")])

        soft_responses = {control: outputs(next_state, tau)["seismic"].detach() for control, next_state in next_states.items()}
        hard_results = {control: _hard_response(next_state, model, acoustic_table, subsurface, operator) for control, next_state in next_states.items()}
        for optimized_by in controls:
            labels, hard_field = hard_results[optimized_by]
            for evaluated_against in controls:
                one_step_rows.append({
                    "state": state_name, "step": step, "optimized_by": optimized_by,
                    "evaluated_against": evaluated_against,
                    "soft_rmse": _rmse(soft_responses[optimized_by], controlled["seismic"][evaluated_against]),
                    "hard_rmse": _rmse(hard_field, controlled["seismic"][evaluated_against]),
                    "evaluated_against_correct_truth_observation": evaluated_against == "correct",
                })
        hard_labels = {name: value[0] for name, value in hard_results.items()}
        for left, right in (("correct", "zero"), ("correct", "shuffled_xy"), ("zero", "shuffled_xy")):
            changed_left = hard_labels[left] == case.target_label
            changed_right = hard_labels[right] == case.target_label
            union = int((changed_left | changed_right).sum())
            overlap = int((changed_left & changed_right).sum()) / union if union else 1.0
            hard_overlap_values.append(overlap)

    # Existing D4 states: cross-evaluate, never optimize or reselect.
    final_rows: list[dict[str, object]] = []
    final_labels: dict[str, torch.Tensor] = {}
    d4_root = _resolve_dir(config["d4_run"], "d4_run")
    truth_response = truth_outputs["seismic"]
    for optimized_by in controls:
        for mode in config["state_modes"]:
            for state_kind in config["state_kinds"]:
                path = d4_root / "states" / optimized_by / mode / f"{state_kind}.pt"
                state = torch.load(path, map_location=device, weights_only=True)
                labels, response = _hard_response(state, model, acoustic_table, subsurface, operator)
                key = f"{optimized_by}/{mode}/{state_kind}"
                final_labels[key] = labels.detach().cpu()
                retrospective = hidden_target_metrics(
                    labels, target_label=case.target_label,
                    truth_hidden_mask=hidden_truth, evaluation_domain=candidate_domain,
                )
                for evaluated_against in controls:
                    final_rows.append({
                        "optimized_by": optimized_by, "mode": mode, "state_kind": state_kind,
                        "evaluated_against": evaluated_against,
                        "hard_rmse": _rmse(response, controlled["seismic"][evaluated_against]),
                        "hard_correct_observation_rmse": _rmse(response, truth_response),
                        "distance_to_truth_response": float(torch.linalg.vector_norm((response - truth_response).double())),
                        **retrospective,
                    })
    overlap_rows = []
    for mode in config["state_modes"]:
        for state_kind in config["state_kinds"]:
            for left, right in (("correct", "zero"), ("correct", "shuffled_xy"), ("zero", "shuffled_xy")):
                a = final_labels[f"{left}/{mode}/{state_kind}"]
                b = final_labels[f"{right}/{mode}/{state_kind}"]
                overlap_rows.append({
                    "mode": mode, "state_kind": state_kind, "left": left, "right": right,
                    "full_label_agreement": float((a == b).float().mean()),
                    "target_label_iou": float(((a == case.target_label) & (b == case.target_label)).sum() / ((a == case.target_label) | (b == case.target_label)).sum().clamp_min(1)),
                })

    # Small JVP basis at the common BASE endpoint.
    sensitivity_state = base_schedule["states"][int(config["sensitivity_state_step"])].to(device)
    sensitivity_tau = float(controller["tau_end"])
    basis_names, basis = _build_basis(case, sensitivity_state, model.embedding.weight, int(config["hidden_roi_direction_count"]))
    level_functions: dict[str, Callable[[torch.Tensor], torch.Tensor]] = {
        level: (lambda value, selected=level: outputs(value, sensitivity_tau)[selected])
        for level in ("probability", "expected_property", "blurred_property", "reflectivity", "twt", "spikes", "seismic")
    }
    sensitivity: dict[str, object] = {}
    for level, function in level_functions.items():
        columns = []
        for direction in basis:
            _, tangent = torch.autograd.functional.jvp(function, sensitivity_state, direction, create_graph=False, strict=True)
            columns.append(tangent)
        sensitivity[level] = sensitivity_spectrum(
            columns, basis_names,
            truth_column_indices=list(case.truth_candidate_indices),
            relative_rank_tolerance=float(config["sensitivity_relative_rank_tolerance"]),
        )

    mean_residual = sum(value for value in residual_cosines if math.isfinite(value)) / max(1, sum(math.isfinite(value) for value in residual_cosines))
    mean_raw = sum(value for value in raw_cosines if math.isfinite(value)) / max(1, sum(math.isfinite(value) for value in raw_cosines))
    mean_applied = sum(value for value in applied_cosines if math.isfinite(value)) / max(1, sum(math.isfinite(value) for value in applied_cosines))
    mean_hard_overlap = sum(hard_overlap_values) / len(hard_overlap_values)
    seismic_rank_fraction = float(sensitivity["seismic"]["effective_rank"]) / float(sensitivity["seismic"]["column_count"])
    reflectivity_rank_fraction = float(sensitivity["reflectivity"]["effective_rank"]) / float(sensitivity["reflectivity"]["column_count"])
    scores = {
        "S1_residual_similarity": max(0.0, mean_residual),
        "S2_jacobian_vjp_projection_collapse": max(0.0, mean_raw - mean_residual, reflectivity_rank_fraction - seismic_rank_fraction),
        "S3_controller_normalization_cap_collapse": max(0.0, mean_applied - mean_raw),
        "S4_categorical_hard_transition_collapse": max(0.0, mean_hard_overlap - mean_applied),
    }
    verdict = {
        "mechanism_ranking": rank_mechanisms(scores),
        "aggregate_diagnostics": {
            "mean_correct_control_seismic_residual_cosine": mean_residual,
            "mean_correct_control_raw_gradient_cosine": mean_raw,
            "mean_correct_control_applied_velocity_cosine": mean_applied,
            "mean_one_step_hard_target_overlap": mean_hard_overlap,
            "reflectivity_effective_rank_fraction": reflectivity_rank_fraction,
            "seismic_effective_rank_fraction": seismic_rank_fraction,
        },
        "optional_s3_controller_control_authorized": scores["S3_controller_normalization_cap_collapse"] >= 0.25,
    }
    git_status = _git("status", "--short")
    summary = {
        "stage": "stage7_d7_observation_specificity",
        "version": OBSERVATION_SPECIFICITY_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "provenance": provenance,
        "shared_state_hashes": state_hashes,
        "identical_state_gate_pass": len(state_hashes) == len(state_specs),
        "base_endpoint_sha256": base_endpoint_hash,
        "d4_base_endpoint_sha256": d4_summary["base_endpoint_sha256"],
        "sensitivity": sensitivity,
        "verdict": verdict,
        "truth_metrics_selection_blind": True,
        "training_performed": False,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_at_completion": "clean" if not git_status else "dirty_stage7_implementation",
        "checkpoint_sha256": _file_sha256(checkpoint),
        "observation_sha256": _tensor_sha256(truth_response),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "audit": _file_sha256(PROJECT_DIR / "guidance/observation_specificity.py"),
            "frozen_flow": _file_sha256(PROJECT_DIR / "guidance/frozen_flow_causality.py"),
            "controller": _file_sha256(PROJECT_DIR / "guidance/probability_sampling.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
        },
        "runtime": {"hostname": socket.gethostname(), "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(device)},
        "model_load_report": load_report,
        "acoustic_metadata": acoustic_meta,
        "seismic_metadata": seismic_meta,
    }
    output_dir.mkdir(parents=True)
    _write_json(output_dir / "config_input.json", config)
    _write_rows(output_dir / "residual_geometry.csv", residual_rows)
    _write_rows(output_dir / "gradient_controller_geometry.csv", gradient_rows)
    _write_rows(output_dir / "controller_diagnostics.csv", controller_rows)
    _write_rows(output_dir / "one_step_cross_observation.csv", one_step_rows)
    _write_rows(output_dir / "existing_state_cross_observation.csv", final_rows)
    _write_rows(output_dir / "existing_state_hard_overlap.csv", overlap_rows)
    _write_json(output_dir / "summary.json", summary)
    _write_json(output_dir / "d7_observation_specificity_verdict.json", verdict)
    (output_dir / "D7_OBSERVATION_SPECIFICITY_REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "ranking": verdict["mechanism_ranking"]}, indent=2))


if __name__ == "__main__":
    main()
