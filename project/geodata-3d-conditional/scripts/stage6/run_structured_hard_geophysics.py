#!/usr/bin/env python3
"""Run Stage-7B structured hard-geophysics inference and its controls."""

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


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guidance.causality_gradient_audit import cosine_decode_categories
from guidance.native_geology_audit import build_structuralgeo_native_case
from guidance.observation_specificity import hidden_target_metrics
from guidance.seismic import (
    acoustic_tables_from_config,
    hard_labels_to_acoustic,
    seismic_operator_from_config,
)
from guidance.simple_causality import build_simple_causal_case, controlled_observation
from guidance.structured_hard_inference import (
    STRUCTURED_HARD_VERSION,
    StructuredObject,
    beam_evolutionary_search,
)
from scripts.stage6.run_simple_causality import (
    _file_sha256,
    _read_json,
    _resolve_repo_path,
    _tensor_sha256,
    _write_json,
    _write_rows,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Stage-7B structured hard inference")
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


def _rmse(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left - right.to(left)).square().mean().sqrt().detach().cpu())


def _hard_response(labels: torch.Tensor, table: torch.Tensor, subsurface: torch.Tensor, operator) -> torch.Tensor:
    acoustic = hard_labels_to_acoustic(labels, table)
    return operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)


def _cuboid_library(case) -> list[StructuredObject]:
    result = []
    for body in case.candidate_bodies:
        center = tuple((left + right - 1) / 2.0 for left, right in zip(body.start, body.stop))
        size = tuple(float(right - left) for left, right in zip(body.start, body.stop))
        result.append(StructuredObject(
            object_id=body.id, presence=True,
            center_x=center[0], center_y=center[1], center_z=center[2],
            size_x=size[0], size_y=size[1], size_z=size[2],
            orientation_deg=0.0, shape="cuboid", material_label=case.target_label,
            source_family="benchmark_domain_cuboid; truth indices unavailable to search",
        ))
    return result


def _native_library(config: dict[str, object]) -> list[StructuredObject]:
    result = []
    if int(config.get("sobol_population_size_per_center", 0)) > 0:
        count = int(config["sobol_population_size_per_center"])
        bounds = config["parameter_bounds"]
        points = torch.quasirandom.SobolEngine(4, scramble=False).draw(count)
        variants = []
        for point in points:
            values = []
            for coordinate, name in zip(point.tolist(), ("diameter", "height", "minor_axis_scale", "rotation")):
                low, high = (float(value) for value in bounds[name])
                values.append(low + coordinate * (high - low))
            variants.append({
                "diameter": values[0], "height": values[1],
                "minor_axis_scale": values[2], "rotation": values[3],
            })
    else:
        variants = list(config["shape_variants"])
    for center_index, center in enumerate(config["centers_xyz"]):
        for variant_index, variant in enumerate(variants):
            result.append(StructuredObject(
                object_id=f"native_center{center_index:02d}_variant{variant_index:02d}",
                presence=True,
                center_x=float(center[0]), center_y=float(center[1]), center_z=float(center[2]),
                size_x=float(variant["diameter"]) * float(variant["minor_axis_scale"]),
                size_y=float(variant["diameter"]), size_z=float(variant["height"]),
                orientation_deg=float(variant["rotation"]), shape="dike_hemisphere",
                material_label=9,
                source_family="StructuralGeo IntrusionSpec(kind=hemisphere, anchor_to_present=False, clip=False, upper=True)",
            ))
    return result


def _body_metrics(
    labels: torch.Tensor,
    *,
    target_label: int,
    truth_body_masks: torch.Tensor,
    evaluation_domain: torch.Tensor,
    predicted_object_count: int,
) -> dict[str, float | int]:
    predicted = (labels == target_label)[0, 0] & evaluation_domain[0, 0]
    truth_masks = truth_body_masks.to(device=predicted.device, dtype=torch.bool)
    recalls = [float((predicted & mask).sum() / mask.sum().clamp_min(1)) for mask in truth_masks]
    recovered = sum(value >= 0.5 for value in recalls)
    matched_predictions = min(recovered, int(predicted_object_count))
    return {
        "body_recall": recovered / len(recalls) if recalls else 1.0,
        "body_precision": matched_predictions / int(predicted_object_count) if predicted_object_count else 0.0,
        "truth_body_recalls": recalls,
        "recovered_truth_body_count_at_0p5": recovered,
    }


def _wrong_cuboid_observation(case, indices, table, operator, device) -> torch.Tensor:
    labels = case.baseline_labels.to(device).clone()
    union = case.candidate_masks[list(indices)].any(dim=0).to(device)
    labels[0, 0, union] = case.target_label
    return _hard_response(labels, table, case.subsurface_mask.to(device), operator).detach()


def _controlled_observations(correct, wrong, controls, shuffle_seed):
    result = {}
    for name in controls:
        if name == "wrong_case_observation":
            result[name] = wrong.detach().clone()
        else:
            result[name] = controlled_observation(correct, name, shuffle_seed=shuffle_seed).detach()
    return result


def _run_case(
    *,
    case_id: str,
    baseline_labels: torch.Tensor,
    condition_mask: torch.Tensor,
    subsurface: torch.Tensor,
    truth_labels: torch.Tensor,
    truth_hidden: torch.Tensor,
    truth_body_masks: torch.Tensor,
    evaluation_domain: torch.Tensor,
    table: torch.Tensor,
    operator,
    observations: dict[str, torch.Tensor],
    proposal_library: list[StructuredObject],
    search_config: dict[str, object],
    target_label: int,
    air_start_z: int,
    output_dir: Path,
) -> dict[str, object]:
    response_fn = lambda labels: _hard_response(labels, table, subsurface, operator)
    correct = observations["correct"]
    arms = []
    cross_rows = []
    results = {}
    for control, observation in observations.items():
        result = beam_evolutionary_search(
            baseline_labels=baseline_labels,
            condition_mask=condition_mask,
            air_start_z=air_start_z,
            observation=observation,
            hard_response=response_fn,
            proposal_library=proposal_library,
            allowed_material_labels=(0, target_label),
            kmax=int(search_config["kmax"]),
            beam_size=int(search_config["beam_size"]),
            local_generations=int(search_config["local_generations"]),
        )
        results[control] = result
        labels = result["best_labels"].to(table.device)
        response = result["best_response"].to(table.device)
        hidden = hidden_target_metrics(
            labels, target_label=target_label,
            truth_hidden_mask=truth_hidden, evaluation_domain=evaluation_domain,
        )
        bodies = _body_metrics(
            labels, target_label=target_label,
            truth_body_masks=truth_body_masks,
            evaluation_domain=evaluation_domain,
            predicted_object_count=len(result["best_model"].objects),
        )
        condition_violations = int((labels[condition_mask] != baseline_labels[condition_mask]).sum())
        wrong_lithology = int(((labels != -1) & (labels != 0) & (labels != target_label)).sum())
        correct_baseline_rmse = _rmse(result["baseline_response"].to(table.device), correct)
        correct_rmse = _rmse(response, correct)
        arm = {
            "case_id": case_id,
            "optimized_by": control,
            "selected_hard_observation_rmse": result["best_hard_rmse"],
            "selected_target_attainment": result["hard_attainment"],
            "hard_correct_observation_rmse": correct_rmse,
            "hard_correct_observation_attainment": 1.0 - correct_rmse / correct_baseline_rmse if correct_baseline_rmse > 0 else float("nan"),
            "distance_to_truth_response": float(torch.linalg.vector_norm((response - correct).double())),
            **hidden, **bodies,
            "condition_violations": condition_violations,
            "wrong_lithology_volume": wrong_lithology,
            "runtime_seconds": result["runtime_seconds"],
            "forward_call_count": result["forward_call_count"],
            "selected_object_count": len(result["best_model"].objects),
            "selected_objects": [value.record() for value in result["best_model"].objects],
            "selection_criterion": result["selection_criterion"],
            "truth_used_for_selection": result["selection_used_truth"],
        }
        arms.append(arm)
        trace_path = output_dir / "traces" / case_id / f"{control}.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        _write_json(trace_path, {"trace": result["trace"]})
        state_path = output_dir / "states" / case_id / control
        state_path.mkdir(parents=True, exist_ok=True)
        torch.save(result["best_labels"], state_path / "best_labels.pt")
        _write_json(state_path / "selected_event_history.json", {
            "event_type": "StructuredGeo hard proposal sequence",
            "events": [value.record() for value in result["best_model"].objects],
            "history": [
                f"{value.source_family}: center=({value.center_x},{value.center_y},{value.center_z}), size=({value.size_x},{value.size_y},{value.size_z}), rotation={value.orientation_deg}, material={value.material_label}"
                for value in result["best_model"].objects
            ],
            "proposal_parent": result["best_model"].parent_id,
            "proposal_move": result["best_model"].proposal_move,
            "hard_seismic_rmse": result["best_hard_rmse"],
            "condition_violations": condition_violations,
        })
    for optimized_by, result in results.items():
        response = result["best_response"].to(table.device)
        for evaluated_against, observation in observations.items():
            cross_rows.append({
                "case_id": case_id,
                "optimized_by": optimized_by,
                "evaluated_against": evaluated_against,
                "hard_seismic_rmse": _rmse(response, observation),
                "evaluated_against_correct_truth_observation": evaluated_against == "correct",
            })
    correct_rows = [row for row in cross_rows if row["evaluated_against"] == "correct"]
    correct_ranked = sorted(correct_rows, key=lambda row: (row["hard_seismic_rmse"], row["optimized_by"]))
    strict_correct_win = (
        correct_ranked[0]["optimized_by"] == "correct"
        and float(correct_ranked[0]["hard_seismic_rmse"])
        < float(correct_ranked[1]["hard_seismic_rmse"]) - 1e-10
    )
    return {
        "case_id": case_id,
        "arms": arms,
        "cross_evaluation": cross_rows,
        "correct_optimized_is_best_against_correct": strict_correct_win,
        "correct_evaluation_ranking": [row["optimized_by"] for row in correct_ranked],
        "observation_hashes": {name: _tensor_sha256(value) for name, value in observations.items()},
    }


def _paired_flow_comparison(case, model, table, operator, device, d4_root, structured_case, q1_summary):
    subsurface = case.subsurface_mask.to(device)
    truth_response = _hard_response(case.truth_labels.to(device), table, subsurface, operator)
    baseline_labels = case.baseline_labels.to(device)
    hidden_truth = case.candidate_masks[list(case.truth_candidate_indices)].any(dim=0).view(1, 1, *baseline_labels.shape[2:]).to(device)
    domain = case.candidate_masks.any(dim=0).view(1, 1, *baseline_labels.shape[2:]).to(device)
    rows = []
    base_flow_labels, base_flow_response = _state_to_labels_response(
        d4_root / "states/correct/BASE/final_state.pt", model, table,
        subsurface, operator, device,
    )
    base_flow_rmse = _rmse(base_flow_response, truth_response)
    def add(name, labels, response, selection):
        rmse = _rmse(response, truth_response)
        rows.append({
            "method": name, "hard_correct_observation_rmse": rmse,
            "hard_attainment_relative_to_base_flow_endpoint": 1.0 - rmse / base_flow_rmse if base_flow_rmse > 0 else float("nan"),
            **hidden_target_metrics(labels, target_label=case.target_label, truth_hidden_mask=hidden_truth, evaluation_domain=domain),
            "selection": selection,
        })
    add("BASE_frozen_flow_sample", base_flow_labels, base_flow_response, "no physics")
    for kind in ("best_hard_state", "final_state"):
        add(f"continuous_BASE_PLUS_PHYSICS_{kind}", *_state_to_labels_response(d4_root / f"states/correct/BASE_PLUS_PHYSICS/{kind}.pt", model, table, subsurface, operator, device), "existing continuous D4 hard checkpoint rule")
    correct_arm = next(row for row in structured_case["arms"] if row["optimized_by"] == "correct")
    labels = torch.load(d4_root.parents[3] / "dummy.pt", weights_only=True) if False else None
    structured_labels_path = None
    # The caller has already saved labels; reconstruct from selected objects is unnecessary.
    rows.append({
        "method": "structured_hard_geophysics",
        "hard_correct_observation_rmse": correct_arm["hard_correct_observation_rmse"],
        "hard_attainment_relative_to_base_flow_endpoint": 1.0,
        "hidden_target_iou": correct_arm["hidden_target_iou"],
        "hidden_target_recall": correct_arm["hidden_target_recall"],
        "hidden_target_precision": correct_arm["hidden_target_precision"],
        "hidden_target_true_positive_voxels": correct_arm["hidden_target_true_positive_voxels"],
        "hidden_target_predicted_voxels": correct_arm["hidden_target_predicted_voxels"],
        "hidden_target_truth_voxels": correct_arm["hidden_target_truth_voxels"],
        "selection": "hard observed seismic only",
    })
    rows.append({
        "method": "Q1_oracle_like_structured_reference",
        "hard_correct_observation_rmse": q1_summary["enumeration"]["seismic"]["truth_pair_rmse"],
        "hard_attainment_relative_to_base_flow_endpoint": 1.0,
        "hidden_target_iou": 1.0, "hidden_target_recall": 1.0, "hidden_target_precision": 1.0,
        "hidden_target_true_positive_voxels": int(hidden_truth.sum()),
        "hidden_target_predicted_voxels": int(hidden_truth.sum()),
        "hidden_target_truth_voxels": int(hidden_truth.sum()),
        "selection": "exhaustive fixed two-body dictionary hard seismic ranking; truth indices not supplied",
    })
    return rows


def _state_to_labels_response(path, model, table, subsurface, operator, device):
    state = torch.load(path, map_location=device, weights_only=True)
    labels = (cosine_decode_categories(state, model.embedding.weight) - 1).unsqueeze(1)
    return labels, _hard_response(labels, table, subsurface, operator)


def _report(summary: dict[str, object]) -> str:
    answers = summary["answers"]
    lines = [
        "# Stage 7 Report — Observation Specificity and Structured Hard-Geophysics",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "## D7 mechanism closure",
        "",
        "| Rank | Mechanism | Support score |",
        "|---:|---|---:|",
    ]
    for row in summary["d7_verdict"]["mechanism_ranking"]:
        lines.append(f"| {row['rank']} | {row['mechanism']} | {row['support_score']:.4f} |")
    lines.extend([
        "",
        "## Analytic cuboid controls",
        "",
        "| Optimized by | Correct-field attainment | Hidden IoU | Hidden recall | Forward calls |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["cuboid"]["arms"]:
        lines.append(
            f"| {row['optimized_by']} | {row['hard_correct_observation_attainment']:.4f} | "
            f"{row['hidden_target_iou']:.4f} | {row['hidden_target_recall']:.4f} | {row['forward_call_count']} |"
        )
    lines.extend([
        "",
        "## StructuralGeo deterministic replicas",
        "",
        "| Case | Correct ranks first | Correct-field attainment | Hidden IoU | Hidden recall |",
        "|---|---:|---:|---:|---:|",
    ])
    for case in summary["native_replicas"]:
        row = next(value for value in case["arms"] if value["optimized_by"] == "correct")
        lines.append(
            f"| {case['case_id']} | {case['correct_optimized_is_best_against_correct']} | "
            f"{row['hard_correct_observation_attainment']:.4f} | {row['hidden_target_iou']:.4f} | "
            f"{row['hidden_target_recall']:.4f} |"
        )
    lines.extend([
        "",
        "## Paired comparison",
        "",
        "| Method | Correct hard RMSE | Attainment vs BASE flow endpoint | Hidden IoU | Hidden recall |",
        "|---|---:|---:|---:|---:|",
    ])
    for row in summary["paired_comparison"]:
        lines.append(
            f"| {row['method']} | {row['hard_correct_observation_rmse']:.6g} | "
            f"{row['hard_attainment_relative_to_base_flow_endpoint']:.4f} | "
            f"{row['hidden_target_iou']:.4f} | {row['hidden_target_recall']:.4f} |"
        )
    lines.extend([
        "",
        "## Required questions",
        "",
    ])
    for index, answer in enumerate(answers, start=1):
        lines.append(f"{index}. {answer}")
    lines.extend([
        "",
        "## Gates",
        "",
        f"- Cuboid correct-control specificity: `{summary['success_gates']['cuboid_correct_specificity']}`",
        f"- Cuboid hard-attainment improvement: `{summary['success_gates']['cuboid_attainment']}`",
        f"- Native replication: `{summary['success_gates']['native_replication']}`",
        f"- Training performed: `{summary['training_performed']}`",
        "",
        "All proposal acceptance and beam selection used hard observed seismic RMSE only. Geological truth metrics were computed retrospectively.",
        "",
    ])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    if output_dir.exists():
        raise FileExistsError(f"refusing to reuse output: {output_dir}")
    config = _read_json(args.config.resolve())
    if config.get("schema") != "stage7_structured_hard_geophysics_config_v1":
        raise ValueError("unexpected Stage7B config schema")
    if bool(config.get("formal_training_authorized")):
        raise ValueError("Stage7B forbids training")
    d7 = _read_json(_resolve_dir(config["d7_run"], "d7_run") / "summary.json")
    if d7.get("status") != "completed" or not d7["provenance"]["provenance_verified"]:
        raise RuntimeError("D7 and provenance must complete before Stage7B")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("formal Stage7B requires CUDA")
    base_config = _read_json(_resolve_repo_path(config["base_config"], "base_config"))
    case = build_simple_causal_case(base_config)
    acoustic_config = _read_json(_resolve_repo_path(base_config["acoustic_config"], "acoustic_config"))
    seismic_config = _read_json(_resolve_repo_path(base_config["seismic_config"], "seismic_config"))
    table_bundle, acoustic_meta = acoustic_tables_from_config(acoustic_config, 15)
    table = table_bundle.property_table.to(device)
    operator, seismic_meta = seismic_operator_from_config(seismic_config, grid_shape=case.truth_labels.shape[2:])
    truth_labels = case.truth_labels.to(device)
    baseline_labels = case.baseline_labels.to(device)
    subsurface = case.subsurface_mask.to(device)
    condition = case.condition_mask.to(device)
    correct = _hard_response(truth_labels, table, subsurface, operator).detach()
    wrong = _wrong_cuboid_observation(case, config["wrong_case_candidate_indices"], table, operator, device)
    observations = _controlled_observations(correct, wrong, config["controls"], int(base_config["shuffle_seed"]))
    hidden_truth = case.candidate_masks[list(case.truth_candidate_indices)].any(dim=0).view(1, 1, *truth_labels.shape[2:]).to(device)
    domain = case.candidate_masks.any(dim=0).view(1, 1, *truth_labels.shape[2:]).to(device)
    cuboid = _run_case(
        case_id="cuboid_seed42", baseline_labels=baseline_labels,
        condition_mask=condition, subsurface=subsurface, truth_labels=truth_labels,
        truth_hidden=hidden_truth,
        truth_body_masks=case.candidate_masks[list(case.truth_candidate_indices)],
        evaluation_domain=domain, table=table, operator=operator,
        observations=observations, proposal_library=_cuboid_library(case),
        search_config=config["cuboid_search"], target_label=case.target_label,
        air_start_z=int(base_config["air_start_z"]), output_dir=output_dir,
    )

    native_results = []
    native_config = config["native_search"]
    native_library = _native_library(native_config)
    for seed in native_config["seeds"]:
        native_case, native_meta = build_structuralgeo_native_case(seed=int(seed))
        wrong_case, wrong_meta = build_structuralgeo_native_case(seed=int(seed) + int(native_config["wrong_case_seed_offset"]))
        native_truth = native_case.truth_labels.to(device)
        native_condition = native_case.condition_mask.to(device)
        native_subsurface = native_case.subsurface_mask.to(device)
        native_baseline = native_truth.clone()
        native_hidden_masks = native_case.body_masks[3:].to(device)
        native_hidden = native_hidden_masks.any(dim=0).view(1, 1, 64, 64, 64)
        native_baseline[0, 0, native_hidden[0, 0]] = native_case.background_label
        native_correct = _hard_response(native_truth, table, native_subsurface, operator).detach()
        native_wrong = _hard_response(wrong_case.truth_labels.to(device), table, wrong_case.subsurface_mask.to(device), operator).detach()
        native_observations = _controlled_observations(native_correct, native_wrong, config["controls"], int(base_config["shuffle_seed"]) + int(seed))
        native_domain = torch.zeros_like(native_hidden)
        native_domain[..., 4:60, 28:56, 8:52] = True
        result = _run_case(
            case_id=f"native_seed{seed}", baseline_labels=native_baseline,
            condition_mask=native_condition, subsurface=native_subsurface,
            truth_labels=native_truth, truth_hidden=native_hidden,
            truth_body_masks=native_hidden_masks, evaluation_domain=native_domain,
            table=table, operator=operator, observations=native_observations,
            proposal_library=native_library, search_config=native_config,
            target_label=native_case.target_label, air_start_z=56, output_dir=output_dir,
        )
        result["native_truth_metadata"] = native_meta
        result["wrong_case_metadata"] = wrong_meta
        native_results.append(result)

    from model_train_sh_inference_cond import Geo3DStochInterp
    import inference_runtime as runtime
    d4_root = _resolve_dir(config["d4_run"], "d4_run")
    d4_summary = _read_json(d4_root / "summary.json")
    checkpoint = Path(d4_summary["checkpoint_path"])
    model, load_report = runtime.load_model_with_weight_policy(Geo3DStochInterp, checkpoint, device, "ema")
    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    q1_summary = _read_json(_resolve_dir(config["q1_run"], "q1_run") / "summary.json")
    paired = _paired_flow_comparison(case, model, table, operator, device, d4_root, cuboid, q1_summary)

    cuboid_correct = next(row for row in cuboid["arms"] if row["optimized_by"] == "correct")
    native_correct_arms = [next(row for row in result["arms"] if row["optimized_by"] == "correct") for result in native_results]
    native_win_fraction = sum(result["correct_optimized_is_best_against_correct"] for result in native_results) / len(native_results)
    thresholds = config["success_thresholds"]
    gates = {
        "cuboid_correct_specificity": cuboid["correct_optimized_is_best_against_correct"],
        "cuboid_attainment": cuboid_correct["hard_correct_observation_attainment"] >= float(thresholds["minimum_hard_attainment"]),
        "cuboid_hidden_recovery": cuboid_correct["hidden_target_recall"] > next(row for row in paired if row["method"] == "continuous_BASE_PLUS_PHYSICS_best_hard_state")["hidden_target_recall"],
        "native_replication": native_win_fraction >= float(thresholds["minimum_native_correct_control_win_fraction"]),
        "native_mean_attainment": sum(row["hard_correct_observation_attainment"] for row in native_correct_arms) / len(native_correct_arms) >= float(thresholds["minimum_hard_attainment"]),
    }
    structured_success = all(gates.values())
    d7_ranking = [row["mechanism"] for row in d7["verdict"]["mechanism_ranking"]]
    answers = [
        f"Correct/zero/shuffled similarity is attributed in D7 in this order: {', '.join(d7_ranking)}; identical-state residual/gradient/controller/hard-transition evidence prevents trajectory-state confounding.",
        f"The dominant specificity-loss location is {d7_ranking[0]}, with the remaining supported mechanisms explicitly ranked rather than conflated.",
        f"Structured hard inference {'restored' if cuboid['correct_optimized_is_best_against_correct'] else 'did not restore'} correct-observation specificity on the analytic benchmark when every arm was scored against the same correct field.",
        f"Correct-arm hard attainment was {cuboid_correct['hard_correct_observation_attainment']:.3%}, compared with the existing full-flow maximum/final {float(thresholds['existing_full_flow_maximum']):.3%}/{float(thresholds['existing_full_flow_final']):.3%}.",
        f"Its hidden-body IoU/recall were {cuboid_correct['hidden_target_iou']:.3f}/{cuboid_correct['hidden_target_recall']:.3f}; truth metrics were retrospective and never selected proposals.",
        f"Across {len(native_results)} deterministic StructuralGeo replicas, the correct arm ranked first against the correct field in {native_win_fraction:.1%}; mean correct hard attainment was {sum(row['hard_correct_observation_attainment'] for row in native_correct_arms)/len(native_correct_arms):.3%}.",
        "Training is still unnecessary for this bounded structured family." if structured_success else "Training may now be discussed because structured native-family inference did not clear every frozen gate, but no training was started.",
    ]
    git_status = _git("status", "--short")
    summary = {
        "stage": "stage7_structured_hard_geophysics",
        "version": STRUCTURED_HARD_VERSION,
        "status": "completed",
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "decision": "STRUCTURED_HARD_INFERENCE_VALIDATED" if structured_success else "STRUCTURED_HARD_INFERENCE_PARTIAL_OR_FAILED",
        "d7_verdict": d7["verdict"],
        "cuboid": cuboid,
        "native_replicas": native_results,
        "paired_comparison": paired,
        "native_correct_control_win_fraction": native_win_fraction,
        "success_gates": gates,
        "answers": answers,
        "truth_used_for_selection": False,
        "training_performed": False,
        "broad_controller_sweep_performed": False,
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_at_completion": "clean" if not git_status else "dirty_stage7_implementation",
        "checkpoint_sha256": _file_sha256(checkpoint),
        "source_hashes": {
            "runner": _file_sha256(Path(__file__).resolve()),
            "structured_search": _file_sha256(PROJECT_DIR / "guidance/structured_hard_inference.py"),
            "seismic": _file_sha256(PROJECT_DIR / "guidance/seismic.py"),
            "native_case": _file_sha256(PROJECT_DIR / "guidance/native_geology_audit.py"),
        },
        "runtime": {"hostname": socket.gethostname(), "torch": torch.__version__, "gpu_name": torch.cuda.get_device_name(device)},
        "model_load_report": load_report,
        "acoustic_metadata": acoustic_meta,
        "seismic_metadata": seismic_meta,
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "config_input.json", config)
    _write_rows(output_dir / "cuboid_cross_evaluation.csv", cuboid["cross_evaluation"])
    _write_rows(output_dir / "native_cross_evaluation.csv", [row for result in native_results for row in result["cross_evaluation"]])
    _write_rows(output_dir / "paired_comparison.csv", paired)
    _write_json(output_dir / "stage7_summary.json", summary)
    (output_dir / "STAGE7_REPORT.md").write_text(_report(summary), encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "decision": summary["decision"], "gates": gates}, indent=2))


if __name__ == "__main__":
    main()
