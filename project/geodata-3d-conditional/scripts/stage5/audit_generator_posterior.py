#!/usr/bin/env python3
"""Truth audit for a completed Phase-5c direct generator-posterior chain."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
import statistics
import sys
from typing import Dict, Mapping

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.probability_evaluation import class_transition_records
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.property_evaluation import (
    paired_per_class_deltas,
    paired_property_metric_deltas,
    paired_truth_component_recovery_deltas,
    per_class_hard_metrics,
    sample_property_hard_metrics,
    truth_component_recovery_rows,
)
from guidance.seismic import tensor_sha256
from scripts.stage4.run_seismic_guidance import (
    add_hard_seismic_metrics,
    load_observation_assets,
    read_json,
    write_json,
    write_rows,
)
from scripts.stage5.run_generator_posterior import PHASE5C_RUN_SCHEMA


PHASE5C_AUDIT_SCHEMA = "phase5c_generator_posterior_truth_audit_v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Audit a completed Phase-5c chain against synthetic truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _read_rows(path: Path) -> list[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return [dict(row) for row in csv.DictReader(stream)]


def _asset_path(record: Mapping[str, object]) -> Path:
    path = Path(str(record["path"]))
    return path if path.is_absolute() else PROJECT_DIR.parents[1] / path


def _validate_asset(record: Mapping[str, object]) -> Path:
    path = _asset_path(record)
    if runtime.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"Phase-5c source asset hash mismatch: {path}")
    return path


def _major_four_mean(
    rows: list[Mapping[str, object]], sample_id: int
) -> float:
    values = [
        float(row["recall"])
        for row in rows
        if int(row["sample_id"]) == sample_id
        and int(row["truth_component_rank"]) <= 4
    ]
    return float(statistics.mean(values)) if values else float("nan")


def _delta_row(
    baseline: Mapping[str, object], current: Mapping[str, object]
) -> Dict[str, object]:
    row = paired_property_metric_deltas(
        {**baseline, "sample_id": int(current["sample_id"])}, current
    )
    for field in (
        "hard_seismic_loss",
        "hard_seismic_rmse_amplitude",
        "hard_seismic_mae_amplitude",
    ):
        row[f"delta_{field}"] = float(current[field]) - float(baseline[field])
    return row


def main() -> None:
    args = parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError(
            f"audit output directory is not empty; refusing to overwrite: {args.output_dir}"
        )
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested for audit but unavailable")

    run_config = read_json(args.run_dir / "config.json")
    if run_config.get("schema") != PHASE5C_RUN_SCHEMA:
        raise ValueError("invalid Phase-5c run schema")
    if run_config.get("run_status") != "completed":
        raise ValueError("Phase-5c run is not complete")
    if run_config.get("truth_metrics_computed_by_sampler") is not False:
        raise ValueError("sampler/truth-audit separation was violated")
    source_assets = run_config.get("asset_records")
    if not isinstance(source_assets, Mapping):
        raise ValueError("Phase-5c run lacks source assets")
    validated_paths = {
        name: _validate_asset(record)
        for name, record in source_assets.items()
        if isinstance(record, Mapping)
    }
    truth_path = validated_paths["truth_model"]
    boreholes_path = validated_paths["boreholes"]
    observation_dir = validated_paths["observation_manifest"].parent

    retained_path = args.run_dir / "retained_samples.pt"
    retained = runtime.load_tensor(retained_path, map_location="cpu")
    if tensor_sha256(retained) != run_config.get("retained_samples_sha256"):
        raise ValueError("retained-sample tensor hash mismatch")
    if retained.ndim != 5 or retained.shape[0] < 2 or retained.shape[1] != 1:
        raise ValueError("retained samples must have shape [N,1,X,Y,Z]")
    final_latent = runtime.load_tensor(args.run_dir / "final_latent.pt", map_location="cpu")
    if tensor_sha256(final_latent) != run_config.get("final_latent_sha256"):
        raise ValueError("final latent hash mismatch")
    if tensor_sha256(retained[-1].long()) != run_config.get("final_sample_sha256"):
        raise ValueError("final retained sample hash mismatch")
    if tensor_sha256(retained[0].long()) != run_config.get("initial_sample_sha256"):
        raise ValueError("initial retained sample hash mismatch")

    trace = _read_rows(args.run_dir / "chain_trace.csv")
    expected_proposals = int(run_config["resolved_protocol"]["chain_proposals"])
    if len(trace) != expected_proposals or retained.shape[0] != expected_proposals + 1:
        raise ValueError("chain trace and retained sample counts disagree")
    recorded_losses = [float(run_config["initial_hard_seismic_loss"])] + [
        float(row["current_hard_seismic_loss"]) for row in trace
    ]
    minimum_likelihood_index = min(
        range(len(recorded_losses)), key=recorded_losses.__getitem__
    )

    truth_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location="cpu"), str(truth_path)
    ).long()
    boreholes_cpu = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location="cpu"), str(boreholes_path)
    ).long()
    conditioning_report = runtime.validate_conditioning_pair(
        truth_cpu,
        boreholes_cpu,
        num_categories=15,
        target_label=args.target_label,
    )
    tensors, observation_manifest, forward_operator, _ = load_observation_assets(
        observation_dir,
        truth_cpu,
        truth_path=truth_path,
        num_categories=15,
    )
    condition_mask = (boreholes_cpu != -1) | (truth_cpu == -1)
    property_confidence = ((truth_cpu != -1) & ~condition_mask).float()
    target_mask, target_metadata = build_target_mask(
        truth_cpu, target_label=args.target_label, component_mode="all"
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)
    property_table = tensors["acoustic_property_table.pt"]
    channel_weights = torch.ones(property_table.shape[0])
    class_ids = list(range(14))

    metrics_rows: list[Dict[str, object]] = []
    per_class_rows: list[Dict[str, object]] = []
    component_rows: list[Dict[str, object]] = []
    fields: list[torch.Tensor] = []
    baseline = retained[0].long()
    for sample_id in range(retained.shape[0]):
        prediction = retained[sample_id].long()
        metrics = sample_property_hard_metrics(
            prediction=prediction,
            truth_model=truth_cpu,
            condition_mask=condition_mask,
            target_mask=target_mask,
            target_roi_mask=target_roi,
            target_label=args.target_label,
            property_table=property_table,
            property_confidence=property_confidence,
            property_sigmas=(0.0,),
            property_scale_weights=(1.0,),
            property_channel_weights=channel_weights,
            sample_id=sample_id,
            baseline_prediction=baseline if sample_id else None,
        )
        fields.append(
            add_hard_seismic_metrics(
                metrics,
                prediction=prediction,
                target_acoustic=tensors["truth_acoustic.pt"],
                condition_mask=condition_mask,
                property_table=property_table,
                subsurface_mask=tensors["subsurface_mask.pt"],
                forward_operator=forward_operator,
                observed=tensors["observed_seismic.pt"],
                sample_mask=tensors["sample_mask.pt"],
                uncertainty=tensors["uncertainty_amplitude.pt"],
                device=device,
            )
        )
        if not torch.isclose(
            torch.tensor(float(metrics["hard_seismic_loss"])),
            torch.tensor(recorded_losses[sample_id]),
            rtol=1e-6,
            atol=1e-6,
        ):
            raise ValueError(f"recomputed seismic loss differs at retained index {sample_id}")
        metrics["retained_role"] = (
            "initial_baseline"
            if sample_id == 0
            else "minimum_likelihood"
            if sample_id == minimum_likelihood_index
            else "final"
            if sample_id == retained.shape[0] - 1
            else "retained"
        )
        metrics_rows.append(metrics)
        per_class_rows.extend(
            per_class_hard_metrics(
                prediction, truth_cpu, sample_id, class_ids=class_ids
            )
        )
        component_rows.extend(
            truth_component_recovery_rows(
                prediction, truth_cpu, args.target_label, sample_id
            )
        )

    baseline_metrics = metrics_rows[0]
    minimum_metrics = metrics_rows[minimum_likelihood_index]
    final_metrics = metrics_rows[-1]
    delta_rows = [
        _delta_row(baseline_metrics, row) for row in metrics_rows[1:]
    ]
    transitions = class_transition_records(
        baseline, retained[minimum_likelihood_index].long(), minimum_likelihood_index
    )
    baseline_classes = [row for row in per_class_rows if int(row["sample_id"]) == 0]
    minimum_classes = [
        row
        for row in per_class_rows
        if int(row["sample_id"]) == minimum_likelihood_index
    ]
    class_deltas = paired_per_class_deltas(
        [
            {**row, "sample_id": minimum_likelihood_index}
            for row in baseline_classes
        ],
        minimum_classes,
    )
    baseline_components = [
        row for row in component_rows if int(row["sample_id"]) == 0
    ]
    minimum_components = [
        row
        for row in component_rows
        if int(row["sample_id"]) == minimum_likelihood_index
    ]
    component_deltas = paired_truth_component_recovery_deltas(
        [
            {**row, "sample_id": minimum_likelihood_index}
            for row in baseline_components
        ],
        minimum_components,
    )

    baseline_major = _major_four_mean(component_rows, 0)
    minimum_major = _major_four_mean(component_rows, minimum_likelihood_index)
    integrity_pass = (
        bool(run_config["historical_baseline_validation"][
            "exact_initial_hard_regression"
        ])
        and int(run_config["max_condition_violations"]) == 0
        and all(int(row["condition_violation_count"]) == 0 for row in metrics_rows)
    )
    mechanism_active = (
        int(run_config["accepted_proposals"]) >= 1
        and int(run_config["unique_retained_hard_samples"]) >= 2
        and any(
            int(row["retained_changed_from_initial_voxels"]) > 0 for row in trace
        )
    )
    physics_improved = float(minimum_metrics["hard_seismic_loss"]) < float(
        baseline_metrics["hard_seismic_loss"]
    )
    geology_directions = {
        "truth_present_mean_iou_improved": float(
            minimum_metrics["truth_present_mean_iou"]
        )
        > float(baseline_metrics["truth_present_mean_iou"]),
        "target_iou_improved": float(minimum_metrics["target_iou"])
        > float(baseline_metrics["target_iou"]),
        "target_recall_improved": float(minimum_metrics["target_recall"])
        > float(baseline_metrics["target_recall"]),
        "major_four_mean_recall_improved": minimum_major > baseline_major,
    }
    geology_promotion_pass = all(geology_directions.values())
    promoted = integrity_pass and mechanism_active and physics_improved and geology_promotion_pass
    summary: Dict[str, object] = {
        "schema": PHASE5C_AUDIT_SCHEMA,
        "run_dir": str(args.run_dir),
        "case_role": "legacy_mechanism_screen_not_held_out",
        "integrity_pass": integrity_pass,
        "mechanism_active": mechanism_active,
        "physics_improved": physics_improved,
        "geology_directions": geology_directions,
        "geology_promotion_pass": geology_promotion_pass,
        "promoted_to_new_held_out_study": promoted,
        "decision": (
            "PASS: authorize a separately frozen new-data posterior study"
            if promoted
            else "FAIL: do not expand this full-dimensional pCN implementation; proceed to Phase 6"
        ),
        "accepted_proposals": int(run_config["accepted_proposals"]),
        "chain_proposals": expected_proposals,
        "acceptance_fraction": float(run_config["acceptance_fraction"]),
        "unique_retained_hard_samples": int(
            run_config["unique_retained_hard_samples"]
        ),
        "minimum_likelihood_retained_index": minimum_likelihood_index,
        "baseline": {
            "hard_seismic_loss": baseline_metrics["hard_seismic_loss"],
            "global_voxel_accuracy": baseline_metrics["global_voxel_accuracy"],
            "truth_present_mean_iou": baseline_metrics["truth_present_mean_iou"],
            "target_iou": baseline_metrics["target_iou"],
            "target_precision": baseline_metrics["target_precision"],
            "target_recall": baseline_metrics["target_recall"],
            "major_four_mean_recall": baseline_major,
        },
        "minimum_likelihood": {
            "hard_seismic_loss": minimum_metrics["hard_seismic_loss"],
            "global_voxel_accuracy": minimum_metrics["global_voxel_accuracy"],
            "truth_present_mean_iou": minimum_metrics[
                "truth_present_mean_iou"
            ],
            "target_iou": minimum_metrics["target_iou"],
            "target_precision": minimum_metrics["target_precision"],
            "target_recall": minimum_metrics["target_recall"],
            "major_four_mean_recall": minimum_major,
        },
        "final": {
            "retained_index": retained.shape[0] - 1,
            "hard_seismic_loss": final_metrics["hard_seismic_loss"],
            "truth_present_mean_iou": final_metrics["truth_present_mean_iou"],
            "target_iou": final_metrics["target_iou"],
            "target_recall": final_metrics["target_recall"],
        },
        "target_metadata": target_metadata,
        "conditioning_report": conditioning_report,
        "observation_manifest_sha256": runtime.file_sha256(
            observation_dir / "manifest.json"
        ),
        "run_config_sha256": runtime.file_sha256(args.run_dir / "config.json"),
        "retained_samples_sha256": tensor_sha256(retained),
        "hard_seismic_fields_sha256": tensor_sha256(torch.cat(fields, dim=0)),
        "continuous_or_physics_loss_alone_is_not_success": True,
        "publication_evidence": False,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "summary.json", summary)
    write_rows(args.output_dir / "retained_metrics.csv", metrics_rows)
    write_rows(args.output_dir / "paired_deltas.csv", delta_rows)
    write_rows(args.output_dir / "per_class_metrics.csv", per_class_rows)
    write_rows(args.output_dir / "minimum_likelihood_per_class_deltas.csv", class_deltas)
    write_rows(args.output_dir / "truth_component_recovery.csv", component_rows)
    write_rows(
        args.output_dir / "minimum_likelihood_component_deltas.csv", component_deltas
    )
    write_rows(args.output_dir / "minimum_likelihood_class_transitions.csv", transitions)
    report = [
        "# Phase 5c direct generator-posterior audit",
        "",
        f"- Decision: **{summary['decision']}**",
        "- This is a legacy mechanism screen, not held-out publication evidence.",
        f"- Acceptance: {summary['accepted_proposals']}/{summary['chain_proposals']} "
        f"({summary['acceptance_fraction']:.4f}); unique retained hard models: "
        f"{summary['unique_retained_hard_samples']}.",
        f"- Minimum-likelihood retained index: {minimum_likelihood_index}.",
        f"- Hard seismic loss: {baseline_metrics['hard_seismic_loss']:.8g} -> "
        f"{minimum_metrics['hard_seismic_loss']:.8g}.",
        f"- Truth-present mIoU: {baseline_metrics['truth_present_mean_iou']:.8g} -> "
        f"{minimum_metrics['truth_present_mean_iou']:.8g}.",
        f"- Label-{args.target_label} IoU / precision / recall: "
        f"{baseline_metrics['target_iou']:.8g} / "
        f"{baseline_metrics['target_precision']:.8g} / "
        f"{baseline_metrics['target_recall']:.8g} -> "
        f"{minimum_metrics['target_iou']:.8g} / "
        f"{minimum_metrics['target_precision']:.8g} / "
        f"{minimum_metrics['target_recall']:.8g}.",
        f"- Four-major-body mean recall: {baseline_major:.8g} -> {minimum_major:.8g}.",
        f"- Integrity/mechanism/physics/geology gates: {integrity_pass} / "
        f"{mechanism_active} / {physics_improved} / {geology_promotion_pass}.",
        "",
        "Continuous or seismic-loss improvement alone is not a successful geological result.",
    ]
    (args.output_dir / "REPORT.md").write_text("\n".join(report) + "\n", encoding="utf-8")
    print(f"Phase-5c audit complete: {summary['decision']}")


if __name__ == "__main__":
    main()
