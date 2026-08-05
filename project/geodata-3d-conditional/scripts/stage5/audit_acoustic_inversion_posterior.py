#!/usr/bin/env python3
"""Audit the completed Phase-5a posterior against truth after construction."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
import sys
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.property_evaluation import (
    per_class_hard_metrics,
    truth_component_recovery_rows,
)
from guidance.seismic import (
    hard_labels_to_acoustic,
    seismic_field_loss,
    seismic_operator_from_config,
    tensor_sha256,
)
from guidance.seismic_inversion import (
    build_exact_condition_acoustic,
    nearest_codebook_labels,
)
from scripts.stage4.run_seismic_guidance import read_json, write_json, write_rows
from scripts.stage4.audit_seismic_identifiability import validate_output_directory
from scripts.stage5.build_acoustic_inversion_posterior import (
    OUTPUT_TENSOR_FILES,
    PHASE5A_BUILD_SCHEMA,
)


PHASE5A_AUDIT_SCHEMA = "phase5a_acoustic_inversion_truth_audit_v1"


def parse_args() -> argparse.Namespace:
    default_builder = (
        PROJECT_DIR
        / "experiments/stage5_acoustic_inversion/outputs/cond_generation_0"
        / "model_based_fixed12_v1"
    )
    parser = argparse.ArgumentParser(
        description="Audit a completed Phase-5a posterior against synthetic truth.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--builder-dir", type=Path, default=default_builder)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--observation-dir", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _recorded_path(record: Mapping[str, object]) -> Path:
    path = Path(str(record.get("path", "")))
    return path if path.is_absolute() else REPOSITORY_ROOT / path


def _validate_file_record(record: Mapping[str, object], expected: Path | None = None) -> Path:
    path = expected or _recorded_path(record)
    if runtime.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"asset hash mismatch: {path}")
    return path


def _load_generated_tensors(
    builder_dir: Path, manifest: Mapping[str, object]
) -> dict[str, torch.Tensor]:
    records = manifest.get("generated_tensors")
    if not isinstance(records, Mapping) or set(records) != set(OUTPUT_TENSOR_FILES):
        raise ValueError("builder generated tensor inventory is incomplete or unexpected")
    tensors: dict[str, torch.Tensor] = {}
    for filename in OUTPUT_TENSOR_FILES:
        record = records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"invalid generated tensor record: {filename}")
        path = builder_dir / filename
        value = runtime.load_tensor(path)
        if list(value.shape) != record.get("shape") or str(value.dtype) != record.get("dtype"):
            raise ValueError(f"generated tensor shape/dtype mismatch: {filename}")
        if tensor_sha256(value) != record.get("tensor_sha256"):
            raise ValueError(f"generated tensor content mismatch: {filename}")
        _validate_file_record(record, expected=path)
        tensors[filename] = value
    return tensors


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _rmse(value: torch.Tensor) -> float:
    return float(value.square().mean().sqrt().detach().cpu())


def _mae(value: torch.Tensor) -> float:
    return float(value.abs().mean().detach().cpu())


def _safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else float("nan")


def _projection_metrics(
    labels: torch.Tensor,
    truth: torch.Tensor,
    *,
    target_label: int,
    member_id: int,
    variant: str,
) -> dict[str, object]:
    valid = truth != -1
    target = truth == target_label
    predicted = labels == target_label
    intersection = int((target & predicted).sum().item())
    union = int((target | predicted).sum().item())
    predicted_count = int(predicted.sum().item())
    truth_count = int(target.sum().item())
    return {
        "member_id": member_id,
        "variant": variant,
        "global_voxel_accuracy": float((labels[valid] == truth[valid]).float().mean()),
        "target_iou": _safe_ratio(intersection, union),
        "target_precision": _safe_ratio(intersection, predicted_count),
        "target_recall": _safe_ratio(intersection, truth_count),
        "target_predicted_voxels": predicted_count,
        "target_truth_voxels": truth_count,
    }


def _condition_violation_count(
    acoustic: torch.Tensor, target: torch.Tensor, condition_mask: torch.Tensor
) -> int:
    expanded = condition_mask.to(acoustic.device).expand(
        acoustic.shape[0], 2, *condition_mask.shape[2:]
    )
    exact = target.to(acoustic).expand_as(acoustic)
    return int(((acoustic != exact) & expanded).any(dim=1).sum().item())


def _seismic_metrics(
    acoustic: torch.Tensor,
    *,
    operator,
    subsurface: torch.Tensor,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
) -> dict[str, float]:
    field = operator(acoustic[:, 0:1], acoustic[:, 1:2], subsurface)
    loss, diagnostics = seismic_field_loss(field, observed, sample_mask, uncertainty)
    return {
        "loss": float(loss.detach().cpu()),
        "rmse": float(diagnostics["seismic_rmse_amplitude"].detach().cpu()),
        "mae": float(diagnostics["seismic_mae_amplitude"].detach().cpu()),
    }


def _close(first: float, second: float) -> bool:
    return math.isclose(first, second, rel_tol=1e-10, abs_tol=1e-12)


def main() -> None:
    args = parse_args()
    output_dir = args.output_dir or args.builder_dir / "audit"
    validate_output_directory(output_dir, overwrite=args.overwrite)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable; use --device cpu")

    manifest = read_json(args.builder_dir / "manifest.json")
    build_config = read_json(args.builder_dir / "config.json")
    if manifest.get("schema") != PHASE5A_BUILD_SCHEMA or manifest.get("status") != "complete":
        raise ValueError("Phase-5a builder manifest is not complete")
    if build_config.get("schema") != PHASE5A_BUILD_SCHEMA or build_config.get("status") != "complete":
        raise ValueError("Phase-5a builder config is not complete")
    anti_leakage = manifest.get("anti_leakage")
    expected_anti_leakage = {
        "truth_geology_loaded": False,
        "truth_acoustic_loaded": False,
        "unconstrained_truth_used": False,
    }
    if not isinstance(anti_leakage, Mapping) or any(
        anti_leakage.get(field) is not expected
        for field, expected in expected_anti_leakage.items()
    ):
        raise ValueError("builder anti-leakage declaration is invalid")
    source_assets = manifest.get("source_assets")
    if not isinstance(source_assets, Mapping):
        raise ValueError("builder manifest lacks source assets")
    for name in (
        "observation_manifest",
        "boreholes",
        "checkpoint_not_loaded",
        "inversion_config",
        "builder_source",
        "inversion_source",
        "seismic_source",
    ):
        record = source_assets.get(name)
        if not isinstance(record, Mapping):
            raise ValueError(f"builder manifest lacks source asset: {name}")
        _validate_file_record(record)
    build_record = manifest.get("build_config")
    metrics_record = manifest.get("member_metrics")
    if not isinstance(build_record, Mapping) or not isinstance(metrics_record, Mapping):
        raise ValueError("builder manifest lacks config/metrics records")
    _validate_file_record(build_record, args.builder_dir / "config.json")
    _validate_file_record(metrics_record, args.builder_dir / "member_inversion_metrics.csv")
    tensors = _load_generated_tensors(args.builder_dir, manifest)

    observation_manifest_path = _recorded_path(source_assets["observation_manifest"])
    observation_dir = args.observation_dir or observation_manifest_path.parent
    if observation_dir / "manifest.json" != observation_manifest_path:
        raise ValueError("requested observation directory differs from builder source")
    observation_manifest = read_json(observation_manifest_path)
    observation_records = observation_manifest.get("generated_tensors")
    observation_sources = observation_manifest.get("source_assets")
    if not isinstance(observation_records, Mapping) or not isinstance(
        observation_sources, Mapping
    ):
        raise ValueError("invalid observation manifest")
    truth_record = observation_sources.get("truth_model")
    if not isinstance(truth_record, Mapping):
        raise ValueError("observation manifest lacks truth source")
    truth_path = args.truth_model or _recorded_path(truth_record)
    _validate_file_record(truth_record, truth_path)
    truth = runtime.normalize_single_geology(runtime.load_tensor(truth_path), str(truth_path)).long()

    def load_observation_tensor(filename: str) -> torch.Tensor:
        record = observation_records.get(filename)
        if not isinstance(record, Mapping):
            raise ValueError(f"observation tensor record missing: {filename}")
        value = runtime.load_tensor(observation_dir / filename)
        if tensor_sha256(value) != record.get("sha256"):
            raise ValueError(f"observation tensor mismatch: {filename}")
        return value

    table = load_observation_tensor("acoustic_property_table.pt")
    truth_acoustic = load_observation_tensor("truth_acoustic.pt").to(torch.float64)
    expected_truth_acoustic = hard_labels_to_acoustic(truth, table).to(torch.float64)
    if not torch.equal(truth_acoustic, expected_truth_acoustic):
        raise ValueError("truth acoustic is inconsistent with truth/codebook")
    subsurface = load_observation_tensor("subsurface_mask.pt").bool()
    observed = load_observation_tensor("observed_seismic.pt").to(device)
    sample_mask = load_observation_tensor("sample_mask.pt").to(device)
    uncertainty = load_observation_tensor("uncertainty_amplitude.pt").to(device)
    operator, _ = seismic_operator_from_config(
        observation_manifest["observation_config_resolved"], grid_shape=truth.shape[2:]
    )
    boreholes_path = _recorded_path(source_assets["boreholes"])
    boreholes = runtime.normalize_single_geology(runtime.load_tensor(boreholes_path), str(boreholes_path)).long()
    condition_target, condition_mask = build_exact_condition_acoustic(
        boreholes, subsurface, table
    )
    if not torch.equal(condition_mask, tensors["condition_mask.pt"].bool()):
        raise ValueError("saved condition mask differs from audited inputs")
    if not torch.equal(condition_target, tensors["condition_acoustic.pt"]):
        raise ValueError("saved condition acoustic differs from audited inputs")

    prior_members = tensors["prior_acoustic_members.pt"].to(device)
    inverted_members = tensors["inverted_acoustic_members.pt"].to(device)
    if prior_members.shape != inverted_members.shape or prior_members.shape[0] != 12:
        raise ValueError("audit requires matching fixed-12 acoustic member tensors")
    if not torch.equal(prior_members[:, 1], inverted_members[:, 1]):
        raise ValueError("Phase-5a v1 must not update slowness")
    if not torch.isfinite(prior_members).all() or not torch.isfinite(inverted_members).all():
        raise ValueError("acoustic members contain non-finite values")

    unconstrained = subsurface & (~condition_mask)
    target_unconstrained = unconstrained & (truth == int(args.target_label))
    if int(unconstrained.sum()) == 0 or int(target_unconstrained.sum()) == 0:
        raise ValueError("truth audit masks must be non-empty")
    truth_log_impedance = truth_acoustic[:, 0:1].log().to(device)
    unconstrained_device = unconstrained.to(device)
    target_device = target_unconstrained.to(device)
    condition_mask_device = condition_mask.to(device)
    condition_target_device = condition_target.to(device)
    subsurface_device = subsurface.to(device)
    table_device = table.to(device)
    truth_device = truth.to(device)

    build_rows = _read_rows(args.builder_dir / "member_inversion_metrics.csv")
    if len(build_rows) != 12:
        raise ValueError("builder member metric table must contain 12 rows")
    member_rows: list[dict[str, object]] = []
    projection_rows: list[dict[str, object]] = []
    per_class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for member_id in range(12):
        prior = prior_members[member_id : member_id + 1]
        inverted = inverted_members[member_id : member_id + 1]
        prior_log = prior[:, 0:1].log()
        inverted_log = inverted[:, 0:1].log()
        prior_seismic = _seismic_metrics(
            prior,
            operator=operator,
            subsurface=subsurface_device,
            observed=observed,
            sample_mask=sample_mask,
            uncertainty=uncertainty,
        )
        inverted_seismic = _seismic_metrics(
            inverted,
            operator=operator,
            subsurface=subsurface_device,
            observed=observed,
            sample_mask=sample_mask,
            uncertainty=uncertainty,
        )
        build_row = build_rows[member_id]
        for field, value in (
            ("prior_seismic_rmse", prior_seismic["rmse"]),
            ("inverted_seismic_rmse", inverted_seismic["rmse"]),
        ):
            if not _close(float(build_row[field]), value):
                raise ValueError(f"audited seismic metric differs from builder: {field}")
        prior_rmse = _rmse((prior_log - truth_log_impedance)[unconstrained_device])
        inverted_rmse = _rmse(
            (inverted_log - truth_log_impedance)[unconstrained_device]
        )
        prior_target_mae = _mae(
            (prior_log - truth_log_impedance)[target_device]
        )
        inverted_target_mae = _mae(
            (inverted_log - truth_log_impedance)[target_device]
        )
        prior_labels = nearest_codebook_labels(prior, table_device, subsurface_device)
        inverted_labels = nearest_codebook_labels(
            inverted, table_device, subsurface_device
        )
        member_rows.append(
            {
                "member_id": member_id,
                "candidate_id": build_row["candidate_id"],
                "prior_condition_violation_count": _condition_violation_count(
                    prior, condition_target_device, condition_mask_device
                ),
                "inverted_condition_violation_count": _condition_violation_count(
                    inverted, condition_target_device, condition_mask_device
                ),
                "prior_seismic_rmse": prior_seismic["rmse"],
                "inverted_seismic_rmse": inverted_seismic["rmse"],
                "seismic_rmse_improved": inverted_seismic["rmse"] < prior_seismic["rmse"],
                "prior_unconstrained_log_impedance_rmse": prior_rmse,
                "inverted_unconstrained_log_impedance_rmse": inverted_rmse,
                "log_impedance_rmse_improved": inverted_rmse < prior_rmse,
                "prior_target_log_impedance_mae": prior_target_mae,
                "inverted_target_log_impedance_mae": inverted_target_mae,
                "target_log_impedance_mae_improved": (
                    inverted_target_mae < prior_target_mae
                ),
            }
        )
        for variant, labels in (("prior", prior_labels), ("inverted", inverted_labels)):
            projection_rows.append(
                _projection_metrics(
                    labels,
                    truth_device,
                    target_label=args.target_label,
                    member_id=member_id,
                    variant=variant,
                )
            )
            class_rows = per_class_hard_metrics(
                labels.cpu(), truth, sample_id=member_id, class_ids=range(table.shape[1] - 1)
            )
            for row in class_rows:
                row["variant"] = variant
                per_class_rows.append(row)
            recovered = truth_component_recovery_rows(
                labels.cpu(), truth, args.target_label, sample_id=member_id
            )
            for row in recovered:
                row["variant"] = variant
                component_rows.append(row)

    prior_mean = tensors["prior_acoustic_mean.pt"].to(device)
    posterior_mean = tensors["posterior_acoustic_mean.pt"].to(device)
    prior_mean_rmse = _rmse(
        (prior_mean[:, 0:1].log() - truth_log_impedance)[unconstrained_device]
    )
    posterior_mean_rmse = _rmse(
        (posterior_mean[:, 0:1].log() - truth_log_impedance)[unconstrained_device]
    )
    prior_mean_target_mae = _mae(
        (prior_mean[:, 0:1].log() - truth_log_impedance)[target_device]
    )
    posterior_mean_target_mae = _mae(
        (posterior_mean[:, 0:1].log() - truth_log_impedance)[target_device]
    )
    posterior_std = tensors["posterior_log_impedance_std.pt"].to(device)
    spread_values = posterior_std[unconstrained_device]
    spread_valid = bool(
        torch.isfinite(spread_values).all()
        and (spread_values >= 0).all()
        and float(spread_values.max()) > 0
    )
    seismic_improved = sum(bool(row["seismic_rmse_improved"]) for row in member_rows)
    property_improved = sum(
        bool(row["log_impedance_rmse_improved"]) for row in member_rows
    )
    conditions_exact = all(
        int(row["prior_condition_violation_count"]) == 0
        and int(row["inverted_condition_violation_count"]) == 0
        for row in member_rows
    ) and _condition_violation_count(
        posterior_mean, condition_target_device, condition_mask_device
    ) == 0
    checks = {
        "all_members_and_mean_conditions_exact": conditions_exact,
        "seismic_rmse_improved_at_least_9_of_12": seismic_improved >= 9,
        "member_log_impedance_rmse_improved_at_least_9_of_12": (
            property_improved >= 9
        ),
        "posterior_mean_log_impedance_rmse_improved": (
            posterior_mean_rmse < prior_mean_rmse
        ),
        "posterior_mean_target_log_impedance_mae_improved": (
            posterior_mean_target_mae < prior_mean_target_mae
        ),
        "posterior_spread_finite_nonnegative_nonzero": spread_valid,
    }
    promoted = all(checks.values())
    summary = {
        "schema": PHASE5A_AUDIT_SCHEMA,
        "decision": (
            "PASS: eligible for a strictly paired Phase-2-style property-guidance bridge test"
            if promoted
            else "FAIL: do not bridge this inversion posterior into flow guidance"
        ),
        "promoted_to_property_guidance_bridge_test": promoted,
        "not_a_geological_recovery_claim": True,
        "checks": checks,
        "member_counts": {
            "total": 12,
            "seismic_rmse_improved": seismic_improved,
            "log_impedance_rmse_improved": property_improved,
            "target_log_impedance_mae_improved": sum(
                bool(row["target_log_impedance_mae_improved"]) for row in member_rows
            ),
        },
        "ensemble_mean_metrics": {
            "prior_unconstrained_log_impedance_rmse": prior_mean_rmse,
            "posterior_unconstrained_log_impedance_rmse": posterior_mean_rmse,
            "prior_target_log_impedance_mae": prior_mean_target_mae,
            "posterior_target_log_impedance_mae": posterior_mean_target_mae,
            "posterior_spread_mean": float(spread_values.mean().detach().cpu()),
            "posterior_spread_max": float(spread_values.max().detach().cpu()),
            "posterior_mean_condition_violation_count": _condition_violation_count(
                posterior_mean, condition_target_device, condition_mask_device
            ),
        },
        "truth_usage": {
            "builder_used_unconstrained_truth": False,
            "audit_opened_truth_after_completed_manifest": True,
            "truth_used_for_selection_or_regularization": False,
            "truth_used_for_frozen_gate_evaluation": True,
        },
        "target_label": int(args.target_label),
        "unconstrained_subsurface_voxels": int(unconstrained.sum()),
        "unconstrained_target_voxels": int(target_unconstrained.sum()),
    }

    for variant, acoustic in (("prior_mean", prior_mean), ("posterior_mean", posterior_mean)):
        labels = nearest_codebook_labels(acoustic, table_device, subsurface_device)
        projection_rows.append(
            _projection_metrics(
                labels,
                truth_device,
                target_label=args.target_label,
                member_id=-1,
                variant=variant,
            )
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(output_dir / "member_truth_audit.csv", member_rows)
    write_rows(output_dir / "hard_projection_metrics.csv", projection_rows)
    write_rows(output_dir / "hard_projection_per_class.csv", per_class_rows)
    write_rows(output_dir / "hard_projection_truth_components.csv", component_rows)
    write_json(output_dir / "summary.json", summary)
    lines = [
        "# Phase 5a acoustic inversion audit",
        "",
        f"Decision: **{summary['decision']}**",
        "",
        "This decision only controls whether the posterior may enter a later strictly paired",
        "property-guidance test. It is not a claim that hard geology has been recovered.",
        "",
        "## Frozen checks",
        "",
    ]
    lines.extend(
        f"- {'PASS' if passed else 'FAIL'} — `{name}`"
        for name, passed in checks.items()
    )
    lines.extend(
        [
            "",
            "## Key counts",
            "",
            f"- seismic RMSE improved: {seismic_improved}/12",
            f"- unconstrained log-impedance RMSE improved: {property_improved}/12",
            f"- posterior/prior mean log-impedance RMSE: {posterior_mean_rmse:.8g} / {prior_mean_rmse:.8g}",
            f"- posterior/prior target log-impedance MAE: {posterior_mean_target_mae:.8g} / {prior_mean_target_mae:.8g}",
            "",
        ]
    )
    (output_dir / "REPORT.md").write_text("\n".join(lines), encoding="utf-8")
    audit_manifest = {
        "schema": PHASE5A_AUDIT_SCHEMA,
        "status": "complete",
        "source_assets": {
            "builder_manifest": runtime.asset_record(args.builder_dir / "manifest.json"),
            "builder_config": runtime.asset_record(args.builder_dir / "config.json"),
            "truth_model": runtime.asset_record(truth_path),
            "observation_manifest": runtime.asset_record(observation_manifest_path),
            "auditor_source": runtime.asset_record(Path(__file__)),
        },
        "outputs": {
            name: runtime.asset_record(output_dir / name)
            for name in (
                "summary.json",
                "member_truth_audit.csv",
                "hard_projection_metrics.csv",
                "hard_projection_per_class.csv",
                "hard_projection_truth_components.csv",
                "REPORT.md",
            )
        },
    }
    write_json(output_dir / "manifest.json", audit_manifest)
    print(summary["decision"])
    print(f"Audit complete: {output_dir}")


if __name__ == "__main__":
    main()
