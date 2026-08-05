#!/usr/bin/env python3
"""Audit Phase-4d seismic identifiability on a frozen prior candidate pool."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import random
import statistics
import sys
from typing import Mapping, Sequence

import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from guidance.probability_volume import build_target_mask, dilate_mask
from guidance.property_evaluation import (
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


PHASE4D_SCHEMA = "phase4d_seismic_identifiability_v1"
EXPECTED_SEEDS = (42, 142, 242)
SAMPLES_PER_SEED = 4
MAJOR_COMPONENT_RANKS = (1, 2, 3, 4)
QUALITY_FIELDS = (
    "global_voxel_accuracy",
    "truth_present_mean_iou",
    "target_iou",
    "target_recall",
    "major_component_mean_recall",
)
RANKING_GATE_FIELDS = (
    "target_iou",
    "target_recall",
    "major_component_mean_recall",
)
SOURCE_EQUAL_FIELDS = (
    "protocol_version",
    "phase2_protocol_version",
    "stage",
    "checkpoint_sha256",
    "model_weight_source",
    "ema_applied",
    "truth_model_sha256",
    "boreholes_sha256",
    "property_config_sha256",
    "property_table_sha256",
    "n_samples",
    "n_steps",
    "integrator",
    "initial_noise_policy",
    "tau_start",
    "tau_end",
    "tau_schedule",
    "condition_projection",
)


def _default_baseline_dirs() -> list[Path]:
    root = (
        PROJECT_DIR
        / "experiments/stage2_property/runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
    )
    return [
        root / f"seed{seed}_n4_s32_a025_c025/baseline"
        for seed in EXPECTED_SEEDS
    ]


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage4_seismic_identifiability"
    parser = argparse.ArgumentParser(
        description="Rank a frozen alpha-zero prior pool by hard seismic loss.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--baseline-dir",
        action="append",
        type=Path,
        default=None,
        help="Repeat exactly three times for seeds 42, 142 and 242.",
    )
    parser.add_argument(
        "--phase4c-anchor-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
        / "seed42_n1_s32_a025_c025/baseline",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=PROJECT_DIR / "samples/jupyter-demo/cond_generation_0",
    )
    parser.add_argument(
        "--ckpt-path",
        type=Path,
        default=PROJECT_DIR / "demo_model/conditional-weights.ckpt",
    )
    parser.add_argument(
        "--observation-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage4_seismic/observations/cond_generation_0"
        / "distinct_upper_bound_v1_fix2",
    )
    parser.add_argument("--target-label", type=int, default=9)
    parser.add_argument("--target-roi-radius", type=int, default=6)
    parser.add_argument("--permutations", type=int, default=10_000)
    parser.add_argument("--permutation-seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root / "reports/cond_generation_0/fixed12_v1",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _mean(rows: Sequence[Mapping[str, object]], field: str) -> float:
    return sum(float(row[field]) for row in rows) / len(rows)


def _average_ranks(values: Sequence[float]) -> list[float]:
    if not values:
        raise ValueError("rank correlation requires at least one value")
    if any(not math.isfinite(float(value)) for value in values):
        raise ValueError("rank values must be finite")
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        value = float(values[order[start]])
        while end < len(order) and float(values[order[end]]) == value:
            end += 1
        average = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_rank_correlation(
    first: Sequence[float], second: Sequence[float]
) -> float:
    """Return Spearman rho with deterministic average ranks for ties."""
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    x = _average_ranks(first)
    y = _average_ranks(second)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0 or y_norm == 0:
        return float("nan")
    return numerator / (x_norm * y_norm)


def one_sided_negative_permutation_pvalue(
    loss: Sequence[float],
    quality: Sequence[float],
    *,
    permutations: int,
    seed: int,
) -> float:
    """Estimate P(rho_null <= rho_observed) with a fixed Monte Carlo stream."""
    if permutations <= 0:
        raise ValueError("permutations must be positive")
    observed = spearman_rank_correlation(loss, quality)
    if not math.isfinite(observed):
        return float("nan")
    generator = random.Random(seed)
    shuffled = [float(value) for value in quality]
    extreme = 0
    for _ in range(permutations):
        generator.shuffle(shuffled)
        rho = spearman_rank_correlation(loss, shuffled)
        if math.isfinite(rho) and rho <= observed + 1e-15:
            extreme += 1
    return (extreme + 1) / (permutations + 1)


def geological_support_checks(row: Mapping[str, object]) -> dict[str, bool]:
    """Evaluate the frozen absolute support thresholds for one candidate."""
    return {
        "conditions_exact": int(row["condition_violation_count"]) == 0,
        "target_iou": float(row["target_iou"]) >= 0.30,
        "target_precision": float(row["target_precision"]) >= 0.75,
        "target_recall": float(row["target_recall"]) >= 0.30,
        "major_component_min_recall": (
            float(row["major_component_min_recall"]) >= 0.25
        ),
        "major_component_mean_recall": (
            float(row["major_component_mean_recall"]) >= 0.40
        ),
    }


def validate_output_directory(path: Path, *, overwrite: bool) -> None:
    """Refuse accidental reuse of a non-empty evidence directory."""
    if path.exists() and any(path.iterdir()) and not overwrite:
        raise FileExistsError(f"output directory is not empty: {path}")


def build_selection_summary(
    rows: Sequence[Mapping[str, object]],
    *,
    permutations: int = 10_000,
    permutation_seed: int = 0,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    """Rank candidates without truth and audit the frozen support/ranking gates."""
    if len(rows) < 3:
        raise ValueError("selection audit requires at least three candidates")
    identities = {
        (int(row["seed"]), int(row["local_sample_id"])) for row in rows
    }
    if len(identities) != len(rows):
        raise ValueError("candidate identities must be unique")
    required = (
        "hard_seismic_loss",
        "target_iou",
        "target_precision",
        "target_recall",
        "major_component_min_recall",
        "major_component_mean_recall",
        "global_voxel_accuracy",
        "truth_present_mean_iou",
    )
    for row in rows:
        for field in required:
            if not math.isfinite(float(row[field])):
                raise ValueError(f"candidate metric must be finite: {field}")

    ordered_source = sorted(
        rows,
        key=lambda row: (
            float(row["hard_seismic_loss"]),
            int(row["seed"]),
            int(row["local_sample_id"]),
        ),
    )
    ranking: list[dict[str, object]] = []
    for rank, row in enumerate(ordered_source, start=1):
        ranked = dict(row)
        ranked["seismic_rank"] = rank
        ranking.append(ranked)

    losses = [float(row["hard_seismic_loss"]) for row in rows]
    correlations: dict[str, object] = {}
    for field_index, field in enumerate(QUALITY_FIELDS):
        quality = [float(row[field]) for row in rows]
        rho = spearman_rank_correlation(losses, quality)
        pvalue = one_sided_negative_permutation_pvalue(
            losses,
            quality,
            permutations=permutations,
            seed=permutation_seed + field_index,
        )
        correlations[field] = {
            "spearman_rho": rho if math.isfinite(rho) else None,
            "one_sided_negative_permutation_pvalue": (
                pvalue if math.isfinite(pvalue) else None
            ),
            "permutations": permutations,
        }

    support_rows: list[dict[str, object]] = []
    support_ids: list[str] = []
    for row in rows:
        checks = geological_support_checks(row)
        passed = all(checks.values())
        support_rows.append(
            {
                "candidate_id": str(row["candidate_id"]),
                "passed": passed,
                "checks": checks,
            }
        )
        if passed:
            support_ids.append(str(row["candidate_id"]))

    top_three = ranking[:3]
    ensemble_mean = {field: _mean(rows, field) for field in QUALITY_FIELDS}
    top_three_mean = {field: _mean(top_three, field) for field in QUALITY_FIELDS}
    ensemble_median = {
        field: statistics.median(float(row[field]) for row in rows)
        for field in QUALITY_FIELDS
    }
    oracle_best = {
        field: {
            "candidate_id": str(max(rows, key=lambda row: float(row[field]))["candidate_id"]),
            "value": max(float(row[field]) for row in rows),
        }
        for field in QUALITY_FIELDS
    }

    correlation_checks = {
        field: (
            correlations[field]["spearman_rho"] is not None
            and float(correlations[field]["spearman_rho"]) <= -0.50
        )
        for field in RANKING_GATE_FIELDS
    }
    top_three_checks = {
        field: top_three_mean[field] > ensemble_mean[field]
        for field in RANKING_GATE_FIELDS
    }
    top_one_checks = geological_support_checks(ranking[0])
    ranking_checks = {
        "target_correlations": all(correlation_checks.values()),
        "top_three_target_enrichment": all(top_three_checks.values()),
        "top_one_complete_support": all(top_one_checks.values()),
    }
    support_passed = bool(support_ids)
    ranking_passed = all(ranking_checks.values())
    promoted = support_passed and ranking_passed
    if promoted:
        decision = "PASS: support and seismic ranking gates pass"
    elif not support_passed:
        decision = "FAIL: frozen prior pool lacks geological support"
    else:
        decision = "FAIL: seismic loss does not select the supported geology"

    summary = {
        "decision": decision,
        "promoted": promoted,
        "n_candidates": len(rows),
        "selection_score": "ascending_hard_seismic_loss_only",
        "selected_candidate_id": str(ranking[0]["candidate_id"]),
        "selected_hard_seismic_loss": float(ranking[0]["hard_seismic_loss"]),
        "support_gate": {
            "passed": support_passed,
            "passing_candidate_ids": support_ids,
            "candidates": support_rows,
        },
        "ranking_gate": {
            "passed": ranking_passed,
            "checks": ranking_checks,
            "correlation_checks": correlation_checks,
            "top_three_enrichment_checks": top_three_checks,
            "top_one_support_checks": top_one_checks,
        },
        "correlations": correlations,
        "ensemble_mean": ensemble_mean,
        "ensemble_median": ensemble_median,
        "top_three_mean": top_three_mean,
        "oracle_best": oracle_best,
    }
    return ranking, summary


def validate_source_config(
    config: Mapping[str, object],
    *,
    expected_seed: int,
    expected_truth_hash: str,
    expected_boreholes_hash: str,
    expected_checkpoint_hash: str,
) -> None:
    """Validate one historical alpha-zero source run without loading the model."""
    expected = {
        "seed": expected_seed,
        "n_samples": SAMPLES_PER_SEED,
        "samples_written": SAMPLES_PER_SEED,
        "n_steps": 32,
        "alpha": 0.0,
        "run_status": "completed",
        "ema_applied": True,
        "model_weight_source": "ema",
        "integrator": runtime.PAIRED_INTEGRATOR,
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "max_post_projection_condition_violations": 0,
        "truth_model_sha256": expected_truth_hash,
        "boreholes_sha256": expected_boreholes_hash,
        "checkpoint_sha256": expected_checkpoint_hash,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(
                f"seed {expected_seed} source {field}={config.get(field)!r}, expected {value!r}"
            )
    report = config.get("model_load_report")
    if not isinstance(report, Mapping):
        raise ValueError(f"seed {expected_seed} lacks a model-load report")
    if report.get("weight_source") != "ema" or report.get("ema_applied") is not True:
        raise ValueError(f"seed {expected_seed} did not use the EMA policy")
    if report.get("ema_missing_trainable") != [] or report.get("ema_shape_mismatches") != []:
        raise ValueError(f"seed {expected_seed} has invalid EMA coverage")
    if len(config.get("sample_sha256", [])) != SAMPLES_PER_SEED:
        raise ValueError(f"seed {expected_seed} has incomplete sample hashes")
    if len(config.get("initial_noise_sha256", [])) != SAMPLES_PER_SEED:
        raise ValueError(f"seed {expected_seed} has incomplete initial-noise hashes")


def _validate_source_pool(
    baseline_dirs: Sequence[Path],
    *,
    truth_hash: str,
    boreholes_hash: str,
    checkpoint_hash: str,
    anchor_dir: Path,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    if len(baseline_dirs) != len(EXPECTED_SEEDS):
        raise ValueError("Phase 4d requires exactly three source baseline directories")
    indexed: dict[int, tuple[Path, dict[str, object]]] = {}
    for directory in baseline_dirs:
        config = read_json(directory / "config.json")
        seed = int(config.get("seed", -1))
        if seed in indexed:
            raise ValueError(f"duplicate source seed: {seed}")
        indexed[seed] = (directory, config)
    if set(indexed) != set(EXPECTED_SEEDS):
        raise ValueError(f"source seeds must be exactly {EXPECTED_SEEDS}")

    reference: Mapping[str, object] | None = None
    candidates: list[dict[str, object]] = []
    source_records: list[dict[str, object]] = []
    for seed in EXPECTED_SEEDS:
        directory, config = indexed[seed]
        validate_source_config(
            config,
            expected_seed=seed,
            expected_truth_hash=truth_hash,
            expected_boreholes_hash=boreholes_hash,
            expected_checkpoint_hash=checkpoint_hash,
        )
        if reference is None:
            reference = config
        else:
            equal, reason = runtime.require_equal_fields(
                reference, config, SOURCE_EQUAL_FIELDS
            )
            if not equal:
                raise ValueError(f"source baseline semantics differ: {reason}")
        hashes = list(config["sample_sha256"])
        source = {
            "seed": seed,
            "directory": str(directory),
            "config": runtime.asset_record(directory / "config.json"),
            "samples": [],
        }
        for local_sample_id in range(SAMPLES_PER_SEED):
            sample_path = directory / f"sample_{local_sample_id}.pt"
            sample = runtime.normalize_single_geology(
                runtime.load_tensor(sample_path), str(sample_path)
            ).long()
            digest = tensor_sha256(sample.squeeze(0))
            if digest != hashes[local_sample_id]:
                # Historical runners hashed the saved [1,X,Y,Z] decoded tensor.
                original = runtime.load_tensor(sample_path)
                digest = tensor_sha256(original)
            if digest != hashes[local_sample_id]:
                raise ValueError(
                    f"source sample tensor hash mismatch: seed={seed}, sample={local_sample_id}"
                )
            candidate_id = f"seed{seed}_sample{local_sample_id}"
            candidates.append(
                {
                    "candidate_id": candidate_id,
                    "seed": seed,
                    "local_sample_id": local_sample_id,
                    "sample_path": sample_path,
                    "sample_sha256": digest,
                }
            )
            source["samples"].append(runtime.asset_record(sample_path))
        source_records.append(source)

    anchor_config = read_json(anchor_dir / "config.json")
    for field, expected in {
        "seed": 42,
        "n_samples": 1,
        "samples_written": 1,
        "n_steps": 32,
        "alpha": 0.0,
        "run_status": "completed",
        "ema_applied": True,
        "model_weight_source": "ema",
        "max_post_projection_condition_violations": 0,
    }.items():
        if anchor_config.get(field) != expected:
            raise ValueError(f"invalid Phase-4c alpha-zero anchor: {field}")
    anchor = runtime.normalize_single_geology(
        runtime.load_tensor(anchor_dir / "sample_0.pt"), "Phase-4c anchor"
    ).long()
    seed42 = runtime.normalize_single_geology(
        runtime.load_tensor(candidates[0]["sample_path"]), "seed-42 source sample 0"
    ).long()
    if not torch.equal(anchor, seed42):
        raise ValueError("Phase-4c alpha-zero anchor differs from seed-42 source sample 0")
    source_records.append(
        {
            "role": "phase4c_alpha_zero_anchor",
            "directory": str(anchor_dir),
            "config": runtime.asset_record(anchor_dir / "config.json"),
            "sample": runtime.asset_record(anchor_dir / "sample_0.pt"),
        }
    )
    return candidates, source_records


def _substitution_sensitivity(
    *,
    truth: torch.Tensor,
    condition_mask: torch.Tensor,
    target_acoustic: torch.Tensor,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator,
    observed: torch.Tensor,
    sample_mask: torch.Tensor,
    uncertainty: torch.Tensor,
    device: torch.device,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    truth_present = sorted(
        int(value.item()) for value in torch.unique(truth) if int(value.item()) != -1
    )
    destination_labels = range(0, property_table.shape[1] - 1)
    rows: list[dict[str, object]] = []
    for source_label in truth_present:
        mutable = (truth == source_label) & ~condition_mask
        changed_voxels = int(mutable.sum().item())
        if changed_voxels == 0:
            continue
        for destination_label in destination_labels:
            if destination_label == source_label:
                continue
            perturbed = truth.clone()
            perturbed[mutable] = destination_label
            row: dict[str, object] = {
                "source_label": source_label,
                "destination_label": destination_label,
                "changed_voxels": changed_voxels,
                "condition_violation_count": int(
                    (perturbed[condition_mask] != truth[condition_mask]).sum().item()
                ),
            }
            add_hard_seismic_metrics(
                row,
                prediction=perturbed,
                target_acoustic=target_acoustic,
                condition_mask=condition_mask,
                property_table=property_table,
                subsurface_mask=subsurface_mask,
                forward_operator=forward_operator,
                observed=observed,
                sample_mask=sample_mask,
                uncertainty=uncertainty,
                device=device,
            )
            rows.append(row)

    by_source: dict[str, object] = {}
    for source_label in truth_present:
        current = [row for row in rows if int(row["source_label"]) == source_label]
        if not current:
            continue
        ordered = sorted(
            current,
            key=lambda row: (
                float(row["hard_seismic_loss"]), int(row["destination_label"])
            ),
        )
        by_source[str(source_label)] = {
            "changed_voxels": int(ordered[0]["changed_voxels"]),
            "least_visible_destination": int(ordered[0]["destination_label"]),
            "minimum_hard_seismic_loss": float(ordered[0]["hard_seismic_loss"]),
            "minimum_hard_seismic_rmse_amplitude": float(
                ordered[0]["hard_seismic_rmse_amplitude"]
            ),
            "median_hard_seismic_rmse_amplitude": statistics.median(
                float(row["hard_seismic_rmse_amplitude"]) for row in current
            ),
            "maximum_hard_seismic_rmse_amplitude": max(
                float(row["hard_seismic_rmse_amplitude"]) for row in current
            ),
        }
    return rows, {
        "truth_derived_oracle_perturbation": True,
        "selection_score": False,
        "source_labels": truth_present,
        "destination_labels": list(destination_labels),
        "by_source": by_source,
    }


def _render_report(summary: Mapping[str, object]) -> str:
    selection = summary["selection"]
    selected = summary["selected_candidate"]
    target_correlations = selection["correlations"]
    label9_substitution = summary["substitution_sensitivity"]["by_source"].get("9")
    lines = [
        "# Phase-4d seismic identifiability and posterior-selection audit",
        "",
        "## Decision",
        "",
        f"**{selection['decision']}**",
        "",
        "No sample was generated or modified. Ranking used hard seismic loss only; truth was revealed only for audit.",
        "",
        "## Frozen pool",
        "",
        f"- Candidates: `{selection['n_candidates']}` from seeds 42/142/242.",
        f"- Geological support gate: `{selection['support_gate']['passed']}`.",
        f"- Seismic ranking gate: `{selection['ranking_gate']['passed']}`.",
        f"- Promotion: `{selection['promoted']}`.",
        "",
        "## Seismic-selected top candidate",
        "",
        f"- Candidate: `{selected['candidate_id']}`.",
        f"- Hard seismic RMSE: `{float(selected['hard_seismic_rmse_amplitude']):.6f}`.",
        f"- Global accuracy / truth-present mIoU: `{float(selected['global_voxel_accuracy']):.4f}` / `{float(selected['truth_present_mean_iou']):.4f}`.",
        f"- Label-9 IoU / precision / recall: `{float(selected['target_iou']):.4f}` / `{float(selected['target_precision']):.4f}` / `{float(selected['target_recall']):.4f}`.",
        f"- Major-component minimum / mean recall: `{float(selected['major_component_min_recall']):.4f}` / `{float(selected['major_component_mean_recall']):.4f}`.",
        "",
        "## Rank relationship",
        "",
    ]
    for field in QUALITY_FIELDS:
        result = target_correlations[field]
        lines.append(
            f"- loss vs {field}: rho `{result['spearman_rho']}`, one-sided p `{result['one_sided_negative_permutation_pvalue']}`."
        )
    if label9_substitution is not None:
        lines.extend(
            [
                "",
                "## Label-9 whole-class substitution oracle",
                "",
                f"- Unconditioned truth voxels changed: `{label9_substitution['changed_voxels']}`.",
                f"- Least-visible replacement label: `{label9_substitution['least_visible_destination']}`.",
                f"- Minimum substitution RMSE: `{float(label9_substitution['minimum_hard_seismic_rmse_amplitude']):.6f}`.",
                "",
                "Whole-class substitution sensitivity does not establish local lithology uniqueness.",
            ]
        )
    lines.extend(["", "A lower continuous field loss alone is not geological recovery.", ""])
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    if args.target_roi_radius < 0:
        raise ValueError("target-roi-radius must be non-negative")
    if args.permutations <= 0:
        raise ValueError("permutations must be positive")
    validate_output_directory(args.output_dir, overwrite=args.overwrite)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")

    truth_path = args.samples_dir / "true_model.pt"
    boreholes_path = args.samples_dir / "boreholes.pt"
    truth = runtime.normalize_single_geology(runtime.load_tensor(truth_path), "truth").long()
    boreholes = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path), "boreholes"
    ).long()
    conditioning_report = runtime.validate_conditioning_pair(
        truth, boreholes, num_categories=15, target_label=args.target_label
    )
    truth_hash = runtime.file_sha256(truth_path)
    boreholes_hash = runtime.file_sha256(boreholes_path)
    checkpoint_hash = runtime.file_sha256(args.ckpt_path)
    baseline_dirs = args.baseline_dir or _default_baseline_dirs()
    candidates, source_records = _validate_source_pool(
        baseline_dirs,
        truth_hash=truth_hash,
        boreholes_hash=boreholes_hash,
        checkpoint_hash=checkpoint_hash,
        anchor_dir=args.phase4c_anchor_dir,
    )

    tensors, observation_manifest, forward_operator, resolved_observation = (
        load_observation_assets(
            args.observation_dir,
            truth,
            truth_path=truth_path,
            num_categories=15,
        )
    )
    property_table = tensors["acoustic_property_table.pt"]
    target_acoustic = tensors["truth_acoustic.pt"]
    subsurface_mask = tensors["subsurface_mask.pt"]
    observed = tensors["observed_seismic.pt"]
    sample_mask = tensors["sample_mask.pt"]
    uncertainty = tensors["uncertainty_amplitude.pt"]
    condition_mask = (boreholes != -1) | (truth == -1)
    confidence = ((truth != -1) & ~condition_mask).float()
    target_mask, _ = build_target_mask(
        truth, target_label=args.target_label, component_mode="all"
    )
    target_roi = dilate_mask(target_mask, args.target_roi_radius)

    candidate_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    for global_sample_id, candidate in enumerate(candidates):
        prediction = runtime.normalize_single_geology(
            runtime.load_tensor(candidate["sample_path"]), str(candidate["sample_path"])
        ).long()
        metrics = sample_property_hard_metrics(
            prediction=prediction,
            truth_model=truth,
            condition_mask=condition_mask,
            target_mask=target_mask,
            target_roi_mask=target_roi,
            target_label=args.target_label,
            property_table=property_table,
            property_confidence=confidence,
            property_sigmas=(0.0,),
            property_scale_weights=(1.0,),
            property_channel_weights=torch.ones(2),
            sample_id=global_sample_id,
        )
        add_hard_seismic_metrics(
            metrics,
            prediction=prediction,
            target_acoustic=target_acoustic,
            condition_mask=condition_mask,
            property_table=property_table,
            subsurface_mask=subsurface_mask,
            forward_operator=forward_operator,
            observed=observed,
            sample_mask=sample_mask,
            uncertainty=uncertainty,
            device=device,
        )
        components = truth_component_recovery_rows(
            prediction, truth, args.target_label, global_sample_id
        )
        major = [
            float(row["recall"])
            for row in components
            if int(row["truth_component_rank"]) in MAJOR_COMPONENT_RANKS
        ]
        if len(major) != len(MAJOR_COMPONENT_RANKS):
            raise ValueError("truth target lacks the four required major components")
        metrics.update(
            {
                "candidate_id": candidate["candidate_id"],
                "seed": candidate["seed"],
                "local_sample_id": candidate["local_sample_id"],
                "sample_sha256": candidate["sample_sha256"],
                "path": str(candidate["sample_path"]),
                "major_component_min_recall": min(major),
                "major_component_mean_recall": sum(major) / len(major),
            }
        )
        candidate_rows.append(metrics)
        for row in per_class_hard_metrics(
            prediction, truth, global_sample_id, class_ids=range(14)
        ):
            row.update(
                candidate_id=candidate["candidate_id"],
                seed=candidate["seed"],
                local_sample_id=candidate["local_sample_id"],
            )
            class_rows.append(row)
        for row in components:
            row.update(
                candidate_id=candidate["candidate_id"],
                seed=candidate["seed"],
                local_sample_id=candidate["local_sample_id"],
            )
            component_rows.append(row)

    ranking, selection = build_selection_summary(
        candidate_rows,
        permutations=args.permutations,
        permutation_seed=args.permutation_seed,
    )
    substitution_rows, substitution_summary = _substitution_sensitivity(
        truth=truth,
        condition_mask=condition_mask,
        target_acoustic=target_acoustic,
        property_table=property_table,
        subsurface_mask=subsurface_mask,
        forward_operator=forward_operator,
        observed=observed,
        sample_mask=sample_mask,
        uncertainty=uncertainty,
        device=device,
    )

    selected = ranking[0]
    summary = {
        "schema": PHASE4D_SCHEMA,
        "status": "completed",
        "scope": "offline fixed-prior-pool seismic identifiability diagnostic",
        "truth_derived": True,
        "measured_geophysics": False,
        "inverse_crime": True,
        "generated_or_modified_samples": False,
        "selection_used_truth": False,
        "selection": selection,
        "selected_candidate": {
            field: selected[field]
            for field in (
                "candidate_id",
                "seed",
                "local_sample_id",
                "hard_seismic_loss",
                "hard_seismic_rmse_amplitude",
                "global_voxel_accuracy",
                "truth_present_mean_iou",
                "target_iou",
                "target_precision",
                "target_recall",
                "major_component_min_recall",
                "major_component_mean_recall",
            )
        },
        "substitution_sensitivity": substitution_summary,
        "conditions_exact": all(
            int(row["condition_violation_count"]) == 0 for row in candidate_rows
        )
        and all(
            int(row["condition_violation_count"]) == 0 for row in substitution_rows
        ),
        "candidate_ids": [candidate["candidate_id"] for candidate in candidates],
        "permutation_count": args.permutations,
        "permutation_seed": args.permutation_seed,
        "observation_manifest_sha256": runtime.file_sha256(
            args.observation_dir / "manifest.json"
        ),
        "limitations": [
            "fixed pool contains only 12 prior samples",
            "synthetic inverse-crime convolutional seismic is not measured data",
            "truth substitution is an oracle sensitivity diagnostic",
            "reranking cannot recover geology absent from the proposal pool",
        ],
    }
    manifest = {
        "schema": PHASE4D_SCHEMA,
        "status": "complete",
        "protocol": runtime.asset_record(PROJECT_DIR / "docs/PHASE4D_SPEC.md"),
        "runner": runtime.asset_record(Path(__file__)),
        "truth": runtime.asset_record(truth_path),
        "boreholes": runtime.asset_record(boreholes_path),
        "checkpoint": runtime.asset_record(args.ckpt_path),
        "observation_manifest": runtime.asset_record(
            args.observation_dir / "manifest.json"
        ),
        "source_runs": source_records,
        "conditioning_report": conditioning_report,
        "resolved_observation": resolved_observation,
        "observation_id": resolved_observation.get("id"),
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_rows(args.output_dir / "candidate_metrics.csv", candidate_rows)
    write_rows(args.output_dir / "seismic_ranking.csv", ranking)
    write_rows(args.output_dir / "per_class_metrics.csv", class_rows)
    write_rows(args.output_dir / "truth_component_recovery.csv", component_rows)
    write_rows(args.output_dir / "truth_substitution_sensitivity.csv", substitution_rows)
    write_json(args.output_dir / "summary.json", summary)
    write_json(args.output_dir / "manifest.json", manifest)
    report = _render_report(summary)
    (args.output_dir / "REPORT.md").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
