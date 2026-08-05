#!/usr/bin/env python3
"""Aggregate and gate the completed Phase-2a ideal-property strict pairs.

Immutable baseline/guided run directories are read only. Derived CSV, JSON and
Markdown artifacts are written beneath the Phase-2 report directory.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Mapping, Sequence


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.stage2.run_property_guidance import paired_property_config_verdict


DEFAULT_SEEDS = (42, 142, 242)
MAJOR_TRUTH_COMPONENT_RANKS = (1, 2, 3, 4)
TRUTH_PRESENT_CLASS_COUNT = 8
REPORT_FILENAMES = (
    "REPORT.md",
    "summary.json",
    "paired_samples.csv",
    "class_summary.csv",
    "component_summary.csv",
    "seed_summary.csv",
)


def parse_args() -> argparse.Namespace:
    default_root = (
        PROJECT_DIR
        / "experiments/stage2_property/runs/cond_generation_0"
        / "ideal_density_susceptibility_label9_contrast_v1/phase2a_v1"
    )
    parser = argparse.ArgumentParser(
        description="Aggregate the Phase-2a 12-pair ideal-property confirmation.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--runs-root", type=Path, default=default_root)
    parser.add_argument("--seed", type=int, action="append", dest="seeds")
    parser.add_argument("--n-samples", type=int, default=4)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--run-name-template",
        default="seed{seed}_n{n_samples}_s{n_steps}_a025_c025",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_DIR
        / "experiments/stage2_property/reports/phase2a_v1_12pair",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace named derived reports only; immutable runs are untouched.",
    )
    return parser.parse_args()


def _numeric(value: str) -> int | float | str:
    if value == "":
        return value
    if value == "True":
        return True
    if value == "False":
        return False
    try:
        integer = int(value)
        if str(integer) == value:
            return integer
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        return value


def read_numeric_csv(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", newline="", encoding="utf-8") as stream:
        return [
            {field: _numeric(value) for field, value in row.items()}
            for row in csv.DictReader(stream)
        ]


def read_json(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object: {path}")
    return payload


def write_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    with path.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")


def _index_rows(
    rows: Sequence[Mapping[str, object]],
    key_field: str = "sample_id",
) -> dict[int, Mapping[str, object]]:
    indexed = {int(row[key_field]): row for row in rows}
    if len(indexed) != len(rows):
        raise ValueError(f"duplicate {key_field} values")
    return indexed


def _finite_numeric_row(row: Mapping[str, object]) -> bool:
    return all(
        math.isfinite(value)
        for value in row.values()
        if isinstance(value, float)
    )


def _stats(values: Sequence[float]) -> dict[str, object]:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "min": min(finite),
        "max": max(finite),
    }


def sample_gate_audit(
    guided: Mapping[str, object],
    delta: Mapping[str, object],
    class_deltas: Sequence[Mapping[str, object]],
    component_rows: Sequence[Mapping[str, object]],
    final_churn_fraction: float,
) -> dict[str, object]:
    """Evaluate the frozen per-pair Phase-2a gates."""
    truth_classes = [row for row in class_deltas if bool(row["truth_present"])]
    improved_classes = sum(float(row["delta_iou"]) > 0 for row in truth_classes)
    major = [
        row
        for row in component_rows
        if int(row["truth_component_rank"]) in MAJOR_TRUTH_COMPONENT_RANKS
    ]
    major_recalls = [float(row["guided_recall"]) for row in major]
    volume_fraction = float(guided["predicted_target_volume"]) / float(
        guided["target_volume"]
    )
    checks = {
        "primary_directions": (
            float(delta["delta_global_voxel_accuracy"]) > 0
            and float(delta["delta_truth_present_mean_iou"]) > 0
            and float(delta["delta_hard_property_loss"]) < 0
            and float(delta["delta_target_iou"]) > 0
            and float(delta["delta_target_precision"]) > 0
            and float(delta["delta_target_recall"]) > 0
        ),
        "majority_classes": (
            len(truth_classes) == TRUTH_PRESENT_CLASS_COUNT
            and improved_classes >= 5
        ),
        "target_thresholds": (
            float(guided["target_precision"]) >= 0.75
            and float(guided["target_recall"]) >= 0.30
            and float(guided["target_iou"]) >= 0.30
            and 0.35 <= volume_fraction <= 1.20
        ),
        "major_component_recovery": (
            len(major_recalls) == len(MAJOR_TRUTH_COMPONENT_RANKS)
            and min(major_recalls, default=0.0) >= 0.25
            and sum(major_recalls) / len(major_recalls) >= 0.40
        ),
        "size_stratified_topology": (
            float(guided["target_tiny_component_mass_fraction_le_5"]) <= 0.10
            and float(guided["target_top8_component_mass_fraction"]) >= 0.75
        ),
        "endpoint_churn": float(final_churn_fraction) <= 0.015,
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "truth_present_classes": len(truth_classes),
        "improved_truth_present_classes": improved_classes,
        "target_volume_fraction": volume_fraction,
        "major_component_min_recall": min(major_recalls, default=float("nan")),
        "major_component_mean_recall": (
            sum(major_recalls) / len(major_recalls)
            if major_recalls
            else float("nan")
        ),
        "final_churn_fraction": float(final_churn_fraction),
    }


def _validate_configs(
    baseline: Mapping[str, object],
    guided: Mapping[str, object],
    seed: int,
    n_samples: int,
    n_steps: int,
) -> None:
    paired, reason = paired_property_config_verdict(baseline, guided)
    if not paired:
        raise ValueError(f"seed {seed} strict pairing failed: {reason}")
    if guided.get("pairing_validation", {}).get("paired") is not True:
        raise ValueError(f"seed {seed} saved pairing verdict is not true")
    for name, config, alpha in (
        ("baseline", baseline, 0.0),
        ("guided", guided, 0.25),
    ):
        required = {
            "seed": seed,
            "n_samples": n_samples,
            "n_steps": n_steps,
            "run_status": "completed",
            "samples_written": n_samples,
            "ema_applied": True,
            "model_weight_source": "ema",
            "max_post_projection_condition_violations": 0,
        }
        for field, expected in required.items():
            if config.get(field) != expected:
                raise ValueError(
                    f"seed {seed} {name} {field}={config.get(field)!r}, "
                    f"expected {expected!r}"
                )
        if float(config.get("alpha", float("nan"))) != alpha:
            raise ValueError(f"seed {seed} {name} alpha is not {alpha}")
        if float(config.get("max_guidance_ratio", float("nan"))) != 0.25:
            raise ValueError(f"seed {seed} {name} cap is not 0.25")
    if baseline.get("initial_noise_sha256") != guided.get("initial_noise_sha256"):
        raise ValueError(f"seed {seed} paired initial-noise hashes differ")


def _last_trace_rows(
    rows: Sequence[Mapping[str, object]],
    n_samples: int,
    n_steps: int,
) -> dict[int, Mapping[str, object]]:
    if not rows or any(not _finite_numeric_row(row) for row in rows):
        raise ValueError("guidance trace contains missing or non-finite numbers")
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["sample_id"])].append(row)
        if int(row["post_projection_condition_violations"]) != 0:
            raise ValueError("guidance trace contains condition violations")
        if int(row["hard_change_outside_confidence"]) != 0:
            raise ValueError("guidance trace changes outside property confidence")
    if set(grouped) != set(range(n_samples)):
        raise ValueError("guidance trace sample IDs are incomplete")
    output: dict[int, Mapping[str, object]] = {}
    for sample_id, sample_rows in grouped.items():
        if len(sample_rows) != n_steps:
            raise ValueError(f"sample {sample_id} does not have {n_steps} trace rows")
        output[sample_id] = max(sample_rows, key=lambda row: int(row["step"]))
        if int(output[sample_id]["step"]) != n_steps - 1:
            raise ValueError(f"sample {sample_id} trace endpoint is incomplete")
    return output


def _aggregate_metric(
    baseline_rows: Sequence[Mapping[str, object]],
    guided_rows: Sequence[Mapping[str, object]],
    field: str,
) -> dict[str, object]:
    baseline_values = [float(row[field]) for row in baseline_rows]
    guided_values = [float(row[field]) for row in guided_rows]
    deltas = [after - before for before, after in zip(baseline_values, guided_values)]
    return {
        "baseline": _stats(baseline_values),
        "guided": _stats(guided_values),
        "delta": _stats(deltas),
        "positive_delta_count": sum(value > 0 for value in deltas),
        "negative_delta_count": sum(value < 0 for value in deltas),
    }


def _class_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if bool(row["truth_present"]):
            grouped[int(row["class_id"])].append(row)
    output: list[dict[str, object]] = []
    for class_id in sorted(grouped):
        values = grouped[class_id]
        deltas = [float(row["delta_iou"]) for row in values]
        output.append(
            {
                "class_id": class_id,
                "n_pairs": len(values),
                "mean_baseline_iou": _stats(
                    [float(row["baseline_iou"]) for row in values]
                )["mean"],
                "mean_guided_iou": _stats(
                    [float(row["guided_iou"]) for row in values]
                )["mean"],
                "mean_delta_iou": _stats(deltas)["mean"],
                "improved_pairs": sum(value > 0 for value in deltas),
                "worsened_pairs": sum(value < 0 for value in deltas),
                "unchanged_pairs": sum(value == 0 for value in deltas),
            }
        )
    return output


def _component_summary(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    grouped: dict[int, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["truth_component_rank"])].append(row)
    output: list[dict[str, object]] = []
    for rank in sorted(grouped):
        values = grouped[rank]
        output.append(
            {
                "truth_component_rank": rank,
                "truth_component_voxels": int(values[0]["truth_component_voxels"]),
                "n_pairs": len(values),
                "mean_baseline_recall": _stats(
                    [float(row["baseline_recall"]) for row in values]
                )["mean"],
                "mean_guided_recall": _stats(
                    [float(row["guided_recall"]) for row in values]
                )["mean"],
                "min_guided_recall": _stats(
                    [float(row["guided_recall"]) for row in values]
                )["min"],
                "mean_delta_recall": _stats(
                    [float(row["delta_recall"]) for row in values]
                )["mean"],
            }
        )
    return output


def _format_change(before: float, after: float) -> str:
    return f"`{before:.4f} -> {after:.4f}`"


def _report_markdown(summary: Mapping[str, object]) -> str:
    metrics = summary["metrics"]
    gates = summary["gates"]
    class_rows = summary["class_summary"]
    component_rows = summary["component_summary"]
    lines = [
        "# Phase-2a 12-pair ideal-property confirmation",
        "",
        "## Decision",
        "",
        f"**{summary['decision']}**",
        "",
        (
            "This validates a truth-derived, full-resolution two-channel 3-D "
            "property upper bound with the frozen EMA model. It does not validate "
            "measured geophysics or a 2-D field inversion."
        ),
        "",
        "## Strict evidence",
        "",
        f"- Strict pairs: {summary['n_pairs']} across seeds {summary['seeds']}.",
        f"- All per-pair frozen gates pass: {gates['all_pair_gates_pass']}.",
        f"- All seed diversity gates pass: {gates['all_seed_diversity_gates_pass']}.",
        "- EMA, paired noise/assets, finite traces, exact conditions and confidence locality pass.",
        "",
        "## Aggregate hard results",
        "",
    ]
    for field, label in (
        ("global_voxel_accuracy", "Global voxel accuracy"),
        ("truth_present_mean_iou", "Truth-present fixed-set mIoU"),
        ("global_mean_iou", "Historical dynamic-union mIoU"),
        ("hard_property_loss", "Hard-property loss"),
        ("target_iou", "Label-9 IoU"),
        ("target_precision", "Label-9 precision"),
        ("target_recall", "Label-9 recall"),
        ("target_centroid_distance", "Label-9 centroid distance"),
    ):
        before = float(metrics[field]["baseline"]["mean"])
        after = float(metrics[field]["guided"]["mean"])
        lines.append(f"- {label}: {_format_change(before, after)}.")
    lines.extend(["", "## Class and component findings", ""])
    label2 = next(row for row in class_rows if int(row["class_id"]) == 2)
    lines.append(
        "- Label 2 is the consistent secondary-class tradeoff: "
        f"mean IoU delta {float(label2['mean_delta_iou']):.4f}, "
        f"worse in {label2['worsened_pairs']}/{summary['n_pairs']} pairs."
    )
    lines.append(
        "- Every pair improves at least five of eight truth-present classes; "
        "label 13 remains effectively unrecovered."
    )
    for row in component_rows[:4]:
        lines.append(
            f"- Truth component {row['truth_component_rank']} "
            f"({row['truth_component_voxels']} voxels): mean guided recall "
            f"{float(row['mean_guided_recall']):.4f}, minimum "
            f"{float(row['min_guided_recall']):.4f}."
        )
    topology = summary["topology"]
    lines.extend(
        [
            (
                "- Tiny-component mass fraction range: "
                f"{topology['tiny_mass_fraction']['min']:.4f}-"
                f"{topology['tiny_mass_fraction']['max']:.4f}."
            ),
            (
                "- Top-eight target-mass fraction range: "
                f"{topology['top8_mass_fraction']['min']:.4f}-"
                f"{topology['top8_mass_fraction']['max']:.4f}."
            ),
            (
                "- Final hard-churn fraction range: "
                f"{topology['final_churn_fraction']['min']:.4f}-"
                f"{topology['final_churn_fraction']['max']:.4f}."
            ),
            "",
            "## Caveats and next phase",
            "",
            "- The target bodies remain incomplete and more fragmented than truth.",
            "- One repeated guided sample differed by 10 hard voxels across CUDA processes; bitwise determinism is not claimed.",
            "- Phase 2b should first test overlapping/less distinctive property codebooks; Phase 3 can then add resolution, blur, missing regions and noise before any 2-D joint physics experiment.",
            "- Continuous property loss never substitutes for the recorded hard-label and geometry gates.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    seeds = tuple(args.seeds or DEFAULT_SEEDS)
    existing = [path for path in args.output_dir.iterdir()] if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite for derived reports: "
            f"{args.output_dir}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)

    paired_rows: list[dict[str, object]] = []
    baseline_rows: list[dict[str, object]] = []
    guided_rows: list[dict[str, object]] = []
    class_rows: list[dict[str, object]] = []
    component_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    common_source_hashes: dict[str, object] | None = None

    for seed in seeds:
        run_name = args.run_name_template.format(
            seed=seed,
            n_samples=args.n_samples,
            n_steps=args.n_steps,
        )
        pair_root = args.runs_root / run_name
        baseline_dir = pair_root / args.baseline_name
        guided_dir = pair_root / args.guided_name
        baseline_config = read_json(baseline_dir / "config.json")
        guided_config = read_json(guided_dir / "config.json")
        _validate_configs(
            baseline_config,
            guided_config,
            seed,
            args.n_samples,
            args.n_steps,
        )
        source_hashes = {
            field: guided_config[field]
            for field in (
                "checkpoint_sha256",
                "property_config_sha256",
                "property_table_sha256",
                "target_properties_sha256",
                "property_confidence_sha256",
                "property_volume_source_sha256",
                "property_sampling_source_sha256",
                "property_evaluation_source_sha256",
                "runner_source_sha256",
            )
        }
        if common_source_hashes is None:
            common_source_hashes = source_hashes
        elif source_hashes != common_source_hashes:
            raise ValueError(f"seed {seed} source/property hashes differ")

        baseline_metrics = _index_rows(
            read_numeric_csv(guided_dir / "paired_baseline_metrics.csv")
        )
        guided_metrics = _index_rows(read_numeric_csv(guided_dir / "sample_metrics.csv"))
        deltas = _index_rows(read_numeric_csv(guided_dir / "paired_deltas.csv"))
        class_deltas = read_numeric_csv(guided_dir / "paired_per_class_deltas.csv")
        components = read_numeric_csv(
            guided_dir / "paired_truth_component_recovery_deltas.csv"
        )
        trace_endpoints = _last_trace_rows(
            read_numeric_csv(guided_dir / "guidance_trace.csv"),
            args.n_samples,
            args.n_steps,
        )
        if not (
            set(baseline_metrics)
            == set(guided_metrics)
            == set(deltas)
            == set(range(args.n_samples))
        ):
            raise ValueError(f"seed {seed} sample metric IDs are incomplete")

        ensemble = read_json(guided_dir / "ensemble_summary.json")
        current_ensemble = ensemble.get("current", {})
        diversity_pass = (
            int(current_ensemble.get("unique_decoded_samples", 0))
            == args.n_samples
            and float(
                current_ensemble.get(
                    "mean_pairwise_hard_disagreement_outside_roi",
                    0.0,
                )
            )
            > 0
        )
        seed_rows.append(
            {
                "seed": seed,
                "strict_pairing": True,
                "unique_guided_samples": int(
                    current_ensemble["unique_decoded_samples"]
                ),
                "guided_outside_roi_disagreement": float(
                    current_ensemble[
                        "mean_pairwise_hard_disagreement_outside_roi"
                    ]
                ),
                "guided_inside_roi_disagreement": float(
                    current_ensemble["mean_pairwise_hard_disagreement_inside_roi"]
                ),
                "diversity_gate_pass": diversity_pass,
            }
        )

        for sample_id in range(args.n_samples):
            sample_classes = [
                row for row in class_deltas if int(row["sample_id"]) == sample_id
            ]
            sample_components = [
                row for row in components if int(row["sample_id"]) == sample_id
            ]
            endpoint = trace_endpoints[sample_id]
            audit = sample_gate_audit(
                guided_metrics[sample_id],
                deltas[sample_id],
                sample_classes,
                sample_components,
                float(endpoint["hard_change_fraction"]),
            )
            pair_row = {
                "seed": seed,
                "sample_id": sample_id,
                "pair_id": f"{seed}:{sample_id}",
                **guided_metrics[sample_id],
                **deltas[sample_id],
                "truth_present_classes": audit["truth_present_classes"],
                "improved_truth_present_classes": audit[
                    "improved_truth_present_classes"
                ],
                "target_volume_fraction": audit["target_volume_fraction"],
                "major_component_min_recall": audit[
                    "major_component_min_recall"
                ],
                "major_component_mean_recall": audit[
                    "major_component_mean_recall"
                ],
                "final_churn_fraction": audit["final_churn_fraction"],
                "pair_gate_pass": audit["passed"],
            }
            for gate_name, passed in audit["checks"].items():
                pair_row[f"gate_{gate_name}"] = passed
            paired_rows.append(pair_row)
            baseline_rows.append(
                {"seed": seed, **baseline_metrics[sample_id]}
            )
            guided_rows.append({"seed": seed, **guided_metrics[sample_id]})
        class_rows.extend({"seed": seed, **row} for row in class_deltas)
        component_rows.extend({"seed": seed, **row} for row in components)

    if len(paired_rows) != len(seeds) * args.n_samples:
        raise ValueError("aggregate pair count is incomplete")

    metric_fields = (
        "global_voxel_accuracy",
        "global_mean_iou",
        "truth_present_mean_iou",
        "hard_property_loss",
        "hard_property_mae",
        "target_iou",
        "target_precision",
        "target_recall",
        "predicted_target_volume",
        "target_absolute_volume_error_fraction",
        "target_centroid_distance",
        "target_connected_components",
        "largest_component_fraction",
        "inside_roi_voxel_accuracy",
        "outside_roi_voxel_accuracy",
    )
    metrics = {
        field: _aggregate_metric(baseline_rows, guided_rows, field)
        for field in metric_fields
    }
    class_summary = _class_summary(class_rows)
    component_summary = _component_summary(component_rows)
    all_pair_gates = all(bool(row["pair_gate_pass"]) for row in paired_rows)
    all_seed_diversity = all(bool(row["diversity_gate_pass"]) for row in seed_rows)
    summary: dict[str, object] = {
        "decision": (
            "PASS: Phase-2a ideal 3-D property upper bound validated with caveats"
            if all_pair_gates and all_seed_diversity
            else "FAIL: Phase-2a pre-registered confirmation gate not met"
        ),
        "scope": "truth-derived full-resolution two-channel 3-D property oracle",
        "is_measured_geophysics": False,
        "seeds": list(seeds),
        "n_samples_per_seed": args.n_samples,
        "n_pairs": len(paired_rows),
        "n_steps": args.n_steps,
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "source_and_property_hashes": common_source_hashes,
        "gates": {
            "all_pair_gates_pass": all_pair_gates,
            "pair_pass_count": sum(bool(row["pair_gate_pass"]) for row in paired_rows),
            "all_seed_diversity_gates_pass": all_seed_diversity,
            "seed_diversity_pass_count": sum(
                bool(row["diversity_gate_pass"]) for row in seed_rows
            ),
        },
        "metrics": metrics,
        "topology": {
            "tiny_mass_fraction": _stats(
                [
                    float(row["target_tiny_component_mass_fraction_le_5"])
                    for row in guided_rows
                ]
            ),
            "top8_mass_fraction": _stats(
                [
                    float(row["target_top8_component_mass_fraction"])
                    for row in guided_rows
                ]
            ),
            "final_churn_fraction": _stats(
                [float(row["final_churn_fraction"]) for row in paired_rows]
            ),
        },
        "class_summary": class_summary,
        "component_summary": component_summary,
        "seed_summary": seed_rows,
        "limitations": [
            "truth-derived ideal property oracle, not measured geophysics",
            "target geometry remains incomplete and fragmented",
            "label 2 IoU consistently declines",
            "label 13 remains effectively unrecovered",
            "cross-process bitwise CUDA determinism is not claimed",
        ],
    }

    write_csv(args.output_dir / "paired_samples.csv", paired_rows)
    write_csv(args.output_dir / "class_summary.csv", class_summary)
    write_csv(args.output_dir / "component_summary.csv", component_summary)
    write_csv(args.output_dir / "seed_summary.csv", seed_rows)
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(
        _report_markdown(summary),
        encoding="utf-8",
    )
    print(args.output_dir / "REPORT.md")
    print(summary["decision"])


if __name__ == "__main__":
    main()
