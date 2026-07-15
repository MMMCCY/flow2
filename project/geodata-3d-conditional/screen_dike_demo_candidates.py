"""Prepare and summarize dike-like target-label demo candidate screening runs."""

from __future__ import annotations

import argparse
import csv
import math
import shlex
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence

import torch

from geology_io_utils import (
    label_mask,
    load_volume,
    read_csv_rows,
    read_json,
    write_csv_rows,
    write_json,
)


OUTPUT_FIELDS = [
    "candidate",
    "case_name",
    "target_label",
    "status",
    "recommendation",
    "reason",
    "truth_target_volume",
    "truth_target_volume_fraction",
    "borehole_target_hits",
    "observability_normalized_delta",
    "baseline_target_volume_mean",
    "baseline_target_volume_ratio",
    "baseline_target_volume_std",
    "baseline_target_iou_mean",
    "baseline_target_recall_mean",
    "baseline_probability_overlap_inside_truth",
    "guided_target_volume_mean",
    "guided_target_volume_ratio",
    "guided_target_iou_mean",
    "guided_target_recall_mean",
    "target_iou_improvement",
    "target_recall_improvement",
    "target_centroid_distance_improvement",
    "baseline_geo_misfit_mean",
    "guided_geo_misfit_mean",
    "geo_misfit_improvement",
    "baseline_voxel_accuracy_mean",
    "guided_voxel_accuracy_mean",
    "baseline_mean_iou_mean",
    "guided_mean_iou_mean",
    "baseline_borehole_consistency_mean",
    "guided_borehole_consistency_mean",
    "paired_by_seed",
    "candidate_dir",
]


@dataclass(frozen=True)
class Candidate:
    case_name: str
    target_label: int

    @property
    def slug(self) -> str:
        return f"{self.case_name}_label{self.target_label}"


def parse_candidate(value: str) -> Candidate:
    if ":" not in value:
        raise argparse.ArgumentTypeError("candidate must have form case_name:target_label")
    case_name, label_text = value.split(":", 1)
    if not case_name:
        raise argparse.ArgumentTypeError("candidate case_name is empty")
    try:
        target_label = int(label_text)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"invalid target label in candidate: {value}") from exc
    return Candidate(case_name=case_name, target_label=target_label)


def _alpha_slug(alpha: float) -> str:
    text = f"{float(alpha):g}".replace(".", "_").replace("-", "m")
    return f"alpha{text}"


def _command(python_executable: str, script_dir: Path, script_name: str, args: Sequence[object]) -> List[str]:
    return [python_executable, str(script_dir / script_name), *(str(arg) for arg in args)]


def _format_command(command: Sequence[str]) -> str:
    return shlex.join(str(part) for part in command)


def _mean(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return sum(finite) / len(finite) if finite else float("nan")


def _std(values: Sequence[float]) -> float:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return float("nan")
    mean = sum(finite) / len(finite)
    return math.sqrt(sum((value - mean) ** 2 for value in finite) / len(finite))


def _float(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    value = row.get(key, default)
    if value in ("", None):
        return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _column_mean(rows: Sequence[Mapping[str, object]], key: str) -> float:
    return _mean([_float(row, key) for row in rows])


def _truth_stats(truth_model: Path, boreholes: Optional[Path], target_label: int, device: str) -> Dict[str, object]:
    truth = load_volume(truth_model, device=device, single=True).long()
    mask = label_mask(truth, target_label)
    target_volume = int(mask.sum().item())
    total = int(truth[0, 0].numel())
    borehole_hits = None
    if boreholes is not None and boreholes.exists():
        borehole_volume = load_volume(boreholes, device=device, single=True).long()
        borehole_hits = int(((borehole_volume == int(target_label)) & mask).sum().item())
    return {
        "truth_target_volume": target_volume,
        "truth_target_volume_fraction": target_volume / total if total > 0 else float("nan"),
        "borehole_target_hits": borehole_hits,
    }


def _paths_for_candidate(args: argparse.Namespace, candidate: Candidate) -> Dict[str, Path]:
    candidate_dir = args.output_dir / candidate.slug
    alpha_name = _alpha_slug(args.alpha)
    return {
        "candidate_dir": candidate_dir,
        "truth_model": args.truth_root / candidate.case_name / "true_model.pt",
        "boreholes": args.truth_root / candidate.case_name / "boreholes.pt",
        "density_config": candidate_dir / "density_config.json",
        "observed_dir": candidate_dir / "observed_gravity",
        "observed_gravity": candidate_dir / "observed_gravity" / "observed_gravity.pt",
        "baseline_dir": candidate_dir / "baseline_alpha0",
        "guided_dir": candidate_dir / f"guided_{alpha_name}",
        "final_dir": candidate_dir / "screening",
        "observability_dir": candidate_dir / "screening" / "observability",
        "baseline_global": candidate_dir / "screening" / "baseline_global_evaluation",
        "guided_global": candidate_dir / "screening" / "guided_global_evaluation",
        "baseline_target": candidate_dir / "screening" / "baseline_target",
        "guided_target": candidate_dir / "screening" / "guided_target",
        "sample_selection": candidate_dir / "screening" / "sample_selection",
    }


def build_candidate_commands(args: argparse.Namespace, candidate: Candidate) -> List[List[str]]:
    paths = _paths_for_candidate(args, candidate)
    script_dir = Path(__file__).resolve().parent
    truth_model = paths["truth_model"]
    boreholes = paths["boreholes"]
    density_config = paths["density_config"]
    observed_gravity = paths["observed_gravity"]
    baseline_dir = paths["baseline_dir"]
    guided_dir = paths["guided_dir"]
    commands: List[List[str]] = []

    commands.append(
        _command(
            args.python,
            script_dir,
            "create_density_config.py",
            [
                "--truth-model",
                truth_model,
                "--target-label",
                candidate.target_label,
                "--target-density",
                args.target_density,
                "--background-scale",
                args.background_scale,
                "--output-json",
                density_config,
                "--device",
                args.device,
            ],
        )
    )
    commands.append(
        _command(
            args.python,
            script_dir,
            "generate_observed_gravity.py",
            [
                "--truth-model",
                truth_model,
                "--density-config",
                density_config,
                "--output-dir",
                paths["observed_dir"],
                "--kernel-size",
                args.kernel_size,
                "--device",
                args.device,
            ],
        )
    )
    base_sampling_args = [
        "--ckpt-path",
        args.ckpt_path,
        "--samples-dir",
        truth_model.parent,
        "--truth-model",
        truth_model,
        "--boreholes",
        boreholes,
        "--density-config",
        density_config,
        "--observed-gravity",
        observed_gravity,
        "--n-samples",
        args.n_samples,
        "--n-steps",
        args.n_steps,
        "--guidance-mode",
        "relative",
        "--mu",
        args.mu,
        "--tau",
        args.tau,
        "--guidance-start",
        args.guidance_start,
        "--guidance-schedule",
        args.guidance_schedule,
        "--kernel-size",
        args.kernel_size,
        "--grad-clip-norm",
        args.grad_clip_norm,
        "--seed",
        args.seed,
        "--device",
        args.device,
    ]
    commands.append(
        _command(
            args.python,
            script_dir,
            "guided_geophysical_sampling.py",
            [
                *base_sampling_args,
                "--output-dir",
                baseline_dir,
                "--alpha",
                0.0,
            ],
        )
    )
    commands.append(
        _command(
            args.python,
            script_dir,
            "guided_geophysical_sampling.py",
            [
                *base_sampling_args,
                "--output-dir",
                guided_dir,
                "--alpha",
                args.alpha,
                "--baseline-dir",
                baseline_dir,
            ],
        )
    )
    for run_dir, output_dir in (
        (baseline_dir, paths["baseline_global"]),
        (guided_dir, paths["guided_global"]),
    ):
        commands.append(
            _command(
                args.python,
                script_dir,
                "evaluate_geophysics.py",
                [
                    "--samples-dir",
                    run_dir,
                    "--truth-model",
                    truth_model,
                    "--boreholes",
                    boreholes,
                    "--observed-gravity",
                    observed_gravity,
                    "--density-config",
                    density_config,
                    "--output-dir",
                    output_dir,
                    "--kernel-size",
                    args.kernel_size,
                    "--device",
                    args.device,
                ],
            )
        )
    commands.append(
        _command(
            args.python,
            script_dir,
            "analyze_dike_observability.py",
            [
                "--truth-model",
                truth_model,
                "--boreholes",
                boreholes,
                "--target-label",
                candidate.target_label,
                "--density-config",
                density_config,
                "--output-dir",
                paths["observability_dir"],
                "--kernel-size",
                args.kernel_size,
                "--device",
                args.device,
            ],
        )
    )
    for run_dir, metrics_dir, output_dir in (
        (baseline_dir, paths["baseline_global"], paths["baseline_target"]),
        (guided_dir, paths["guided_global"], paths["guided_target"]),
    ):
        commands.append(
            _command(
                args.python,
                script_dir,
                "evaluate_target_feature.py",
                [
                    "--samples-dir",
                    run_dir,
                    "--truth-model",
                    truth_model,
                    "--target-label",
                    candidate.target_label,
                    "--metrics-csv",
                    metrics_dir / "metrics.csv",
                    "--output-dir",
                    output_dir,
                    "--device",
                    args.device,
                ],
            )
        )
    commands.append(
        _command(
            args.python,
            script_dir,
            "select_dike_demo_samples.py",
            [
                "--baseline-metrics",
                paths["baseline_global"] / "metrics.csv",
                "--guided-metrics",
                paths["guided_global"] / "metrics.csv",
                "--baseline-target-metrics",
                paths["baseline_target"] / "target_metrics.csv",
                "--guided-target-metrics",
                paths["guided_target"] / "target_metrics.csv",
                "--baseline-dir",
                baseline_dir,
                "--guided-dir",
                guided_dir,
                "--output-dir",
                paths["sample_selection"],
            ],
        )
    )
    return commands


def write_run_script(args: argparse.Namespace, candidates: Sequence[Candidate]) -> Path:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    script_path = args.output_dir / "run_candidate_screening.sh"
    lines = [
        "#!/usr/bin/env bash",
        "set -euo pipefail",
        "export MPLCONFIGDIR=${MPLCONFIGDIR:-/tmp/matplotlib-geoflow}",
        "",
    ]
    for candidate in candidates:
        paths = _paths_for_candidate(args, candidate)
        lines.extend(
            [
                f"echo '=== {candidate.slug} ==='",
                f"mkdir -p {_format_command([str(paths['candidate_dir'])])}",
            ]
        )
        for command in build_candidate_commands(args, candidate):
            lines.append(_format_command(command))
        lines.append("")
    lines.append(
        _format_command(
            [
                args.python,
                str(Path(__file__).resolve()),
                "--truth-root",
                str(args.truth_root),
                "--ckpt-path",
                str(args.ckpt_path),
                "--output-dir",
                str(args.output_dir),
                "--device",
                str(args.device),
                "--summarize-only",
                *sum((["--candidate", f"{candidate.case_name}:{candidate.target_label}"] for candidate in candidates), []),
            ]
        )
    )
    script_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    script_path.chmod(0o755)
    return script_path


def _load_json_if_exists(path: Path) -> Dict[str, object]:
    return read_json(path) if path.exists() else {}


def _load_rows_if_exists(path: Path) -> List[Dict[str, str]]:
    return read_csv_rows(path) if path.exists() else []


def _classify(row: Dict[str, object], args: argparse.Namespace) -> tuple[str, str]:
    missing = []
    for field in (
        "observability_normalized_delta",
        "baseline_target_volume_ratio",
        "geo_misfit_improvement",
        "target_iou_improvement",
    ):
        if row.get(field) in ("", None) or not math.isfinite(float(row[field])):
            missing.append(field)
    if missing:
        return "pending", f"missing completed metrics: {', '.join(missing)}"

    reasons = []
    truth_volume = float(row["truth_target_volume"])
    borehole_hits = row.get("borehole_target_hits")
    borehole_hits_value = float(borehole_hits) if borehole_hits not in ("", None) else 0.0
    observability = float(row["observability_normalized_delta"])
    prior_ratio = float(row["baseline_target_volume_ratio"])
    geo_improvement = float(row["geo_misfit_improvement"])
    iou_improvement = float(row["target_iou_improvement"])
    recall_improvement = float(row["target_recall_improvement"])
    centroid_improvement = float(row["target_centroid_distance_improvement"])

    if truth_volume < args.min_target_voxels:
        reasons.append("truth target volume is too small")
    if borehole_hits_value < args.min_borehole_hits:
        reasons.append("insufficient sparse target-label borehole evidence")
    if observability < args.min_observability_delta:
        reasons.append("weak lightweight gravity-proxy observability")
    if prior_ratio < args.min_prior_volume_ratio:
        reasons.append("weak learned-prior support for target label")
    if prior_ratio > args.max_prior_volume_ratio:
        reasons.append("baseline over-generates target label")
    if geo_improvement < args.min_geo_improvement:
        reasons.append("guided run does not reduce proxy misfit enough")

    geology_preserved = True
    for baseline_field, guided_field in (
        ("baseline_voxel_accuracy_mean", "guided_voxel_accuracy_mean"),
        ("baseline_mean_iou_mean", "guided_mean_iou_mean"),
        ("baseline_borehole_consistency_mean", "guided_borehole_consistency_mean"),
    ):
        baseline = row.get(baseline_field)
        guided = row.get(guided_field)
        if baseline not in ("", None) and guided not in ("", None):
            if math.isfinite(float(baseline)) and math.isfinite(float(guided)):
                if float(guided) < float(baseline) - args.geology_tolerance:
                    geology_preserved = False
    if not geology_preserved:
        reasons.append("guided run degrades global geology metrics")

    target_improved = (
        iou_improvement >= args.min_target_iou_improvement
        or recall_improvement >= args.min_target_recall_improvement
        or centroid_improvement >= args.min_centroid_improvement
    )
    if reasons:
        if geo_improvement >= args.min_geo_improvement:
            return "limitation_case", "; ".join(reasons)
        return "not_recommended", "; ".join(reasons)
    if target_improved:
        return "main_demo_candidate", "proxy misfit improves, target metrics improve, and geology metrics are preserved"
    return "limitation_case", "proxy misfit improves but target-label metrics do not improve enough"


def summarize_candidate(args: argparse.Namespace, candidate: Candidate) -> Dict[str, object]:
    paths = _paths_for_candidate(args, candidate)
    truth_model = paths["truth_model"]
    boreholes = paths["boreholes"] if paths["boreholes"].exists() else None
    if not truth_model.exists():
        raise FileNotFoundError(f"truth_model not found for candidate {candidate.slug}: {truth_model}")
    row: Dict[str, object] = {
        "candidate": candidate.slug,
        "case_name": candidate.case_name,
        "target_label": candidate.target_label,
        "candidate_dir": str(paths["candidate_dir"]),
    }
    row.update(_truth_stats(truth_model, boreholes, candidate.target_label, args.summary_device))

    observability = _load_json_if_exists(paths["observability_dir"] / "summary.json")
    if observability:
        row["observability_normalized_delta"] = observability.get("lightweight_gravity_proxy_normalized_delta", "")
        row["borehole_target_hits"] = observability.get("borehole_target_hits", row.get("borehole_target_hits", ""))
    else:
        row["observability_normalized_delta"] = ""

    baseline_global = _load_rows_if_exists(paths["baseline_global"] / "metrics.csv")
    guided_global = _load_rows_if_exists(paths["guided_global"] / "metrics.csv")
    baseline_target = _load_rows_if_exists(paths["baseline_target"] / "target_metrics.csv")
    guided_target = _load_rows_if_exists(paths["guided_target"] / "target_metrics.csv")
    baseline_summary = _load_json_if_exists(paths["baseline_target"] / "summary.json")
    guided_summary = _load_json_if_exists(paths["guided_target"] / "summary.json")
    for prefix, rows in (("baseline", baseline_global), ("guided", guided_global)):
        row[f"{prefix}_geo_misfit_mean"] = _column_mean(rows, "geo_misfit") if rows else ""
        row[f"{prefix}_voxel_accuracy_mean"] = _column_mean(rows, "voxel_accuracy") if rows else ""
        row[f"{prefix}_mean_iou_mean"] = _column_mean(rows, "mean_iou") if rows else ""
        row[f"{prefix}_borehole_consistency_mean"] = _column_mean(rows, "borehole_consistency") if rows else ""
    for prefix, rows, summary in (
        ("baseline", baseline_target, baseline_summary),
        ("guided", guided_target, guided_summary),
    ):
        volume_values = [_float(record, "predicted_target_volume") for record in rows]
        row[f"{prefix}_target_volume_mean"] = _mean(volume_values) if rows else ""
        row[f"{prefix}_target_volume_std"] = _std(volume_values) if rows else ""
        truth_volume = float(row["truth_target_volume"])
        row[f"{prefix}_target_volume_ratio"] = (
            float(row[f"{prefix}_target_volume_mean"]) / truth_volume
            if rows and truth_volume > 0
            else ""
        )
        row[f"{prefix}_target_iou_mean"] = _column_mean(rows, "target_iou") if rows else ""
        row[f"{prefix}_target_recall_mean"] = _column_mean(rows, "target_recall") if rows else ""
        row[f"{prefix}_target_centroid_distance_mean"] = _column_mean(rows, "target_centroid_distance") if rows else ""
        row[f"{prefix}_probability_overlap_inside_truth"] = summary.get(
            "ensemble_probability_overlap_inside_truth",
            "",
        )

    row["baseline_probability_overlap_inside_truth"] = row.get("baseline_probability_overlap_inside_truth", "")
    row["geo_misfit_improvement"] = (
        float(row["baseline_geo_misfit_mean"]) - float(row["guided_geo_misfit_mean"])
        if row.get("baseline_geo_misfit_mean") not in ("", None)
        and row.get("guided_geo_misfit_mean") not in ("", None)
        else ""
    )
    row["target_iou_improvement"] = (
        float(row["guided_target_iou_mean"]) - float(row["baseline_target_iou_mean"])
        if row.get("baseline_target_iou_mean") not in ("", None)
        and row.get("guided_target_iou_mean") not in ("", None)
        else ""
    )
    row["target_recall_improvement"] = (
        float(row["guided_target_recall_mean"]) - float(row["baseline_target_recall_mean"])
        if row.get("baseline_target_recall_mean") not in ("", None)
        and row.get("guided_target_recall_mean") not in ("", None)
        else ""
    )
    row["target_centroid_distance_improvement"] = (
        float(row["baseline_target_centroid_distance_mean"]) - float(row["guided_target_centroid_distance_mean"])
        if row.get("baseline_target_centroid_distance_mean") not in ("", None)
        and row.get("guided_target_centroid_distance_mean") not in ("", None)
        else ""
    )
    paired = _load_json_if_exists(paths["sample_selection"] / "summary.json").get("paired_by_seed", "")
    row["paired_by_seed"] = paired
    completed = bool(baseline_global and guided_global and baseline_target and guided_target and observability)
    row["status"] = "complete" if completed else "pending_sampling_or_metrics"
    recommendation, reason = _classify(row, args)
    row["recommendation"] = recommendation
    row["reason"] = reason
    return {field: row.get(field, "") for field in OUTPUT_FIELDS}


def write_summary(args: argparse.Namespace, candidates: Sequence[Candidate]) -> List[Dict[str, object]]:
    rows = [summarize_candidate(args, candidate) for candidate in candidates]
    write_csv_rows(args.output_dir / "candidate_screening.csv", rows, OUTPUT_FIELDS)
    grouped: Dict[str, List[str]] = {}
    for row in rows:
        grouped.setdefault(str(row["recommendation"]), []).append(str(row["candidate"]))
    write_json(
        args.output_dir / "candidate_screening_summary.json",
        {
            "output_csv": str(args.output_dir / "candidate_screening.csv"),
            "recommendations": grouped,
            "thresholds": {
                "min_target_voxels": args.min_target_voxels,
                "min_borehole_hits": args.min_borehole_hits,
                "min_prior_volume_ratio": args.min_prior_volume_ratio,
                "max_prior_volume_ratio": args.max_prior_volume_ratio,
                "min_observability_delta": args.min_observability_delta,
                "min_geo_improvement": args.min_geo_improvement,
                "min_target_iou_improvement": args.min_target_iou_improvement,
                "min_target_recall_improvement": args.min_target_recall_improvement,
                "min_centroid_improvement": args.min_centroid_improvement,
            },
            "description": (
                "Candidate screening for dike-like demo labels. Geophysical terms refer "
                "to the lightweight gravity-proxy, not quantitative gravity inversion."
            ),
        },
    )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare and summarize dike guidance candidate screening runs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-root", type=Path, required=True)
    parser.add_argument("--candidate", type=parse_candidate, action="append", required=True)
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--summary-device", default="cpu")
    parser.add_argument("--target-density", type=float, default=3.5)
    parser.add_argument("--background-scale", type=float, default=1.0)
    parser.add_argument("--n-samples", type=int, default=16)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=0.05)
    parser.add_argument("--mu", type=float, default=0.01)
    parser.add_argument("--tau", type=float, default=0.1)
    parser.add_argument("--guidance-start", type=float, default=0.5)
    parser.add_argument("--guidance-schedule", default="late_quadratic")
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--summarize-only", action="store_true")
    parser.add_argument("--min-target-voxels", type=int, default=1000)
    parser.add_argument("--min-borehole-hits", type=int, default=1)
    parser.add_argument("--min-prior-volume-ratio", type=float, default=0.20)
    parser.add_argument("--max-prior-volume-ratio", type=float, default=2.00)
    parser.add_argument("--min-observability-delta", type=float, default=0.01)
    parser.add_argument("--min-geo-improvement", type=float, default=0.02)
    parser.add_argument("--min-target-iou-improvement", type=float, default=0.001)
    parser.add_argument("--min-target-recall-improvement", type=float, default=0.001)
    parser.add_argument("--min-centroid-improvement", type=float, default=0.5)
    parser.add_argument("--geology-tolerance", type=float, default=0.02)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.truth_root = args.truth_root.resolve()
    args.ckpt_path = args.ckpt_path.resolve()
    args.output_dir = args.output_dir.resolve()
    candidates: Sequence[Candidate] = args.candidate
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not args.summarize_only:
        script_path = write_run_script(args, candidates)
        print(f"Saved run script: {script_path}")
    rows = write_summary(args, candidates)
    print(f"Saved candidate screening CSV: {args.output_dir / 'candidate_screening.csv'}")
    for row in rows:
        print(f"{row['candidate']}: {row['recommendation']} ({row['status']}) - {row['reason']}")


if __name__ == "__main__":
    main()
