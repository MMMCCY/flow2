"""One-command assembly for dike guidance demo post-processing artifacts."""

from __future__ import annotations

import argparse
import csv
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Mapping, Optional

from geology_io_utils import infer_paired_by_seed, load_density_config, read_csv_rows, read_json, write_json


def _script_dir() -> Path:
    return Path(__file__).resolve().parent


def _command(script_name: str, args: List[str]) -> List[str]:
    return [sys.executable, str(_script_dir() / script_name), *args]


def _run(command: List[str], dry_run: bool) -> None:
    print(shlex.join(command))
    if not dry_run:
        subprocess.run(command, check=True)


def _default_metrics(run_dir: Path) -> Path:
    candidate = run_dir / "evaluation" / "metrics.csv"
    return candidate if candidate.exists() else run_dir / "metrics.csv"


def _maybe_read_json(path: Path) -> Dict[str, object]:
    return read_json(path) if path.exists() else {}


def _selected_rows(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _select_residual_sample_id(rows: List[Mapping[str, object]]) -> Optional[int]:
    preferred = (
        "max_geo_improvement_preserve_geology",
        "best_guided_geo",
        "representative_guided",
        "best_target_improvement",
    )
    for role in preferred:
        for row in rows:
            if row.get("role") == role and row.get("sample_id") not in ("", None):
                return int(row["sample_id"])
    if rows and rows[0].get("sample_id") not in ("", None):
        return int(rows[0]["sample_id"])
    return None


def _sample_id_args(rows: List[Mapping[str, object]], limit: int = 6) -> List[str]:
    ids = []
    for row in rows:
        if row.get("sample_id") in ("", None):
            continue
        sample_id = int(row["sample_id"])
        if sample_id not in ids:
            ids.append(sample_id)
        if len(ids) >= limit:
            break
    result: List[str] = []
    for sample_id in ids:
        result.extend(["--sample-id", str(sample_id)])
    return result


def _sweep_summary_candidate(baseline_dir: Path, guided_dir: Path) -> Optional[Path]:
    for candidate in (
        guided_dir.parent / "guidance_sweep_summary.csv",
        baseline_dir.parent / "guidance_sweep_summary.csv",
    ):
        if candidate.exists():
            return candidate
    return None


def _metric_snapshot(path: Path) -> Dict[str, object]:
    rows = read_csv_rows(path) if path.exists() else []
    if not rows:
        return {}
    try:
        best = min(rows, key=lambda row: float(row.get("geo_misfit", "inf")))
    except ValueError:
        best = rows[0]
    return {
        "n_rows": len(rows),
        "best_sample_id": best.get("sample_id", ""),
        "best_geo_misfit": best.get("geo_misfit", ""),
        "best_mean_iou": best.get("mean_iou", ""),
    }


def _target_snapshot(path: Path) -> Dict[str, object]:
    rows = read_csv_rows(path) if path.exists() else []
    if not rows:
        return {}
    try:
        best = max(rows, key=lambda row: float(row.get("target_iou", "-inf")))
    except ValueError:
        best = rows[0]
    return {
        "best_sample_id": best.get("sample_id", ""),
        "best_target_iou": best.get("target_iou", ""),
        "best_target_recall": best.get("target_recall", ""),
        "best_target_volume_error": best.get("target_volume_error", ""),
    }


def _global_evaluation_command(
    run_dir: Path,
    truth_model: Path,
    boreholes: Optional[Path],
    observed_gravity: Optional[Path],
    density_config: Optional[Path],
    output_dir: Path,
    kernel_size: int,
    device: str,
) -> List[str]:
    args = [
        "--samples-dir",
        str(run_dir),
        "--truth-model",
        str(truth_model),
        "--output-dir",
        str(output_dir),
        "--kernel-size",
        str(kernel_size),
        "--device",
        device,
    ]
    if boreholes:
        args.extend(["--boreholes", str(boreholes)])
    if observed_gravity:
        args.extend(["--observed-gravity", str(observed_gravity)])
    if density_config:
        args.extend(["--density-config", str(density_config)])
    return _command("evaluate_geophysics.py", args)


def _load_density_config_for_demo(path: Optional[Path], dry_run: bool, warnings: List[str]) -> Optional[Dict[str, object]]:
    if path is None:
        return None
    if path.exists():
        return load_density_config(path)
    if dry_run:
        warnings.append(f"dry-run: density_config was not read because it does not exist yet: {path}")
        return None
    return load_density_config(path)


def _resolve_target_label(
    target_label: Optional[int],
    density_config: Optional[Mapping[str, object]],
    allow_auto_target_selection: bool,
) -> tuple[Optional[int], str]:
    config_label = None
    if density_config is not None and density_config.get("target_label") is not None:
        config_label = int(density_config["target_label"])
    if target_label is not None and config_label is not None and int(target_label) != config_label:
        raise SystemExit(
            f"--target-label {target_label} conflicts with density_config target_label {config_label}"
        )
    if target_label is not None:
        return int(target_label), "--target-label"
    if config_label is not None:
        return config_label, "density_config.target_label"
    if allow_auto_target_selection:
        return None, "select_dike_demo_case.py"
    raise SystemExit(
        "manual target label is required: run visualize_truth_model_labels.py and "
        "create_density_config.py, then pass --density-config; alternatively pass "
        "--target-label. Use --allow-auto-target-selection only for legacy exploratory runs."
    )


def _write_report(
    output_dir: Path,
    commands: List[List[str]],
    paired: bool,
    paired_reason: str,
    target_label: object,
    target_label_source: str,
    density_config_path: Optional[Path],
    observed_gravity_path: Optional[Path],
    warnings: List[str],
) -> None:
    observability = _maybe_read_json(output_dir / "observability" / "summary.json")
    baseline_global = _metric_snapshot(output_dir / "input_baseline_metrics.csv")
    guided_global = _metric_snapshot(output_dir / "input_guided_metrics.csv")
    baseline_target = _target_snapshot(output_dir / "baseline_target" / "target_metrics.csv")
    guided_target = _target_snapshot(output_dir / "guided_target" / "target_metrics.csv")
    lines = [
        "# Dike Guidance Demo Report",
        "",
        "This report summarizes post-processing artifacts for baseline vs guided dike-like target reconstruction.",
        "",
        "Terminology: all geophysical fields are lightweight gravity-proxy fields.",
        "",
        f"- target_label: `{target_label}`",
        f"- target_label_source: `{target_label_source}`",
        f"- density_config: `{density_config_path}`",
        f"- observed_gravity: `{observed_gravity_path}`",
        f"- recommended_for_demo: `{str(observability.get('recommended_for_demo', 'unknown')).lower()}`",
        f"- observability_reason: {observability.get('reason', '')}",
        f"- paired_by_seed: `{str(paired).lower()}`",
        f"- paired_by_seed_reason: {paired_reason}",
        f"- baseline best geo_misfit: {baseline_global.get('best_geo_misfit', '')}",
        f"- guided best geo_misfit: {guided_global.get('best_geo_misfit', '')}",
        f"- baseline best target_iou/recall/volume_error: {baseline_target.get('best_target_iou', '')} / {baseline_target.get('best_target_recall', '')} / {baseline_target.get('best_target_volume_error', '')}",
        f"- guided best target_iou/recall/volume_error: {guided_target.get('best_target_iou', '')} / {guided_target.get('best_target_recall', '')} / {guided_target.get('best_target_volume_error', '')}",
        "",
        "## Warnings",
        "",
    ]
    lines.extend(f"- {warning}" for warning in warnings)
    if not warnings:
        lines.append("- none")
    lines.extend([
        "",
        "## Artifacts",
        "",
        "- `truth_label_qa/`: manual truth_model label QA figures and label summary",
        "- `observed_gravity/`: observed lightweight gravity-proxy generated from truth_model + density_config when requested",
        "- `case_selection/manifest.json`: legacy automatic target-label evidence when `--allow-auto-target-selection` is used",
        "- `observability/summary.json`: target-label lightweight gravity-proxy observability",
        "- `baseline_target/target_metrics.csv`: baseline target-label metrics",
        "- `guided_target/target_metrics.csv`: guided target-label metrics",
        "- `figures/`: ensemble probability and target realization figures",
        "- `residuals/`: gravity-proxy residual comparison",
        "- `sample_selection/selected_samples.csv`: samples selected for display",
        "- `sweep/combined_guidance_response.png`: guidance sweep response when `guidance_sweep_summary.csv` exists",
        "",
        "## Commands",
        "",
    ])
    lines.extend(f"```bash\n{shlex.join(command)}\n```" for command in commands)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "demo_report.md").write_text("\n\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assemble dike guidance demo figures and summaries from saved outputs.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--observed-gravity", type=Path, default=None)
    parser.add_argument("--density-config", type=Path, default=None)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--guided-metrics", type=Path, default=None)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument(
        "--allow-auto-target-selection",
        action="store_true",
        help="Legacy exploratory mode: run select_dike_demo_case.py when no manual target label is supplied.",
    )
    parser.add_argument(
        "--generate-observed-gravity",
        action="store_true",
        help="Deprecated compatibility flag; observed_gravity.pt is generated automatically when density_config is provided.",
    )
    parser.add_argument(
        "--skip-observed-gravity-generation",
        action="store_true",
        help="Do not auto-generate observed_gravity.pt when --density-config is provided.",
    )
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    baseline_metrics = args.baseline_metrics or _default_metrics(args.baseline_dir)
    guided_metrics = args.guided_metrics or _default_metrics(args.guided_dir)
    output_dir = args.output_dir
    commands: List[List[str]] = []
    warnings: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)

    density_config = _load_density_config_for_demo(args.density_config, args.dry_run, warnings)
    target_label, target_label_source = _resolve_target_label(
        args.target_label,
        density_config,
        args.allow_auto_target_selection,
    )

    truth_qa_args = [
        "--truth-model",
        str(args.truth_model),
        "--output-dir",
        str(output_dir / "truth_label_qa"),
        "--device",
        args.device,
    ]
    if args.boreholes:
        truth_qa_args.extend(["--boreholes", str(args.boreholes)])
    command = _command("visualize_truth_model_labels.py", truth_qa_args)
    commands.append(command)
    _run(command, args.dry_run)

    if target_label is None and args.allow_auto_target_selection:
        warnings.append(
            "automatic target-label selection was used; do not use this mode for paper claims without manual label QA"
        )
        select_args = [
            "--truth-model",
            str(args.truth_model),
            "--output-dir",
            str(output_dir / "case_selection"),
            "--kernel-size",
            str(args.kernel_size),
            "--device",
            args.device,
        ]
        if args.boreholes:
            select_args.extend(["--boreholes", str(args.boreholes)])
        select_args.extend(["--samples-dir", str(args.baseline_dir)])
        select_args.extend(["--baseline-dir", str(args.baseline_dir), "--guided-dir", str(args.guided_dir)])
        command = _command("select_dike_demo_case.py", select_args)
        commands.append(command)
        _run(command, args.dry_run)
        if not args.dry_run:
            target_label = int(read_json(output_dir / "case_selection" / "manifest.json")["selected"]["target_label"])
        target_label_source = "select_dike_demo_case.py"
    if target_label is None:
        target_label_text = "<target_label>"
    else:
        target_label_text = str(target_label)

    observed_gravity_path = args.observed_gravity
    if (
        observed_gravity_path is None
        and args.density_config is not None
        and not args.skip_observed_gravity_generation
    ):
        observed_command = _command(
            "generate_observed_gravity.py",
            [
                "--truth-model",
                str(args.truth_model),
                "--density-config",
                str(args.density_config),
                "--output-dir",
                str(output_dir / "observed_gravity"),
                "--kernel-size",
                str(args.kernel_size),
                "--device",
                args.device,
            ],
        )
        commands.append(observed_command)
        _run(observed_command, args.dry_run)
        observed_gravity_path = output_dir / "observed_gravity" / "observed_gravity.pt"

    if args.density_config is None:
        warnings.append(
            "no density_config was provided; downstream lightweight gravity-proxy outputs use the default LithologyPropertyMap"
        )

    if not baseline_metrics.exists():
        baseline_eval_dir = output_dir / "baseline_global_evaluation"
        command = _global_evaluation_command(
            run_dir=args.baseline_dir,
            truth_model=args.truth_model,
            boreholes=args.boreholes,
            observed_gravity=observed_gravity_path,
            density_config=args.density_config,
            output_dir=baseline_eval_dir,
            kernel_size=args.kernel_size,
            device=args.device,
        )
        commands.append(command)
        _run(command, args.dry_run)
        baseline_metrics = baseline_eval_dir / "metrics.csv"
    if not guided_metrics.exists():
        guided_eval_dir = output_dir / "guided_global_evaluation"
        command = _global_evaluation_command(
            run_dir=args.guided_dir,
            truth_model=args.truth_model,
            boreholes=args.boreholes,
            observed_gravity=observed_gravity_path,
            density_config=args.density_config,
            output_dir=guided_eval_dir,
            kernel_size=args.kernel_size,
            device=args.device,
        )
        commands.append(command)
        _run(command, args.dry_run)
        guided_metrics = guided_eval_dir / "metrics.csv"

    observability_args = [
        "--truth-model",
        str(args.truth_model),
        "--target-label",
        target_label_text,
        "--output-dir",
        str(output_dir / "observability"),
        "--kernel-size",
        str(args.kernel_size),
        "--device",
        args.device,
    ]
    if args.boreholes:
        observability_args.extend(["--boreholes", str(args.boreholes)])
    if args.density_config:
        observability_args.extend(["--density-config", str(args.density_config)])
    command = _command("analyze_dike_observability.py", observability_args)
    commands.append(command)
    _run(command, args.dry_run)
    if not args.dry_run:
        observability_summary = read_json(output_dir / "observability" / "summary.json")
        if not bool(observability_summary.get("recommended_for_demo", False)):
            warnings.append(
                "observability gate failed: target is not recommended for supporting dike guidance conclusions; QA artifacts were still generated"
            )

    for run_name, run_dir, metrics_path in (
        ("baseline_target", args.baseline_dir, baseline_metrics),
        ("guided_target", args.guided_dir, guided_metrics),
    ):
        command = (
            _command(
                "evaluate_target_feature.py",
                [
                    "--samples-dir",
                    str(run_dir),
                    "--truth-model",
                    str(args.truth_model),
                    "--target-label",
                    target_label_text,
                    "--metrics-csv",
                    str(metrics_path),
                    "--output-dir",
                    str(output_dir / run_name),
                    "--device",
                    args.device,
                ],
            )
        )
        commands.append(command)
        _run(command, args.dry_run)

    selection_command = _command(
            "select_dike_demo_samples.py",
            [
                "--baseline-metrics",
                str(baseline_metrics),
                "--guided-metrics",
                str(guided_metrics),
                "--baseline-target-metrics",
                str(output_dir / "baseline_target" / "target_metrics.csv"),
                "--guided-target-metrics",
                str(output_dir / "guided_target" / "target_metrics.csv"),
                "--baseline-dir",
                str(args.baseline_dir),
                "--guided-dir",
                str(args.guided_dir),
                "--output-dir",
                str(output_dir / "sample_selection"),
            ],
        )
    commands.append(selection_command)
    _run(selection_command, args.dry_run)

    selected_rows = [] if args.dry_run else _selected_rows(output_dir / "sample_selection" / "selected_samples.csv")
    residual_sample_id = None if args.dry_run else _select_residual_sample_id(selected_rows)
    if residual_sample_id is None:
        if args.dry_run:
            residual_sample_id = "<selected_sample_id>"
        else:
            raise SystemExit("sample selection did not produce a usable sample_id")

    visualize_args = [
        "--baseline-dir",
        str(args.baseline_dir),
        "--guided-dir",
        str(args.guided_dir),
        "--truth-model",
        str(args.truth_model),
        "--target-label",
        target_label_text,
        "--output-dir",
        str(output_dir / "figures"),
        "--device",
        args.device,
    ]
    if args.boreholes:
        visualize_args.extend(["--boreholes", str(args.boreholes)])
    visualize_args.extend(_sample_id_args(selected_rows) if not args.dry_run else ["--sample-id", str(residual_sample_id)])
    visualize_command = _command("visualize_dike_ensemble.py", visualize_args)
    commands.append(visualize_command)
    _run(visualize_command, args.dry_run)

    residual_args = [
        "--baseline-dir",
        str(args.baseline_dir),
        "--guided-dir",
        str(args.guided_dir),
        "--sample-id",
        str(residual_sample_id),
        "--kernel-size",
        str(args.kernel_size),
        "--output-dir",
        str(output_dir / "residuals"),
        "--device",
        args.device,
    ]
    if observed_gravity_path:
        residual_args.extend(["--observed-gravity", str(observed_gravity_path)])
    else:
        residual_args.extend(["--truth-model", str(args.truth_model)])
    if args.density_config:
        residual_args.extend(["--density-config", str(args.density_config)])
    residual_command = _command("compare_gravity_residuals.py", residual_args)
    commands.append(residual_command)
    _run(residual_command, args.dry_run)

    sweep_summary = _sweep_summary_candidate(args.baseline_dir, args.guided_dir)
    if sweep_summary is not None:
        sweep_command = _command(
            "plot_guidance_sweep.py",
            ["--summary-csv", str(sweep_summary), "--output-dir", str(output_dir / "sweep")],
        )
        commands.append(sweep_command)
        _run(sweep_command, args.dry_run)

    # Preserve the input global metrics next to the report for summary reads.
    if not args.dry_run:
        for source, dest in (
            (baseline_metrics, output_dir / "input_baseline_metrics.csv"),
            (guided_metrics, output_dir / "input_guided_metrics.csv"),
        ):
            if source.exists():
                dest.write_text(source.read_text(encoding="utf-8"), encoding="utf-8")

    paired, paired_reason = infer_paired_by_seed(args.baseline_dir, args.guided_dir)
    write_json(
        output_dir / "manifest.json",
        {
            "target_label": target_label,
            "target_label_source": target_label_source,
            "paired_by_seed": paired,
            "paired_by_seed_reason": paired_reason,
            "baseline_dir": str(args.baseline_dir),
            "guided_dir": str(args.guided_dir),
            "truth_model": str(args.truth_model),
            "density_config": str(args.density_config) if args.density_config else None,
            "observed_gravity": str(observed_gravity_path) if observed_gravity_path else None,
            "residual_sample_id": residual_sample_id,
            "description": (
                "One-command dike guidance demo manifest. All geophysical fields "
                "are lightweight gravity-proxy outputs."
            ),
        },
    )
    _write_report(
        output_dir,
        commands,
        paired,
        paired_reason,
        target_label,
        target_label_source,
        args.density_config,
        observed_gravity_path,
        warnings,
    )
    print(f"Saved demo manifest/report: {output_dir}")


if __name__ == "__main__":
    main()
