"""Summarize geophysical guidance sweep evaluation outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence


OUTPUT_FIELDS = [
    "run_name",
    "guidance_mode",
    "mu",
    "alpha",
    "n_samples",
    "geo_misfit_mean",
    "geo_misfit_std",
    "geo_misfit_min",
    "voxel_accuracy_mean",
    "voxel_accuracy_std",
    "mean_iou_mean",
    "mean_iou_std",
    "borehole_consistency_mean",
    "borehole_consistency_std",
    "best10_geo_misfit_mean",
    "best10_voxel_accuracy_mean",
    "best10_mean_iou_mean",
    "worst10_geo_misfit_mean",
    "worst10_voxel_accuracy_mean",
    "worst10_mean_iou_mean",
]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize multiple guided geophysical evaluation folders.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--root-dir", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def _to_float(value: object) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def _finite_values(rows: Sequence[Dict[str, str]], field: str) -> List[float]:
    values = []
    for row in rows:
        value = _to_float(row.get(field))
        if value is not None:
            values.append(value)
    return values


def _mean(values: Sequence[float]) -> object:
    if not values:
        return ""
    return sum(values) / len(values)


def _std(values: Sequence[float]) -> object:
    if not values:
        return ""
    mean = sum(values) / len(values)
    variance = sum((value - mean) ** 2 for value in values) / len(values)
    return math.sqrt(variance)


def _minimum(values: Sequence[float]) -> object:
    return min(values) if values else ""


def _format_setting(value: object) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)


def _folder_number(text: str) -> Optional[float]:
    if not text:
        return None
    if "_" in text:
        text = text.replace("_", ".")
    elif text.startswith("0") and len(text) > 1:
        text = "0." + text[1:]
    return _to_float(text)


def _settings_from_name(run_name: str) -> Dict[str, object]:
    settings: Dict[str, object] = {"guidance_mode": "", "mu": "", "alpha": ""}
    mu_match = re.search(r"(?:^|_)mu_([0-9][0-9_]*)", run_name)
    alpha_match = re.search(r"(?:^|_)alpha_([0-9][0-9_]*)", run_name)

    if run_name.startswith("relative") or alpha_match:
        settings["guidance_mode"] = "relative"
    elif mu_match:
        settings["guidance_mode"] = "absolute"

    if mu_match:
        mu = _folder_number(mu_match.group(1))
        settings["mu"] = "" if mu is None else mu
    if alpha_match:
        alpha = _folder_number(alpha_match.group(1))
        settings["alpha"] = "" if alpha is None else alpha
    return settings


def _settings_from_config(run_dir: Path) -> Dict[str, object]:
    for path in (run_dir / "config.json", run_dir / "evaluation" / "config.json"):
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8") as stream:
            config = json.load(stream)
        return {
            "guidance_mode": config.get("guidance_mode", ""),
            "mu": config.get("mu", ""),
            "alpha": config.get("alpha", ""),
        }
    return {}


def _read_metrics(path: Path) -> List[Dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def _best_worst_rows(rows: Sequence[Dict[str, str]]) -> tuple[List[Dict[str, str]], List[Dict[str, str]]]:
    ranked = [
        (value, row)
        for row in rows
        if (value := _to_float(row.get("geo_misfit"))) is not None
    ]
    ranked.sort(key=lambda item: item[0])
    if not ranked:
        return [], []
    count = max(1, math.ceil(len(ranked) * 0.1))
    best = [row for _, row in ranked[:count]]
    worst = [row for _, row in ranked[-count:]]
    return best, worst


def _summarize_run(root_dir: Path, metrics_path: Path) -> Dict[str, object]:
    run_dir = metrics_path.parent.parent if metrics_path.parent.name == "evaluation" else metrics_path.parent
    run_name = str(run_dir.relative_to(root_dir)) if run_dir != root_dir else run_dir.name
    rows = _read_metrics(metrics_path)
    settings = _settings_from_name(run_dir.name)
    settings.update(
        {
            key: value
            for key, value in _settings_from_config(run_dir).items()
            if value not in ("", None)
        }
    )

    best_rows, worst_rows = _best_worst_rows(rows)
    summary: Dict[str, object] = {
        "run_name": run_name,
        "guidance_mode": settings.get("guidance_mode", ""),
        "mu": settings.get("mu", ""),
        "alpha": settings.get("alpha", ""),
        "n_samples": len(rows),
    }
    for field in ("geo_misfit", "voxel_accuracy", "mean_iou", "borehole_consistency"):
        values = _finite_values(rows, field)
        summary[f"{field}_mean"] = _mean(values)
        summary[f"{field}_std"] = _std(values)
        if field == "geo_misfit":
            summary[f"{field}_min"] = _minimum(values)

    for prefix, group_rows in (("best10", best_rows), ("worst10", worst_rows)):
        for field in ("geo_misfit", "voxel_accuracy", "mean_iou"):
            summary[f"{prefix}_{field}_mean"] = _mean(
                _finite_values(group_rows, field)
            )
    return summary


def _find_metrics(root_dir: Path) -> List[Path]:
    paths = sorted(root_dir.rglob("metrics.csv"))
    return [path for path in paths if path.parent != root_dir]


def _write_summary(rows: Iterable[Dict[str, object]], output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=OUTPUT_FIELDS)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in OUTPUT_FIELDS})


def _print_table(rows: Sequence[Dict[str, object]]) -> None:
    columns = [
        "run_name",
        "guidance_mode",
        "mu",
        "alpha",
        "n_samples",
        "geo_misfit_mean",
        "voxel_accuracy_mean",
        "mean_iou_mean",
        "borehole_consistency_mean",
    ]
    widths = {
        column: max(
            len(column),
            *(
                len(_format_setting(row.get(column, "")))
                for row in rows
            ),
        )
        for column in columns
    }
    print("  ".join(column.ljust(widths[column]) for column in columns))
    for row in rows:
        print(
            "  ".join(
                _format_setting(row.get(column, "")).ljust(widths[column])
                for column in columns
            )
        )


def main() -> None:
    args = _parse_args()
    metrics_paths = _find_metrics(args.root_dir)
    if not metrics_paths:
        raise SystemExit(f"no metrics.csv files found under {args.root_dir}")

    summaries = [_summarize_run(args.root_dir, path) for path in metrics_paths]
    _write_summary(summaries, args.output_csv)
    _print_table(summaries)
    print(f"\nSaved summary: {args.output_csv}")


if __name__ == "__main__":
    main()
