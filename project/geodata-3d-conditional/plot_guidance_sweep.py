"""Plot lightweight gravity-proxy guidance sweep response curves."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List


def _float(value: object) -> float:
    try:
        if value in ("", None):
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def read_sweep(path: Path) -> List[Dict[str, object]]:
    if not path.exists():
        raise FileNotFoundError(f"guidance sweep summary not found: {path}")
    with path.open("r", newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise ValueError(f"guidance sweep summary is empty: {path}")
    return rows


def save_guidance_response(rows: List[Dict[str, object]], output_path: Path) -> Dict[str, object]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x_key = "alpha" if any(math.isfinite(_float(row.get("alpha"))) for row in rows) else "mu"
    rows = sorted(rows, key=lambda row: _float(row.get(x_key)))
    x_values = [_float(row.get(x_key)) for row in rows]
    specs = [
        ("geo_misfit_mean", "Mean lightweight gravity-proxy misfit"),
        ("mean_iou_mean", "Mean IoU"),
        ("borehole_consistency_mean", "Borehole consistency"),
    ]
    figure, axes = plt.subplots(1, len(specs), figsize=(5 * len(specs), 4))
    if len(specs) == 1:
        axes = [axes]
    for axis, (field, label) in zip(axes, specs):
        y_values = [_float(row.get(field)) for row in rows]
        axis.plot(x_values, y_values, marker="o")
        axis.set_xlabel(x_key)
        axis.set_ylabel(label)
        axis.set_title(label)
        axis.grid(alpha=0.25)
    figure.suptitle("Guidance sweep response")
    figure.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=180)
    plt.close(figure)
    return {
        "x_key": x_key,
        "n_runs": len(rows),
        "figure": str(output_path),
        "description": "Guidance sweep summary for lightweight gravity-proxy guided sampling.",
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot guidance sweep response from guidance_sweep_summary.csv.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--summary-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    summary = save_guidance_response(read_sweep(args.summary_csv), args.output_dir / "combined_guidance_response.png")
    print(f"Saved guidance response: {summary['figure']}")


if __name__ == "__main__":
    main()
