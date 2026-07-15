"""Compare baseline and guided lightweight gravity-proxy residual fields."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Mapping, Sequence

import torch

from geophysics import LithologyPropertyMap, SimpleGravityForward, normalized_misfit
from geology_io_utils import (
    find_sample_files,
    infer_paired_by_seed,
    load_density_config,
    load_sample_stack,
    load_tensor,
    load_volume,
    property_map_from_density_config,
    read_csv_rows,
    write_json,
)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _save_field_plot(
    array: torch.Tensor,
    path: Path,
    title: str,
    cmap: str,
    vmin: float | None = None,
    vmax: float | None = None,
) -> None:
    plt = _setup_matplotlib()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(array.detach().cpu().float().T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _best_sample_id(metrics_path: Path) -> int | None:
    if not metrics_path.exists():
        return None
    rows = read_csv_rows(metrics_path)
    ranked = []
    for row in rows:
        try:
            ranked.append((float(row["geo_misfit"]), int(row["sample_id"])))
        except (KeyError, TypeError, ValueError):
            continue
    if not ranked:
        return None
    return min(ranked, key=lambda item: item[0])[1]


def _sample_by_id(samples: torch.Tensor, records: Sequence[Mapping[str, object]], sample_id: int) -> torch.Tensor:
    for record in records:
        if int(record["sample_id"]) == int(sample_id):
            index = int(record["stack_index"])
            return samples[index : index + 1]
    raise ValueError(f"sample_id {sample_id} not found")


def save_residual_comparison(
    baseline_sample: torch.Tensor,
    guided_sample: torch.Tensor,
    observed_gravity: torch.Tensor,
    output_dir: Path,
    sample_id: int,
    kernel_size: int = 9,
    density_config: Mapping[str, object] | None = None,
) -> dict[str, object]:
    plt = _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    property_map = property_map_from_density_config(density_config)
    forward_model = SimpleGravityForward(kernel_size=kernel_size)
    baseline_gravity = forward_model(property_map(baseline_sample))
    guided_gravity = forward_model(property_map(guided_sample))
    observed = observed_gravity.reshape(1, 1, *observed_gravity.shape[-2:]).to(baseline_gravity)
    baseline_residual = baseline_gravity - observed
    guided_residual = guided_gravity - observed
    residual_difference = baseline_residual - guided_residual
    baseline_misfit = normalized_misfit(baseline_gravity, observed, reduction="mean")
    guided_misfit = normalized_misfit(guided_gravity, observed, reduction="mean")
    residual_abs_max = float(
        torch.stack(
            [
                baseline_residual.abs().max(),
                guided_residual.abs().max(),
                residual_difference.abs().max(),
            ]
        )
        .max()
        .item()
    )

    panels = [
        (observed[0, 0], "Observed lightweight gravity-proxy"),
        (baseline_gravity[0, 0], "Baseline predicted proxy"),
        (baseline_residual[0, 0], "Baseline proxy residual"),
        (guided_gravity[0, 0], "Guided predicted proxy"),
        (guided_residual[0, 0], "Guided proxy residual"),
    ]
    figure, axes = plt.subplots(1, len(panels), figsize=(4 * len(panels), 4))
    for axis, (array, title) in zip(axes, panels):
        cmap = "coolwarm" if "residual" in title.lower() else "viridis"
        vmin = -residual_abs_max if "residual" in title.lower() else None
        vmax = residual_abs_max if "residual" in title.lower() else None
        image = axis.imshow(array.detach().cpu().T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle(f"Sample {sample_id}: baseline vs guided gravity-proxy residuals")
    figure.tight_layout()
    figure_path = output_dir / "gravity_proxy_residual_comparison.png"
    figure.savefig(figure_path, dpi=180)
    plt.close(figure)
    individual_paths = {
        "observed_gravity": output_dir / "observed_gravity.png",
        "baseline_predicted_gravity": output_dir / f"baseline_predicted_gravity_sample_{sample_id}.png",
        "guided_predicted_gravity": output_dir / f"guided_predicted_gravity_sample_{sample_id}.png",
        "baseline_residual": output_dir / f"baseline_residual_sample_{sample_id}.png",
        "guided_residual": output_dir / f"guided_residual_sample_{sample_id}.png",
        "residual_difference": output_dir / f"residual_difference_sample_{sample_id}.png",
    }
    _save_field_plot(observed[0, 0], individual_paths["observed_gravity"], "Observed lightweight gravity-proxy", "viridis")
    _save_field_plot(
        baseline_gravity[0, 0],
        individual_paths["baseline_predicted_gravity"],
        f"Baseline predicted proxy sample {sample_id}",
        "viridis",
    )
    _save_field_plot(
        guided_gravity[0, 0],
        individual_paths["guided_predicted_gravity"],
        f"Guided predicted proxy sample {sample_id}",
        "viridis",
    )
    _save_field_plot(
        baseline_residual[0, 0],
        individual_paths["baseline_residual"],
        f"Baseline proxy residual sample {sample_id}",
        "coolwarm",
        -residual_abs_max,
        residual_abs_max,
    )
    _save_field_plot(
        guided_residual[0, 0],
        individual_paths["guided_residual"],
        f"Guided proxy residual sample {sample_id}",
        "coolwarm",
        -residual_abs_max,
        residual_abs_max,
    )
    _save_field_plot(
        residual_difference[0, 0],
        individual_paths["residual_difference"],
        f"Baseline minus guided residual sample {sample_id}",
        "coolwarm",
        -residual_abs_max,
        residual_abs_max,
    )
    baseline_residual_rms = baseline_residual.square().mean().sqrt()
    guided_residual_rms = guided_residual.square().mean().sqrt()
    baseline_abs_mean = baseline_residual.abs().mean()
    guided_abs_mean = guided_residual.abs().mean()

    return {
        "sample_id": int(sample_id),
        "kernel_size": int(kernel_size),
        "density_config": density_config.get("name") if density_config else None,
        "baseline_lightweight_gravity_proxy_misfit": float(baseline_misfit.item()),
        "guided_lightweight_gravity_proxy_misfit": float(guided_misfit.item()),
        "misfit_improvement": float(baseline_misfit.item() - guided_misfit.item()),
        "residual_rms_reduction": float((baseline_residual_rms - guided_residual_rms).item()),
        "residual_abs_mean_reduction": float((baseline_abs_mean - guided_abs_mean).item()),
        "residual_symmetric_color_limit": residual_abs_max,
        "figure": str(figure_path),
        "individual_figures": {key: str(value) for key, value in individual_paths.items()},
        "description": (
            "Residual comparison uses the existing lightweight gravity-proxy."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot baseline vs guided lightweight gravity-proxy residuals.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--observed-gravity", type=Path, default=None)
    parser.add_argument("--sample-id", type=int, default=None)
    parser.add_argument("--baseline-metrics", type=Path, default=None)
    parser.add_argument("--guided-metrics", type=Path, default=None)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--density-config", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.observed_gravity is None and args.truth_model is None:
        raise SystemExit("provide --observed-gravity or --truth-model")
    baseline, baseline_records = load_sample_stack(find_sample_files(args.baseline_dir), device=args.device)
    guided, guided_records = load_sample_stack(find_sample_files(args.guided_dir), device=args.device)
    sample_id = args.sample_id
    if sample_id is None and args.guided_metrics is not None:
        sample_id = _best_sample_id(args.guided_metrics)
    if sample_id is None and args.baseline_metrics is not None:
        sample_id = _best_sample_id(args.baseline_metrics)
    if sample_id is None:
        sample_id = int(baseline_records[0]["sample_id"])

    if args.observed_gravity is not None:
        observed = load_tensor(args.observed_gravity, device=args.device)
    else:
        truth = load_volume(args.truth_model, device=args.device, single=True)
        density_config = load_density_config(args.density_config)
        observed = SimpleGravityForward(kernel_size=args.kernel_size)(property_map_from_density_config(density_config)(truth))
    density_config = load_density_config(args.density_config)
    summary = save_residual_comparison(
        baseline_sample=_sample_by_id(baseline, baseline_records, sample_id),
        guided_sample=_sample_by_id(guided, guided_records, sample_id),
        observed_gravity=observed,
        output_dir=args.output_dir,
        sample_id=sample_id,
        kernel_size=args.kernel_size,
        density_config=density_config,
    )
    paired, reason = infer_paired_by_seed(args.baseline_dir, args.guided_dir)
    summary["paired_by_seed"] = paired
    summary["paired_by_seed_reason"] = reason
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved gravity-proxy residual comparison: {args.output_dir}")


if __name__ == "__main__":
    main()
