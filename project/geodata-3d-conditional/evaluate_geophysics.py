"""Post-hoc feasibility evaluation for decoded geology realizations."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import torch

from geophysics import (
    GravityGradientForward,
    LithologyPropertyMap,
    MagneticTMIForward,
    SimpleGravityForward,
    mean_iou,
    normalized_misfit,
    rank_realizations_by_geophysics,
    spearman_correlation,
    voxel_accuracy,
)

try:
    from geology_io_utils import (
        load_density_config,
        load_susceptibility_config,
        property_map_from_density_config,
        property_map_from_susceptibility_config,
    )
except ImportError:  # keep this script usable before optional demo helpers exist
    load_density_config = None
    load_susceptibility_config = None
    property_map_from_density_config = None
    property_map_from_susceptibility_config = None


MetricRecord = Dict[str, object]


def _numeric_suffix(path: Path) -> int:
    match = re.search(r"_(\d+)\.pt$", path.name)
    return int(match.group(1)) if match else -1


def find_realization_files(
    samples_dir: Path,
    prefixes: Optional[Sequence[str]] = None,
) -> List[Path]:
    """Find generated tensors with sample, sol, or run prefixes."""
    search_prefixes = tuple(prefixes or ("sample", "sol", "run"))
    paths: List[Path] = []
    for prefix in search_prefixes:
        paths.extend(samples_dir.glob(f"{prefix}_*.pt"))

    unique_paths = sorted(
        set(paths),
        key=lambda path: (
            path.stem.rsplit("_", 1)[0],
            _numeric_suffix(path),
            path.name,
        ),
    )
    if not unique_paths:
        searched = ", ".join(f"{prefix}_*.pt" for prefix in search_prefixes)
        raise FileNotFoundError(
            f"no realizations matching {searched} found in {samples_dir}"
        )
    return unique_paths


def _normalize_saved_realization(volume: torch.Tensor, source: object) -> torch.Tensor:
    """Normalize one saved volume to [1, X, Y, Z]."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"expected a tensor in {source}, got {type(volume).__name__}")
    if volume.dim() == 3:
        return volume.unsqueeze(0)
    if volume.dim() == 4 and volume.shape[0] == 1:
        return volume
    if volume.dim() == 5 and volume.shape[:2] == (1, 1):
        return volume.squeeze(0)
    raise ValueError(
        f"unsupported shape {tuple(volume.shape)} in {source}; expected "
        "[X,Y,Z], [1,X,Y,Z], or [1,1,X,Y,Z]"
    )


def _as_batched_realizations(volume: torch.Tensor, source: object) -> torch.Tensor:
    """Normalize one or more saved volumes to [B, 1, X, Y, Z]."""
    if not isinstance(volume, torch.Tensor):
        raise TypeError(f"expected a tensor in {source}, got {type(volume).__name__}")
    if volume.dim() == 3:
        return volume.unsqueeze(0).unsqueeze(0)
    if volume.dim() == 4:
        return volume.unsqueeze(0) if volume.shape[0] == 1 else volume.unsqueeze(1)
    if volume.dim() == 5 and volume.shape[1] == 1:
        return volume
    raise ValueError(
        f"unsupported shape {tuple(volume.shape)} in {source}; expected "
        "[X,Y,Z], [1,X,Y,Z], [B,X,Y,Z], or [B,1,X,Y,Z]"
    )


def load_realizations(paths: Iterable[Path], device: str = "cpu") -> torch.Tensor:
    """Load saved realizations and return a [B, 1, X, Y, Z] tensor."""
    normalized = []
    expected_shape = None
    for path in paths:
        volume = torch.load(path, map_location=device)
        volume = _normalize_saved_realization(volume, path)
        if expected_shape is None:
            expected_shape = tuple(volume.shape)
        elif tuple(volume.shape) != expected_shape:
            raise ValueError(
                f"shape mismatch in {path}: got {tuple(volume.shape)}, "
                f"expected {expected_shape}"
            )
        normalized.append(volume)

    if not normalized:
        raise ValueError("at least one realization path is required")
    return torch.stack(normalized, dim=0)


def load_truth_model(path: Path, device: str = "cpu") -> torch.Tensor:
    """Load a truth tensor and return [1, 1, X, Y, Z]."""
    truth = torch.load(path, map_location=device)
    return _normalize_saved_realization(truth, path).unsqueeze(0)


def _metric_records(
    paths: Sequence[Path],
    geo_misfits: torch.Tensor,
    ranked_indices: torch.Tensor,
    accuracies: torch.Tensor,
    mean_ious: torch.Tensor,
    borehole_consistencies: Optional[torch.Tensor] = None,
    extra_metrics: Optional[Dict[str, torch.Tensor]] = None,
) -> List[MetricRecord]:
    geo_ranks = torch.empty_like(ranked_indices)
    geo_ranks[ranked_indices] = torch.arange(
        1,
        ranked_indices.numel() + 1,
        dtype=ranked_indices.dtype,
        device=ranked_indices.device,
    )

    records = []
    for index, path in enumerate(paths):
        numeric_id = _numeric_suffix(path)
        record = {
            "sample_id": numeric_id if numeric_id >= 0 else index,
            "path": str(path),
            "geo_misfit": float(geo_misfits[index].item()),
            "geo_rank": int(geo_ranks[index].item()),
            "voxel_accuracy": float(accuracies[index].item()),
            "mean_iou": float(mean_ious[index].item()),
        }
        if borehole_consistencies is not None:
            record["borehole_consistency"] = float(
                borehole_consistencies[index].item()
            )
        if extra_metrics:
            for key, values in extra_metrics.items():
                record[key] = float(values[index].item())
        records.append(record)
    return records


def _print_ranking(records: Sequence[MetricRecord], top_k: int) -> None:
    print("geo_rank,sample_id,geo_misfit,voxel_accuracy,mean_iou,path")
    for record in sorted(records, key=lambda row: int(row["geo_rank"]))[:top_k]:
        print(
            f"{record['geo_rank']},{record['sample_id']},"
            f"{record['geo_misfit']:.6g},{record['voxel_accuracy']:.6g},"
            f"{record['mean_iou']:.6g},{record['path']}"
        )


def _finite_stats(values: torch.Tensor) -> str:
    finite = values[torch.isfinite(values)]
    if finite.numel() == 0:
        return "mean=nan,std=nan,min=nan,max=nan"
    return (
        f"mean={finite.mean().item():.6g},"
        f"std={finite.std(unbiased=False).item():.6g},"
        f"min={finite.min().item():.6g},"
        f"max={finite.max().item():.6g}"
    )


def _print_group_summary(
    name: str,
    indices: torch.Tensor,
    geo_misfits: torch.Tensor,
    accuracies: torch.Tensor,
    mean_ious: torch.Tensor,
    borehole_consistencies: Optional[torch.Tensor] = None,
) -> None:
    print(f"{name} (n={indices.numel()})")
    print(f"  geo_misfit: {_finite_stats(geo_misfits[indices])}")
    print(f"  voxel_accuracy: {_finite_stats(accuracies[indices])}")
    print(f"  mean_iou: {_finite_stats(mean_ious[indices])}")
    if borehole_consistencies is not None:
        print(
            "  borehole_consistency: "
            f"{_finite_stats(borehole_consistencies[indices])}"
        )


def borehole_consistency(
    predictions: torch.Tensor,
    truth_model: torch.Tensor,
    boreholes: torch.Tensor,
) -> torch.Tensor:
    """Return per-sample agreement on conditioned and ignored truth voxels."""
    predicted_labels = _as_batched_realizations(predictions, "predictions").long()
    truth_labels = _as_batched_realizations(truth_model, "truth_model").long()
    borehole_labels = _as_batched_realizations(boreholes, "boreholes").long()

    if predicted_labels.shape[1:] != truth_labels.shape[1:]:
        raise ValueError("predictions and truth_model must have matching dimensions")
    if borehole_labels.shape[1:] != truth_labels.shape[1:]:
        raise ValueError("boreholes and truth_model must have matching dimensions")
    if truth_labels.shape[0] == 1 and predicted_labels.shape[0] > 1:
        truth_labels = truth_labels.expand(predicted_labels.shape[0], -1, -1, -1, -1)
        borehole_labels = borehole_labels.expand(
            predicted_labels.shape[0], -1, -1, -1, -1
        )
    elif truth_labels.shape[0] != predicted_labels.shape[0]:
        raise ValueError("truth_model batch size must be 1 or match predictions")

    mask = (borehole_labels != -1) | (truth_labels == -1)
    correct = (predicted_labels == truth_labels) & mask
    valid_count = mask.flatten(1).sum(dim=1)
    correct_count = correct.flatten(1).sum(dim=1)
    consistency = correct_count.float() / valid_count.clamp_min(1).float()
    return torch.where(
        valid_count > 0,
        consistency,
        torch.full_like(consistency, torch.nan),
    )


def pairwise_disagreement(
    realizations: torch.Tensor,
    ignore_label: Optional[int] = -1,
    max_pairs: Optional[int] = None,
) -> float:
    """
    Compute average pairwise voxel disagreement among generated samples.
    Return one scalar.
    """
    labels = _as_batched_realizations(realizations, "realizations").long()
    sample_count = labels.shape[0]
    if sample_count < 2:
        return float("nan")
    if max_pairs is not None and max_pairs <= 0:
        raise ValueError("max_pairs must be positive when provided")

    disagreements = []
    pair_count = 0
    for first in range(sample_count - 1):
        for second in range(first + 1, sample_count):
            first_sample = labels[first]
            second_sample = labels[second]
            valid = torch.ones_like(first_sample, dtype=torch.bool)
            if ignore_label is not None:
                valid &= first_sample != int(ignore_label)
                valid &= second_sample != int(ignore_label)
            valid_count = valid.sum()
            if valid_count.item() > 0:
                changed = (first_sample != second_sample) & valid
                disagreements.append(changed.float().sum() / valid_count.float())
            pair_count += 1
            if max_pairs is not None and pair_count >= max_pairs:
                break
        if max_pairs is not None and pair_count >= max_pairs:
            break

    if not disagreements:
        return float("nan")
    return float(torch.stack(disagreements).mean().item())


def print_summary(
    geo_misfits: torch.Tensor,
    ranked_indices: torch.Tensor,
    accuracies: torch.Tensor,
    mean_ious: torch.Tensor,
    best_fraction: float,
    worst_fraction: float,
    borehole_consistencies: Optional[torch.Tensor] = None,
    ensemble_pairwise_disagreement: Optional[float] = None,
) -> None:
    """Print all/best/worst summaries and rank correlations."""
    sample_count = geo_misfits.numel()
    best_count = max(1, math.ceil(sample_count * best_fraction))
    worst_count = max(1, math.ceil(sample_count * worst_fraction))
    all_indices = torch.arange(sample_count, device=geo_misfits.device)
    best_indices = ranked_indices[:best_count]
    worst_indices = ranked_indices[-worst_count:].flip(0)

    print("\nSummary statistics")
    _print_group_summary(
        "all_samples",
        all_indices,
        geo_misfits,
        accuracies,
        mean_ious,
        borehole_consistencies,
    )
    _print_group_summary(
        f"best_fraction={best_fraction:g}",
        best_indices,
        geo_misfits,
        accuracies,
        mean_ious,
        borehole_consistencies,
    )
    _print_group_summary(
        f"worst_fraction={worst_fraction:g}",
        worst_indices,
        geo_misfits,
        accuracies,
        mean_ious,
        borehole_consistencies,
    )
    if ensemble_pairwise_disagreement is not None:
        print(
            "ensemble_pairwise_disagreement: "
            f"{ensemble_pairwise_disagreement:.6g}"
        )

    accuracy_corr = spearman_correlation(geo_misfits, accuracies)
    iou_corr = spearman_correlation(geo_misfits, mean_ious)
    print("\nSpearman correlations")
    print(f"  geo_misfit vs voxel_accuracy: {accuracy_corr:.6g}")
    print(f"  geo_misfit vs mean_iou: {iou_corr:.6g}")


def save_metrics_csv(records: Sequence[MetricRecord], output_dir: Path) -> Path:
    """Write per-realization metrics to metrics.csv."""
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "metrics.csv"
    fieldnames = [
        "sample_id",
        "path",
        "geo_misfit",
        "geo_rank",
        "voxel_accuracy",
        "mean_iou",
    ]
    if any("borehole_consistency" in record for record in records):
        fieldnames.append("borehole_consistency")
    extra_fields = sorted(
        {
            key
            for record in records
            for key in record
            if key not in set(fieldnames)
        }
    )
    fieldnames.extend(extra_fields)
    with output_path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    return output_path


def _save_array_plot(array: torch.Tensor, title: str, path: Path, cmap: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(array.detach().cpu().T, origin="lower", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)


def save_plots(
    output_dir: Path,
    geo_misfits: torch.Tensor,
    accuracies: torch.Tensor,
    mean_ious: torch.Tensor,
    observed_gravity: torch.Tensor,
    predicted_gravity: torch.Tensor,
    truth_model: torch.Tensor,
    realizations: torch.Tensor,
    best_index: int,
    worst_index: int,
) -> None:
    """Save post-hoc scatter plots, proxy fields, and central geology slices."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    scatter_specs = (
        (
            accuracies,
            "Voxel accuracy",
            "geo_misfit_vs_voxel_accuracy.png",
        ),
        (mean_ious, "Mean IoU", "geo_misfit_vs_mean_iou.png"),
    )
    for y_values, y_label, filename in scatter_specs:
        figure, axis = plt.subplots(figsize=(6, 5))
        axis.scatter(
            geo_misfits.detach().cpu(),
            y_values.detach().cpu(),
            alpha=0.75,
        )
        axis.set_xlabel("Lightweight gravity-proxy misfit")
        axis.set_ylabel(y_label)
        axis.set_title(f"Proxy misfit vs {y_label}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(output_dir / filename, dpi=160)
        plt.close(figure)

    true_gravity = observed_gravity.reshape(-1, *observed_gravity.shape[-2:])[0]
    _save_array_plot(
        true_gravity,
        "Reference lightweight gravity proxy",
        output_dir / "true_gravity.png",
        "viridis",
    )
    _save_array_plot(
        predicted_gravity[best_index, 0],
        "Best realization lightweight gravity proxy",
        output_dir / "best_gravity.png",
        "viridis",
    )
    _save_array_plot(
        predicted_gravity[worst_index, 0],
        "Worst realization lightweight gravity proxy",
        output_dir / "worst_gravity.png",
        "viridis",
    )

    z_index = truth_model.shape[-1] // 2
    _save_array_plot(
        truth_model[0, 0, :, :, z_index],
        f"Truth geology slice z={z_index}",
        output_dir / "true_slice.png",
        "tab20",
    )
    _save_array_plot(
        realizations[best_index, 0, :, :, z_index],
        f"Best geology slice z={z_index}",
        output_dir / "best_slice.png",
        "tab20",
    )
    _save_array_plot(
        realizations[worst_index, 0, :, :, z_index],
        f"Worst geology slice z={z_index}",
        output_dir / "worst_slice.png",
        "tab20",
    )


def run_demo(
    device: str = "cpu",
    kernel_size: int = 7,
    ignore_label: Optional[int] = -1,
) -> None:
    """Run a deterministic post-hoc feasibility demo without a trained model."""
    true_model = torch.ones((1, 1, 16, 16, 12), dtype=torch.long, device=device)
    true_model[:, :, 5:11, 5:11, 2:7] = 10

    exact = true_model[0]
    shifted = torch.ones_like(exact)
    shifted[:, 7:13, 5:11, 2:7] = 10
    missing = torch.ones_like(exact)
    shallow = torch.ones_like(exact)
    shallow[:, 5:11, 5:11, 5:10] = 10

    realizations = torch.stack([exact, shifted, missing, shallow], dim=0)
    property_map = LithologyPropertyMap()
    forward_model = SimpleGravityForward(kernel_size=kernel_size)
    observed_gravity = forward_model(property_map(true_model))
    ranking = rank_realizations_by_geophysics(
        realizations,
        observed_gravity,
        property_map=property_map,
        forward_model=forward_model,
    )
    accuracies = voxel_accuracy(
        realizations, true_model, ignore_label=ignore_label
    )
    mean_ious = mean_iou(realizations, true_model, ignore_label=ignore_label)

    labels = ["exact", "shifted", "missing", "shallow"]
    print("rank,index,geo_misfit,voxel_accuracy,mean_iou,label")
    for rank, (index, misfit) in enumerate(
        zip(ranking.ranked_indices, ranking.ranked_misfits),
        start=1,
    ):
        sample_index = int(index.item())
        print(
            f"{rank},{sample_index},{misfit.item():.6g},"
            f"{accuracies[sample_index].item():.6g},"
            f"{mean_ious[sample_index].item():.6g},{labels[sample_index]}"
        )


def evaluate_directory(
    samples_dir: Path,
    sample_prefixes: Optional[Sequence[str]] = None,
    truth_model_path: Optional[Path] = None,
    boreholes_path: Optional[Path] = None,
    observed_gravity_path: Optional[Path] = None,
    observed_magnetic_path: Optional[Path] = None,
    observed_gravity_gradient_path: Optional[Path] = None,
    density_config_path: Optional[Path] = None,
    susceptibility_config_path: Optional[Path] = None,
    gravity_weight: float = 1.0,
    magnetic_weight: float = 0.0,
    gravity_gradient_weight: float = 0.0,
    top_k: int = 10,
    device: str = "cpu",
    kernel_size: int = 9,
    output_dir: Optional[Path] = None,
    save_csv: bool = False,
    save_plot_files: bool = False,
    best_fraction: float = 0.1,
    worst_fraction: float = 0.1,
    ignore_label: Optional[int] = -1,
) -> List[MetricRecord]:
    """Evaluate and rank a directory of decoded geology realizations."""
    for name, fraction in (
        ("best_fraction", best_fraction),
        ("worst_fraction", worst_fraction),
    ):
        if not 0.0 < fraction <= 1.0:
            raise ValueError(f"{name} must be in the interval (0, 1]")
    for name, weight in (
        ("gravity_weight", gravity_weight),
        ("magnetic_weight", magnetic_weight),
        ("gravity_gradient_weight", gravity_gradient_weight),
    ):
        if float(weight) < 0.0:
            raise ValueError(f"{name} must be non-negative")

    realization_paths = find_realization_files(samples_dir, sample_prefixes)
    realizations = load_realizations(realization_paths, device=device)
    truth_path = truth_model_path or samples_dir / "true_model.pt"
    if not truth_path.exists():
        raise FileNotFoundError(f"truth model not found at {truth_path}")
    truth_model = load_truth_model(truth_path, device=device)
    if truth_model.shape[2:] != realizations.shape[2:]:
        raise ValueError(
            "truth model and realizations must have matching spatial dimensions"
        )
    resolved_boreholes_path = boreholes_path
    if resolved_boreholes_path is None:
        candidate = samples_dir / "boreholes.pt"
        if candidate.exists():
            resolved_boreholes_path = candidate
    borehole_consistencies = None
    if resolved_boreholes_path is not None:
        if not resolved_boreholes_path.exists():
            raise FileNotFoundError(f"boreholes not found at {resolved_boreholes_path}")
        boreholes = load_truth_model(resolved_boreholes_path, device=device)
        if boreholes.shape[2:] != truth_model.shape[2:]:
            raise ValueError(
                "boreholes and truth model must have matching spatial dimensions"
            )
        borehole_consistencies = borehole_consistency(
            realizations,
            truth_model,
            boreholes,
        )

    density_config = load_density_config(density_config_path) if density_config_path and load_density_config else None
    susceptibility_config = (
        load_susceptibility_config(susceptibility_config_path)
        if susceptibility_config_path and load_susceptibility_config
        else None
    )
    property_map = (
        property_map_from_density_config(density_config)
        if density_config is not None and property_map_from_density_config
        else LithologyPropertyMap()
    )
    susceptibility_map = (
        property_map_from_susceptibility_config(susceptibility_config)
        if susceptibility_config is not None and property_map_from_susceptibility_config
        else LithologyPropertyMap(
            properties={
                label: value * 0.01
                for label, value in LithologyPropertyMap.DEFAULT_DENSITY_CONTRASTS.items()
            },
            default_value=0.0,
        )
    )
    forward_model = SimpleGravityForward(kernel_size=kernel_size)
    if observed_gravity_path is not None:
        observed_gravity = torch.load(observed_gravity_path, map_location=device)
    else:
        observed_gravity = forward_model(property_map(truth_model))

    ranking = rank_realizations_by_geophysics(
        realizations,
        observed_gravity,
        property_map=property_map,
        forward_model=forward_model,
    )
    extra_metrics: Dict[str, torch.Tensor] = {}
    weighted_terms = [float(gravity_weight) * ranking.all_misfits]

    if (
        observed_gravity_gradient_path is not None
        or float(gravity_gradient_weight) > 0.0
    ):
        gradient_forward = GravityGradientForward(kernel_size=kernel_size)
        observed_gravity_gradient = (
            torch.load(observed_gravity_gradient_path, map_location=device)
            if observed_gravity_gradient_path is not None
            else gradient_forward(property_map(truth_model))
        )
        predicted_gravity_gradient = gradient_forward(property_map(realizations))
        gravity_gradient_misfit = normalized_misfit(
            predicted_gravity_gradient,
            observed_gravity_gradient,
            reduction="none",
        )
        extra_metrics["gravity_gradient_proxy_misfit"] = gravity_gradient_misfit
        weighted_terms.append(float(gravity_gradient_weight) * gravity_gradient_misfit)

    if observed_magnetic_path is not None or susceptibility_config_path is not None or float(magnetic_weight) > 0.0:
        magnetic_forward = MagneticTMIForward(kernel_size=kernel_size)
        observed_magnetic = (
            torch.load(observed_magnetic_path, map_location=device)
            if observed_magnetic_path is not None
            else magnetic_forward(susceptibility_map(truth_model))
        )
        predicted_magnetic = magnetic_forward(susceptibility_map(realizations))
        magnetic_misfit = normalized_misfit(
            predicted_magnetic,
            observed_magnetic,
            reduction="none",
        )
        extra_metrics["magnetic_proxy_misfit"] = magnetic_misfit
        weighted_terms.append(float(magnetic_weight) * magnetic_misfit)

    if len(weighted_terms) > 1 or float(gravity_weight) != 1.0:
        extra_metrics["joint_proxy_misfit"] = torch.stack(weighted_terms).sum(dim=0)

    accuracies = voxel_accuracy(
        realizations,
        truth_model,
        ignore_label=ignore_label,
    )
    mean_ious = mean_iou(
        realizations,
        truth_model,
        ignore_label=ignore_label,
    )
    records = _metric_records(
        realization_paths,
        ranking.all_misfits,
        ranking.ranked_indices,
        accuracies,
        mean_ious,
        borehole_consistencies,
        extra_metrics=extra_metrics,
    )
    diversity = pairwise_disagreement(realizations, ignore_label=ignore_label)

    _print_ranking(records, min(top_k, len(records)))
    print_summary(
        ranking.all_misfits,
        ranking.ranked_indices,
        accuracies,
        mean_ious,
        best_fraction,
        worst_fraction,
        borehole_consistencies=borehole_consistencies,
        ensemble_pairwise_disagreement=diversity,
    )

    resolved_output_dir = output_dir or samples_dir / "geophysics_evaluation"
    if save_csv or output_dir is not None:
        csv_path = save_metrics_csv(records, resolved_output_dir)
        print(f"\nSaved metrics: {csv_path}")
    if save_plot_files:
        save_plots(
            resolved_output_dir,
            ranking.all_misfits,
            accuracies,
            mean_ious,
            observed_gravity,
            ranking.predicted_gravity,
            truth_model,
            realizations,
            best_index=ranking.best_index,
            worst_index=int(ranking.ranked_indices[-1].item()),
        )
        print(f"Saved plots: {resolved_output_dir}")

    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate decoded geology ensembles using a lightweight gravity "
            "proxy and truth-based geology metrics."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run a tiny synthetic demo instead of reading a samples directory.",
    )
    parser.add_argument(
        "--samples-dir",
        type=Path,
        default=None,
        help="Directory containing sample_*.pt, sol_*.pt, or run_*.pt files.",
    )
    parser.add_argument(
        "--sample-prefix",
        action="append",
        default=None,
        help="Sample prefix to load. May be repeated.",
    )
    parser.add_argument(
        "--truth-model",
        type=Path,
        default=None,
        help="Truth categorical model used for geology metrics.",
    )
    parser.add_argument(
        "--boreholes",
        type=Path,
        default=None,
        help=(
            "Optional borehole tensor for consistency metrics. If omitted, "
            "samples_dir/boreholes.pt is used when it exists."
        ),
    )
    parser.add_argument(
        "--observed-gravity",
        type=Path,
        default=None,
        help=(
            "Precomputed observed proxy tensor. If omitted, true_model.pt is "
            "mapped through the lightweight forward proxy."
        ),
    )
    parser.add_argument(
        "--observed-magnetic",
        type=Path,
        default=None,
        help=(
            "Optional precomputed observed lightweight magnetic-proxy tensor. "
            "If omitted but magnetic evaluation is requested, it is generated "
            "from truth_model and susceptibility_config."
        ),
    )
    parser.add_argument(
        "--observed-gravity-gradient",
        type=Path,
        default=None,
        help=(
            "Optional precomputed observed lightweight gravity-gradient-proxy tensor. "
            "If omitted but gravity-gradient evaluation is requested, it is "
            "generated from truth_model and density_config."
        ),
    )
    parser.add_argument(
        "--density-config",
        type=Path,
        default=None,
        help=(
            "Optional controlled density_config.json for lightweight gravity "
            "proxy evaluation when --observed-gravity is omitted and for predicted fields."
        ),
    )
    parser.add_argument(
        "--susceptibility-config",
        type=Path,
        default=None,
        help=(
            "Optional controlled susceptibility_config.json for lightweight "
            "magnetic-proxy evaluation."
        ),
    )
    parser.add_argument(
        "--gravity-weight",
        type=float,
        default=1.0,
        help="Weight used only for optional joint_proxy_misfit output.",
    )
    parser.add_argument(
        "--magnetic-weight",
        type=float,
        default=0.0,
        help=(
            "Weight used for optional joint_proxy_misfit output. A positive "
            "value also requests magnetic-proxy evaluation."
        ),
    )
    parser.add_argument(
        "--gravity-gradient-weight",
        type=float,
        default=0.0,
        help=(
            "Weight used for optional joint_proxy_misfit output. A positive "
            "value also requests gravity-gradient-proxy evaluation."
        ),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Output directory. Providing it saves metrics.csv; otherwise "
            "--save-csv/--save-plots use samples_dir/geophysics_evaluation."
        ),
    )
    parser.add_argument(
        "--save-csv",
        action="store_true",
        help="Save metrics.csv.",
    )
    parser.add_argument(
        "--save-plots",
        action="store_true",
        help="Save scatter, proxy-field, and geology-slice plots.",
    )
    parser.add_argument(
        "--best-fraction",
        type=float,
        default=0.1,
        help="Fraction of lowest-misfit samples in the best summary.",
    )
    parser.add_argument(
        "--worst-fraction",
        type=float,
        default=0.1,
        help="Fraction of highest-misfit samples in the worst summary.",
    )
    parser.add_argument(
        "--ignore-label",
        type=int,
        default=-1,
        help="Target label excluded from voxel accuracy and IoU.",
    )
    parser.add_argument(
        "--no-ignore-label",
        action="store_const",
        const=None,
        dest="ignore_label",
        help="Include every target label in geology metrics.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of ranked realizations to print.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Torch device for evaluation.",
    )
    parser.add_argument(
        "--kernel-size",
        type=int,
        default=9,
        help="Odd 2D kernel size for the lightweight gravity proxy.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.demo:
        run_demo(
            device=args.device,
            kernel_size=args.kernel_size,
            ignore_label=args.ignore_label,
        )
        return
    if args.samples_dir is None:
        raise SystemExit("provide --samples-dir or use --demo")

    evaluate_directory(
        samples_dir=args.samples_dir,
        sample_prefixes=args.sample_prefix,
        truth_model_path=args.truth_model,
        boreholes_path=args.boreholes,
        observed_gravity_path=args.observed_gravity,
        observed_magnetic_path=args.observed_magnetic,
        observed_gravity_gradient_path=args.observed_gravity_gradient,
        density_config_path=args.density_config,
        susceptibility_config_path=args.susceptibility_config,
        gravity_weight=args.gravity_weight,
        magnetic_weight=args.magnetic_weight,
        gravity_gradient_weight=args.gravity_gradient_weight,
        top_k=args.top_k,
        device=args.device,
        kernel_size=args.kernel_size,
        output_dir=args.output_dir,
        save_csv=args.save_csv,
        save_plot_files=args.save_plots,
        best_fraction=args.best_fraction,
        worst_fraction=args.worst_fraction,
        ignore_label=args.ignore_label,
    )


if __name__ == "__main__":
    main()
