"""Select a dike-like target feature for gravity-proxy guidance figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Optional

import torch

from geophysics import LithologyPropertyMap, SimpleGravityForward, normalized_misfit
from geology_io_utils import (
    as_batched_volume,
    connected_components_3d,
    density_value,
    dominant_label,
    find_sample_files,
    load_sample_stack,
    load_volume,
    target_probability,
    write_csv_rows,
    write_json,
)


CANDIDATE_FIELDS = [
    "target_label",
    "component_id",
    "voxel_count",
    "volume_fraction",
    "bbox_min",
    "bbox_max",
    "centroid",
    "borehole_target_hits",
    "replacement_label",
    "target_density",
    "replacement_density",
    "density_contrast",
    "lightweight_gravity_proxy_observability",
    "baseline_probability_inside_component",
    "guided_probability_inside_component",
    "guided_minus_baseline_probability_inside_component",
    "ensemble_probability_inside_component",
    "dike_like_score",
    "bbox_extent_x",
    "bbox_extent_y",
    "bbox_extent_z",
    "bbox_elongation",
    "covariance_thinness",
    "covariance_planarity",
    "shape_score",
]


def _component_mask(shape: torch.Size, coords: torch.Tensor, device: torch.device) -> torch.Tensor:
    mask = torch.zeros(shape, dtype=torch.bool, device=device)
    if coords.numel() > 0:
        mask[coords[:, 0], coords[:, 1], coords[:, 2]] = True
    return mask


def _shape_scores(coords: torch.Tensor) -> Dict[str, float]:
    coords_f = coords.float()
    bbox_extent = coords_f.max(dim=0).values - coords_f.min(dim=0).values + 1
    sorted_extent = torch.sort(bbox_extent).values
    if coords_f.shape[0] < 3:
        eigvals = torch.ones(3)
    else:
        centered = coords_f - coords_f.mean(dim=0, keepdim=True)
        covariance = centered.T @ centered / max(coords_f.shape[0] - 1, 1)
        eigvals = torch.linalg.eigvalsh(covariance).clamp_min(1e-6)
    eigvals = torch.sort(eigvals).values
    thinness = float(torch.sqrt(eigvals[-1] / eigvals[0]).item())
    planarity = float(torch.sqrt(eigvals[1] / eigvals[0]).item())
    elongation = float((sorted_extent[-1] / sorted_extent[0].clamp_min(1)).item())
    return {
        "bbox_extent_x": float(bbox_extent[0].item()),
        "bbox_extent_y": float(bbox_extent[1].item()),
        "bbox_extent_z": float(bbox_extent[2].item()),
        "bbox_elongation": elongation,
        "covariance_thinness": thinness,
        "covariance_planarity": planarity,
        "shape_score": math.log1p(elongation) + math.log1p(thinness) + 0.5 * math.log1p(planarity),
    }


def candidate_records(
    truth_model: torch.Tensor,
    boreholes: Optional[torch.Tensor] = None,
    samples: Optional[torch.Tensor] = None,
    baseline_samples: Optional[torch.Tensor] = None,
    guided_samples: Optional[torch.Tensor] = None,
    min_voxels: int = 32,
    max_volume_fraction: float = 0.35,
    ignore_labels: tuple[int, ...] = (-1,),
    kernel_size: int = 9,
) -> List[Dict[str, object]]:
    truth = as_batched_volume(truth_model, "truth_model").long()
    if truth.shape[0] != 1:
        raise ValueError("truth_model must contain one volume")
    truth_3d = truth[0, 0]
    total_voxels = int(truth_3d.numel())
    labels = sorted(int(value.item()) for value in torch.unique(truth_3d))
    excluded = set(int(value) for value in ignore_labels)
    boreholes_3d = None
    if boreholes is not None:
        boreholes_3d = as_batched_volume(boreholes, "boreholes").long()[0, 0]
        if boreholes_3d.shape != truth_3d.shape:
            raise ValueError("boreholes and truth_model must have matching spatial shape")
    for name, ensemble in (("samples", samples), ("baseline_samples", baseline_samples), ("guided_samples", guided_samples)):
        if ensemble is not None and ensemble.shape[-3:] != truth_3d.shape:
            raise ValueError(f"{name} and truth_model must have matching spatial shape")

    property_map = LithologyPropertyMap()
    forward_model = SimpleGravityForward(kernel_size=kernel_size)
    observed = forward_model(property_map(truth))
    records: List[Dict[str, object]] = []
    for label in labels:
        if label in excluded:
            continue
        replacement_label = dominant_label(truth, exclude=tuple(excluded | {label}))
        label_components = connected_components_3d((truth_3d == label).cpu())
        label_probability = target_probability(samples, label)[0, 0] if samples is not None else None
        baseline_probability = target_probability(baseline_samples, label)[0, 0] if baseline_samples is not None else None
        guided_probability = target_probability(guided_samples, label)[0, 0] if guided_samples is not None else None
        target_density = density_value(label, property_map.properties)
        replacement_density = density_value(replacement_label, property_map.properties)
        density_contrast = abs(target_density - replacement_density)
        for component in label_components:
            voxel_count = int(component["voxel_count"])
            volume_fraction = voxel_count / total_voxels
            if voxel_count < min_voxels or volume_fraction > max_volume_fraction:
                continue
            coords = component["coords"]
            comp_mask = _component_mask(truth_3d.shape, coords, truth_3d.device)
            altered = truth.clone()
            altered[0, 0][comp_mask] = int(replacement_label)
            altered_gravity = forward_model(property_map(altered))
            observability = normalized_misfit(altered_gravity, observed, reduction="mean")
            borehole_hits = 0
            if boreholes_3d is not None:
                borehole_hits = int(((boreholes_3d == label) & comp_mask).sum().item())
            probability_inside = None
            if label_probability is not None:
                probability_inside = float(
                    label_probability[comp_mask.to(label_probability.device)].mean().item()
                )
            baseline_probability_inside = None
            guided_probability_inside = None
            probability_improvement = None
            if baseline_probability is not None:
                baseline_probability_inside = float(
                    baseline_probability[comp_mask.to(baseline_probability.device)].mean().item()
                )
            if guided_probability is not None:
                guided_probability_inside = float(
                    guided_probability[comp_mask.to(guided_probability.device)].mean().item()
                )
            if baseline_probability_inside is not None and guided_probability_inside is not None:
                probability_improvement = guided_probability_inside - baseline_probability_inside
            shape = _shape_scores(coords)
            sparse_penalty = borehole_hits / max(voxel_count, 1)
            score = (
                shape["shape_score"]
                + 5.0 * float(observability.item())
                + 0.1 * math.log1p(voxel_count)
                - 10.0 * sparse_penalty
            )
            records.append(
                {
                    "target_label": label,
                    "component_id": int(component["component_id"]),
                    "voxel_count": voxel_count,
                    "volume_fraction": volume_fraction,
                    "bbox_min": component["bbox_min"],
                    "bbox_max": component["bbox_max"],
                    "centroid": component["centroid"],
                    "borehole_target_hits": borehole_hits,
                    "replacement_label": int(replacement_label),
                    "target_density": target_density,
                    "replacement_density": replacement_density,
                    "density_contrast": density_contrast,
                    "lightweight_gravity_proxy_observability": float(observability.item()),
                    "baseline_probability_inside_component": baseline_probability_inside,
                    "guided_probability_inside_component": guided_probability_inside,
                    "guided_minus_baseline_probability_inside_component": probability_improvement,
                    "ensemble_probability_inside_component": probability_inside,
                    "dike_like_score": score,
                    **shape,
                }
            )
    return sorted(records, key=lambda row: float(row["dike_like_score"]), reverse=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select a dike-like target label/component for demo figures.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--samples-dir", type=Path, default=None)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--guided-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-voxels", type=int, default=32)
    parser.add_argument("--max-volume-fraction", type=float, default=0.35)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.min_voxels <= 0:
        raise SystemExit("--min-voxels must be positive")
    if not 0.0 < args.max_volume_fraction <= 1.0:
        raise SystemExit("--max-volume-fraction must be in (0, 1]")
    truth = load_volume(args.truth_model, device=args.device, single=True)
    boreholes = load_volume(args.boreholes, device=args.device, single=True) if args.boreholes else None
    samples = None
    if args.samples_dir is not None:
        sample_paths = find_sample_files(args.samples_dir)
        samples, _ = load_sample_stack(sample_paths, device=args.device)
    baseline_samples = None
    guided_samples = None
    if args.baseline_dir is not None:
        baseline_samples, _ = load_sample_stack(find_sample_files(args.baseline_dir), device=args.device)
    if args.guided_dir is not None:
        guided_samples, _ = load_sample_stack(find_sample_files(args.guided_dir), device=args.device)
    candidates = candidate_records(
        truth_model=truth,
        boreholes=boreholes,
        samples=samples,
        baseline_samples=baseline_samples,
        guided_samples=guided_samples,
        min_voxels=args.min_voxels,
        max_volume_fraction=args.max_volume_fraction,
        kernel_size=args.kernel_size,
    )
    if not candidates:
        raise SystemExit("no dike-like target candidates found with the current filters")
    manifest = {
        "selected": candidates[0],
        "candidates": candidates,
        "truth_model": str(args.truth_model),
        "boreholes": str(args.boreholes) if args.boreholes else None,
        "samples_dir": str(args.samples_dir) if args.samples_dir else None,
        "description": (
            "The selected target is a dike-like categorical component chosen by "
            "geometry, sparse conditioning, and lightweight gravity-proxy "
            "observability. The categorical id is not assumed to be a semantic "
            "dike label unless independently verified."
        ),
    }
    write_csv_rows(args.output_dir / "candidates.csv", candidates, CANDIDATE_FIELDS)
    write_json(args.output_dir / "manifest.json", manifest)
    print(f"Selected target_label={candidates[0]['target_label']} component_id={candidates[0]['component_id']}")
    print(f"Saved manifest: {args.output_dir / 'manifest.json'}")


if __name__ == "__main__":
    main()
