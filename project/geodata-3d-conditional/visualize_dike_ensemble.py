"""Create headless baseline-vs-guided target-label ensemble figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

import torch

from geology_io_utils import (
    find_sample_files,
    infer_paired_by_seed,
    label_mask,
    load_sample_stack,
    load_volume,
    target_probability,
    write_json,
)


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _mip_xy(volume: torch.Tensor) -> torch.Tensor:
    return volume.detach().cpu().float().amax(dim=-1)


def _surface_mask(mask: torch.Tensor) -> torch.Tensor:
    values = mask.detach().cpu().bool()
    if values.sum().item() == 0:
        return values
    padded = torch.nn.functional.pad(values.unsqueeze(0).unsqueeze(0), (1, 1, 1, 1, 1, 1), value=False)[0, 0]
    interior = (
        padded[1:-1, 1:-1, 1:-1]
        & padded[:-2, 1:-1, 1:-1]
        & padded[2:, 1:-1, 1:-1]
        & padded[1:-1, :-2, 1:-1]
        & padded[1:-1, 2:, 1:-1]
        & padded[1:-1, 1:-1, :-2]
        & padded[1:-1, 1:-1, 2:]
    )
    return values & ~interior


def _choose_backend() -> str:
    try:
        import pyvista  # noqa: F401

        return "pyvista_offscreen"
    except Exception:
        try:
            import plotly  # noqa: F401

            return "plotly_html_available_matplotlib_png"
        except Exception:
            return "matplotlib_voxel"


def _set_3d_axes(axis, shape: torch.Size) -> None:
    axis.set_xlim(0, shape[0])
    axis.set_ylim(0, shape[1])
    axis.set_zlim(0, shape[2])
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.view_init(elev=24, azim=-55)


def _draw_voxels(axis, mask: torch.Tensor, color: str, alpha: float = 0.65) -> None:
    surface = _surface_mask(mask)
    if surface.any():
        axis.voxels(surface.numpy(), facecolors=color, edgecolor="none", alpha=alpha)


def _save_voxel_grid(
    masks: Sequence[tuple[torch.Tensor, str, str, float]],
    path: Path,
    title: str,
    ncols: int = 3,
) -> None:
    plt = _setup_matplotlib()
    count = max(1, len(masks))
    ncols = max(1, min(ncols, count))
    nrows = (count + ncols - 1) // ncols
    figure = plt.figure(figsize=(4.2 * ncols, 4.0 * nrows))
    reference_shape = masks[0][0].shape if masks else torch.Size((1, 1, 1))
    for index, (mask, subtitle, color, alpha) in enumerate(masks, start=1):
        axis = figure.add_subplot(nrows, ncols, index, projection="3d")
        _draw_voxels(axis, mask, color, alpha)
        _set_3d_axes(axis, mask.shape)
        axis.set_title(subtitle)
    for index in range(len(masks) + 1, nrows * ncols + 1):
        axis = figure.add_subplot(nrows, ncols, index, projection="3d")
        _set_3d_axes(axis, reference_shape)
        axis.set_axis_off()
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _record_sample_mask(
    ensemble: torch.Tensor,
    index_map: Mapping[int, int],
    sample_id: int,
    target_label: int,
) -> torch.Tensor:
    index = index_map[sample_id]
    return label_mask(ensemble[index : index + 1], target_label)[0, 0]


def _record_index(records: Sequence[Mapping[str, object]]) -> Dict[int, int]:
    return {int(record["sample_id"]): int(record["stack_index"]) for record in records}


def _select_sample_ids(
    baseline_records: Sequence[Mapping[str, object]],
    guided_records: Sequence[Mapping[str, object]],
    requested: Sequence[int],
    max_samples: int,
) -> List[int]:
    baseline_ids = set(_record_index(baseline_records))
    guided_ids = set(_record_index(guided_records))
    common = sorted(baseline_ids & guided_ids)
    if not common:
        raise ValueError("baseline and guided ensembles have no common sample ids")
    if requested:
        missing = [sample_id for sample_id in requested if sample_id not in common]
        if missing:
            raise ValueError(f"requested sample ids not present in both ensembles: {missing}")
        return list(requested)
    return common[:max_samples]


def save_ensemble_figures(
    baseline: torch.Tensor,
    baseline_records: Sequence[Mapping[str, object]],
    guided: torch.Tensor,
    guided_records: Sequence[Mapping[str, object]],
    truth_model: torch.Tensor,
    target_label: int,
    output_dir: Path,
    boreholes: torch.Tensor | None = None,
    sample_ids: Sequence[int] = (),
    max_samples: int = 6,
    thresholds: Sequence[float] = (0.05, 0.33, 0.62, 0.90),
    paired_by_seed: bool = False,
) -> Dict[str, object]:
    plt = _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    truth_mask = label_mask(truth_model, target_label)[0, 0]
    baseline_prob = target_probability(baseline, target_label)[0, 0]
    guided_prob = target_probability(guided, target_label)[0, 0]
    borehole_mask = None
    if boreholes is not None:
        borehole_mask = label_mask(boreholes, target_label)[0, 0]

    figure, axes = plt.subplots(1, 4 if borehole_mask is not None else 3, figsize=(14, 4))
    axes = list(axes)
    panels = [
        (_mip_xy(truth_mask), "Truth target label"),
        (_mip_xy(baseline_prob), "Baseline target-label probability"),
        (_mip_xy(guided_prob), "Guided target-label probability"),
    ]
    if borehole_mask is not None:
        panels.insert(1, (_mip_xy(borehole_mask), "Sparse target-label conditioning"))
    for axis, (array, title) in zip(axes, panels):
        image = axis.imshow(array.T, origin="lower", cmap="magma", vmin=0, vmax=1)
        axis.set_title(title)
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        figure.colorbar(image, ax=axis, shrink=0.75)
    figure.suptitle("Baseline vs guided dike-like target probability (lightweight gravity-proxy guidance)")
    figure.tight_layout()
    probability_path = output_dir / "ensemble_probability.png"
    figure.savefig(probability_path, dpi=180)
    plt.close(figure)

    chosen_ids = _select_sample_ids(baseline_records, guided_records, sample_ids, max_samples)
    baseline_index = _record_index(baseline_records)
    guided_index = _record_index(guided_records)
    figure, axes = plt.subplots(2, max(1, len(chosen_ids)), figsize=(3.0 * max(1, len(chosen_ids)), 6))
    if len(chosen_ids) == 1:
        axes = [[axes[0]], [axes[1]]]
    for column, sample_id in enumerate(chosen_ids):
        for row, (ensemble, index_map, label) in enumerate(
            ((baseline, baseline_index, "Baseline"), (guided, guided_index, "Guided"))
        ):
            mask = label_mask(ensemble[index_map[sample_id] : index_map[sample_id] + 1], target_label)[0, 0]
            axes[row][column].imshow(_mip_xy(mask).T, origin="lower", cmap="gray", vmin=0, vmax=1)
            axes[row][column].set_title(f"{label} sample {sample_id}")
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
    figure.suptitle("Target-label realization MIPs")
    figure.tight_layout()
    samples_path = output_dir / "target_realization_panels.png"
    figure.savefig(samples_path, dpi=180)
    plt.close(figure)

    figure, axes = plt.subplots(2, len(thresholds), figsize=(3.2 * len(thresholds), 6))
    for column, threshold in enumerate(thresholds):
        for row, (probability, label) in enumerate(((baseline_prob, "Baseline"), (guided_prob, "Guided"))):
            array = _mip_xy(probability >= float(threshold))
            axes[row][column].imshow(array.T, origin="lower", cmap="viridis", vmin=0, vmax=1)
            axes[row][column].contour(_mip_xy(truth_mask).T, levels=[0.5], colors="white", linewidths=0.8)
            axes[row][column].set_title(f"{label} p>={threshold:g}")
            axes[row][column].set_xticks([])
            axes[row][column].set_yticks([])
    figure.suptitle("Target-label probability thresholds with truth outline")
    figure.tight_layout()
    threshold_path = output_dir / "probability_thresholds.png"
    figure.savefig(threshold_path, dpi=180)
    plt.close(figure)
    backend = _choose_backend()
    actual_backend = "matplotlib_voxel"

    truth_body_path = output_dir / "truth_target_body_3d.png"
    truth_masks = [(truth_mask, "Truth target body", "gold", 0.70)]
    if borehole_mask is not None:
        truth_masks.append((borehole_mask, "Sparse observed target evidence", "red", 0.90))
    _save_voxel_grid(truth_masks, truth_body_path, "Truth target and sparse evidence", ncols=2)

    baseline_masks = [
        (_record_sample_mask(baseline, baseline_index, sample_id, target_label), f"baseline {sample_id}", "tab:blue", 0.55)
        for sample_id in chosen_ids
    ]
    guided_masks = [
        (_record_sample_mask(guided, guided_index, sample_id, target_label), f"guided {sample_id}", "tab:green", 0.55)
        for sample_id in chosen_ids
    ]
    baseline_fig8 = output_dir / "figure8_like_baseline_realizations.png"
    guided_fig8 = output_dir / "figure8_like_guided_realizations.png"
    combined_fig8 = output_dir / "figure8_like_baseline_vs_guided.png"
    _save_voxel_grid(baseline_masks, baseline_fig8, "Figure 8-like baseline target realizations", ncols=3)
    _save_voxel_grid(guided_masks, guided_fig8, "Figure 8-like guided target realizations", ncols=3)
    _save_voxel_grid(
        baseline_masks[:3] + guided_masks[:3],
        combined_fig8,
        "Figure 8-like baseline vs guided target realizations",
        ncols=3,
    )

    threshold_colors = ("#fee08b", "#fdae61", "#f46d43", "#a50026")
    baseline_probability_masks = [
        (baseline_prob >= float(threshold), f"baseline p>={threshold:g}", threshold_colors[index], 0.35 + 0.1 * index)
        for index, threshold in enumerate(thresholds)
    ]
    guided_probability_masks = [
        (guided_prob >= float(threshold), f"guided p>={threshold:g}", threshold_colors[index], 0.35 + 0.1 * index)
        for index, threshold in enumerate(thresholds)
    ]
    baseline_fig9 = output_dir / "figure9_like_baseline_probability_isosurfaces.png"
    guided_fig9 = output_dir / "figure9_like_guided_probability_isosurfaces.png"
    combined_fig9 = output_dir / "figure9_like_baseline_vs_guided_probability.png"
    _save_voxel_grid(baseline_probability_masks, baseline_fig9, "Figure 9-like baseline probability thresholds", ncols=4)
    _save_voxel_grid(guided_probability_masks, guided_fig9, "Figure 9-like guided probability thresholds", ncols=4)
    _save_voxel_grid(
        baseline_probability_masks + guided_probability_masks,
        combined_fig9,
        "Figure 9-like baseline vs guided probability thresholds",
        ncols=4,
    )

    changed_paths: Dict[str, str] = {}
    if paired_by_seed:
        for sample_id in chosen_ids[:max_samples]:
            baseline_index_value = baseline_index[sample_id]
            guided_index_value = guided_index[sample_id]
            baseline_sample = baseline[baseline_index_value, 0].long()
            guided_sample = guided[guided_index_value, 0].long()
            changed = baseline_sample != guided_sample
            target_changed = changed & ((baseline_sample == int(target_label)) | (guided_sample == int(target_label)))
            non_target_changed = changed & ~target_changed
            changed_path = output_dir / f"changed_voxels_3d_sample_{sample_id}.png"
            _save_voxel_grid(
                [
                    (target_changed, "target-label changed voxels", "red", 0.70),
                    (non_target_changed, "non-target changed voxels", "tab:gray", 0.35),
                ],
                changed_path,
                f"Paired changed voxels sample {sample_id}",
                ncols=2,
            )
            changed_paths[f"changed_voxels_3d_sample_{sample_id}"] = str(changed_path)

    return {
        "target_label": int(target_label),
        "n_baseline_samples": int(baseline.shape[0]),
        "n_guided_samples": int(guided.shape[0]),
        "sample_ids": chosen_ids,
        "figures": {
            "ensemble_probability": str(probability_path),
            "target_realization_panels": str(samples_path),
            "probability_thresholds": str(threshold_path),
            "truth_target_body_3d": str(truth_body_path),
            "figure8_like_baseline_realizations": str(baseline_fig8),
            "figure8_like_guided_realizations": str(guided_fig8),
            "figure8_like_baseline_vs_guided": str(combined_fig8),
            "figure9_like_baseline_probability_isosurfaces": str(baseline_fig9),
            "figure9_like_guided_probability_isosurfaces": str(guided_fig9),
            "figure9_like_baseline_vs_guided_probability": str(combined_fig9),
            **changed_paths,
        },
        "visualization_backend": actual_backend,
        "visualization_backend_preferred": backend,
        "description": (
            "Headless paper-style target-label ensemble visualization. Figures "
            "show categorical dike-like target probability under baseline and "
            "inference-time lightweight gravity-proxy guidance."
        ),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize baseline vs guided target-label ensembles.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--guided-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sample-id", type=int, action="append", default=[])
    parser.add_argument("--max-samples", type=int, default=6)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_samples <= 0:
        raise SystemExit("--max-samples must be positive")
    baseline, baseline_records = load_sample_stack(find_sample_files(args.baseline_dir), device=args.device)
    guided, guided_records = load_sample_stack(find_sample_files(args.guided_dir), device=args.device)
    truth = load_volume(args.truth_model, device=args.device, single=True)
    boreholes = load_volume(args.boreholes, device=args.device, single=True) if args.boreholes else None
    paired, reason = infer_paired_by_seed(args.baseline_dir, args.guided_dir)
    summary = save_ensemble_figures(
        baseline=baseline,
        baseline_records=baseline_records,
        guided=guided,
        guided_records=guided_records,
        truth_model=truth,
        target_label=args.target_label,
        output_dir=args.output_dir,
        boreholes=boreholes,
        sample_ids=args.sample_id,
        max_samples=args.max_samples,
        paired_by_seed=paired,
    )
    summary["paired_by_seed"] = paired
    summary["paired_by_seed_reason"] = reason
    write_json(args.output_dir / "manifest.json", summary)
    print(f"Saved ensemble figures and manifest: {args.output_dir}")


if __name__ == "__main__":
    main()
