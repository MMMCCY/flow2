"""Render truth, baseline, and guided sample triplets as 3D geology figures."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import torch

from geology_io_utils import as_single_volume, load_volume, write_json


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _surface_mask(mask: torch.Tensor) -> torch.Tensor:
    values = mask.detach().cpu().bool()
    if values.sum().item() == 0:
        return values
    padded = torch.nn.functional.pad(
        values.unsqueeze(0).unsqueeze(0),
        (1, 1, 1, 1, 1, 1),
        value=False,
    )[0, 0]
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


def _sample_coords(coords: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    if coords.shape[0] <= max_points:
        return coords
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(coords.shape[0], generator=generator)[:max_points]
    return coords[order]


def _set_clean_axis(axis, shape: Sequence[int], title: str, show_axes: bool) -> None:
    axis.set_title(title, pad=8)
    axis.set_xlim(0, shape[0] - 1)
    axis.set_ylim(0, shape[1] - 1)
    axis.set_zlim(0, shape[2] - 1)
    axis.view_init(elev=23, azim=-52)
    try:
        axis.set_box_aspect(shape)
    except AttributeError:
        pass
    if show_axes:
        axis.set_xlabel("X")
        axis.set_ylabel("Y")
        axis.set_zlabel("Z")
    else:
        axis.set_axis_off()


def _label_color_map(labels: Iterable[int]):
    plt = _setup_matplotlib()
    cmap = plt.get_cmap("tab20")
    color_map = {}
    for index, label in enumerate(sorted(int(value) for value in labels)):
        color_map[label] = cmap(index % 20)
    return color_map


def _draw_full_geology(
    axis,
    volume_3d: torch.Tensor,
    color_map: Mapping[int, object],
    exclude_labels: Sequence[int],
    max_points_per_label: int,
    seed: int,
) -> Dict[str, int]:
    stats: Dict[str, int] = {}
    excluded = {int(label) for label in exclude_labels}
    labels = sorted(int(value.item()) for value in torch.unique(volume_3d.long()))
    for index, label in enumerate(labels):
        if label in excluded:
            continue
        mask = volume_3d.long() == label
        surface = _surface_mask(mask)
        coords = torch.nonzero(surface, as_tuple=False)
        if coords.numel() == 0:
            continue
        stats[str(label)] = int(coords.shape[0])
        coords = _sample_coords(coords, max_points_per_label, seed + index + label)
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=1.1,
            alpha=0.42,
            color=color_map[label],
            depthshade=False,
        )
    return stats


def _draw_target_body(
    axis,
    volume_3d: torch.Tensor,
    target_label: int,
    max_points: int,
    seed: int,
) -> int:
    mask = volume_3d.long() == int(target_label)
    coords = torch.nonzero(mask, as_tuple=False)
    if coords.numel() == 0:
        return 0
    count = int(coords.shape[0])
    coords = _sample_coords(coords, max_points, seed)
    axis.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        s=3.0,
        alpha=0.72,
        color="#d7191c",
        depthshade=False,
    )
    return count


def _load_sample(path: Path, device: str) -> torch.Tensor:
    return as_single_volume(torch.load(path, map_location=device), path).cpu()


def save_triplet_figure(
    truth_model: torch.Tensor,
    baseline_sample: torch.Tensor,
    guided_sample: torch.Tensor,
    target_label: int,
    output_dir: Path,
    sample_id: int,
    exclude_labels: Sequence[int],
    max_points_per_label: int,
    max_target_points: int,
    seed: int,
    show_axes: bool = False,
) -> Dict[str, object]:
    plt = _setup_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    volumes = [
        ("Truth model", truth_model[0, 0].cpu().long()),
        (f"Baseline sample {sample_id}", baseline_sample[0, 0].cpu().long()),
        (f"Guided sample {sample_id}", guided_sample[0, 0].cpu().long()),
    ]
    all_labels = sorted(
        {
            int(value.item())
            for _, volume in volumes
            for value in torch.unique(volume.long())
            if int(value.item()) not in {int(label) for label in exclude_labels}
        }
    )
    color_map = _label_color_map(all_labels)
    shape = tuple(int(value) for value in volumes[0][1].shape)
    render_stats: Dict[str, object] = {}

    figure = plt.figure(figsize=(14.5, 9.5))
    for column, (title, volume) in enumerate(volumes, start=1):
        axis = figure.add_subplot(2, 3, column, projection="3d")
        render_stats[f"{title}_surface_points_by_label"] = _draw_full_geology(
            axis,
            volume,
            color_map=color_map,
            exclude_labels=exclude_labels,
            max_points_per_label=max_points_per_label,
            seed=seed + 100 * column,
        )
        _set_clean_axis(axis, shape, title, show_axes=show_axes)

        axis = figure.add_subplot(2, 3, column + 3, projection="3d")
        target_count = _draw_target_body(
            axis,
            volume,
            target_label=target_label,
            max_points=max_target_points,
            seed=seed + 1000 * column,
        )
        render_stats[f"{title}_target_voxels"] = target_count
        _set_clean_axis(axis, shape, f"{title}: label {target_label}", show_axes=show_axes)

    figure.suptitle(
        f"Truth, baseline, and guided 3D geology with target label {target_label}",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.96))
    combined_path = output_dir / f"sample_{sample_id}_truth_baseline_guided_3d_six_panel.png"
    figure.savefig(combined_path, dpi=220)
    plt.close(figure)

    panel_paths = {}
    for index, (title, volume) in enumerate(volumes, start=1):
        safe_name = title.lower().replace(" ", "_")
        for mode in ("full", "target"):
            figure = plt.figure(figsize=(7.5, 7.0))
            axis = figure.add_subplot(111, projection="3d")
            if mode == "full":
                _draw_full_geology(
                    axis,
                    volume,
                    color_map=color_map,
                    exclude_labels=exclude_labels,
                    max_points_per_label=max_points_per_label * 2,
                    seed=seed + 2000 * index,
                )
                title_text = title
            else:
                _draw_target_body(
                    axis,
                    volume,
                    target_label=target_label,
                    max_points=max_target_points * 2,
                    seed=seed + 3000 * index,
                )
                title_text = f"{title}: label {target_label}"
            _set_clean_axis(axis, shape, title_text, show_axes=show_axes)
            figure.tight_layout()
            path = output_dir / f"sample_{sample_id}_{safe_name}_{mode}_3d.png"
            figure.savefig(path, dpi=240)
            plt.close(figure)
            panel_paths[f"{safe_name}_{mode}"] = str(path)

    manifest = {
        "sample_id": int(sample_id),
        "target_label": int(target_label),
        "combined_figure": str(combined_path),
        "panel_figures": panel_paths,
        "exclude_labels": [int(label) for label in exclude_labels],
        "max_points_per_label": int(max_points_per_label),
        "max_target_points": int(max_target_points),
        "render_stats": render_stats,
        "description": "3D categorical geology and target-label-only triplet visualization for truth, baseline, and guided outputs.",
    }
    write_json(output_dir / f"sample_{sample_id}_triplet_3d_manifest.json", manifest)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render truth/baseline/guided full geology and target-label-only 3D panels.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--baseline-sample", type=Path, required=True)
    parser.add_argument("--guided-sample", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--sample-id", type=int, default=12)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--exclude-label", type=int, action="append", default=[-1])
    parser.add_argument("--max-points-per-label", type=int, default=60000)
    parser.add_argument("--max-target-points", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--show-axes", action="store_true")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_volume(args.truth_model, device=args.device, single=True).cpu()
    baseline = _load_sample(args.baseline_sample, args.device)
    guided = _load_sample(args.guided_sample, args.device)
    if truth.shape != baseline.shape or truth.shape != guided.shape:
        raise ValueError(
            f"shape mismatch: truth={tuple(truth.shape)}, "
            f"baseline={tuple(baseline.shape)}, guided={tuple(guided.shape)}"
        )
    manifest = save_triplet_figure(
        truth_model=truth,
        baseline_sample=baseline,
        guided_sample=guided,
        target_label=args.target_label,
        output_dir=args.output_dir,
        sample_id=args.sample_id,
        exclude_labels=args.exclude_label,
        max_points_per_label=args.max_points_per_label,
        max_target_points=args.max_target_points,
        seed=args.seed,
        show_axes=args.show_axes,
    )
    print(f"Saved 3D triplet figure: {manifest['combined_figure']}")


if __name__ == "__main__":
    main()
