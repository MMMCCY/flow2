"""Visualize truth_model labels for manual dike-like target selection."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import torch

from geology_io_utils import label_mask, load_volume, target_component_stats, write_csv_rows, write_json


LABEL_FIELDS = [
    "label",
    "voxel_count",
    "volume_fraction",
    "borehole_hits",
    "target_connected_components",
    "largest_component_fraction",
]


def _setup_matplotlib():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    return plt


def _slice_indices(depth: int) -> List[int]:
    return sorted(set(max(0, min(depth - 1, int(value))) for value in (0, depth // 4, depth // 2, (3 * depth) // 4, depth - 1)))


def _save_truth_slices(truth: torch.Tensor, path: Path) -> None:
    plt = _setup_matplotlib()
    model = truth[0, 0].detach().cpu()
    z_values = _slice_indices(model.shape[-1])
    figure, axes = plt.subplots(1, len(z_values), figsize=(4 * len(z_values), 4))
    if len(z_values) == 1:
        axes = [axes]
    for axis, z_index in zip(axes, z_values):
        image = axis.imshow(model[:, :, z_index].T, origin="lower", cmap="tab20")
        axis.set_title(f"truth z={z_index}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.colorbar(image, ax=axes, shrink=0.75)
    figure.suptitle("Truth model categorical labels")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_label_overview(truth: torch.Tensor, labels: List[int], path: Path, max_labels: int) -> None:
    plt = _setup_matplotlib()
    model = truth[0, 0]
    shown = labels[:max_labels]
    ncols = min(5, max(1, len(shown)))
    nrows = (len(shown) + ncols - 1) // ncols
    figure, axes = plt.subplots(nrows, ncols, figsize=(3.2 * ncols, 3.2 * nrows))
    axes_list = axes.reshape(-1) if hasattr(axes, "reshape") else [axes]
    for axis, label in zip(axes_list, shown):
        mask = (model == int(label)).float().amax(dim=-1).detach().cpu()
        axis.imshow(mask.T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"label {label} MIP")
        axis.set_xticks([])
        axis.set_yticks([])
    for axis in axes_list[len(shown):]:
        axis.set_axis_off()
    figure.suptitle("Per-label XY maximum-intensity projections")
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _configure_3d_axis(axis, shape: Sequence[int], title: str) -> None:
    axis.set_title(title)
    axis.set_xlim(0, shape[0] - 1)
    axis.set_ylim(0, shape[1] - 1)
    axis.set_zlim(0, shape[2] - 1)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    axis.set_zlabel("Z")
    axis.view_init(elev=22, azim=-55)
    try:
        axis.set_box_aspect(shape)
    except AttributeError:
        pass


def _sample_coords(coords: torch.Tensor, max_points: int, seed: int) -> torch.Tensor:
    if coords.shape[0] <= max_points:
        return coords
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    order = torch.randperm(coords.shape[0], generator=generator)[:max_points]
    return coords[order]


def _save_3d_label_overview(
    truth: torch.Tensor,
    labels: List[int],
    path: Path,
    max_labels: int,
    max_points_per_label: int,
    exclude_labels: Sequence[int],
    seed: int,
) -> None:
    plt = _setup_matplotlib()
    model = truth[0, 0].detach().cpu().long()
    shown = [label for label in labels if int(label) not in set(exclude_labels)][:max_labels]
    figure = plt.figure(figsize=(9, 8))
    axis = figure.add_subplot(111, projection="3d")
    cmap = plt.get_cmap("tab20")
    for index, label in enumerate(shown):
        coords = torch.nonzero(model == int(label), as_tuple=False)
        if coords.numel() == 0:
            continue
        coords = _sample_coords(coords, max_points_per_label, seed + int(label) + index)
        color = cmap(index % 20)
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=1.5,
            alpha=0.35,
            color=color,
            label=f"label {label}",
            depthshade=False,
        )
    _configure_3d_axis(axis, model.shape, "Truth model 3D label overview")
    if shown:
        axis.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), markerscale=4)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_per_label_3d_views(
    truth: torch.Tensor,
    labels: List[int],
    output_dir: Path,
    max_labels: int,
    max_points: int,
    exclude_labels: Sequence[int],
    seed: int,
) -> List[str]:
    plt = _setup_matplotlib()
    model = truth[0, 0].detach().cpu().long()
    shown = [label for label in labels if int(label) not in set(exclude_labels)][:max_labels]
    paths: List[str] = []
    output_dir.mkdir(parents=True, exist_ok=True)
    for index, label in enumerate(shown):
        coords = torch.nonzero(model == int(label), as_tuple=False)
        if coords.numel() == 0:
            continue
        coords = _sample_coords(coords, max_points, seed + int(label) + index)
        figure = plt.figure(figsize=(7, 6.5))
        axis = figure.add_subplot(111, projection="3d")
        axis.scatter(
            coords[:, 0],
            coords[:, 1],
            coords[:, 2],
            s=2.0,
            alpha=0.45,
            color=plt.get_cmap("tab20")(index % 20),
            depthshade=False,
        )
        _configure_3d_axis(axis, model.shape, f"label {label} 3D sampled voxels")
        figure.tight_layout()
        path = output_dir / f"label_{label}_3d.png"
        figure.savefig(path, dpi=180)
        plt.close(figure)
        paths.append(str(path))
    return paths


def label_summary(truth: torch.Tensor, boreholes: torch.Tensor | None = None) -> List[Dict[str, object]]:
    model = truth.long()
    labels, counts = torch.unique(model, return_counts=True)
    total = int(model.numel())
    borehole_model = boreholes.long() if boreholes is not None else None
    rows: List[Dict[str, object]] = []
    for label_tensor, count_tensor in zip(labels, counts):
        label = int(label_tensor.item())
        mask = label_mask(model, label)
        stats = target_component_stats(mask)
        borehole_hits = ""
        if borehole_model is not None:
            borehole_hits = int((borehole_model == label).sum().item())
        rows.append(
            {
                "label": label,
                "voxel_count": int(count_tensor.item()),
                "volume_fraction": int(count_tensor.item()) / total,
                "borehole_hits": borehole_hits,
                "target_connected_components": stats["target_connected_components"],
                "largest_component_fraction": stats["largest_component_fraction"],
            }
        )
    return sorted(rows, key=lambda row: int(row["voxel_count"]), reverse=True)


def _find_truth_models(root: Path) -> List[Path]:
    if not root.exists():
        raise FileNotFoundError(f"truth root not found: {root}")
    paths = sorted(root.glob("*/true_model.pt"))
    if not paths:
        paths = sorted(root.rglob("true_model.pt"))
    if not paths:
        raise FileNotFoundError(f"no true_model.pt files found under {root}")
    return paths


def _case_output_dir(base_output_dir: Path, truth_path: Path, root: Path | None) -> Path:
    if root is None:
        return base_output_dir
    try:
        relative_parent = truth_path.parent.relative_to(root)
    except ValueError:
        relative_parent = Path(truth_path.parent.name)
    return base_output_dir / relative_parent


def _matching_boreholes(truth_path: Path, explicit_boreholes: Path | None, root_mode: bool) -> Path | None:
    if explicit_boreholes is not None and not root_mode:
        return explicit_boreholes
    candidate = truth_path.parent / "boreholes.pt"
    return candidate if candidate.exists() else None


def visualize_one_case(
    truth_model: Path,
    boreholes_path: Path | None,
    output_dir: Path,
    max_labels: int,
    max_3d_points_per_label: int,
    exclude_3d_label: Sequence[int],
    seed: int,
    device: str,
) -> Dict[str, object]:
    truth = load_volume(truth_model, device=device, single=True)
    boreholes = load_volume(boreholes_path, device=device, single=True) if boreholes_path else None
    rows = label_summary(truth, boreholes)
    labels = [int(row["label"]) for row in rows]
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv_rows(output_dir / "label_summary.csv", rows, LABEL_FIELDS)
    _save_truth_slices(truth, output_dir / "truth_model_label_slices.png")
    _save_label_overview(truth, labels, output_dir / "label_mip_overview.png", max_labels)
    _save_3d_label_overview(
        truth,
        labels,
        output_dir / "label_3d_overview.png",
        max_labels=max_labels,
        max_points_per_label=max_3d_points_per_label,
        exclude_labels=exclude_3d_label,
        seed=seed,
    )
    per_label_paths = _save_per_label_3d_views(
        truth,
        labels,
        output_dir / "label_3d",
        max_labels=max_labels,
        max_points=max_3d_points_per_label,
        exclude_labels=exclude_3d_label,
        seed=seed,
    )
    manifest = {
        "truth_model": str(truth_model),
        "boreholes": str(boreholes_path) if boreholes_path else None,
        "label_summary_csv": str(output_dir / "label_summary.csv"),
        "truth_model_label_slices": str(output_dir / "truth_model_label_slices.png"),
        "label_mip_overview": str(output_dir / "label_mip_overview.png"),
        "label_3d_overview": str(output_dir / "label_3d_overview.png"),
        "per_label_3d_figures": per_label_paths,
        "exclude_3d_label": [int(label) for label in exclude_3d_label],
        "max_3d_points_per_label": int(max_3d_points_per_label),
        "manual_step": "Inspect 3D figures, 2D figures, and label_summary.csv, then pass the confirmed dike-like id as --target-label.",
    }
    write_json(output_dir / "manual_target_label_manifest.json", manifest)
    return manifest


def _relative_link(target: Path, base: Path) -> str:
    try:
        return target.relative_to(base).as_posix()
    except ValueError:
        return target.as_posix()


def _write_batch_gallery(output_dir: Path, manifests: Iterable[Dict[str, object]]) -> None:
    lines = [
        "# Truth Model 3D Label QA Gallery",
        "",
        "Inspect the 3D overview first, then open per-label 3D figures for candidate dike-like labels.",
        "",
    ]
    for manifest in manifests:
        case_dir = Path(str(manifest["label_summary_csv"])).parent
        title = case_dir.name
        overview_3d = _relative_link(Path(str(manifest["label_3d_overview"])), output_dir)
        slices = _relative_link(Path(str(manifest["truth_model_label_slices"])), output_dir)
        mip = _relative_link(Path(str(manifest["label_mip_overview"])), output_dir)
        summary = _relative_link(Path(str(manifest["label_summary_csv"])), output_dir)
        lines.extend(
            [
                f"## {title}",
                "",
                f"- label summary: [{summary}]({summary})",
                f"- 2D slices: [{slices}]({slices})",
                f"- label MIP: [{mip}]({mip})",
                "",
                f"![{title} 3D overview]({overview_3d})",
                "",
                "Per-label 3D figures:",
                "",
            ]
        )
        for figure in manifest.get("per_label_3d_figures", []):
            figure_path = Path(str(figure))
            rel = _relative_link(figure_path, output_dir)
            lines.append(f"- [{figure_path.name}]({rel})")
        lines.append("")
    (output_dir / "truth_model_3d_gallery.md").write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Visualize truth_model labels to manually confirm a dike-like target label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--truth-model", type=Path)
    source.add_argument("--truth-root", type=Path, help="Directory containing case subdirectories with true_model.pt files.")
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-labels", type=int, default=20)
    parser.add_argument("--max-3d-points-per-label", type=int, default=6000)
    parser.add_argument(
        "--exclude-3d-label",
        type=int,
        action="append",
        default=[-1],
        help="Label excluded from 3D scatter views. Repeat to exclude more labels.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root_mode = args.truth_root is not None
    truth_models = _find_truth_models(args.truth_root) if root_mode else [args.truth_model]
    manifests = []
    for truth_model in truth_models:
        case_output_dir = _case_output_dir(args.output_dir, truth_model, args.truth_root)
        boreholes_path = _matching_boreholes(truth_model, args.boreholes, root_mode)
        manifest = visualize_one_case(
            truth_model=truth_model,
            boreholes_path=boreholes_path,
            output_dir=case_output_dir,
            max_labels=args.max_labels,
            max_3d_points_per_label=args.max_3d_points_per_label,
            exclude_3d_label=args.exclude_3d_label,
            seed=args.seed,
            device=args.device,
        )
        manifests.append(manifest)
    write_json(
        args.output_dir / "batch_manifest.json",
        {
            "truth_root": str(args.truth_root) if args.truth_root else None,
            "truth_models": [str(path) for path in truth_models],
            "case_manifests": [manifest["label_summary_csv"].replace("label_summary.csv", "manual_target_label_manifest.json") for manifest in manifests],
            "manual_step": "Inspect each case directory, especially label_3d_overview.png and label_3d/*.png, then choose a case and dike-like target label.",
        },
    )
    _write_batch_gallery(args.output_dir, manifests)
    print(f"Saved truth label QA for {len(truth_models)} case(s): {args.output_dir}")


if __name__ == "__main__":
    main()
