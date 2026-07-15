"""Analyze whether a target label is observable by the lightweight gravity proxy."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional

import torch

from geophysics import LithologyPropertyMap, SimpleGravityForward, normalized_misfit
from geology_io_utils import (
    as_batched_volume,
    density_value,
    load_density_config,
    label_mask,
    local_replacement_label,
    load_volume,
    property_map_from_density_config,
    read_json,
    write_json,
)


def _property_map_with_target(target_label: int, target_density: Optional[float]) -> LithologyPropertyMap:
    if target_density is None:
        return LithologyPropertyMap()
    properties = dict(LithologyPropertyMap.DEFAULT_DENSITY_CONTRASTS)
    properties[int(target_label)] = float(target_density)
    return LithologyPropertyMap(properties=properties)


def _save_array_plot(array: torch.Tensor, path: Path, title: str, cmap: str = "viridis", symmetric: bool = False) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    data = array.detach().cpu().float()
    vmax = float(data.abs().max().item()) if symmetric else None
    vmin = -vmax if symmetric else None
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(data.T, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def _save_mask_slices(mask: torch.Tensor, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    mask_3d = mask.detach().cpu().bool()
    z_values = [0, mask_3d.shape[-1] // 3, (2 * mask_3d.shape[-1]) // 3, mask_3d.shape[-1] - 1]
    z_values = sorted(set(max(0, min(mask_3d.shape[-1] - 1, int(value))) for value in z_values))
    figure, axes = plt.subplots(1, len(z_values), figsize=(4 * len(z_values), 4))
    if len(z_values) == 1:
        axes = [axes]
    for axis, z_index in zip(axes, z_values):
        axis.imshow(mask_3d[:, :, z_index].float().T, origin="lower", cmap="gray", vmin=0, vmax=1)
        axis.set_title(f"z={z_index}")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(title)
    figure.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, dpi=180)
    plt.close(figure)


def observability_summary(
    truth_model: torch.Tensor,
    target_label: int,
    boreholes: Optional[torch.Tensor] = None,
    replacement_label: Optional[int] = None,
    target_density: Optional[float] = None,
    density_config: Optional[Dict[str, object]] = None,
    kernel_size: int = 9,
) -> Dict[str, object]:
    """Compare truth gravity-proxy with target-deleted/replaced proxy response."""
    summary, _, _, _, _ = observability_analysis(
        truth_model=truth_model,
        target_label=target_label,
        boreholes=boreholes,
        replacement_label=replacement_label,
        target_density=target_density,
        density_config=density_config,
        kernel_size=kernel_size,
    )
    return summary


def observability_analysis(
    truth_model: torch.Tensor,
    target_label: int,
    boreholes: Optional[torch.Tensor] = None,
    replacement_label: Optional[int] = None,
    target_density: Optional[float] = None,
    density_config: Optional[Dict[str, object]] = None,
    kernel_size: int = 9,
) -> tuple[Dict[str, object], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return summary plus truth/replaced/delta proxy fields and target mask."""
    truth = as_batched_volume(truth_model, "truth_model").long()
    if truth.shape[0] != 1:
        raise ValueError("truth_model must contain exactly one volume")
    target = label_mask(truth, target_label)
    target_volume = int(target.sum().item())
    if target_volume == 0:
        raise ValueError(f"target_label {target_label} is absent from truth_model")

    if replacement_label is None:
        resolved_replacement, replacement_source = local_replacement_label(
            truth,
            target,
            target_label=target_label,
            ignore_labels=(-1,),
        )
    else:
        resolved_replacement = int(replacement_label)
        replacement_source = "cli_argument"
    altered = truth.clone()
    altered[target] = resolved_replacement

    property_map = property_map_from_density_config(density_config) if density_config is not None else _property_map_with_target(target_label, target_density)
    forward_model = SimpleGravityForward(kernel_size=kernel_size)
    observed = forward_model(property_map(truth))
    replaced = forward_model(property_map(altered))
    delta = observed - replaced
    normalized_delta = normalized_misfit(replaced, observed, reduction="mean")
    target_density_value = density_value(target_label, property_map.properties)
    replacement_density_value = density_value(resolved_replacement, property_map.properties)
    density_contrast = abs(target_density_value - replacement_density_value)
    target_volume_fraction = target_volume / truth[0, 0].numel()

    borehole_target_hits = None
    if boreholes is not None:
        borehole_volume = as_batched_volume(boreholes, "boreholes").long()
        if borehole_volume.shape != truth.shape:
            raise ValueError("boreholes and truth_model must have matching shape")
        borehole_target_hits = int(((borehole_volume == int(target_label)) & target).sum().item())

    reasons = []
    if target_volume_fraction < 1e-4:
        reasons.append("target volume fraction is very small")
    if density_contrast <= 0.0:
        reasons.append("target and replacement density contrast is zero")
    if float(normalized_delta.item()) < 0.01:
        reasons.append("lightweight gravity-proxy response change is weak")
    if borehole_target_hits == 0:
        reasons.append("no target-label borehole evidence")
    recommended = not any(reason in reasons for reason in (
        "target volume fraction is very small",
        "target and replacement density contrast is zero",
        "lightweight gravity-proxy response change is weak",
    ))
    reason = "recommended: target has measurable lightweight gravity-proxy response" if recommended else "; ".join(reasons)

    return {
        "target_label": int(target_label),
        "target_volume": target_volume,
        "target_volume_fraction": float(target_volume_fraction),
        "replacement_label": int(resolved_replacement),
        "replacement_label_source": replacement_source,
        "target_density_override": target_density,
        "density_config": density_config.get("name") if density_config else None,
        "target_density": target_density_value,
        "replacement_density": replacement_density_value,
        "density_contrast": density_contrast,
        "kernel_size": int(kernel_size),
        "borehole_target_hits": borehole_target_hits,
        "lightweight_gravity_proxy_normalized_delta": float(normalized_delta.item()),
        "lightweight_gravity_proxy_delta_l2": float(torch.linalg.vector_norm(delta.flatten()).item()),
        "lightweight_gravity_proxy_delta_mean_abs": float(delta.abs().mean().item()),
        "lightweight_gravity_proxy_delta_peak_abs": float(delta.abs().max().item()),
        "recommended_for_demo": bool(recommended),
        "reason": reason,
        "description": (
            "Observability is measured by replacing the target label in the "
            "truth model and comparing the lightweight gravity-proxy response. "
            "This is a lightweight gravity-proxy sensitivity diagnostic."
        ),
    }, observed, replaced, delta, target


def save_observability_artifacts(
    output_dir: Path,
    truth_gravity: torch.Tensor,
    removed_target_gravity: torch.Tensor,
    delta_gravity: torch.Tensor,
    target_mask: torch.Tensor,
) -> Dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "truth_gravity": output_dir / "truth_gravity.png",
        "removed_target_gravity": output_dir / "removed_target_gravity.png",
        "delta_gravity": output_dir / "delta_gravity.png",
        "target_mask_slices": output_dir / "target_mask_slices.png",
    }
    _save_array_plot(truth_gravity[0, 0], paths["truth_gravity"], "Truth lightweight gravity-proxy")
    _save_array_plot(
        removed_target_gravity[0, 0],
        paths["removed_target_gravity"],
        "Target-removed lightweight gravity-proxy",
    )
    _save_array_plot(delta_gravity[0, 0], paths["delta_gravity"], "Target proxy response delta", cmap="coolwarm", symmetric=True)
    _save_mask_slices(target_mask[0, 0], paths["target_mask_slices"], "Target-label mask slices")
    return {key: str(value) for key, value in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure target-label observability under the lightweight gravity-proxy.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--target-label", type=int, default=None)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--replacement-label", type=int, default=None)
    parser.add_argument("--target-density", type=float, default=None)
    parser.add_argument("--density-config", type=Path, default=None)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    density_config = load_density_config(args.density_config)
    target_label = args.target_label
    if target_label is None and density_config is not None and density_config.get("target_label") is not None:
        target_label = int(density_config["target_label"])
    if target_label is None and args.manifest is not None:
        target_label = int(read_json(args.manifest)["selected"]["target_label"])
    if target_label is None:
        raise SystemExit("provide --target-label, --density-config with target_label, or --manifest")
    truth = load_volume(args.truth_model, device=args.device, single=True)
    boreholes = load_volume(args.boreholes, device=args.device, single=True) if args.boreholes else None
    summary, truth_gravity, removed_target_gravity, delta_gravity, target_mask = observability_analysis(
        truth_model=truth,
        target_label=target_label,
        boreholes=boreholes,
        replacement_label=args.replacement_label,
        target_density=args.target_density,
        density_config=density_config,
        kernel_size=args.kernel_size,
    )
    summary["figures"] = save_observability_artifacts(
        args.output_dir,
        truth_gravity,
        removed_target_gravity,
        delta_gravity,
        target_mask,
    )
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved observability summary: {args.output_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
