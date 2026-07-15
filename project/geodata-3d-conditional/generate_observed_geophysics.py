"""Generate observed lightweight geophysical proxy fields from truth_model."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

import torch

from geophysics import GravityGradientForward, MagneticTMIForward, SimpleGravityForward
from geology_io_utils import (
    density_config_metadata,
    load_density_config,
    load_susceptibility_config,
    load_volume,
    property_map_from_density_config,
    property_map_from_susceptibility_config,
    susceptibility_config_metadata,
    write_json,
)


def _save_field_plot(field: torch.Tensor, path: Path, title: str, cmap: str = "viridis") -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    array = field.detach().cpu().float().reshape(-1, *field.shape[-2:])[0]
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(array.T, origin="lower", cmap=cmap)
    axis.set_title(title)
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate observed lightweight gravity/magnetic proxy tensors from truth_model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--density-config", type=Path, default=None)
    parser.add_argument("--susceptibility-config", type=Path, default=None)
    parser.add_argument(
        "--physics-mode",
        choices=("gravity", "gravity_gradient", "magnetic", "joint"),
        default="joint",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_volume(args.truth_model, device=args.device, single=True)
    density_config = load_density_config(args.density_config)
    susceptibility_config = load_susceptibility_config(args.susceptibility_config)
    density_map = property_map_from_density_config(density_config)
    susceptibility_map = property_map_from_susceptibility_config(susceptibility_config)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, str] = {}
    figures: Dict[str, str] = {}
    if args.physics_mode in {"gravity", "joint"}:
        observed = SimpleGravityForward(kernel_size=args.kernel_size)(density_map(truth)).detach().cpu()
        path = args.output_dir / "observed_gravity.pt"
        torch.save(observed, path)
        figure_path = args.output_dir / "observed_gravity.png"
        _save_field_plot(observed, figure_path, "Observed lightweight gravity-proxy")
        outputs["observed_gravity"] = str(path)
        figures["observed_gravity"] = str(figure_path)
    if args.physics_mode in {"gravity_gradient", "joint"}:
        observed = GravityGradientForward(kernel_size=args.kernel_size)(density_map(truth)).detach().cpu()
        path = args.output_dir / "observed_gravity_gradient.pt"
        torch.save(observed, path)
        figure_path = args.output_dir / "observed_gravity_gradient.png"
        _save_field_plot(observed, figure_path, "Observed lightweight gravity-gradient-proxy", cmap="coolwarm")
        outputs["observed_gravity_gradient"] = str(path)
        figures["observed_gravity_gradient"] = str(figure_path)
    if args.physics_mode in {"magnetic", "joint"}:
        observed = MagneticTMIForward(kernel_size=args.kernel_size)(susceptibility_map(truth)).detach().cpu()
        path = args.output_dir / "observed_magnetic.pt"
        torch.save(observed, path)
        figure_path = args.output_dir / "observed_magnetic.png"
        _save_field_plot(observed, figure_path, "Observed lightweight magnetic-proxy", cmap="magma")
        outputs["observed_magnetic"] = str(path)
        figures["observed_magnetic"] = str(figure_path)

    write_json(
        args.output_dir / "manifest.json",
        {
            "truth_model": str(args.truth_model),
            "density_config": str(args.density_config) if args.density_config else None,
            "susceptibility_config": str(args.susceptibility_config) if args.susceptibility_config else None,
            "physics_mode": args.physics_mode,
            "kernel_size": args.kernel_size,
            "outputs": outputs,
            "figures": figures,
            **density_config_metadata(density_config),
            **susceptibility_config_metadata(susceptibility_config),
            "description": (
                "Observed fields generated with controlled lightweight geophysical "
                "proxy forward models, not quantitative geophysical inversion."
            ),
        },
    )
    print(f"Saved observed lightweight geophysical proxies: {args.output_dir}")


if __name__ == "__main__":
    main()
