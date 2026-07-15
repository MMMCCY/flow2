"""Generate observed lightweight gravity-proxy from truth_model and density_config."""

from __future__ import annotations

import argparse
from pathlib import Path

from geophysics import SimpleGravityForward
from geology_io_utils import (
    density_config_metadata,
    load_density_config,
    load_volume,
    property_map_from_density_config,
    write_json,
)


def _save_gravity_plot(field, path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    array = field.detach().cpu().float().reshape(-1, *field.shape[-2:])[0]
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(array.T, origin="lower", cmap="viridis")
    axis.set_title("Observed lightweight gravity-proxy")
    axis.set_xlabel("X")
    axis.set_ylabel("Y")
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=180)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate observed_gravity.pt from truth_model using density_config.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--density-config", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--kernel-size", type=int, default=9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_volume(args.truth_model, device=args.device, single=True)
    density_config = load_density_config(args.density_config)
    property_map = property_map_from_density_config(density_config)
    forward_model = SimpleGravityForward(kernel_size=args.kernel_size)
    observed = forward_model(property_map(truth)).detach().cpu()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    observed_path = args.output_dir / "observed_gravity.pt"
    import torch

    torch.save(observed, observed_path)
    figure_path = args.output_dir / "observed_gravity.png"
    _save_gravity_plot(observed, figure_path)
    write_json(
        args.output_dir / "manifest.json",
        {
            "truth_model": str(args.truth_model),
            "density_config": str(args.density_config),
            "observed_gravity": str(observed_path),
            "observed_gravity_figure": str(figure_path),
            "kernel_size": args.kernel_size,
            **density_config_metadata(density_config),
            "description": "Observed field generated with controlled lightweight gravity-proxy density_config.",
        },
    )
    print(f"Saved observed gravity proxy: {observed_path}")


if __name__ == "__main__":
    main()
