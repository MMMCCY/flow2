"""Create a controlled density_config for lightweight gravity-proxy demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geophysics import LithologyPropertyMap
from geology_io_utils import density_value, load_volume, write_json


def build_density_config(
    truth_model: torch.Tensor,
    target_label: int,
    target_density: float | None = None,
    background_scale: float = 1.0,
    default_density: float = 0.0,
) -> dict[str, object]:
    labels = sorted(int(value.item()) for value in torch.unique(truth_model.long()))
    densities = {}
    for label in labels:
        base = density_value(label, LithologyPropertyMap.DEFAULT_DENSITY_CONTRASTS)
        densities[label] = float(base * background_scale)
    if target_density is not None:
        densities[int(target_label)] = float(target_density)
    if int(target_label) not in densities:
        raise ValueError(f"target_label {target_label} is absent from truth_model labels")
    target_value = densities[int(target_label)]
    other_values = [value for label, value in densities.items() if int(label) != int(target_label)]
    nearest_other = min(other_values, key=lambda value: abs(value - target_value)) if other_values else default_density
    return {
        "name": "manual_target_density_config",
        "description": (
            "Controlled lightweight gravity-proxy density configuration for "
            "manual dike-like target-label demo."
        ),
        "target_label": int(target_label),
        "target_density": target_value,
        "nearest_non_target_density": nearest_other,
        "target_to_nearest_non_target_density_contrast": abs(target_value - nearest_other),
        "default_density": float(default_density),
        "densities": {str(label): value for label, value in densities.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create density_config.json after manually confirming target label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--target-density", type=float, default=None)
    parser.add_argument("--background-scale", type=float, default=1.0)
    parser.add_argument("--default-density", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_volume(args.truth_model, device=args.device, single=True)
    config = build_density_config(
        truth,
        target_label=args.target_label,
        target_density=args.target_density,
        background_scale=args.background_scale,
        default_density=args.default_density,
    )
    write_json(args.output_json, config)
    print(f"Saved density config: {args.output_json}")


if __name__ == "__main__":
    main()
