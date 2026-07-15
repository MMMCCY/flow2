"""Create a controlled susceptibility_config for lightweight magnetic-proxy demos."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch

from geology_io_utils import load_volume, write_json


def build_susceptibility_config(
    truth_model: torch.Tensor,
    target_label: int,
    target_susceptibility: float = 5.0,
    background_scale: float = 0.01,
    default_susceptibility: float = 0.0,
) -> dict[str, object]:
    labels = sorted(int(value.item()) for value in torch.unique(truth_model.long()))
    susceptibilities = {}
    for label in labels:
        if label == -1:
            susceptibilities[label] = 0.0
        else:
            susceptibilities[label] = float(max(label + 1, 1) * background_scale)
    if int(target_label) not in susceptibilities:
        raise ValueError(f"target_label {target_label} is absent from truth_model labels")
    susceptibilities[int(target_label)] = float(target_susceptibility)
    target_value = susceptibilities[int(target_label)]
    other_values = [value for label, value in susceptibilities.items() if int(label) != int(target_label)]
    nearest_other = min(other_values, key=lambda value: abs(value - target_value)) if other_values else default_susceptibility
    return {
        "name": "manual_target_susceptibility_config",
        "description": (
            "Controlled lightweight magnetic-proxy susceptibility configuration "
            "for manual dike-like target-label demo."
        ),
        "target_label": int(target_label),
        "target_susceptibility": target_value,
        "nearest_non_target_susceptibility": nearest_other,
        "target_to_nearest_non_target_susceptibility_contrast": abs(target_value - nearest_other),
        "default_susceptibility": float(default_susceptibility),
        "susceptibilities": {str(label): value for label, value in susceptibilities.items()},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create susceptibility_config.json after manually confirming target label.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--truth-model", type=Path, required=True)
    parser.add_argument("--target-label", type=int, required=True)
    parser.add_argument("--target-susceptibility", type=float, default=5.0)
    parser.add_argument("--background-scale", type=float, default=0.01)
    parser.add_argument("--default-susceptibility", type=float, default=0.0)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    truth = load_volume(args.truth_model, device=args.device, single=True)
    config = build_susceptibility_config(
        truth,
        target_label=args.target_label,
        target_susceptibility=args.target_susceptibility,
        background_scale=args.background_scale,
        default_susceptibility=args.default_susceptibility,
    )
    write_json(args.output_json, config)
    print(f"Saved susceptibility config: {args.output_json}")


if __name__ == "__main__":
    main()
