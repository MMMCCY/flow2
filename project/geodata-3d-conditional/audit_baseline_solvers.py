"""Compare the original adaptive reference with the paired fixed-step baseline.

The adaptive Dopri5 path represents the historical inference implementation.
The fixed Euler path is the canonical no-guidance control for inference-time
guidance experiments.  They intentionally answer different questions, so this
audit quantifies their solver-only difference from identical model weights,
conditioning, and initial noise.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List

import torch

import inference_runtime as runtime
from flowtrain.solvers.solvers import ODEFlowSolver


def fixed_euler_prior_sample(
    model,
    x0: torch.Tensor,
    conditioning: torch.Tensor,
    n_steps: int,
) -> torch.Tensor:
    """Match the prior-only update used by ``guided_euler_sample``."""
    if n_steps <= 0:
        raise ValueError("n_steps must be positive")
    x = x0
    conditioning = conditioning.to(device=x.device, dtype=x.dtype).expand(
        x.shape[0], -1, -1, -1, -1
    )
    dt = 1.0 / n_steps
    for step in range(n_steps):
        t_value = (step + 0.5) / n_steps
        t = torch.full((x.shape[0],), t_value, device=x.device, dtype=x.dtype)
        with torch.no_grad():
            velocity = model.net(x, conditioning, t)
            x = x + dt * velocity
    return x


def adaptive_reference_sample(
    model,
    x0: torch.Tensor,
    conditioning: torch.Tensor,
    saved_steps: int = 8,
) -> torch.Tensor:
    """Reproduce the historical adaptive-Dopri5 inference semantics."""
    conditioning = conditioning.to(device=x0.device, dtype=x0.dtype).expand(
        x0.shape[0], -1, -1, -1, -1
    )

    def dxdt_cond(x: torch.Tensor, time: torch.Tensor) -> torch.Tensor:
        return model.net(x, conditioning, time)

    solver = ODEFlowSolver(model=dxdt_cond, rtol=1e-6)
    return solver.solve(
        x0,
        t0=0.0001,
        tf=0.9999,
        n_steps=saved_steps,
    )[-1]


def solver_difference_record(
    model,
    paired_state: torch.Tensor,
    reference_state: torch.Tensor,
    sample_id: int,
) -> Dict[str, object]:
    delta = paired_state - reference_state
    reference_norm = reference_state.flatten(1).norm(dim=1)
    paired_decoded = model.decode(paired_state).detach().cpu() - 1
    reference_decoded = model.decode(reference_state).detach().cpu() - 1
    changed = paired_decoded != reference_decoded
    return {
        "sample_id": sample_id,
        "continuous_rmse": float(delta.square().mean().sqrt().item()),
        "continuous_relative_l2": float(
            (
                delta.flatten(1).norm(dim=1)
                / reference_norm.clamp_min(torch.finfo(delta.dtype).eps)
            )
            .mean()
            .item()
        ),
        "decoded_changed_voxel_fraction": float(changed.float().mean().item()),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Quantify the solver-only difference between historical adaptive "
            "Dopri5 inference and the strict paired fixed-Euler baseline."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--ckpt-path", type=Path, required=True)
    parser.add_argument("--samples-dir", type=Path, required=True)
    parser.add_argument("--truth-model", type=Path, default=None)
    parser.add_argument("--boreholes", type=Path, default=None)
    parser.add_argument("--model-weights", choices=("ema", "raw"), default="ema")
    parser.add_argument("--n-samples", type=int, default=1)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.n_samples <= 0 or args.n_steps <= 0:
        raise ValueError("--n-samples and --n-steps must be positive")
    device = torch.device(args.device)

    repository_root = Path(__file__).resolve().parents[2]
    if str(repository_root) not in sys.path:
        sys.path.insert(0, str(repository_root))
    from model_train_sh_inference_cond import Geo3DStochInterp

    truth_path = args.truth_model or args.samples_dir / "true_model.pt"
    boreholes_path = args.boreholes or args.samples_dir / "boreholes.pt"
    truth = runtime.normalize_single_geology(
        runtime.load_tensor(truth_path, map_location=device),
        str(truth_path),
    ).to(device)
    boreholes = runtime.normalize_single_geology(
        runtime.load_tensor(boreholes_path, map_location=device),
        str(boreholes_path),
    ).to(device)

    model, model_report = runtime.load_model_with_weight_policy(
        model_class=Geo3DStochInterp,
        checkpoint_path=args.ckpt_path,
        map_location=device,
        weight_source=args.model_weights,
    )
    model = model.to(device)
    conditioning_report = runtime.validate_conditioning_pair(
        truth,
        boreholes,
        num_categories=model.num_categories,
    )

    condition_mask = (boreholes != -1) | (truth == -1)
    embedded_truth = model.embed(truth)
    conditioning = embedded_truth * condition_mask.expand(
        -1, embedded_truth.shape[1], -1, -1, -1
    )

    generator = torch.Generator(device="cpu").manual_seed(args.seed)
    records: List[Dict[str, object]] = []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for sample_id in range(args.n_samples):
        x0 = torch.randn(
            1,
            model.embedding_dim,
            *model.data_shape,
            generator=generator,
            dtype=embedded_truth.dtype,
        ).to(device)
        paired_state = fixed_euler_prior_sample(
            model,
            x0.clone(),
            conditioning,
            n_steps=args.n_steps,
        )
        reference_state = adaptive_reference_sample(
            model,
            x0.clone(),
            conditioning,
        )
        record = solver_difference_record(
            model,
            paired_state,
            reference_state,
            sample_id,
        )
        records.append(record)
        torch.save(
            (model.decode(paired_state).detach().cpu() - 1)[0],
            args.output_dir / f"paired_fixed_euler_{sample_id}.pt",
        )
        torch.save(
            (model.decode(reference_state).detach().cpu() - 1)[0],
            args.output_dir / f"reference_dopri5_{sample_id}.pt",
        )

    summary = {
        "protocol_version": runtime.PROTOCOL_VERSION,
        "purpose": (
            "Measure solver-only differences. Do not treat fixed Euler and "
            "adaptive Dopri5 outputs as paired guidance realizations."
        ),
        "checkpoint": model_report["checkpoint"],
        "model_weight_source": args.model_weights,
        "ema_applied": model_report["ema_applied"],
        "truth_model": runtime.asset_record(truth_path),
        "boreholes": runtime.asset_record(boreholes_path),
        "seed": args.seed,
        "n_samples": args.n_samples,
        "paired_integrator": runtime.PAIRED_INTEGRATOR,
        "paired_n_steps": args.n_steps,
        "reference_integrator": "adaptive_dopri5_rtol_1e-6_t0_0.0001_tf_0.9999",
        "initial_noise_policy": runtime.INITIAL_NOISE_POLICY,
        "conditioning_report": conditioning_report,
        "model_load_report": model_report,
        "records": records,
    }
    with (args.output_dir / "solver_baseline_audit.json").open("w") as handle:
        json.dump(summary, handle, indent=2)


if __name__ == "__main__":
    main()
