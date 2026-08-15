#!/usr/bin/env python3
"""Retrospective paired audit of label-9 omission and commission correction.

The script consumes only saved truth/baseline/guided tensors.  It does not run
Flow inference or alter any experiment artifact.
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage


PROJECT = Path(__file__).resolve().parents[2]
ROOT = PROJECT.parents[1]
OUT = PROJECT / "experiments/stage15_binary_seismic_consensus/reports/prior_error_correction_v1"
TRUTH = PROJECT / "samples/jupyter-demo/cond_generation_0/true_model.pt"
P1 = PROJECT / "experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4/calibrated_reference_windowed/seed42_n4_s32"
P2 = PROJECT / "experiments/stage2_property/runs/cond_generation_0/ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n4_s32_a025_c025"
S15 = PROJECT / "experiments/stage15_binary_seismic_consensus/trace_boundary"


def load(path: Path) -> np.ndarray:
    value = torch.load(path, map_location="cpu", weights_only=True)
    return value.detach().cpu().numpy().squeeze().astype(np.int16, copy=False)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def components(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, structure=ndimage.generate_binary_structure(3, 1))[1])


def audit(truth: np.ndarray, baseline: np.ndarray, guided: np.ndarray) -> dict[str, float | int]:
    domain = truth != -1
    target = (truth == 9) & domain
    base = (baseline == 9) & domain
    pred = (guided == 9) & domain
    base_tp = target & base
    base_fn = target & ~base
    base_fp = ~target & base & domain
    recovered_fn = base_fn & pred
    residual_fn = base_fn & ~pred
    preserved_tp = base_tp & pred
    lost_tp = base_tp & ~pred
    removed_fp = base_fp & ~pred
    retained_fp = base_fp & pred
    new_fp = ~target & ~base & pred & domain
    tp = target & pred
    fp = ~target & pred & domain
    fn = target & ~pred
    return {
        "truth_target_voxels": int(target.sum()),
        "baseline_target_voxels": int(base.sum()),
        "guided_target_voxels": int(pred.sum()),
        "baseline_tp": int(base_tp.sum()),
        "baseline_fn": int(base_fn.sum()),
        "baseline_fp": int(base_fp.sum()),
        "recovered_fn": int(recovered_fn.sum()),
        "residual_fn": int(residual_fn.sum()),
        "preserved_tp": int(preserved_tp.sum()),
        "lost_tp": int(lost_tp.sum()),
        "removed_fp": int(removed_fp.sum()),
        "retained_fp": int(retained_fp.sum()),
        "new_fp": int(new_fp.sum()),
        "fn_recovery_rate": float(recovered_fn.sum() / max(base_fn.sum(), 1)),
        "fp_removal_rate": float(removed_fp.sum() / max(base_fp.sum(), 1)),
        "tp_preservation_rate": float(preserved_tp.sum() / max(base_tp.sum(), 1)),
        "new_fp_per_recovered_fn": float(new_fp.sum() / max(recovered_fn.sum(), 1)),
        "baseline_iou": float(base_tp.sum() / max((base_tp | base_fn | base_fp).sum(), 1)),
        "guided_iou": float(tp.sum() / max((tp | fn | fp).sum(), 1)),
        "baseline_precision": float(base_tp.sum() / max(base.sum(), 1)),
        "guided_precision": float(tp.sum() / max(pred.sum(), 1)),
        "baseline_recall": float(base_tp.sum() / max(target.sum(), 1)),
        "guided_recall": float(tp.sum() / max(target.sum(), 1)),
        "truth_components": components(target),
        "baseline_components": components(base),
        "guided_components": components(pred),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth = load(TRUTH)
    records: list[dict[str, object]] = []
    sources: list[dict[str, str]] = [{"role": "truth", "path": str(TRUTH), "sha256": sha256(TRUTH)}]
    pairs = [
        ("Phase1", 42, 0, P1 / "baseline/sample_0.pt", P1 / "alpha025/sample_0.pt"),
        ("Phase2", 42, 0, P2 / "baseline/sample_0.pt", P2 / "alpha025/sample_0.pt"),
        ("Stage15-H", 42, 0, S15 / "flow_property_seed42_v4/baseline/sample_0.pt", S15 / "flow_property_seed42_v4/guided/sample_0.pt"),
        ("Stage15-H", 142, 0, S15 / "flow_property_seed142_v4/baseline/sample_0.pt", S15 / "flow_property_seed142_v4/guided/sample_0.pt"),
        ("Stage15-H", 242, 0, S15 / "flow_property_seed242_v4/baseline/sample_0.pt", S15 / "flow_property_seed242_v4/guided/sample_0.pt"),
    ]
    for method, seed, sample, base_path, guided_path in pairs:
        baseline, guided = load(base_path), load(guided_path)
        if baseline.shape != truth.shape or guided.shape != truth.shape:
            raise ValueError(f"shape mismatch: {method} seed {seed}")
        row: dict[str, object] = {"method": method, "seed": seed, "sample_id": sample}
        row.update(audit(truth, baseline, guided))
        records.append(row)
        sources.extend(
            [
                {"role": f"{method} seed{seed} baseline", "path": str(base_path), "sha256": sha256(base_path)},
                {"role": f"{method} seed{seed} guided", "path": str(guided_path), "sha256": sha256(guided_path)},
            ]
        )
    fields = list(records[0])
    with (OUT / "paired_error_correction.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(records)
    stage15 = [row for row in records if row["method"] == "Stage15-H"]
    summary = {
        "schema": "paired_prior_error_correction_v1",
        "definition": {
            "omission": "truth label9 and baseline non-label9",
            "commission": "truth non-label9 and baseline label9",
            "fn_recovery_rate": "recovered baseline omission / baseline omission",
            "fp_removal_rate": "removed baseline commission / baseline commission",
        },
        "seed42_strict_comparison": [row for row in records if row["seed"] == 42],
        "stage15_three_seed_mean": {
            key: float(np.mean([float(row[key]) for row in stage15]))
            for key in (
                "fn_recovery_rate", "fp_removal_rate", "tp_preservation_rate",
                "new_fp_per_recovered_fn", "baseline_iou", "guided_iou",
                "baseline_components", "guided_components",
            )
        },
        "input_files": sources,
        "inference_rerun": False,
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# Paired prior-error correction audit", "", "No inference was rerun.", ""]
    for row in summary["seed42_strict_comparison"]:
        lines.append(
            f"- {row['method']}: FN recovery {100*row['fn_recovery_rate']:.2f}%, "
            f"FP removal {100*row['fp_removal_rate']:.2f}%, TP preservation "
            f"{100*row['tp_preservation_rate']:.2f}%, IoU {row['baseline_iou']:.3f}→{row['guided_iou']:.3f}."
        )
    mean = summary["stage15_three_seed_mean"]
    lines += ["", f"Stage15-H three-seed mean FN recovery: {100*mean['fn_recovery_rate']:.2f}%.", f"Stage15-H three-seed mean FP removal: {100*mean['fp_removal_rate']:.2f}%."]
    (OUT / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps(summary["seed42_strict_comparison"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
