#!/usr/bin/env python3
"""Build one explicitly post-hoc Stage15-G illustrative validation case."""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.binary_inversion_logistic import (
    apply_linear_probability,
    binary_inversion_features,
    coarse_support_count_8,
)
from guidance.inversion_score_probability import upsample_inversion_score
from guidance.seismic import tensor_sha256
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage15.run_inversion_score_probability import generate_case


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
CALIBRATION_DIR = EXPERIMENT_ROOT / "binary_logistic/calibration_n128_8x8x8_v2"
SOURCE_DIR = EXPERIMENT_ROOT / "inversion_probability/calibration_n128_8x8x8_v1"
OUTPUT_DIR = EXPERIMENT_ROOT / "reports/binary_logistic_showcase_seed15180114_v1"


def select_showcase(rows: list[dict[str, str]]) -> dict[str, str]:
    """Post-hoc rule: positive validation case with largest AP-prevalence gain."""
    positive = [row for row in rows if int(row["label9_voxels"]) > 0]
    if not positive:
        raise ValueError("no positive validation case is available")
    return max(positive, key=lambda row: float(row["auprc"]) - float(row["prevalence"]))


def projection(volume: torch.Tensor, axis: int, mode: str) -> torch.Tensor:
    if mode == "sum":
        result = volume.sum(dim=axis)
        return result / result.max().clamp_min(1)
    if mode == "max":
        return volume.amax(dim=axis)
    raise ValueError(mode)


def main() -> None:
    refuse_nonempty(OUTPUT_DIR)
    with (CALIBRATION_DIR / "validation_metrics.csv").open(
        encoding="utf-8", newline=""
    ) as handle:
        rows = list(csv.DictReader(handle))
    selected = select_showcase(rows)
    seed = int(selected["root_seed"])
    case_index = int(selected["case_index"])
    if seed != 15180114:
        raise RuntimeError(f"frozen post-hoc rule selected unexpected seed {seed}")

    source_manifest = read_json(SOURCE_DIR / "calibration_manifest.json")
    source_record = source_manifest["case_records"][case_index]
    q_path = SOURCE_DIR / "cases" / f"case_{case_index:03d}" / "coarse_inversion_score.pt"
    q = runtime.load_tensor(q_path).float()
    if tensor_sha256(q) != source_record["coarse_inversion_score_tensor_sha256"]:
        raise ValueError("showcase inversion score hash mismatch")
    geology, subsurface, _ = generate_case(seed)
    if tensor_sha256(geology) != source_record["geology_tensor_sha256"]:
        raise ValueError("showcase geology replay hash mismatch")
    binary_truth = ((geology == 9) & subsurface).bool()

    checkpoint_path = CALIBRATION_DIR / "binary_logistic_checkpoint.pt"
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    support_count = coarse_support_count_8(subsurface)
    features = binary_inversion_features(q, support_count)
    probability_values = apply_linear_probability(
        features,
        checkpoint["feature_mean"],
        checkpoint["feature_std"],
        checkpoint["linear_weight"],
        checkpoint["linear_bias"],
    )
    domain = support_count[0, 0] > 0
    coarse_probability = torch.zeros((1, 1, 8, 8, 8))
    coarse_probability[0, 0][domain] = probability_values[domain]
    probability = upsample_inversion_score(coarse_probability)
    probability = torch.where(subsurface, probability, torch.zeros_like(probability))

    OUTPUT_DIR.mkdir(parents=True)
    torch.save(probability, OUTPUT_DIR / "label9_probability_volume.pt")
    truth3 = binary_truth[0, 0].float()
    probability3 = probability[0, 0]
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    labels = ("Projection along X", "Projection along Y", "Projection along Z")
    for column, axis in enumerate((0, 1, 2)):
        truth_projection = projection(truth3, axis, "sum")
        probability_projection = projection(probability3, axis, "max")
        axes[0, column].imshow(truth_projection.T, origin="lower", cmap="Reds", vmin=0, vmax=1)
        axes[0, column].set_title(f"Binary label9 truth\n{labels[column]}")
        image = axes[1, column].imshow(
            probability_projection.T,
            origin="lower",
            cmap="viridis",
            vmin=0,
            vmax=float(probability3.max()),
        )
        axes[1, column].contour(
            (truth_projection > 0).T.numpy(), levels=[0.5], colors="white", linewidths=0.8
        )
        axes[1, column].set_title("Predicted P(label9); white = truth support")
        for row in range(2):
            axes[row, column].set_xticks([])
            axes[row, column].set_yticks([])
    fig.colorbar(image, ax=axes[1, :], shrink=0.75, label="P(label9)")
    fig.suptitle(
        "Stage15-G post-hoc illustrative validation case — seed 15180114",
        fontsize=14,
    )
    figure_path = OUTPUT_DIR / "binary_logistic_showcase.png"
    fig.savefig(figure_path, dpi=200)
    plt.close(fig)

    stage_f_rows = {}
    with (
        EXPERIMENT_ROOT
        / "inversion_probability/calibration_n128_8x8x8_v1/validation_metrics.csv"
    ).open(encoding="utf-8", newline="") as handle:
        stage_f_rows = {int(row["root_seed"]): row for row in csv.DictReader(handle)}
    stage_f = stage_f_rows[seed]
    summary = {
        "schema": "stage15_g_posthoc_illustrative_case_v1",
        "case_role": "post_hoc_illustrative_validation_case_not_generalization_evidence",
        "selection_rule": "among label9-positive frozen validation cases, maximize Stage15-G AUPRC minus own prevalence",
        "case_index": case_index,
        "root_seed": seed,
        "label9_voxels": int(selected["label9_voxels"]),
        "prevalence": float(selected["prevalence"]),
        "stage15_f_auprc": float(stage_f["auprc"]),
        "stage15_g_auprc": float(selected["auprc"]),
        "stage15_g_auprc_minus_prevalence": float(selected["auprc"])
        - float(selected["prevalence"]),
        "stage15_g_truth_mean_probability": float(selected["truth_mean_probability"]),
        "stage15_g_background_mean_probability": float(selected["background_mean_probability"]),
        "stage15_g_truth_minus_background_probability": float(
            selected["truth_mean_probability"]
        )
        - float(selected["background_mean_probability"]),
        "probability_range_subsurface": [
            float(probability[subsurface].min()),
            float(probability[subsurface].max()),
        ],
        "truth_used_for_case_selection_and_visual_overlay": True,
        "flow_used": False,
        "new_training_or_inversion_performed": False,
        "inputs": {
            "checkpoint": runtime.asset_record(checkpoint_path),
            "coarse_inversion_score": runtime.asset_record(q_path),
        },
        "outputs": {
            "figure": runtime.asset_record(figure_path),
            "probability_volume": runtime.asset_record(
                OUTPUT_DIR / "label9_probability_volume.pt"
            ),
        },
    }
    write_json(OUTPUT_DIR / "summary.json", summary)
    (OUTPUT_DIR / "SHOWCASE.md").write_text(
        f"""# Stage15-G illustrative binary result

This is an explicitly **post-hoc selected validation example**, not a held-out generalization claim and not a Phase1-equivalent success claim. The fixed selection rule chose seed `{seed}` because it has the largest absolute `AUPRC - prevalence` among the five label9-positive Stage15-G validation cases.

- Binary target: label9 versus background
- Label9 voxels: {summary['label9_voxels']}
- Natural prevalence: {summary['prevalence']:.6f}
- Stage15-F global-histogram AUPRC: {summary['stage15_f_auprc']:.6f}
- Stage15-G linear-mapper AUPRC: {summary['stage15_g_auprc']:.6f}
- AUPRC minus prevalence: {summary['stage15_g_auprc_minus_prevalence']:.6f}
- Truth/background mean P9: {summary['stage15_g_truth_mean_probability']:.6f} / {summary['stage15_g_background_mean_probability']:.6f}

The figure shows that the frozen binary seismic inversion plus the simple linear mapper can highlight part of the true label9 region. It supports feasibility of the inversion-to-label9 bridge in one favorable case; it does not establish robust performance across cases or justify Flow guidance yet.
""",
        encoding="utf-8",
    )
    manifest = base_manifest("stage15_g_posthoc_showcase_run_v1", Path(__file__))
    manifest.update(
        {
            "run_status": "completed",
            "case_role": summary["case_role"],
            "selection_rule": summary["selection_rule"],
            "summary": runtime.asset_record(OUTPUT_DIR / "summary.json"),
            "figure": runtime.asset_record(figure_path),
        }
    )
    write_json(OUTPUT_DIR / "manifest.json", manifest)


if __name__ == "__main__":
    main()
