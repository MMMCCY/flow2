#!/usr/bin/env python3
"""Compute the retrospective Stage 10R mechanism diagnostic addendum.

This program never samples Flow and never calls the seismic inversion.  It
reconstructs the exact pre-inversion acoustic priors from the frozen Stage9A
hard models and applies the already-frozen Stage10 class-probability bridge.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import subprocess
import sys
from typing import Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import Rectangle
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.stats import rankdata
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from geology_io_utils import connected_components_3d
from guidance.geophysical_probability_bridge import (
    scalar_gaussian_sample_bridge,
    validate_probabilities,
)
from guidance.prior_ensemble import file_sha256
from guidance.seismic import (
    hard_labels_to_acoustic,
    overwrite_exact_condition_acoustic,
    tensor_sha256,
)
from guidance.seismic_inversion import labels_to_clean_prior_acoustic
from scripts.stage10.build_probability_bridge import _load_fixed_prior_members
from scripts.stage10.common import (
    EXPERIMENT_DIR,
    inference_case_dir,
    load_frozen_config,
    load_stage10_inference_case,
    retrospective_case_dir,
    stage9_pool_dir,
    target_probability_channel,
    validate_bridge_collection,
)
from scripts.stage10.evaluate_bridge_information import binary_information_metrics
from scripts.stage9.audit_prior_truth import load_retrospective_case
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_json,
    save_tensor_x,
    utc_now,
    write_csv_x,
    write_json_x,
)


ADDENDUM_SCHEMA = "stage10r_mechanism_diagnostic_raw_v1"
DIAGNOSTIC_THRESHOLD = 0.5
TOP_MASS_FRACTION = 0.10
SLICE_Y = 42


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _relative_record(path: Path) -> dict[str, object]:
    return file_record(path, relative_to=REPOSITORY_ROOT)


def _trimmed_channel(value: torch.Tensor, channel: int) -> torch.Tensor:
    return value[:, channel : channel + 1].clone().contiguous()


def _force_known_air(
    probabilities: torch.Tensor,
    subsurface: torch.Tensor,
    *,
    air_channel: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    result = probabilities.clone()
    outside = ~subsurface.bool()
    result = torch.where(outside.expand_as(result), torch.zeros_like(result), result)
    result[:, air_channel : air_channel + 1] = torch.where(
        outside,
        torch.ones_like(result[:, air_channel : air_channel + 1]),
        result[:, air_channel : air_channel + 1],
    )
    result = result / result.sum(dim=1, keepdim=True)
    validate_probabilities(result)
    clipped = result.clamp_min(torch.finfo(result.dtype).tiny)
    entropy = -(clipped * clipped.log()).sum(dim=1, keepdim=True)
    return result.contiguous(), entropy.contiguous()


def reconstruct_prior_bridge(
    config: Mapping[str, object],
    class_model: Mapping[str, object],
    case_id: str,
) -> dict[str, torch.Tensor | list[dict[str, object]]]:
    """Rebuild q_prior and P_prior without accepting truth or running inversion."""
    _, tensors = load_stage10_inference_case(config, case_id)
    indices = [int(value) for value in config["property_inversion"]["prior_candidate_indices"]]
    labels, candidate_records, _ = _load_fixed_prior_members(
        stage9_pool_dir(config, case_id), indices
    )
    subsurface = tensors["subsurface_mask"].bool()
    condition_values = tensors["condition_values"].long()
    condition_mask = tensors["condition_mask"].bool()
    property_table = tensors["acoustic_property_table"].float()
    condition_acoustic = hard_labels_to_acoustic(condition_values, property_table)
    samples = []
    cleanup_records = []
    for member_id, labels_member in enumerate(labels):
        acoustic, cleanup = labels_to_clean_prior_acoustic(
            labels_member.unsqueeze(0), property_table, subsurface
        )
        exact = overwrite_exact_condition_acoustic(
            acoustic, condition_acoustic, condition_mask
        ).float()
        if bool(
            (
                (exact != condition_acoustic)
                & condition_mask.expand_as(exact)
            ).any()
        ):
            raise RuntimeError("pre-inversion prior failed exact property conditions")
        samples.append(exact[:, 0:1].log().clone().contiguous())
        cleanup_records.append(
            {
                "member_id": member_id,
                "candidate_index": indices[member_id],
                **cleanup,
            }
        )
    property_samples = torch.cat(samples, dim=0).float().contiguous()
    probabilities, _ = scalar_gaussian_sample_bridge(property_samples, class_model)
    air_channel = target_probability_channel(class_model, -1)
    probabilities, entropy = _force_known_air(
        probabilities, subsurface, air_channel=air_channel
    )
    target_channel = target_probability_channel(class_model, int(config["target_label"]))
    return {
        "property_samples": property_samples,
        "property_mean": property_samples.mean(dim=0, keepdim=True).contiguous(),
        "property_uncertainty": property_samples.std(
            dim=0, unbiased=False, keepdim=True
        ).contiguous(),
        "probability_all_classes": probabilities,
        "probability_label9": _trimmed_channel(probabilities, target_channel),
        "entropy": entropy,
        "candidate_records": candidate_records,
        "cleanup_records": cleanup_records,
    }


def _spatial_overlap(
    probability: torch.Tensor,
    truth: torch.Tensor,
    mask: torch.Tensor,
    *,
    threshold: float,
) -> tuple[float, float]:
    predicted = (probability >= float(threshold)) & mask.bool()
    target = truth.bool() & mask.bool()
    intersection = int((predicted & target).sum().item())
    union = int((predicted | target).sum().item())
    total = int(predicted.sum().item()) + int(target.sum().item())
    return (
        intersection / union if union else 0.0,
        2.0 * intersection / total if total else 0.0,
    )


def _pearson(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    x_centered = x - x.mean()
    y_centered = y - y.mean()
    denominator = np.linalg.norm(x_centered) * np.linalg.norm(y_centered)
    return float(np.dot(x_centered, y_centered) / denominator) if denominator else 0.0


def _spearman(first: np.ndarray, second: np.ndarray) -> float:
    return _pearson(rankdata(np.asarray(first).reshape(-1)), rankdata(np.asarray(second).reshape(-1)))


def _cosine(first: np.ndarray, second: np.ndarray) -> float:
    x = np.asarray(first, dtype=np.float64).reshape(-1)
    y = np.asarray(second, dtype=np.float64).reshape(-1)
    denominator = np.linalg.norm(x) * np.linalg.norm(y)
    return float(np.dot(x, y) / denominator) if denominator else 0.0


def _weighted_centroid(probability: np.ndarray, mask: np.ndarray) -> tuple[float, float, float]:
    weights = np.where(mask, probability, 0.0).astype(np.float64)
    total = weights.sum()
    if total <= 0:
        return (float("nan"),) * 3
    coordinates = np.indices(weights.shape, dtype=np.float64)
    return tuple(float((coordinates[axis] * weights).sum() / total) for axis in range(3))


def _top_mass_centroid(
    probability: np.ndarray,
    mask: np.ndarray,
    *,
    mass_fraction: float,
) -> tuple[tuple[float, float, float], int]:
    flat_probability = np.where(mask, probability, 0.0).reshape(-1).astype(np.float64)
    total = flat_probability.sum()
    if total <= 0:
        return (float("nan"),) * 3, 0
    order = np.argsort(-flat_probability, kind="stable")
    cumulative = np.cumsum(flat_probability[order])
    count = int(np.searchsorted(cumulative, float(mass_fraction) * total, side="left")) + 1
    selected = np.unravel_index(order[:count], probability.shape)
    weights = flat_probability[order[:count]]
    centroid = tuple(
        float(np.average(np.asarray(selected[axis], dtype=np.float64), weights=weights))
        for axis in range(3)
    )
    return centroid, count


def _map_similarity(
    first: torch.Tensor,
    second: torch.Tensor,
    mask: torch.Tensor,
) -> dict[str, float]:
    selected = mask.bool()
    x = first[selected].cpu().numpy().astype(np.float64)
    y = second[selected].cpu().numpy().astype(np.float64)
    return {
        "pearson": _pearson(x, y),
        "spearman": _spearman(x, y),
        "cosine": _cosine(x, y),
        "mean_absolute_difference": float(np.mean(np.abs(x - y))),
        "rms_difference": float(np.sqrt(np.mean(np.square(x - y)))),
    }


def _bernoulli_divergences(
    prior: torch.Tensor,
    post: torch.Tensor,
    mask: torch.Tensor,
    *,
    eps: float = 1e-7,
) -> dict[str, float]:
    p = prior[mask.bool()].double().clamp(eps, 1.0 - eps)
    q = post[mask.bool()].double().clamp(eps, 1.0 - eps)

    def kl(first: torch.Tensor, second: torch.Tensor) -> torch.Tensor:
        return first * torch.log(first / second) + (1.0 - first) * torch.log(
            (1.0 - first) / (1.0 - second)
        )

    midpoint = 0.5 * (p + q)
    return {
        "kl_prior_to_post_bernoulli_mean": float(kl(p, q).mean().item()),
        "kl_post_to_prior_bernoulli_mean": float(kl(q, p).mean().item()),
        "js_bernoulli_mean": float(
            (0.5 * kl(p, midpoint) + 0.5 * kl(q, midpoint)).mean().item()
        ),
    }


def _truth_centroid(mask: torch.Tensor) -> np.ndarray:
    coordinates = torch.nonzero(mask, as_tuple=False).double()
    if not len(coordinates):
        raise ValueError("truth target is empty")
    return coordinates.mean(dim=0).numpy()


def _component_summary(mask: torch.Tensor) -> dict[str, object]:
    components = connected_components_3d(mask.bool())
    volumes = sorted((int(item["voxel_count"]) for item in components), reverse=True)
    return {
        "count": len(components),
        "volumes_descending": volumes,
        "largest_fraction": volumes[0] / int(mask.sum().item()) if volumes else 0.0,
    }


def _body_centroids_and_volumes(body_masks: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    centroids = []
    volumes = []
    for body in body_masks.bool():
        coordinates = torch.nonzero(body, as_tuple=False).double()
        if not len(coordinates):
            raise ValueError("native truth body is empty")
        centroids.append(coordinates.mean(dim=0).numpy())
        volumes.append(int(body.sum().item()))
    return np.asarray(centroids), np.asarray(volumes, dtype=np.float64)


def _truth_geometry_row(
    first_id: str,
    second_id: str,
    first_truth: torch.Tensor,
    second_truth: torch.Tensor,
    first_bodies: torch.Tensor,
    second_bodies: torch.Tensor,
) -> dict[str, object]:
    first = first_truth[0, 0].bool()
    second = second_truth[0, 0].bool()
    intersection = int((first & second).sum().item())
    union = int((first | second).sum().item())
    first_volume = int(first.sum().item())
    second_volume = int(second.sum().item())
    centroid_distance = float(
        np.linalg.norm(_truth_centroid(first) - _truth_centroid(second))
    )
    first_centroids, first_volumes = _body_centroids_and_volumes(first_bodies)
    second_centroids, second_volumes = _body_centroids_and_volumes(second_bodies)
    distances = np.linalg.norm(
        first_centroids[:, None, :] - second_centroids[None, :, :], axis=2
    )
    first_match, second_match = linear_sum_assignment(distances)
    matched_distances = distances[first_match, second_match]
    matched_volume_ratio = np.minimum(
        first_volumes[first_match], second_volumes[second_match]
    ) / np.maximum(first_volumes[first_match], second_volumes[second_match])
    body_ious = []
    for left, right in zip(first_match, second_match):
        body_a = first_bodies[left].bool()
        body_b = second_bodies[right].bool()
        body_union = int((body_a | body_b).sum().item())
        body_ious.append(int((body_a & body_b).sum().item()) / body_union)
    first_components = _component_summary(first)
    second_components = _component_summary(second)
    return {
        "truth_case_i": first_id,
        "truth_case_j": second_id,
        "label9_iou": intersection / union if union else 0.0,
        "centroid_distance_voxels": centroid_distance,
        "volume_i": first_volume,
        "volume_j": second_volume,
        "volume_ratio_min_over_max": min(first_volume, second_volume)
        / max(first_volume, second_volume),
        "component_count_i": first_components["count"],
        "component_count_j": second_components["count"],
        "largest_component_fraction_i": first_components["largest_fraction"],
        "largest_component_fraction_j": second_components["largest_fraction"],
        "matched_body_centroid_distance_mean": float(matched_distances.mean()),
        "matched_body_centroid_distance_max": float(matched_distances.max()),
        "matched_body_volume_ratio_mean": float(matched_volume_ratio.mean()),
        "matched_body_iou_mean": float(np.mean(body_ious)),
        "matched_body_iou_min": float(np.min(body_ious)),
        "body_matching": "minimum-total-centroid-distance Hungarian assignment",
    }


def _save_figure(fig: plt.Figure, root: Path, name: str) -> dict[str, dict[str, object]]:
    records = {}
    for extension in ("pdf", "svg", "png"):
        path = root / f"{name}.{extension}"
        kwargs: dict[str, object] = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs.update(dpi=600, metadata={"Software": "Stage10R diagnostic"})
        elif extension == "pdf":
            kwargs["metadata"] = {
                "Creator": "Stage10R diagnostic",
                "CreationDate": None,
                "ModDate": None,
            }
        else:
            kwargs["metadata"] = {
                "Creator": "Stage10R diagnostic",
                "Date": "2000-01-01T00:00:00Z",
            }
        fig.savefig(path, **kwargs)
        records[extension] = file_record(path, relative_to=root)
    plt.close(fig)
    return records


def _transfer_figure(
    ap_matrix: np.ndarray,
    case_ids: Sequence[str],
    output_root: Path,
) -> dict[str, dict[str, object]]:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 8,
            "axes.titlesize": 9,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, ax = plt.subplots(figsize=(4.25, 3.55), constrained_layout=True)
    image = ax.imshow(ap_matrix, cmap="YlOrBr", vmin=0.0, vmax=float(ap_matrix.max()))
    labels = [value.replace("native_seed", "") for value in case_ids]
    ax.set_xticks(range(3), labels=[f"truth {value}" for value in labels])
    ax.set_yticks(range(3), labels=[f"bridge {value}" for value in labels])
    ax.set_xlabel("Retrospective truth case")
    ax.set_ylabel("Frozen post-seismic bridge source")
    ax.set_title("Stage 10R all-by-all label-9 AUPRC")
    threshold = 0.55 * float(ap_matrix.max())
    for row in range(3):
        for column in range(3):
            ax.text(
                column,
                row,
                f"{ap_matrix[row, column]:.3f}",
                ha="center",
                va="center",
                color="white" if ap_matrix[row, column] > threshold else "#222222",
                fontweight="bold" if row == column else "normal",
            )
            if row == column:
                ax.add_patch(
                    Rectangle(
                        (column - 0.48, row - 0.48),
                        0.96,
                        0.96,
                        fill=False,
                        edgecolor="#1F4E79",
                        linewidth=1.5,
                    )
                )
    colorbar = fig.colorbar(image, ax=ax, shrink=0.86)
    colorbar.set_label("Average precision")
    ax.text(
        0.0,
        -0.19,
        "Blue outline: matched bridge/truth case. Retrospective diagnostic only; Stage10 remains FAIL.",
        transform=ax.transAxes,
        fontsize=6.8,
        color="#555555",
    )
    return _save_figure(fig, output_root, "stage10r_transfer_matrix")


def _prior_post_figure(
    config: Mapping[str, object],
    priors: Mapping[str, Mapping[str, torch.Tensor]],
    posts: Mapping[str, Mapping[str, torch.Tensor]],
    truths: Mapping[str, torch.Tensor],
    metrics: Mapping[str, Mapping[str, float]],
    output_root: Path,
) -> dict[str, dict[str, object]]:
    differences = [
        (
            posts[case_id]["probability_label9"]
            - priors[case_id]["probability_label9"]
        )
        .numpy()
        .reshape(-1)
        for case_id in config["case_ids"]
    ]
    difference_limit = float(
        np.quantile(np.abs(np.concatenate(differences)), 0.995)
    )
    difference_limit = max(difference_limit, 1e-6)
    probability_cmap = LinearSegmentedColormap.from_list(
        "stage10r_probability", ["#F5F5F5", "#F6C58F", "#D95F02"]
    )
    truth_cmap = LinearSegmentedColormap.from_list(
        "stage10r_truth", ["#F5F5F5", "#D95F02"]
    )
    fig, axes = plt.subplots(3, 4, figsize=(7.0, 5.3), constrained_layout=True)
    diff_image = None
    probability_image = None
    for row, case_id in enumerate(config["case_ids"]):
        prior = priors[case_id]["probability_label9"][0, 0].numpy()
        post = posts[case_id]["probability_label9"][0, 0].numpy()
        truth = truths[case_id][0, 0].numpy()
        prior_projection = prior.max(axis=1)
        post_projection = post.max(axis=1)
        difference_projection = post_projection - prior_projection
        truth_projection = truth.any(axis=1)
        probability_image = axes[row, 0].imshow(
            prior_projection.T,
            origin="lower",
            interpolation="nearest",
            cmap=probability_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        axes[row, 1].imshow(
            post_projection.T,
            origin="lower",
            interpolation="nearest",
            cmap=probability_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        diff_image = axes[row, 2].imshow(
            difference_projection.T,
            origin="lower",
            interpolation="nearest",
            cmap="RdBu_r",
            vmin=-difference_limit,
            vmax=difference_limit,
        )
        axes[row, 3].imshow(
            truth_projection.T,
            origin="lower",
            interpolation="nearest",
            cmap=truth_cmap,
            vmin=0.0,
            vmax=1.0,
        )
        row_metrics = metrics[case_id]
        axes[row, 0].set_ylabel(
            case_id.replace("native_seed", "seed ")
            + f"\n$\\Delta$AP={row_metrics['delta_ap_seismic']:+.3f}"
            + f"  $\\Delta$Brier={row_metrics['delta_brier_seismic']:+.3f}"
        )
        for axis in axes[row]:
            axis.set_xticks((0, 32, 63))
            axis.set_yticks((0, 32, 63))
            axis.set_xlabel("x index")
    titles = (
        "(a) Prior-only $P_9$",
        "(b) Post-seismic $P_9$",
        "(c) Post $-$ prior",
        "(d) Truth label 9\n(retrospective only)",
    )
    for axis, title in zip(axes[0], titles):
        axis.set_title(title)
    probability_colorbar = fig.colorbar(
        probability_image, ax=axes[:, 0:2], orientation="horizontal", shrink=0.72, pad=0.03
    )
    probability_colorbar.set_label("Label-9 probability")
    difference_colorbar = fig.colorbar(
        diff_image, ax=axes[:, 2], orientation="horizontal", shrink=0.78, pad=0.03
    )
    difference_colorbar.set_label("Probability change")
    fig.suptitle(
        "Stage 10R prior vs post-seismic bridge — maximum projection over y; common scales",
        fontsize=9,
    )
    return _save_figure(fig, output_root, "stage10r_prior_vs_post")


def main() -> None:
    config = load_frozen_config()
    original_decision = read_json(EXPERIMENT_DIR / "reports/STAGE10_MACHINE_DECISION.json")
    if original_decision.get("machine_decision") != "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION":
        raise RuntimeError("Stage10 original machine decision changed")
    if any(
        bool(original_decision.get(field))
        for field in ("stage10b_executed", "stage10c_executed", "stage10d_executed")
    ) or int(original_decision.get("flow_forward_count_stage10", -1)) != 0:
        raise RuntimeError("Stage10 frozen stop state changed")
    posts_validated = validate_bridge_collection(config)
    posts = {case_id: values for case_id, (_, values) in posts_validated.items()}
    class_model_path = EXPERIMENT_DIR / "configs/petrophysical_class_model.json"
    class_model = read_json(class_model_path)
    addendum = EXPERIMENT_DIR / "diagnostic_addendum"
    staging = create_staging_directory(addendum)

    # Reconstruct and save every inference-only prior bridge before opening truth.
    priors: dict[str, dict[str, torch.Tensor]] = {}
    prior_manifests = {}
    for case_id in config["case_ids"]:
        reconstructed = reconstruct_prior_bridge(config, class_model, case_id)
        tensor_values = {
            name: value
            for name, value in reconstructed.items()
            if isinstance(value, torch.Tensor)
        }
        priors[case_id] = tensor_values
        case_root = staging / "prior_only" / case_id
        generated = {
            name: save_tensor_x(case_root / f"{name}.pt", value)
            for name, value in tensor_values.items()
        }
        for name, record in generated.items():
            record["path"] = str(
                (case_root / f"{name}.pt").relative_to(staging)
            )
        prior_manifest = {
            "schema": "stage10r_prior_only_bridge_v1",
            "status": "complete_before_retrospective_truth_load",
            "case_id": case_id,
            "candidate_indices": config["property_inversion"]["prior_candidate_indices"],
            "candidate_records": reconstructed["candidate_records"],
            "cleanup_records": reconstructed["cleanup_records"],
            "new_flow_sampling": False,
            "seismic_inversion_called": False,
            "truth_tensor_received": False,
            "class_model": _relative_record(class_model_path),
            "generated_tensors": generated,
        }
        write_json_x(case_root / "manifest.json", prior_manifest)
        prior_manifests[case_id] = file_record(
            case_root / "manifest.json", relative_to=staging
        )
    write_json_x(
        staging / "prior_only/manifest.json",
        {
            "schema": "stage10r_prior_only_bridge_collection_v1",
            "status": "complete_before_retrospective_truth_load",
            "case_manifests": prior_manifests,
            "truth_tensor_received": False,
            "seismic_inversion_called": False,
            "flow_forward_count": 0,
        },
    )

    # Retrospective truth may be loaded only after all prior-only maps are frozen.
    truths = {}
    body_masks = {}
    evaluation_masks = {}
    retrospective_records = {}
    for case_id in config["case_ids"]:
        _, inference = load_stage10_inference_case(config, case_id)
        retrospective_manifest, retrospective = load_retrospective_case(
            retrospective_case_dir(config, case_id), expected_case_id=case_id
        )
        if retrospective_manifest["inference_manifest_sha256"] != file_sha256(
            inference_case_dir(config, case_id) / "manifest.json"
        ):
            raise ValueError("retrospective truth is not linked to inference case")
        truths[case_id] = retrospective["truth_labels"].long() == int(config["target_label"])
        body_masks[case_id] = retrospective["native_body_masks"].bool()
        evaluation_masks[case_id] = (
            inference["subsurface_mask"].bool() & ~inference["condition_mask"].bool()
        )
        retrospective_records[case_id] = _relative_record(
            retrospective_case_dir(config, case_id) / "manifest.json"
        )

    case_ids = list(config["case_ids"])
    ap_matrix = np.zeros((3, 3), dtype=np.float64)
    brier_matrix = np.zeros((3, 3), dtype=np.float64)
    transfer_rows = []
    for bridge_index, bridge_case in enumerate(case_ids):
        probability = posts[bridge_case]["probability_label9"]
        for truth_index, truth_case in enumerate(case_ids):
            metrics = binary_information_metrics(
                probability, truths[truth_case], evaluation_masks[truth_case]
            )
            iou, dice = _spatial_overlap(
                probability,
                truths[truth_case],
                evaluation_masks[truth_case],
                threshold=DIAGNOSTIC_THRESHOLD,
            )
            ap_matrix[bridge_index, truth_index] = metrics["auprc"]
            brier_matrix[bridge_index, truth_index] = metrics["brier"]
            transfer_rows.append(
                {
                    "bridge_case": bridge_case,
                    "truth_case": truth_case,
                    **metrics,
                    "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
                    "threshold_spatial_iou": iou,
                    "threshold_spatial_dice": dice,
                    "is_diagonal": bridge_case == truth_case,
                    "retrospective_only": True,
                    "used_by_original_stage10_gate": False,
                }
            )
    ap_rows = [
        {
            "bridge_case": bridge_case,
            **{f"truth_{case_ids[column].replace('native_seed', '')}": ap_matrix[row, column] for column in range(3)},
        }
        for row, bridge_case in enumerate(case_ids)
    ]
    brier_rows = [
        {
            "bridge_case": bridge_case,
            **{f"truth_{case_ids[column].replace('native_seed', '')}": brier_matrix[row, column] for column in range(3)},
        }
        for row, bridge_case in enumerate(case_ids)
    ]
    write_csv_x(staging / "all_by_all_ap_matrix.csv", ap_rows)
    write_csv_x(staging / "all_by_all_brier_matrix.csv", brier_rows)
    write_csv_x(staging / "all_by_all_secondary_metrics.csv", transfer_rows)

    truth_geometry_rows = []
    for first_index in range(3):
        for second_index in range(first_index + 1, 3):
            first_id = case_ids[first_index]
            second_id = case_ids[second_index]
            truth_geometry_rows.append(
                _truth_geometry_row(
                    first_id,
                    second_id,
                    truths[first_id],
                    truths[second_id],
                    body_masks[first_id],
                    body_masks[second_id],
                )
            )
    write_csv_x(staging / "truth_geometry_similarity.csv", truth_geometry_rows)

    prior_post_rows = []
    prior_post_by_case = {}
    for case_id in case_ids:
        mask = evaluation_masks[case_id]
        truth = truths[case_id]
        prior_probability = priors[case_id]["probability_label9"]
        post_probability = posts[case_id]["probability_label9"]
        prior_metrics = binary_information_metrics(prior_probability, truth, mask)
        post_metrics = binary_information_metrics(post_probability, truth, mask)
        similarity = _map_similarity(prior_probability, post_probability, mask)
        divergences = _bernoulli_divergences(
            prior_probability, post_probability, mask
        )
        prior_entropy = priors[case_id]["entropy"][mask].double()
        post_entropy = posts[case_id]["entropy"][mask].double()
        row = {
            "case_id": case_id,
            "prior_auprc": prior_metrics["auprc"],
            "post_auprc": post_metrics["auprc"],
            "delta_ap_seismic": post_metrics["auprc"] - prior_metrics["auprc"],
            "prior_brier": prior_metrics["brier"],
            "post_brier": post_metrics["brier"],
            "delta_brier_seismic": prior_metrics["brier"] - post_metrics["brier"],
            "prior_roc_auc": prior_metrics["roc_auc"],
            "post_roc_auc": post_metrics["roc_auc"],
            "spatial_pearson_prior_post": similarity["pearson"],
            "spatial_spearman_prior_post": similarity["spearman"],
            "spatial_cosine_prior_post": similarity["cosine"],
            "mean_absolute_probability_change": similarity["mean_absolute_difference"],
            "rms_probability_change": similarity["rms_difference"],
            **divergences,
            "prior_categorical_entropy_mean": float(prior_entropy.mean().item()),
            "post_categorical_entropy_mean": float(post_entropy.mean().item()),
            "delta_categorical_entropy_post_minus_prior": float(
                post_entropy.mean().item() - prior_entropy.mean().item()
            ),
        }
        prior_post_rows.append(row)
        prior_post_by_case[case_id] = row
    write_csv_x(staging / "prior_vs_post_metrics.csv", prior_post_rows)

    bridge_similarity_rows = []
    centroid_rows = []
    for map_type, maps in (("prior_only", priors), ("post_seismic", posts)):
        for case_id in case_ids:
            probability = maps[case_id]["probability_label9"][0, 0].numpy()
            mask = evaluation_masks[case_id][0, 0].numpy()
            weighted = _weighted_centroid(probability, mask)
            top_centroid, top_count = _top_mass_centroid(
                probability, mask, mass_fraction=TOP_MASS_FRACTION
            )
            centroid_rows.append(
                {
                    "map_type": map_type,
                    "case_id": case_id,
                    "weighted_centroid_x": weighted[0],
                    "weighted_centroid_y": weighted[1],
                    "weighted_centroid_z": weighted[2],
                    "top_mass_fraction": TOP_MASS_FRACTION,
                    "top_mass_voxel_count": top_count,
                    "top_mass_centroid_x": top_centroid[0],
                    "top_mass_centroid_y": top_centroid[1],
                    "top_mass_centroid_z": top_centroid[2],
                }
            )
        for first_index in range(3):
            for second_index in range(first_index + 1, 3):
                first_id = case_ids[first_index]
                second_id = case_ids[second_index]
                common_mask = evaluation_masks[first_id] & evaluation_masks[second_id]
                similarity = _map_similarity(
                    maps[first_id]["probability_label9"],
                    maps[second_id]["probability_label9"],
                    common_mask,
                )
                bridge_similarity_rows.append(
                    {
                        "map_type": map_type,
                        "case_i": first_id,
                        "case_j": second_id,
                        "voxel_count": int(common_mask.sum().item()),
                        **similarity,
                    }
                )
    write_csv_x(staging / "bridge_similarity.csv", bridge_similarity_rows)
    write_csv_x(staging / "bridge_top_mass_centroids.csv", centroid_rows)

    diagonal = np.diag(ap_matrix)
    off_diagonal = ap_matrix[~np.eye(3, dtype=bool)]
    transfer_summary = {
        "diagonal_mean_auprc": float(diagonal.mean()),
        "off_diagonal_mean_auprc": float(off_diagonal.mean()),
        "diagonal_minus_off_diagonal_mean_auprc": float(
            diagonal.mean() - off_diagonal.mean()
        ),
        "fraction_diagonal_is_row_maximum": float(
            np.mean([ap_matrix[index, index] == ap_matrix[index].max() for index in range(3)])
        ),
        "fraction_diagonal_is_column_maximum": float(
            np.mean([ap_matrix[index, index] == ap_matrix[:, index].max() for index in range(3)])
        ),
        "diagnostic_threshold": DIAGNOSTIC_THRESHOLD,
        "threshold_role": "retrospective visualization/overlap only; not an original Stage10 gate",
    }
    figures = {}
    figures["transfer_matrix"] = _transfer_figure(ap_matrix, case_ids, staging)
    figures["prior_vs_post"] = _prior_post_figure(
        config, priors, posts, truths, prior_post_by_case, staging
    )
    source_paths = (
        EXPERIMENT_DIR / "reports/STAGE10_REPORT.md",
        EXPERIMENT_DIR / "reports/STAGE10_MACHINE_DECISION.json",
        EXPERIMENT_DIR / "audit/leakage_audit.json",
        EXPERIMENT_DIR / "audit/property_inversion_provenance.json",
        EXPERIMENT_DIR / "bridge/manifest.json",
        EXPERIMENT_DIR / "controls/manifest.json",
        EXPERIMENT_DIR / "diagnostics/stage10a_decision.json",
        EXPERIMENT_DIR / "configs/petrophysical_class_model.json",
        Path(__file__),
    )
    raw_summary = {
        "schema": ADDENDUM_SCHEMA,
        "status": "computed_awaiting_frozen_interpretation",
        "original_stage10_machine_decision": original_decision["machine_decision"],
        "original_stage10_decision_unchanged": True,
        "stage10r_flow_forward_count": 0,
        "stage10r_seismic_inversion_count": 0,
        "stage10_bcd_executed": False,
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_short": _git("status", "--short"),
        "case_ids": case_ids,
        "transfer_summary": transfer_summary,
        "prior_vs_post": prior_post_by_case,
        "truth_geometry_pairwise": truth_geometry_rows,
        "bridge_similarity_pairwise": bridge_similarity_rows,
        "top_mass_centroids": centroid_rows,
        "truth_loaded_only_after_prior_only_maps_written": True,
        "probability_model_reused_without_change": True,
        "source_artifacts": [_relative_record(path) for path in source_paths],
        "retrospective_case_manifests": retrospective_records,
        "figures": figures,
        "completed_at_utc": utc_now(),
    }
    write_json_x(staging / "raw_summary.json", raw_summary)
    write_json_x(
        staging / "manifest.json",
        {
            "schema": "stage10r_diagnostic_manifest_v1",
            "status": "complete_raw_diagnostics",
            "original_stage10_machine_decision": original_decision["machine_decision"],
            "original_stage10_files_modified": False,
            "flow_forward_count": 0,
            "seismic_inversion_count": 0,
            "generated_files": [
                str(path.relative_to(staging))
                for path in sorted(staging.rglob("*"))
                if path.is_file()
            ],
        },
    )
    publish_staging_directory(staging, addendum)
    print(
        json.dumps(
            {
                "status": "COMPLETE_RAW_DIAGNOSTICS",
                "transfer": transfer_summary,
                "delta_ap": {
                    case_id: prior_post_by_case[case_id]["delta_ap_seismic"]
                    for case_id in case_ids
                },
                "delta_brier": {
                    case_id: prior_post_by_case[case_id]["delta_brier_seismic"]
                    for case_id in case_ids
                },
            }
        )
    )


if __name__ == "__main__":
    main()
