"""Frozen conditional-Flow ensemble utilities for Stage 9A.

Inference-visible functions in this module deliberately have no geological
truth input.  Retrospective gate helpers operate only on already-computed
metric rows and cannot create or rank candidates.
"""

from __future__ import annotations

import gzip
import hashlib
import math
from pathlib import Path
from typing import Callable, Mapping, Sequence

import torch

from guidance.generator_posterior import (
    CONDITION_PROJECTION_POLICY,
    projected_fixed_euler_prior_sample,
)
from guidance.seismic import hard_labels_to_acoustic, tensor_sha256


STAGE9A_CONFIG_SCHEMA = "stage9a_flow_prior_support_config_v1"
STAGE9A_POOL_SCHEMA = "stage9a_inference_visible_candidate_pool_v1"
STAGE9A_RANKING_SCHEMA = "stage9a_inference_visible_ranking_v1"
STAGE9A_AUDIT_SCHEMA = "stage9a_retrospective_truth_audit_v1"
STAGE9A_SUMMARY_SCHEMA = "stage9a_flow_prior_support_summary_v1"
PREDICTION_CACHE_POLICY = "torch_float32_lossless_deterministic_gzip_v1"
RANKING_POLICY = "ascending_plain_hard_seismic_rmse_then_candidate_id_v1"
OBSERVATION_NAMES = ("correct", "zero", "shuffled_xy", "wrong_case")
TARGET_METRICS = (
    "label9_iou",
    "label9_recall",
    "major_component_mean_recall",
)
CORRELATION_METRICS = (
    "global_accuracy",
    "truth_present_mean_iou",
    *TARGET_METRICS,
)


def validate_protocol_config(config: Mapping[str, object]) -> dict[str, object]:
    """Validate the frozen Stage9A protocol without opening any case tensor."""
    expected = {
        "schema": STAGE9A_CONFIG_SCHEMA,
        "status": "frozen_before_stage9a_cuda_evidence",
        "model_weight_policy": "ema_trainable_raw_frozen_embedding_v1",
        "integrator": "fixed_euler_midpoint_v1",
        "condition_projection": CONDITION_PROJECTION_POLICY,
        "prediction_cache": PREDICTION_CACHE_POLICY,
        "ranking": RANKING_POLICY,
        "truth_visible_to_candidate_runner": False,
        "truth_visible_to_ranking_runner": False,
        "training_authorized": False,
        "structured_search_authorized": False,
        "gravity_authorized": False,
        "posterior_chain_authorized": False,
    }
    for field, value in expected.items():
        if config.get(field) != value:
            raise ValueError(f"Stage9A config {field} must be {value!r}")
    if int(config.get("n_euler_steps", 0)) != 32:
        raise ValueError("Stage9A requires exactly 32 Euler steps")
    if int(config.get("formal_candidates_per_case", 0)) != 1024:
        raise ValueError("formal Stage9A requires exactly 1024 candidates per case")
    if int(config.get("smoke_candidates_per_case", 0)) not in (2, 3, 4):
        raise ValueError("Stage9A smoke must use 2-4 candidates per case")
    batch_size = int(config.get("candidate_batch_size", 0))
    chunk_size = int(config.get("cache_chunk_size", 0))
    if batch_size <= 0 or chunk_size != batch_size:
        raise ValueError("candidate batch and cache chunk sizes must match and be positive")
    cases = config.get("primary_cases")
    if not isinstance(cases, list) or len(cases) != 3:
        raise ValueError("Stage9A requires exactly three primary cases")
    ids = [str(case.get("case_id")) for case in cases if isinstance(case, Mapping)]
    if len(ids) != 3 or len(set(ids)) != 3:
        raise ValueError("Stage9A primary case IDs must be unique")
    if tuple(config.get("observations", ())) != OBSERVATION_NAMES:
        raise ValueError("Stage9A observation order drifted")
    if tuple(int(value) for value in config.get("best_of_n", ())) != (
        1,
        4,
        16,
        64,
        256,
        1024,
    ):
        raise ValueError("Stage9A best-of-N schedule drifted")
    return {
        "batch_size": batch_size,
        "chunk_size": chunk_size,
        "formal_count": 1024,
        "smoke_count": int(config["smoke_candidates_per_case"]),
        "n_steps": 32,
        "case_ids": ids,
    }


def candidate_id(index: int) -> str:
    if int(index) < 0:
        raise ValueError("candidate index must be non-negative")
    return f"candidate_{int(index):06d}"


def source_seed(
    config: Mapping[str, object], *, case_index: int, candidate_index: int, mode: str
) -> int:
    if mode not in {"smoke", "formal"}:
        raise ValueError("mode must be smoke or formal")
    seed = (
        int(config["formal_source_seed_base"])
        + int(config["case_source_seed_stride"]) * int(case_index)
        + int(candidate_index)
    )
    if mode == "smoke":
        seed += int(config["smoke_source_seed_offset"])
    return seed


def gaussian_source(
    shape: Sequence[int], *, seed: int, dtype: torch.dtype = torch.float32
) -> torch.Tensor:
    """Create one order-independent standard-Gaussian CPU source point."""
    generator = torch.Generator(device="cpu").manual_seed(int(seed))
    return torch.randn(tuple(int(value) for value in shape), generator=generator, dtype=dtype)


def project_hard_conditions(
    labels: torch.Tensor,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    if labels.ndim != 5 or labels.shape[1] != 1:
        raise ValueError("labels must have shape [B,1,X,Y,Z]")
    if condition_values.ndim != 5 or condition_values.shape[1] != 1:
        raise ValueError("condition_values must have shape [1,1,X,Y,Z]")
    if condition_mask.shape != condition_values.shape:
        raise ValueError("condition mask and values must match")
    if condition_values.shape[2:] != labels.shape[2:]:
        raise ValueError("condition spatial shape must match labels")
    values = condition_values.to(device=labels.device, dtype=labels.dtype)
    mask = condition_mask.to(device=labels.device, dtype=torch.bool)
    return torch.where(mask.expand(labels.shape[0], -1, -1, -1, -1), values, labels)


def decode_projected_hard(
    model,
    final_state: torch.Tensor,
    *,
    condition_values: torch.Tensor,
    condition_mask: torch.Tensor,
) -> torch.Tensor:
    decoded = (model.decode(final_state) - 1).unsqueeze(1).long()
    return project_hard_conditions(decoded, condition_values, condition_mask)


def generate_prior_batch(
    model,
    initial_state: torch.Tensor,
    conditioning: torch.Tensor,
    embedded_conditions: torch.Tensor,
    condition_mask: torch.Tensor,
    condition_values: torch.Tensor,
    *,
    n_steps: int,
) -> torch.Tensor:
    """Canonical no-guidance Flow trajectory followed by exact hard projection."""
    final_state = projected_fixed_euler_prior_sample(
        model,
        initial_state,
        conditioning,
        embedded_conditions,
        condition_mask,
        n_steps=int(n_steps),
    )
    if not torch.isfinite(final_state).all():
        raise FloatingPointError("frozen Flow produced non-finite state")
    return decode_projected_hard(
        model,
        final_state,
        condition_values=condition_values,
        condition_mask=condition_mask,
    )


def hard_seismic_response(
    labels: torch.Tensor,
    *,
    property_table: torch.Tensor,
    subsurface_mask: torch.Tensor,
    forward_operator,
) -> torch.Tensor:
    """Stage7-compatible hard petrophysical mapping and seismic forward."""
    acoustic = hard_labels_to_acoustic(labels.long(), property_table.to(labels.device))
    mask = subsurface_mask.to(device=labels.device, dtype=torch.bool)
    if mask.shape[0] == 1 and labels.shape[0] > 1:
        mask = mask.expand(labels.shape[0], -1, -1, -1, -1)
    with torch.no_grad():
        response = forward_operator(acoustic[:, 0:1], acoustic[:, 1:2], mask)
    if response.dtype != torch.float32:
        response = response.float()
    if not torch.isfinite(response).all():
        raise FloatingPointError("hard seismic prediction is non-finite")
    return response


def save_tensor_gzip(path: Path, value: torch.Tensor) -> dict[str, object]:
    """Write one deterministic lossless gzip tensor chunk without overwrite."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = value.detach().cpu().contiguous()
    with path.open("xb") as raw:
        with gzip.GzipFile(
            filename="", fileobj=raw, mode="wb", compresslevel=6, mtime=0
        ) as compressed:
            torch.save(tensor, compressed)
    return {
        "path": path.name,
        "size_bytes": path.stat().st_size,
        "file_sha256": file_sha256(path),
        "tensor_sha256": tensor_sha256(tensor),
        "shape": list(tensor.shape),
        "dtype": str(tensor.dtype),
    }


def load_tensor_gzip(
    path: Path, *, expected: Mapping[str, object] | None = None
) -> torch.Tensor:
    path = Path(path)
    if expected is not None and file_sha256(path) != expected.get("file_sha256"):
        raise ValueError(f"compressed cache file hash mismatch: {path}")
    with path.open("rb") as raw:
        with gzip.GzipFile(filename="", fileobj=raw, mode="rb") as compressed:
            try:
                value = torch.load(compressed, map_location="cpu", weights_only=True)
            except TypeError:
                value = torch.load(compressed, map_location="cpu")
    if not isinstance(value, torch.Tensor):
        raise TypeError(f"compressed cache does not contain a tensor: {path}")
    value = value.contiguous()
    if expected is not None:
        if list(value.shape) != list(expected.get("shape", ())):
            raise ValueError(f"compressed cache shape mismatch: {path}")
        if str(value.dtype) != expected.get("dtype"):
            raise ValueError(f"compressed cache dtype mismatch: {path}")
        if tensor_sha256(value) != expected.get("tensor_sha256"):
            raise ValueError(f"decompressed tensor hash mismatch: {path}")
    return value


def file_sha256(path: Path, chunk_size: int = 8 * 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def plain_rmse(predicted: torch.Tensor, observed: torch.Tensor) -> torch.Tensor:
    """One plain full-field hard RMSE per predicted batch member."""
    if predicted.ndim != 5 or observed.ndim != 5 or observed.shape[0] != 1:
        raise ValueError("predicted and observed fields must be 5-D")
    if predicted.shape[1:] != observed.shape[1:]:
        raise ValueError("predicted and observed seismic shapes differ")
    delta = predicted.float() - observed.to(dtype=torch.float32)
    return delta.flatten(1).square().mean(dim=1).sqrt()


def rank_scores(
    scores: Mapping[str, Mapping[str, float]],
) -> dict[str, list[dict[str, object]]]:
    """Rank inference-visible scores using the frozen deterministic tie break."""
    result: dict[str, list[dict[str, object]]] = {}
    for observation_name in OBSERVATION_NAMES:
        rows = []
        for candidate_name, candidate_scores in scores.items():
            value = float(candidate_scores[observation_name])
            if not math.isfinite(value) or value < 0:
                raise ValueError("hard seismic RMSE must be finite and non-negative")
            rows.append(
                {
                    "candidate_id": str(candidate_name),
                    "hard_seismic_rmse": value,
                }
            )
        rows.sort(key=lambda row: (row["hard_seismic_rmse"], row["candidate_id"]))
        for rank, row in enumerate(rows, start=1):
            row["rank"] = rank
        result[observation_name] = rows
    return result


def _average_ranks(values: Sequence[float]) -> list[float]:
    if not values or any(not math.isfinite(float(value)) for value in values):
        raise ValueError("rank values must be finite and nonempty")
    order = sorted(range(len(values)), key=lambda index: (float(values[index]), index))
    ranks = [0.0] * len(values)
    start = 0
    while start < len(order):
        end = start + 1
        while end < len(order) and float(values[order[end]]) == float(values[order[start]]):
            end += 1
        average = 0.5 * ((start + 1) + end)
        for position in range(start, end):
            ranks[order[position]] = average
        start = end
    return ranks


def spearman_rank_correlation(first: Sequence[float], second: Sequence[float]) -> float:
    if len(first) != len(second) or len(first) < 2:
        raise ValueError("Spearman inputs must have equal length of at least two")
    x = _average_ranks(first)
    y = _average_ranks(second)
    x_mean = sum(x) / len(x)
    y_mean = sum(y) / len(y)
    numerator = sum((a - x_mean) * (b - y_mean) for a, b in zip(x, y))
    x_norm = math.sqrt(sum((a - x_mean) ** 2 for a in x))
    y_norm = math.sqrt(sum((b - y_mean) ** 2 for b in y))
    if x_norm == 0 or y_norm == 0:
        return float("nan")
    return numerator / (x_norm * y_norm)


def support_checks(
    row: Mapping[str, object], thresholds: Mapping[str, object]
) -> dict[str, bool]:
    return {
        "conditions_exact": int(row["condition_violations"])
        <= int(thresholds["condition_violations_maximum"]),
        "label9_iou": float(row["label9_iou"])
        >= float(thresholds["label9_iou_minimum"]),
        "label9_precision": float(row["label9_precision"])
        >= float(thresholds["label9_precision_minimum"]),
        "label9_recall": float(row["label9_recall"])
        >= float(thresholds["label9_recall_minimum"]),
        "major_component_min_recall": float(row["major_component_min_recall"])
        >= float(thresholds["major_component_min_recall_minimum"]),
        "major_component_mean_recall": float(row["major_component_mean_recall"])
        >= float(thresholds["major_component_mean_recall_minimum"]),
    }


def discrimination_checks(
    correlations: Sequence[Mapping[str, object]],
    enrichment: Sequence[Mapping[str, object]],
) -> dict[str, object]:
    corr_index = {
        (str(row["observation"]), str(row["metric"])): float(row["spearman_rho"])
        for row in correlations
    }
    enrich_index = {
        (str(row["observation"]), str(row["metric"]), str(row["subset"])): row
        for row in enrichment
    }
    negative = {
        metric: math.isfinite(corr_index.get(("correct", metric), float("nan")))
        and corr_index[("correct", metric)] < 0
        for metric in TARGET_METRICS
    }
    top5_above = {
        metric: float(enrich_index[("correct", metric, "top_5pct")]["mean"])
        > float(enrich_index[("correct", metric, "full")]["mean"])
        for metric in TARGET_METRICS
    }
    correct_better_controls: dict[str, dict[str, bool]] = {}
    for metric in TARGET_METRICS:
        correct_value = float(
            enrich_index[("correct", metric, "top_5pct")]["enrichment"]
        )
        correct_better_controls[metric] = {
            control: correct_value
            > float(enrich_index[(control, metric, "top_5pct")]["enrichment"])
            for control in ("zero", "shuffled_xy", "wrong_case")
        }
    passed = (
        all(negative.values())
        and all(top5_above.values())
        and all(
            all(control_checks.values())
            for control_checks in correct_better_controls.values()
        )
    )
    return {
        "correct_target_spearman_strictly_negative": negative,
        "correct_top5_mean_strictly_above_full": top5_above,
        "correct_top5_enrichment_strictly_above_controls": correct_better_controls,
        "passed": passed,
    }


def next_action(support_pass: bool, discrimination_pass: bool) -> str:
    if support_pass and discrimination_pass:
        return "STAGE9B_POSTERIOR_WEIGHTING"
    if not support_pass and discrimination_pass:
        return "STAGE9C_ADAPTIVE_PROPOSAL_FEASIBILITY"
    if support_pass and not discrimination_pass:
        return "STOP_REDESIGN_LIKELIHOOD_OR_PETROPHYSICS"
    return "STOP_REASSESS_FROZEN_INFERENCE_ROUTE"


def load_chunks_by_manifest(
    root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    callback: Callable[[Mapping[str, object], torch.Tensor], None],
) -> None:
    """Validated streaming reader shared by ranking and future stages."""
    for record in records:
        value = load_tensor_gzip(Path(root) / str(record["path"]), expected=record)
        callback(record, value)
