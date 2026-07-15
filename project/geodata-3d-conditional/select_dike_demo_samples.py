"""Select representative samples for baseline-vs-guided dike demo figures."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Dict, List, Mapping, Sequence

from geology_io_utils import infer_paired_by_seed, read_csv_rows, rows_by_sample_id, write_csv_rows, write_json


OUTPUT_FIELDS = [
    "role",
    "sample_id",
    "paired_by_seed",
    "selection_score",
    "reason",
    "baseline_geo_misfit",
    "guided_geo_misfit",
    "baseline_voxel_accuracy",
    "guided_voxel_accuracy",
    "baseline_mean_iou",
    "guided_mean_iou",
    "baseline_borehole_consistency",
    "guided_borehole_consistency",
    "baseline_target_iou",
    "guided_target_iou",
    "baseline_target_recall",
    "guided_target_recall",
    "baseline_centroid_distance",
    "guided_centroid_distance",
]


def _float(row: Mapping[str, object], key: str, default: float = float("nan")) -> float:
    try:
        value = row.get(key, default)
        if value in ("", None):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _score_paired(
    baseline_global: Mapping[str, object],
    guided_global: Mapping[str, object],
    baseline_target: Mapping[str, object],
    guided_target: Mapping[str, object],
) -> float:
    geo_improvement = _float(baseline_global, "geo_misfit") - _float(guided_global, "geo_misfit")
    iou_improvement = _float(guided_target, "target_iou") - _float(baseline_target, "target_iou")
    recall_improvement = _float(guided_target, "target_recall") - _float(baseline_target, "target_recall")
    centroid_improvement = _float(baseline_target, "target_centroid_distance") - _float(guided_target, "target_centroid_distance")
    values = [geo_improvement, iou_improvement, recall_improvement, 0.05 * centroid_improvement]
    return sum(value for value in values if math.isfinite(value))


def _score_single(global_row: Mapping[str, object], target_row: Mapping[str, object]) -> float:
    geo = _float(global_row, "geo_misfit")
    iou = _float(target_row, "target_iou")
    recall = _float(target_row, "target_recall")
    centroid = _float(target_row, "target_centroid_distance")
    score = 0.0
    if math.isfinite(geo):
        score -= geo
    if math.isfinite(iou):
        score += iou
    if math.isfinite(recall):
        score += 0.5 * recall
    if math.isfinite(centroid):
        score -= 0.01 * centroid
    return score


def _finite(value: float) -> bool:
    return isinstance(value, (int, float)) and math.isfinite(float(value))


def _geo_improvement(
    baseline_global: Mapping[str, object],
    guided_global: Mapping[str, object],
) -> float:
    return _float(baseline_global, "geo_misfit") - _float(guided_global, "geo_misfit")


def _target_improvement(
    baseline_target: Mapping[str, object],
    guided_target: Mapping[str, object],
) -> float:
    return _float(guided_target, "target_iou") - _float(baseline_target, "target_iou")


def _preserves_geology(
    baseline_global: Mapping[str, object],
    guided_global: Mapping[str, object],
    tolerance: float = 0.02,
) -> bool:
    for field in ("voxel_accuracy", "mean_iou", "borehole_consistency"):
        baseline_value = _float(baseline_global, field)
        guided_value = _float(guided_global, field)
        if _finite(baseline_value) and _finite(guided_value) and guided_value < baseline_value - tolerance:
            return False
    return True


def _best_by(scored: Sequence[tuple[float, int]], fallback_ids: Sequence[int]) -> tuple[float, int] | None:
    finite = [(score, sample_id) for score, sample_id in scored if math.isfinite(score)]
    if finite:
        return max(finite, key=lambda item: (item[0], -item[1]))
    if fallback_ids:
        return 0.0, int(fallback_ids[0])
    return None


def select_samples(
    baseline_metrics: Sequence[Mapping[str, object]],
    guided_metrics: Sequence[Mapping[str, object]],
    baseline_target_metrics: Sequence[Mapping[str, object]],
    guided_target_metrics: Sequence[Mapping[str, object]],
    paired_by_seed: bool,
    top_k: int = 6,
) -> tuple[List[Dict[str, object]], Dict[str, object]]:
    baseline_global = rows_by_sample_id(baseline_metrics)
    guided_global = rows_by_sample_id(guided_metrics)
    baseline_target = rows_by_sample_id(baseline_target_metrics)
    guided_target = rows_by_sample_id(guided_target_metrics)
    rows: List[Dict[str, object]] = []

    if paired_by_seed:
        common_ids = sorted(set(baseline_global) & set(guided_global) & set(baseline_target) & set(guided_target))
        role_specs = []
        role_specs.append(
            (
                "best_guided_geo",
                _best_by([(-_float(guided_global[sid], "geo_misfit"), sid) for sid in common_ids], common_ids),
                "lowest guided lightweight gravity-proxy misfit among paired samples",
            )
        )
        preserve_scored = [
            (_geo_improvement(baseline_global[sid], guided_global[sid]), sid)
            for sid in common_ids
            if _preserves_geology(baseline_global[sid], guided_global[sid])
        ]
        role_specs.append(
            (
                "max_geo_improvement_preserve_geology",
                _best_by(preserve_scored, common_ids),
                "largest paired proxy-misfit improvement while preserving voxel_accuracy, mean_iou, and borehole_consistency within tolerance",
            )
        )
        role_specs.append(
            (
                "representative_guided",
                _best_by([(_score_single(guided_global[sid], guided_target[sid]), sid) for sid in common_ids], common_ids),
                "high-quality guided representative balancing target metrics and proxy misfit",
            )
        )
        role_specs.append(
            (
                "best_target_improvement",
                _best_by([(_target_improvement(baseline_target[sid], guided_target[sid]), sid) for sid in common_ids], common_ids),
                "largest paired target_iou improvement",
            )
        )
        failure_scored = [
            (_float(guided_global[sid], "geo_misfit") + (1.0 - _float(guided_target[sid], "target_iou", 0.0)), sid)
            for sid in common_ids
        ]
        role_specs.append(
            (
                "failure_case",
                _best_by(failure_scored, common_ids),
                "guided sample with relatively weak combined proxy/target result for honest failure-mode display",
            )
        )
        used = 0
        for role, selected, reason in role_specs:
            if selected is None:
                continue
            score, sample_id = selected
            rows.append(_selection_row(role, sample_id, True, score, reason, baseline_global, guided_global, baseline_target, guided_target))
            used += 1
            if used >= top_k:
                break
    else:
        guided_scored = [
            (_score_single(guided_global.get(sample_id, {}), row), sample_id)
            for sample_id, row in guided_target.items()
        ]
        baseline_scored = [
            (_score_single(baseline_global.get(sample_id, {}), row), sample_id)
            for sample_id, row in baseline_target.items()
        ]
        guided_scored.sort(reverse=True)
        baseline_scored.sort(reverse=True)
        guided_geo = sorted(
            [
                (_float(guided_global.get(sample_id, {}), "geo_misfit"), sample_id)
                for sample_id in set(guided_global) | set(guided_target)
            ],
            key=lambda item: item[0],
        )
        guided_target_iou = sorted(
            [
                (_float(row, "target_iou", -math.inf), sample_id)
                for sample_id, row in guided_target.items()
            ],
            reverse=True,
        )
        role_specs = (
            (
                "best_guided_geo",
                ((-guided_geo[0][0], guided_geo[0][1]) if guided_geo else None),
                "unpaired comparison: selected lowest guided proxy-misfit representative without sample-wise improvement claim",
            ),
            (
                "max_geo_improvement_preserve_geology",
                ((-guided_geo[0][0], guided_geo[0][1]) if guided_geo else None),
                "unpaired comparison: sample-wise geo improvement and preserve-geology check are not computed",
            ),
            (
                "representative_guided",
                guided_scored[0] if guided_scored else None,
                "unpaired comparison: selected guided representative independently",
            ),
            (
                "best_target_improvement",
                guided_target_iou[0] if guided_target_iou else None,
                "unpaired comparison: sample-wise target improvement is not computed; selected highest guided target_iou",
            ),
            (
                "failure_case",
                guided_scored[-1] if guided_scored else None,
                "unpaired comparison: selected weak guided representative for failure-mode display",
            ),
            (
                "baseline_representative",
                baseline_scored[0] if baseline_scored else None,
                "unpaired comparison: selected baseline representative independently",
            ),
        )
        for role, selected, reason in role_specs:
            if selected is None:
                continue
            score, sample_id = selected
            rows.append(_selection_row(
                role,
                sample_id,
                False,
                score,
                reason,
                baseline_global,
                guided_global,
                baseline_target,
                guided_target,
            ))

    summary = {
        "paired_by_seed": bool(paired_by_seed),
        "top_k": int(top_k),
        "n_selected": len(rows),
        "description": (
            "Samples are selected from saved global metrics.csv and target_metrics.csv. "
            "When paired_by_seed is false, baseline and guided representatives are "
            "selected separately to avoid implying sample-wise initial-noise pairing."
        ),
    }
    return rows, summary


def _selection_row(
    role: str,
    sample_id: int,
    paired_by_seed: bool,
    score: float,
    reason: str,
    baseline_global: Mapping[int, Mapping[str, object]],
    guided_global: Mapping[int, Mapping[str, object]],
    baseline_target: Mapping[int, Mapping[str, object]],
    guided_target: Mapping[int, Mapping[str, object]],
) -> Dict[str, object]:
    b_global = baseline_global.get(sample_id, {})
    g_global = guided_global.get(sample_id, {})
    b_target = baseline_target.get(sample_id, {})
    g_target = guided_target.get(sample_id, {})
    return {
        "role": role,
        "sample_id": int(sample_id),
        "paired_by_seed": bool(paired_by_seed),
        "selection_score": score,
        "reason": reason,
        "baseline_geo_misfit": _float(b_global, "geo_misfit", ""),
        "guided_geo_misfit": _float(g_global, "geo_misfit", ""),
        "baseline_voxel_accuracy": _float(b_global, "voxel_accuracy", ""),
        "guided_voxel_accuracy": _float(g_global, "voxel_accuracy", ""),
        "baseline_mean_iou": _float(b_global, "mean_iou", ""),
        "guided_mean_iou": _float(g_global, "mean_iou", ""),
        "baseline_borehole_consistency": _float(b_global, "borehole_consistency", ""),
        "guided_borehole_consistency": _float(g_global, "borehole_consistency", ""),
        "baseline_target_iou": _float(b_target, "target_iou", ""),
        "guided_target_iou": _float(g_target, "target_iou", ""),
        "baseline_target_recall": _float(b_target, "target_recall", ""),
        "guided_target_recall": _float(g_target, "target_recall", ""),
        "baseline_centroid_distance": _float(b_target, "target_centroid_distance", ""),
        "guided_centroid_distance": _float(g_target, "target_centroid_distance", ""),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select target-feature demo samples from saved metrics.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--baseline-metrics", type=Path, required=True)
    parser.add_argument("--guided-metrics", type=Path, required=True)
    parser.add_argument("--baseline-target-metrics", type=Path, required=True)
    parser.add_argument("--guided-target-metrics", type=Path, required=True)
    parser.add_argument("--baseline-dir", type=Path, default=None)
    parser.add_argument("--guided-dir", type=Path, default=None)
    parser.add_argument("--paired-by-seed", action="store_true")
    parser.add_argument("--top-k", type=int, default=6)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.top_k <= 0:
        raise SystemExit("--top-k must be positive")
    paired = bool(args.paired_by_seed)
    paired_reason = "provided by --paired-by-seed"
    if args.baseline_dir is not None and args.guided_dir is not None:
        paired, paired_reason = infer_paired_by_seed(args.baseline_dir, args.guided_dir)
    rows, summary = select_samples(
        baseline_metrics=read_csv_rows(args.baseline_metrics),
        guided_metrics=read_csv_rows(args.guided_metrics),
        baseline_target_metrics=read_csv_rows(args.baseline_target_metrics),
        guided_target_metrics=read_csv_rows(args.guided_target_metrics),
        paired_by_seed=paired,
        top_k=args.top_k,
    )
    summary["paired_by_seed_reason"] = paired_reason
    write_csv_rows(args.output_dir / "selected_samples.csv", rows, OUTPUT_FIELDS)
    write_json(args.output_dir / "summary.json", summary)
    print(f"Saved selected samples: {args.output_dir / 'selected_samples.csv'}")


if __name__ == "__main__":
    main()
