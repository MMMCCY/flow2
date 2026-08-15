#!/usr/bin/env python3
"""Audit frozen B2 seismic-loss alignment against frozen B3 truth metrics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
from pathlib import Path
from typing import Mapping, Sequence

from scipy.stats import spearmanr
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.seismic import tensor_sha256
from scripts.stage15.common import read_json, refuse_nonempty, write_csv, write_json


EXPERIMENT_ROOT = PROJECT_DIR / "experiments/stage15_binary_seismic_consensus"
DEFAULT_B2_DIR = EXPERIMENT_ROOT / "inversion/b2_flow_pcn_pilot_v1"
DEFAULT_B3_DIR = EXPERIMENT_ROOT / "reports/b2_truth_evaluation_v1"
DEFAULT_OUTPUT = EXPERIMENT_ROOT / "reports/b2_seismic_geology_alignment_v1"
MACHINE_DECISIONS = (
    "LOWER_SEISMIC_LOSS_ALIGNS_WITH_BETTER_GEOLOGY",
    "SEISMIC_LOSS_NOT_ALIGNED_WITH_GEOLOGY",
)
METRIC_FIELDS = ("iou", "precision", "recall", "centroid_distance")


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def snapshot_input_hashes(paths: Sequence[Path]) -> dict[str, str]:
    return {str(path.resolve()): runtime.file_sha256(path) for path in paths}


def assert_input_hashes_unchanged(before: Mapping[str, str]) -> None:
    changed = [
        path
        for path, expected in before.items()
        if runtime.file_sha256(Path(path)) != expected
    ]
    if changed:
        raise RuntimeError(f"frozen B2/B3 inputs changed during B4 audit: {changed}")


def align_retained_states(
    trace_rows: Sequence[Mapping[str, str]],
    truth_rows: Sequence[Mapping[str, str]],
    retained_binary: torch.Tensor,
) -> list[dict[str, object]]:
    postburn = [row for row in trace_rows if _is_true(str(row["post_burnin_recorded"]))]
    retained_truth = [row for row in truth_rows if row["sample_group"] == "retained"]
    if retained_binary.shape != (96, 1, 64, 64, 64):
        raise ValueError("frozen retained_binary_states must have shape [96,1,64,64,64]")
    if len(postburn) != 96 or len(retained_truth) != 96:
        raise ValueError(
            f"alignment requires 96 B2 and 96 B3 rows, got {len(postburn)} and {len(retained_truth)}"
        )
    by_index = {int(row["sample_index"]): row for row in retained_truth}
    if sorted(by_index) != list(range(96)) or len(by_index) != 96:
        raise ValueError("B3 retained indices must be unique and exactly 0..95")

    aligned: list[dict[str, object]] = []
    for retained_index, trace in enumerate(postburn):
        chain_id = int(trace["chain_id"])
        iteration = int(trace["iteration"])
        expected_index = chain_id * 24 + (iteration - 9)
        if retained_index != expected_index:
            raise ValueError(
                f"B2 retained ordering mismatch at {retained_index}: "
                f"chain={chain_id}, iteration={iteration}, expected={expected_index}"
            )
        truth = by_index[retained_index]
        state = retained_binary[retained_index : retained_index + 1]
        b3_uint8_hash = tensor_sha256(state)
        b2_float32_hash = tensor_sha256(state.float())
        if truth["binary_sha256"] != b3_uint8_hash:
            raise ValueError(f"B3 binary hash mismatch at retained index {retained_index}")
        if trace["current_binary_sha256"] != b2_float32_hash:
            raise ValueError(f"B2 trace binary hash mismatch at retained index {retained_index}")
        if int(truth["target_voxel_count"]) != int(trace["target_voxel_count"]):
            raise ValueError(f"target voxel count mismatch at retained index {retained_index}")
        aligned.append(
            {
                "retained_index": retained_index,
                "chain_id": chain_id,
                "iteration": iteration,
                "current_hard_seismic_loss": float(trace["current_hard_seismic_loss"]),
                "iou": float(truth["iou"]),
                "precision": float(truth["precision"]),
                "recall": float(truth["recall"]),
                "centroid_distance": float(truth["centroid_distance"]),
                "target_voxel_count": int(truth["target_voxel_count"]),
                "b2_current_binary_sha256_float32": b2_float32_hash,
                "b3_binary_sha256_uint8": b3_uint8_hash,
                "index_alignment_verified": True,
                "binary_hash_alignment_verified": True,
            }
        )
    return aligned


def spearman_metrics(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    loss = [float(row["current_hard_seismic_loss"]) for row in rows]
    result: dict[str, object] = {}
    for field in METRIC_FIELDS:
        correlation = spearmanr(loss, [float(row[field]) for row in rows])
        result[field] = {
            "rho": float(correlation.statistic),
            "two_sided_p_value": float(correlation.pvalue),
            "sample_count": len(rows),
        }
    return result


def metric_medians(rows: Sequence[Mapping[str, object]]) -> dict[str, float]:
    return {
        field: statistics.median(float(row[field]) for row in rows)
        for field in METRIC_FIELDS
    }


def quartile_groups(rows: Sequence[Mapping[str, object]]) -> dict[str, object]:
    if len(rows) != 96:
        raise ValueError("quartile audit requires exactly 96 retained states")
    ordered = sorted(
        rows,
        key=lambda row: (
            float(row["current_hard_seismic_loss"]), int(row["retained_index"])
        ),
    )
    low = ordered[:24]
    high = ordered[-24:]
    return {
        "selection_policy": "stable_loss_order_exactly_24_states_per_quartile",
        "lowest_25_percent": {
            "state_count": len(low),
            "loss_min": float(low[0]["current_hard_seismic_loss"]),
            "loss_max": float(low[-1]["current_hard_seismic_loss"]),
            "median_truth_metrics": metric_medians(low),
            "retained_indices": [int(row["retained_index"]) for row in low],
        },
        "highest_25_percent": {
            "state_count": len(high),
            "loss_min": float(high[0]["current_hard_seismic_loss"]),
            "loss_max": float(high[-1]["current_hard_seismic_loss"]),
            "median_truth_metrics": metric_medians(high),
            "retained_indices": [int(row["retained_index"]) for row in high],
        },
    }


def select_extreme(
    rows: Sequence[Mapping[str, object]], field: str, *, minimum: bool
) -> dict[str, object]:
    extreme_value = (
        min(float(row[field]) for row in rows)
        if minimum
        else max(float(row[field]) for row in rows)
    )
    ties = [row for row in rows if float(row[field]) == extreme_value]
    chosen = min(ties, key=lambda row: int(row["retained_index"]))
    return {
        **dict(chosen),
        "selection_field": field,
        "selection_extreme": "minimum" if minimum else "maximum",
        "tie_count": len(ties),
        "tied_retained_indices": [int(row["retained_index"]) for row in ties],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--b2-dir", type=Path, default=DEFAULT_B2_DIR)
    parser.add_argument("--b3-dir", type=Path, default=DEFAULT_B3_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--preview", action="store_true")
    parser.add_argument("--decision", choices=MACHINE_DECISIONS, default=None)
    return parser.parse_args()


def audit_frozen_alignment(
    b2_dir: Path, b3_dir: Path
) -> tuple[dict[str, object], list[dict[str, object]], dict[str, str]]:
    input_paths = (
        b2_dir / "run_manifest.json",
        b2_dir / "chain_trace.csv",
        b2_dir / "retained_binary_states.pt",
        b3_dir / "manifest.json",
        b3_dir / "retained_truth_metrics.csv",
    )
    input_hashes = snapshot_input_hashes(input_paths)
    b2_manifest = read_json(input_paths[0])
    b3_manifest = read_json(input_paths[3])
    if b2_manifest.get("run_status") != "completed":
        raise ValueError("frozen B2 run is not completed")
    if b3_manifest.get("run_status") != "completed":
        raise ValueError("frozen B3 evaluation is not completed")
    if runtime.file_sha256(input_paths[4]) != b3_manifest["retained_truth_metrics"]["sha256"]:
        raise ValueError("B3 retained truth metrics hash differs from its manifest")

    trace_rows = _read_csv(input_paths[1])
    truth_rows = _read_csv(input_paths[4])
    retained_binary = runtime.load_tensor(input_paths[2])
    aligned = align_retained_states(trace_rows, truth_rows, retained_binary)
    correlations = spearman_metrics(aligned)
    quartiles = quartile_groups(aligned)
    minimum_loss = select_extreme(
        aligned, "current_hard_seismic_loss", minimum=True
    )
    maximum_iou = select_extreme(aligned, "iou", minimum=False)
    summary: dict[str, object] = {
        "schema": "stage15_b4_frozen_b2_seismic_geology_alignment_v1",
        "audit_only": True,
        "retained_state_count": len(aligned),
        "state_alignment": {
            "method": "retained_index_plus_dtype_aware_binary_sha256",
            "index_matches": len(aligned),
            "binary_hash_matches": len(aligned),
            "all_states_one_to_one": True,
            "hash_note": (
                "B2 trace hashes the live float32 binary tensor; B3 hashes the frozen "
                "uint8 retained tensor. Both representations were regenerated from each "
                "frozen retained state and matched their respective table."
            ),
        },
        "spearman": correlations,
        "loss_quartiles": quartiles,
        "minimum_seismic_loss_state": minimum_loss,
        "maximum_iou_state": maximum_iou,
        "input_file_sha256_before_and_after": input_hashes,
        "frozen_inputs_unchanged": True,
        "truth_loaded_by_b4": False,
        "pcn_rerun_performed": False,
        "flow_sampling_performed": False,
        "seismic_inversion_performed": False,
        "parameter_sweep_performed": False,
        "thresholds_modified": False,
        "likelihood_weight_modified": False,
        "decision_policy": "manual_interpretation_without_new_numeric_gate",
    }
    assert_input_hashes_unchanged(input_hashes)
    return summary, aligned, input_hashes


def _fmt(value: object) -> str:
    return f"{float(value):.8g}"


def render_report(summary: Mapping[str, object]) -> str:
    correlations = summary["spearman"]
    low = summary["loss_quartiles"]["lowest_25_percent"]
    high = summary["loss_quartiles"]["highest_25_percent"]
    minimum_loss = summary["minimum_seismic_loss_state"]
    maximum_iou = summary["maximum_iou_state"]
    decision = summary["machine_decision"]
    return f"""# Stage15-B4 — Frozen B2 seismic-loss / geology alignment audit

Machine decision: **{decision}**

This audit joined all 96 frozen retained states by retained index and by dtype-aware binary SHA256. It consumed B3 truth metrics without opening truth itself. It did not rerun sampling or inversion and changed no B2/B3 input, threshold, or likelihood weight.

## Spearman correlations

- Loss vs IoU: rho={_fmt(correlations['iou']['rho'])}, p={_fmt(correlations['iou']['two_sided_p_value'])}
- Loss vs precision: rho={_fmt(correlations['precision']['rho'])}, p={_fmt(correlations['precision']['two_sided_p_value'])}
- Loss vs recall: rho={_fmt(correlations['recall']['rho'])}, p={_fmt(correlations['recall']['two_sided_p_value'])}
- Loss vs centroid distance: rho={_fmt(correlations['centroid_distance']['rho'])}, p={_fmt(correlations['centroid_distance']['two_sided_p_value'])}

For alignment, lower loss should accompany higher IoU/precision/recall (negative rho) and smaller centroid distance (positive rho when loss increases).

## Loss quartiles

- Lowest-loss 24 median IoU / precision / recall / centroid distance: {_fmt(low['median_truth_metrics']['iou'])} / {_fmt(low['median_truth_metrics']['precision'])} / {_fmt(low['median_truth_metrics']['recall'])} / {_fmt(low['median_truth_metrics']['centroid_distance'])}
- Highest-loss 24 median IoU / precision / recall / centroid distance: {_fmt(high['median_truth_metrics']['iou'])} / {_fmt(high['median_truth_metrics']['precision'])} / {_fmt(high['median_truth_metrics']['recall'])} / {_fmt(high['median_truth_metrics']['centroid_distance'])}

## Representative states

- Minimum-loss state: retained {minimum_loss['retained_index']}, chain {minimum_loss['chain_id']}, iteration {minimum_loss['iteration']}; loss / IoU / precision / recall / centroid distance: {_fmt(minimum_loss['current_hard_seismic_loss'])} / {_fmt(minimum_loss['iou'])} / {_fmt(minimum_loss['precision'])} / {_fmt(minimum_loss['recall'])} / {_fmt(minimum_loss['centroid_distance'])}
- Maximum-IoU state: retained {maximum_iou['retained_index']}, chain {maximum_iou['chain_id']}, iteration {maximum_iou['iteration']}; loss / IoU / precision / recall / centroid distance: {_fmt(maximum_iou['current_hard_seismic_loss'])} / {_fmt(maximum_iou['iou'])} / {_fmt(maximum_iou['precision'])} / {_fmt(maximum_iou['recall'])} / {_fmt(maximum_iou['centroid_distance'])}

## Interpretation

{summary['decision_rationale']}

The conclusion is a manual interpretation of the frozen audit metrics, not a new numerical pass/fail threshold.
"""


def main() -> None:
    args = parse_args()
    if not args.preview and args.decision is None:
        raise ValueError("final audit requires one of the two --decision values")
    summary, aligned, input_hashes = audit_frozen_alignment(args.b2_dir, args.b3_dir)
    if args.preview:
        print(json.dumps(summary, indent=2, sort_keys=True))
        return

    refuse_nonempty(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary["machine_decision"] = args.decision
    summary["machine_decision_allowed_values"] = list(MACHINE_DECISIONS)
    summary["decision_rationale"] = {
        "LOWER_SEISMIC_LOSS_ALIGNS_WITH_BETTER_GEOLOGY": (
            "Lower frozen hard-binary seismic loss consistently accompanies better "
            "truth-label9 geometry across correlations and loss quartiles."
        ),
        "SEISMIC_LOSS_NOT_ALIGNED_WITH_GEOLOGY": (
            "Frozen hard-binary seismic loss does not consistently rank better truth-label9 "
            "geometry; the relationships are weak, unrelated, or directionally conflicting."
        ),
    }[args.decision]
    write_json(args.output_dir / "summary.json", summary)
    write_csv(args.output_dir / "aligned_states.csv", aligned)
    (args.output_dir / "REPORT.md").write_text(render_report(summary), encoding="utf-8")
    assert_input_hashes_unchanged(input_hashes)


if __name__ == "__main__":
    main()
