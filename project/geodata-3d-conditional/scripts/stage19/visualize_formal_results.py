#!/usr/bin/env python3
"""Render Stage19R formal TEST results in the established Stage15 3-D style."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.paper_figures.style import render_categorical_panel, render_target_panel, show_render
from scripts.stage15.common import refuse_nonempty, write_json

ROOT = PROJECT_DIR / "experiments/stage19_learned_evidence_adapter"
DEFAULT_RUN = ROOT / "formal/inference_v2"
DEFAULT_REGISTRY = ROOT / "evidence_v2/evidence_registry.json"
DEFAULT_METRICS = ROOT / "reports/formal_v2/sample_metrics.csv"
DEFAULT_OUTPUT = ROOT / "reports/formal_v2/visualizations_v1"
SEED = 42
ARMS = (
    "FLOW_ONLY",
    "STAGE18_CONTINUOUS_TARGET",
    "ADAPTER_CORRECT",
    "ADAPTER_ZERO",
    "ADAPTER_WRONG_CASE",
)
ARM_TITLES = {
    "FLOW_ONLY": "Flow-only",
    "STAGE18_CONTINUOUS_TARGET": "Stage18 continuous target",
    "ADAPTER_CORRECT": "Adapter — correct evidence",
    "ADAPTER_ZERO": "Adapter — zero evidence",
    "ADAPTER_WRONG_CASE": "Adapter — wrong-case evidence",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", type=Path, default=DEFAULT_RUN)
    parser.add_argument("--evaluation-registry", type=Path, default=DEFAULT_REGISTRY)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--seed", type=int, default=SEED)
    return parser.parse_args()


def _load_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"expected JSON object: {path}")
    return payload


def _metric_index(path: Path) -> dict[tuple[str, int, str], dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    index = {(row["case_id"], int(row["source_seed"]), row["arm"]): row for row in rows}
    if len(rows) != 180 or len(index) != 180:
        raise ValueError("Stage19R visualization requires exactly 180 unique formal metric rows")
    return index


def _load_model(run_dir: Path, case_id: str, seed: int, arm: str):
    path = run_dir / case_id / f"seed_{seed}" / arm / "decoded_geology.pt"
    return runtime.normalize_single_geology(runtime.load_tensor(path, map_location="cpu"), path.name)[0, 0].numpy(), path


def _render_triplet(
    output: Path,
    case_id: str,
    seed: int,
    truth,
    flow,
    correct,
    flow_metrics: dict[str, str],
    correct_metrics: dict[str, str],
) -> None:
    truth9 = truth == 9
    images = (
        render_categorical_panel(truth),
        render_categorical_panel(flow),
        render_categorical_panel(correct),
        render_target_panel(truth9),
        render_target_panel(flow == 9, ghost_mask=truth9),
        render_target_panel(correct == 9, ghost_mask=truth9),
    )
    titles = (
        "(a) Truth categorical model",
        f"(b) Paired Flow-only — seed {seed}",
        "(c) Learned adapter — correct evidence",
        "(d) Complete truth label9 geometry",
        f"(e) Flow-only label9 — IoU={float(flow_metrics['target_iou']):.3f}, R={float(flow_metrics['target_recall']):.3f}",
        f"(f) Adapter label9 — IoU={float(correct_metrics['target_iou']):.3f}, R={float(correct_metrics['target_recall']):.3f}",
    )
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for axis, image, title in zip(axes.flat, images, titles):
        show_render(axis, image)
        axis.set_title(title, loc="left", fontsize=13)
    fig.suptitle(f"Stage19R learned evidence adapter — {case_id}", fontsize=17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _render_five_arm_target(
    output: Path,
    case_id: str,
    seed: int,
    truth,
    models: dict[str, object],
    metrics: dict[tuple[str, int, str], dict[str, str]],
) -> None:
    truth9 = truth == 9
    images = [render_target_panel(truth9)]
    titles = ["(a) Complete truth label9 geometry"]
    for label, arm in zip("bcdef", ARMS):
        row = metrics[(case_id, seed, arm)]
        images.append(render_target_panel(models[arm] == 9, ghost_mask=truth9))
        titles.append(
            f"({label}) {ARM_TITLES[arm]}\n"
            f"IoU={float(row['target_iou']):.3f}, R={float(row['target_recall']):.3f}, "
            f"seis. RMSE={float(row['hard_seismic_rmse']):.3f}"
        )
    fig, axes = plt.subplots(2, 3, figsize=(17, 10.5), constrained_layout=True)
    for axis, image, title in zip(axes.flat, images, titles):
        show_render(axis, image)
        axis.set_title(title, loc="left", fontsize=12)
    fig.suptitle(f"Stage19R five-arm label9 comparison — {case_id}, seed {seed}", fontsize=17)
    fig.savefig(output, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    run_manifest = _load_json(args.run_dir / "run_manifest.json")
    if run_manifest.get("run_status") != "completed" or run_manifest.get("output_count") != 180:
        raise ValueError("Stage19R formal inference is incomplete")
    if run_manifest.get("truth_loaded_by_runner") is not False:
        raise ValueError("formal inference truth-firewall record failed")
    registry = _load_json(args.evaluation_registry)
    cases = sorted((case for case in registry["cases"] if case["split"] == "test"), key=lambda item: str(item["case_id"]))
    if len(cases) != 12:
        raise ValueError("Stage19R visualization requires exactly 12 TEST cases")
    metrics = _metric_index(args.metrics)
    args.output_dir.mkdir(parents=True)

    figure_records = []
    primary_case = str(cases[0]["case_id"])
    for case in cases:
        case_id = str(case["case_id"])
        truth_path = Path(case["truth_assets"]["truth"]["path"])
        truth = runtime.normalize_single_geology(runtime.load_tensor(truth_path, map_location="cpu"), "truth")[0, 0].numpy()
        models, model_paths = {}, {}
        for arm in ARMS:
            models[arm], model_paths[arm] = _load_model(args.run_dir, case_id, args.seed, arm)
        flow_row = metrics[(case_id, args.seed, "FLOW_ONLY")]
        correct_row = metrics[(case_id, args.seed, "ADAPTER_CORRECT")]
        filename = "truth_flow_adapter_3d.png" if case_id == primary_case else f"truth_flow_adapter_{case_id}_seed{args.seed}_3d.png"
        figure_path = args.output_dir / filename
        _render_triplet(figure_path, case_id, args.seed, truth, models["FLOW_ONLY"], models["ADAPTER_CORRECT"], flow_row, correct_row)
        figure_records.append({
            "case_id": case_id,
            "seed": args.seed,
            "figure": runtime.asset_record(figure_path),
            "truth": runtime.asset_record(truth_path),
            "flow_only": runtime.asset_record(model_paths["FLOW_ONLY"]),
            "adapter_correct": runtime.asset_record(model_paths["ADAPTER_CORRECT"]),
            "flow_target_iou": float(flow_row["target_iou"]),
            "adapter_target_iou": float(correct_row["target_iou"]),
        })

    primary = cases[0]
    truth_path = Path(primary["truth_assets"]["truth"]["path"])
    truth = runtime.normalize_single_geology(runtime.load_tensor(truth_path, map_location="cpu"), "truth")[0, 0].numpy()
    models = {arm: _load_model(args.run_dir, primary_case, args.seed, arm)[0] for arm in ARMS}
    five_arm_path = args.output_dir / "five_arm_target_comparison_3d.png"
    _render_five_arm_target(five_arm_path, primary_case, args.seed, truth, models, metrics)

    write_json(args.output_dir / "visualization_manifest.json", {
        "schema": "stage19r_formal_visualization_v1",
        "run_status": "completed",
        "truth_loaded_for_retrospective_visualization_only": True,
        "formal_inference_truth_loaded_by_runner": False,
        "selection_policy": "lexicographically_first_test_case_and_preregistered_first_source_seed_for_primary_figure; no metric-based selection",
        "primary_case_id": primary_case,
        "source_seed": args.seed,
        "case_figure_count": len(figure_records),
        "case_figures": figure_records,
        "five_arm_figure": runtime.asset_record(five_arm_path),
        "formal_run_manifest": runtime.asset_record(args.run_dir / "run_manifest.json"),
        "evaluation_registry": runtime.asset_record(args.evaluation_registry),
        "sample_metrics": runtime.asset_record(args.metrics),
        "generator_source": runtime.asset_record(Path(__file__)),
    })
    print(f"Stage19R visualizations written to {args.output_dir}")


if __name__ == "__main__":
    main()
