#!/usr/bin/env python3
"""Render the frozen Stage6Q/Stage7 analytic-cuboid causal contrast.

The script is deliberately visualization-only: it loads saved tensors, decodes
the two saved CFM states with the frozen checkpoint embedding, validates every
displayed number against the authoritative Stage7/D4 artifacts, and exports a
paper figure plus a provenance manifest.  It never runs a sampler, a search, or
an acoustic forward model.
"""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys
from typing import Mapping, Sequence

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from scripts.paper_figures.style import (
    CAMERA_PRESETS,
    DOUBLE_COLUMN_MM,
    FIGURE_DATA_DIR,
    FIGURES_DIR,
    LABEL9_COLOR,
    LABEL_COLORS,
    MANIFESTS_DIR,
    OBSERVATION_COLOR,
    TRUTH_OUTLINE_COLOR,
    _add_bounding_box,
    _add_surface,
    _add_wells,
    _new_plotter,
    _set_camera,
    binary_surface,
    configure_matplotlib,
    cutaway_mask,
    ensure_output_dirs,
    generation_record,
    load_volume,
    mm_to_inches,
    output_records,
    read_json,
    save_figure,
    sha256,
    show_render,
    source_record,
    validate_same_shape,
    write_json,
)


SCRIPT_PATH = Path(__file__).resolve()
STYLE_PATH = SCRIPT_PATH.with_name("style.py")
FIGURE_ID = "figure03_cfm_structured_cuboid_contrast"
TITLE = "简单隐藏体基准中的冻结CFM地震制导与结构化硬推理对照"
TARGET_LABEL = 9
CHINESE_FONT = "Noto Sans CJK SC"
CHINESE_FONT_FILE = Path("/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc")

ROOT = PROJECT_DIR / "experiments/stage6_inference_causality"
Q1_DIR = ROOT / "runs/five_body_cuboid_v1/q0_q1_full_v2"
D4_DIR = ROOT / "runs/five_body_cuboid_v1/d4_frozen_flow_trajectory_v1"
D7_DIR = ROOT / "runs/five_body_cuboid_v1/d7_observation_specificity_v1"
STAGE7_DIR = ROOT / "reports/stage7_v1_final_v2"

Q1_CONFIG = Q1_DIR / "config_resolved.json"
Q1_SUMMARY = Q1_DIR / "summary.json"
Q1_REPORT = Q1_DIR / "REPORT.md"
TRUTH_LABELS = Q1_DIR / "tensors/truth_labels.pt"
ANALYTIC_BASELINE = Q1_DIR / "tensors/baseline_labels.pt"
CANDIDATE_MASKS = Q1_DIR / "tensors/candidate_masks.pt"
FIXED_TARGET_MASK = Q1_DIR / "tensors/fixed_target_mask.pt"
CONDITION_MASK = Q1_DIR / "tensors/condition_mask.pt"

D4_CONFIG = D4_DIR / "config_resolved.json"
D4_SUMMARY = D4_DIR / "summary.json"
D4_REPORT = D4_DIR / "REPORT.md"
BASE_STATE = D4_DIR / "states/correct/BASE/final_state.pt"
GUIDED_STATE = D4_DIR / "states/correct/BASE_PLUS_PHYSICS/best_hard_state.pt"
GUIDED_FINAL_STATE = D4_DIR / "states/correct/BASE_PLUS_PHYSICS/final_state.pt"
D4_TRACE = D4_DIR / "traces/correct/BASE_PLUS_PHYSICS.csv"

D7_REPORT = D7_DIR / "D7_OBSERVATION_SPECIFICITY_REPORT.md"
D7_VERDICT = D7_DIR / "d7_observation_specificity_verdict.json"

STAGE7_CONFIG = STAGE7_DIR / "config_input.json"
STAGE7_SUMMARY = STAGE7_DIR / "stage7_summary.json"
STAGE7_REPORT = STAGE7_DIR / "STAGE7_REPORT.md"
PAIRED_CSV = STAGE7_DIR / "paired_comparison.csv"
STRUCTURED_LABELS = STAGE7_DIR / "states/cuboid_seed42/correct/best_labels.pt"
STRUCTURED_HISTORY = STAGE7_DIR / "states/cuboid_seed42/correct/selected_event_history.json"

CHECKPOINT = PROJECT_DIR / "demo_model/conditional-weights.ckpt"

INK = "#202A31"
MUTED = "#5D6971"
GRID = "#D7DBDE"
IOU_COLOR = LABEL9_COLOR
RECALL_COLOR = OBSERVATION_COLOR
CONTEXT_OPACITY = 0.075
TARGET_OPACITY = 0.82
REFERENCE_OPACITY = 0.78
CAMERA_NAME = "perspective_oblique"


def _array_sha256(array: np.ndarray) -> str:
    value = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(str(value.dtype).encode("utf-8"))
    digest.update(json.dumps(list(value.shape)).encode("utf-8"))
    digest.update(value.tobytes())
    return digest.hexdigest()


def _tensor_record(path: Path, array: np.ndarray, role: str, **extra: object) -> dict[str, object]:
    return {
        **source_record(path, role),
        "tensor_content_sha256": _array_sha256(array),
        "shape": list(array.shape),
        "dtype": str(array.dtype),
        **extra,
    }


def _read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _paired_rows() -> dict[str, dict[str, object]]:
    numeric = {
        "hard_correct_observation_rmse",
        "hard_attainment_relative_to_base_flow_endpoint",
        "hidden_target_iou",
        "hidden_target_recall",
        "hidden_target_precision",
    }
    integer = {
        "hidden_target_true_positive_voxels",
        "hidden_target_predicted_voxels",
        "hidden_target_truth_voxels",
    }
    result: dict[str, dict[str, object]] = {}
    for raw in _read_rows(PAIRED_CSV):
        row: dict[str, object] = dict(raw)
        for key in numeric:
            row[key] = float(raw[key])
        for key in integer:
            row[key] = int(raw[key])
        result[str(row["method"])] = row
    return result


def _decode_saved_state(path: Path, embedding_weight: torch.Tensor) -> np.ndarray:
    state = torch.load(path, map_location="cpu", weights_only=True)
    if not torch.is_tensor(state) or tuple(state.shape) != (1, 15, 64, 64, 64):
        raise ValueError(f"unexpected saved CFM state: {path} {getattr(state, 'shape', None)}")
    categories = torch.einsum(
        "bexyz,ce->bcxyz",
        F.normalize(state, dim=1),
        F.normalize(embedding_weight.to(dtype=state.dtype), dim=1),
    ).argmax(dim=1)
    return (categories - 1).squeeze(0).numpy().astype(np.int16, copy=False)


def _hidden_metrics(
    labels: np.ndarray, hidden_truth: np.ndarray, evaluation_domain: np.ndarray
) -> dict[str, float | int]:
    predicted = (np.asarray(labels) == TARGET_LABEL) & evaluation_domain
    truth = np.asarray(hidden_truth, dtype=bool)
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    return {
        "hidden_target_iou": tp / max(tp + fp + fn, 1),
        "hidden_target_recall": tp / max(tp + fn, 1),
        "hidden_target_precision": tp / max(tp + fp, 1),
        "hidden_target_true_positive_voxels": tp,
        "hidden_target_predicted_voxels": int(predicted.sum()),
        "hidden_target_truth_voxels": int(truth.sum()),
    }


def _assert_close(actual: float, expected: float, name: str, tolerance: float = 5e-12) -> None:
    if abs(float(actual) - float(expected)) > tolerance:
        raise ValueError(f"{name} mismatch: {actual} != {expected}")


def _validate_metric_row(
    labels: np.ndarray,
    row: Mapping[str, object],
    hidden_truth: np.ndarray,
    evaluation_domain: np.ndarray,
    method: str,
) -> dict[str, float | int]:
    computed = _hidden_metrics(labels, hidden_truth, evaluation_domain)
    for key, value in computed.items():
        saved = row[key]
        if isinstance(value, int):
            if value != int(saved):
                raise ValueError(f"{method} {key} mismatch: {value} != {saved}")
        else:
            _assert_close(value, float(saved), f"{method} {key}")
    return computed


def _wireframe(plotter, mask: np.ndarray, color: str, opacity: float, width: float) -> None:
    surface = binary_surface(np.asarray(mask, dtype=bool))
    if surface is None:
        return
    plotter.add_mesh(
        surface,
        color=color,
        opacity=opacity,
        style="wireframe",
        line_width=width,
        show_edges=False,
        ambient=0.55,
        diffuse=0.45,
        specular=0.0,
    )


def _render_volume(
    volume: np.ndarray,
    *,
    fixed_target: np.ndarray,
    hidden_truth: np.ndarray,
    well_xy: Sequence[Sequence[int]],
    truth_panel: bool,
) -> np.ndarray:
    """Use one camera/domain/palette while separating reference and prediction roles."""
    volume = np.asarray(volume)
    validate_same_shape(
        {"volume": volume, "fixed target": fixed_target, "hidden truth": hidden_truth}
    )
    camera = CAMERA_PRESETS[CAMERA_NAME]
    plotter = _new_plotter((1100, 900))
    keep = cutaway_mask(volume.shape, float(camera["cut_fraction"]))

    for label in sorted(int(value) for value in np.unique(volume) if int(value) not in (-1, TARGET_LABEL)):
        _add_surface(
            plotter,
            binary_surface((volume == label) & keep),
            LABEL_COLORS.get(label, "#777777"),
            CONTEXT_OPACITY,
        )

    if truth_panel:
        _add_surface(plotter, binary_surface(fixed_target), OBSERVATION_COLOR, TARGET_OPACITY)
        _add_surface(plotter, binary_surface(hidden_truth), LABEL9_COLOR, TARGET_OPACITY)
    else:
        _add_surface(plotter, binary_surface(volume == TARGET_LABEL), LABEL9_COLOR, TARGET_OPACITY)
        _wireframe(plotter, fixed_target, OBSERVATION_COLOR, REFERENCE_OPACITY, 1.6)
        _wireframe(plotter, hidden_truth, TRUTH_OUTLINE_COLOR, REFERENCE_OPACITY, 1.6)

    _add_wells(plotter, well_xy, top_z=float(volume.shape[2]))
    _add_bounding_box(plotter, volume.shape)
    _set_camera(plotter, volume.shape, camera)
    image = plotter.screenshot(return_img=True)
    plotter.close()
    return np.asarray(image)[..., :3]


def _load_and_validate() -> dict[str, object]:
    q1_config = read_json(Q1_CONFIG)
    q1_summary = read_json(Q1_SUMMARY)
    d4_config = read_json(D4_CONFIG)
    d4_summary = read_json(D4_SUMMARY)
    d7_verdict = read_json(D7_VERDICT)
    stage7_config = read_json(STAGE7_CONFIG)
    stage7_summary = read_json(STAGE7_SUMMARY)
    structured_history = read_json(STRUCTURED_HISTORY)

    description = str(q1_config.get("description"))
    hidden_indices = tuple(int(value) for value in q1_config["truth_candidate_indices"])
    fixed_bodies = q1_config["fixed_bodies"]
    well_xy = tuple(tuple(int(v) for v in body["well_xy"]) for body in fixed_bodies)
    if description != (
        "Two-material analytic five-body benchmark; three drilled fixed bodies and two "
        "hidden bodies selected from a frozen twelve-body dictionary."
    ):
        raise ValueError("analytic benchmark semantics changed")
    if hidden_indices != (4, 6) or len(fixed_bodies) != 3:
        raise ValueError("frozen five-body semantics or hidden pair changed")
    if q1_config["grid_shape"] != [64, 64, 64] or int(q1_config["target_label"]) != TARGET_LABEL:
        raise ValueError("unexpected analytic grid or target label")
    if bool(q1_config["formal_training_authorized"]):
        raise ValueError("Q1 unexpectedly authorizes training")

    truth = load_volume(TRUTH_LABELS, dtype=np.int16)
    analytic_baseline = load_volume(ANALYTIC_BASELINE, dtype=np.int16)
    fixed_target = load_volume(FIXED_TARGET_MASK).astype(bool)
    condition_mask = load_volume(CONDITION_MASK).astype(bool)
    candidate_tensor = torch.load(CANDIDATE_MASKS, map_location="cpu", weights_only=True)
    if not torch.is_tensor(candidate_tensor) or tuple(candidate_tensor.shape) != (12, 64, 64, 64):
        raise ValueError("unexpected frozen candidate-mask tensor")
    candidate_masks = candidate_tensor.numpy().astype(bool, copy=False)
    hidden_truth = candidate_masks[list(hidden_indices)].any(axis=0)
    evaluation_domain = candidate_masks.any(axis=0)
    validate_same_shape(
        {
            "truth": truth,
            "analytic baseline": analytic_baseline,
            "fixed target": fixed_target,
            "condition mask": condition_mask,
            "hidden truth": hidden_truth,
            "evaluation domain": evaluation_domain,
        }
    )
    if not np.array_equal(fixed_target, analytic_baseline == TARGET_LABEL):
        raise ValueError("fixed-target mask no longer equals the analytic baseline bodies")
    if not np.array_equal(truth == TARGET_LABEL, fixed_target | hidden_truth):
        raise ValueError("truth is no longer exactly three fixed plus two hidden target bodies")
    if int(hidden_truth.sum()) != int(q1_summary["case_validation"]["expected_hidden_voxels"]):
        raise ValueError("hidden voxel count disagrees with Q1 summary")

    if sha256(CHECKPOINT) != str(d4_config["checkpoint_sha256"]):
        raise ValueError("frozen checkpoint hash differs from D4 configuration")
    if sha256(CHECKPOINT) != str(d4_summary["checkpoint_sha256"]):
        raise ValueError("frozen checkpoint hash differs from D4 summary")
    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    embedding = checkpoint["state_dict"]["embedding.weight"].detach().cpu()
    if tuple(embedding.shape) != (15, 15):
        raise ValueError("unexpected frozen categorical embedding")

    baseline = _decode_saved_state(BASE_STATE, embedding)
    guided = _decode_saved_state(GUIDED_STATE, embedding)
    guided_final = _decode_saved_state(GUIDED_FINAL_STATE, embedding)
    structured = load_volume(STRUCTURED_LABELS, dtype=np.int16)
    validate_same_shape(
        {"truth": truth, "baseline": baseline, "guided": guided, "structured": structured}
    )
    if not np.array_equal(guided, guided_final):
        raise ValueError("D4 saved best-hard and final states no longer decode identically")
    for name, labels in (("baseline", baseline), ("guided", guided), ("structured", structured)):
        if np.any(labels[condition_mask] != truth[condition_mask]):
            raise ValueError(f"{name} violates frozen hard conditions")

    paired = _paired_rows()
    methods = {
        "baseline": "BASE_frozen_flow_sample",
        "guided": "continuous_BASE_PLUS_PHYSICS_best_hard_state",
        "structured": "structured_hard_geophysics",
    }
    computed = {
        key: _validate_metric_row(
            labels,
            paired[method],
            hidden_truth,
            evaluation_domain,
            method,
        )
        for key, labels, method in (
            ("baseline", baseline, methods["baseline"]),
            ("guided", guided, methods["guided"]),
            ("structured", structured, methods["structured"]),
        )
    }

    summary_paired = {row["method"]: row for row in stage7_summary["paired_comparison"]}
    for method in methods.values():
        for key in (
            "hard_correct_observation_rmse",
            "hard_attainment_relative_to_base_flow_endpoint",
            "hidden_target_iou",
            "hidden_target_recall",
        ):
            _assert_close(float(paired[method][key]), float(summary_paired[method][key]), f"{method} {key}")

    d4_run = next(
        row for row in d4_summary["runs"]
        if row["control"] == "correct" and row["mode"] == "BASE_PLUS_PHYSICS"
    )
    _assert_close(float(d4_run["best_hard_attainment"]), 0.30643379548394667, "D4 maximum attainment")
    _assert_close(float(d4_run["final_hard_attainment"]), 0.11524064334027935, "D4 final attainment")
    trace_rows = _read_rows(D4_TRACE)
    trace_attainment = [float(row["hard_attainment"]) for row in trace_rows]
    _assert_close(max(trace_attainment), float(d4_run["best_hard_attainment"]), "D4 trace maximum")
    _assert_close(trace_attainment[-1], float(d4_run["final_hard_attainment"]), "D4 trace final")

    cuboid_correct = next(
        row for row in stage7_summary["cuboid"]["arms"] if row["optimized_by"] == "correct"
    )
    if cuboid_correct["selection_criterion"] != "minimum hard observed seismic RMSE only":
        raise ValueError("Stage7 structured selection criterion changed")
    if cuboid_correct["truth_used_for_selection"] is not False or stage7_summary["truth_used_for_selection"] is not False:
        raise ValueError("Stage7 structured selection records truth use")
    if stage7_summary["training_performed"] is not False:
        raise ValueError("Stage7 unexpectedly records training")
    if stage7_config["id"] != "structured_hard_geophysics_v1":
        raise ValueError("unexpected Stage7 configuration")
    if str(paired[methods["structured"]]["selection"]) != "hard observed seismic only":
        raise ValueError("paired structured selection criterion changed")
    d7_ranking = [row["mechanism"] for row in d7_verdict.get("mechanism_ranking", [])]
    if d7_ranking != [
        "S1_residual_similarity",
        "S4_categorical_hard_transition_collapse",
        "S2_jacobian_vjp_projection_collapse",
        "S3_controller_normalization_cap_collapse",
    ] or d7_verdict.get("optional_s3_controller_control_authorized") is not False:
        raise ValueError("unexpected D7 mechanism verdict")

    stage7_report_text = STAGE7_REPORT.read_text(encoding="utf-8")
    d4_report_text = D4_REPORT.read_text(encoding="utf-8")
    for token in ("0.0441471", "0.0415255", "1.0000", "hard observed seismic RMSE only"):
        if token not in stage7_report_text:
            raise ValueError(f"Stage7 report cross-check token missing: {token}")
    for token in ("0.3064/0.1152", "BASE_PLUS_PHYSICS"):
        if token not in d4_report_text:
            raise ValueError(f"D4 report cross-check token missing: {token}")

    return {
        "q1_config": q1_config,
        "q1_summary": q1_summary,
        "d4_config": d4_config,
        "d4_summary": d4_summary,
        "d4_run": d4_run,
        "d7_verdict": d7_verdict,
        "stage7_summary": stage7_summary,
        "stage7_config": stage7_config,
        "structured_history": structured_history,
        "truth": truth,
        "analytic_baseline": analytic_baseline,
        "candidate_masks": candidate_masks,
        "fixed_target": fixed_target,
        "condition_mask": condition_mask,
        "hidden_truth": hidden_truth,
        "evaluation_domain": evaluation_domain,
        "baseline": baseline,
        "guided": guided,
        "structured": structured,
        "guided_final": guided_final,
        "well_xy": well_xy,
        "hidden_indices": hidden_indices,
        "paired": paired,
        "methods": methods,
        "computed": computed,
    }


def _panel_caption(ax, lines: Sequence[str], *, accent: str | None = None) -> None:
    ax.axis("off")
    if accent is not None:
        ax.plot([0.02, 0.02], [0.12, 0.90], transform=ax.transAxes, color=accent, lw=2.2)
    ax.text(
        0.055 if accent is not None else 0.02,
        0.88,
        "\n".join(lines),
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=5.85,
        color=INK,
        linespacing=1.16,
    )


def _make_figure(data: Mapping[str, object]) -> dict[str, str]:
    images = (
        _render_volume(
            data["truth"], fixed_target=data["fixed_target"], hidden_truth=data["hidden_truth"],
            well_xy=data["well_xy"], truth_panel=True,
        ),
        _render_volume(
            data["baseline"], fixed_target=data["fixed_target"], hidden_truth=data["hidden_truth"],
            well_xy=data["well_xy"], truth_panel=False,
        ),
        _render_volume(
            data["guided"], fixed_target=data["fixed_target"], hidden_truth=data["hidden_truth"],
            well_xy=data["well_xy"], truth_panel=False,
        ),
        _render_volume(
            data["structured"], fixed_target=data["fixed_target"], hidden_truth=data["hidden_truth"],
            well_xy=data["well_xy"], truth_panel=False,
        ),
    )
    paired = data["paired"]
    methods = data["methods"]
    baseline = paired[methods["baseline"]]
    guided = paired[methods["guided"]]
    structured = paired[methods["structured"]]
    d4_run = data["d4_run"]

    fig = plt.figure(figsize=(mm_to_inches(DOUBLE_COLUMN_MM), mm_to_inches(137.0)))
    fig.suptitle(TITLE, x=0.5, y=0.988, fontsize=11.2, fontweight="bold", color=INK)
    outer = fig.add_gridspec(
        2, 1, height_ratios=(1.38, 0.82), left=0.060, right=0.985,
        top=0.925, bottom=0.095, hspace=0.20,
    )
    top = outer[0].subgridspec(1, 4, wspace=0.055)
    titles = (
        "A  简单隐藏体基准",
        "B  冻结CFM基线",
        "C  CFM + 地震连续制导",
        "D  结构化hard-seismic推理",
    )
    captions = (
        (
            "解析five-body/cuboid真值",
            "三个钻井固定体 + 两个隐藏体",
            "待恢复隐藏组合：[4, 6]",
        ),
        (
            f"隐藏 IoU = {float(baseline['hidden_target_iou']):.3f}",
            f"隐藏召回率 = {float(baseline['hidden_target_recall']):.3f}",
            f"正确观测 hard RMSE = {float(baseline['hard_correct_observation_rmse']):.7f}",
        ),
        (
            f"隐藏 IoU = {float(guided['hidden_target_iou']):.4f}",
            f"隐藏召回率 = {float(guided['hidden_target_recall']):.4f}",
            f"正确观测 hard RMSE = {float(guided['hard_correct_observation_rmse']):.7f}",
            "D4自身attainment（逐步定义）*",
            f"最大/末步 = {100*float(d4_run['best_hard_attainment']):.3f}% / {100*float(d4_run['final_hard_attainment']):.3f}%",
        ),
        (
            f"隐藏 IoU / 召回率 = {float(structured['hidden_target_iou']):.3f} / {float(structured['hidden_target_recall']):.3f}",
            f"正确场 hard attainment = {100*float(structured['hard_attainment_relative_to_base_flow_endpoint']):.0f}%",
            f"正确观测 hard RMSE = {float(structured['hard_correct_observation_rmse']):.0f}",
            "非CFM；仅按hard seismic RMSE选择",
            "真值仅用于事后评价",
        ),
    )
    accents = (OBSERVATION_COLOR, "#9A4B43", "#C17A36", "#2F7D5B")
    for index, (image, title, lines, accent) in enumerate(zip(images, titles, captions, accents)):
        cell = top[index].subgridspec(2, 1, height_ratios=(0.70, 0.30), hspace=0.0)
        ax = fig.add_subplot(cell[0])
        show_render(ax, image)
        ax.set_title(title, fontsize=7.5, fontweight="bold", pad=1.2, color=INK)
        caption_ax = fig.add_subplot(cell[1])
        _panel_caption(caption_ax, lines, accent=accent)

    legend_handles = (
        Patch(facecolor=OBSERVATION_COLOR, edgecolor="none", label="三个固定/已观测目标体（A中实心）"),
        Patch(facecolor=LABEL9_COLOR, edgecolor="none", label="hard模型中的label 9 / 真实隐藏体"),
        Line2D([0], [0], color=TRUTH_OUTLINE_COLOR, lw=1.2, label="隐藏真值轮廓（B–D，仅事后叠加）"),
        Line2D([0], [0], color=OBSERVATION_COLOR, lw=1.2, label="固定体参考轮廓（B–D）"),
    )
    fig.legend(
        handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, 0.937),
        ncol=4, frameon=False, fontsize=5.75, handlelength=1.5, columnspacing=1.15,
    )

    bottom = outer[1].subgridspec(1, 2, width_ratios=(1.63, 0.87), wspace=0.16)
    ax = fig.add_subplot(bottom[0])
    labels = ("冻结CFM", "CFM + 制导", "结构化hard-seismic")
    rows = (baseline, guided, structured)
    y = np.arange(3)[::-1]
    for y_value in y:
        ax.axhline(y_value, color=GRID, lw=0.55, zorder=0)
    for y_value, row in zip(y, rows):
        iou = float(row["hidden_target_iou"])
        recall = float(row["hidden_target_recall"])
        ax.plot([iou, recall], [y_value + 0.10, y_value - 0.10], color="#ADB4B8", lw=0.75, zorder=1)
        ax.scatter(iou, y_value + 0.10, s=28, marker="s", color=IOU_COLOR, edgecolor="white", linewidth=0.45, zorder=3)
        ax.scatter(recall, y_value - 0.10, s=30, marker="o", color=RECALL_COLOR, edgecolor="white", linewidth=0.45, zorder=3)
        for value, yy in ((iou, y_value + 0.10), (recall, y_value - 0.10)):
            label = f"{value:.3f}" if value in (0.0, 1.0) else f"{value:.4f}"
            x_offset = 0.018 if value < 0.96 else -0.018
            align = "left" if value < 0.96 else "right"
            ax.text(value + x_offset, yy, label, ha=align, va="center", fontsize=6.2, color=INK)
    ax.set_xlim(-0.24, 1.07)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks([])
    for y_value, label in zip(y, labels):
        ax.text(-0.225, y_value, label, ha="left", va="center", fontsize=6.25, color=INK)
    ax.set_xticks(np.linspace(0, 1, 6))
    ax.set_xlabel("指标值（0–1）", labelpad=2.0)
    ax.set_title("E  定量对照：隐藏目标交并比与召回率", loc="left", fontsize=8.0, fontweight="bold", color=INK, pad=5)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.grid(axis="x", color=GRID, lw=0.45, zorder=0)
    ax.legend(
        handles=(
            Line2D([0], [0], marker="s", color="none", markerfacecolor=IOU_COLOR, markeredgecolor="white", markersize=5.2, label="隐藏目标交并比（IoU）"),
            Line2D([0], [0], marker="o", color="none", markerfacecolor=RECALL_COLOR, markeredgecolor="white", markersize=5.5, label="隐藏目标召回率"),
        ),
        loc="upper right", frameon=False, fontsize=6.2,
    )

    note = fig.add_subplot(bottom[1])
    note.axis("off")
    note.text(0.0, 1.02, "证据口径", transform=note.transAxes, ha="left", va="top", fontsize=8.0, fontweight="bold", color=INK)
    note.text(
        0.0,
        0.88,
        "Stage7 paired-comparison\n"
        f"  Frozen CFM：RMSE {float(baseline['hard_correct_observation_rmse']):.7f}\n"
        f"  CFM + guidance：RMSE {float(guided['hard_correct_observation_rmse']):.7f}\n"
        f"  Structured：RMSE {float(structured['hard_correct_observation_rmse']):.0f}\n\n"
        "Structured hard-seismic\n"
        "  非CFM / 非CFM posterior\n"
        "  候选仅按hard seismic RMSE排序\n"
        "  真值仅用于事后评价\n\n"
        "*D4 maximum/final attainment 使用D4逐步配对定义，\n"
        "  不与Stage7固定BASE端点attainment混列。",
        ha="left",
        va="top",
        fontsize=6.25,
        color=INK,
        linespacing=1.22,
        transform=note.transAxes,
    )
    note.plot([0.0, 0.98], [0.955, 0.955], transform=note.transAxes, color=GRID, lw=0.65)

    fig.text(
        0.5,
        0.035,
        "地震可辨识 ≠ 连续CFM坐标中可有效利用",
        ha="center",
        va="center",
        fontsize=9.0,
        fontweight="bold",
        color="#8A3F36",
    )
    fig.text(
        0.5,
        0.012,
        "所有三维视图：64³同一坐标范围、同一相机、同一类别配色与透明度规则；深色/青色轮廓为事后参考。",
        ha="center",
        va="bottom",
        fontsize=5.5,
        color=MUTED,
    )
    return save_figure(fig, FIGURES_DIR / FIGURE_ID, title=TITLE)


def _configure_chinese_matplotlib() -> None:
    configure_matplotlib()
    font_family = CHINESE_FONT
    if CHINESE_FONT_FILE.is_file():
        mpl.font_manager.fontManager.addfont(str(CHINESE_FONT_FILE))
        # Matplotlib exposes the first TTC face as the JP family although the
        # installed Noto collection contains the complete CJK glyph set.
        font_family = mpl.font_manager.FontProperties(fname=str(CHINESE_FONT_FILE)).get_name()
    resolved = Path(mpl.font_manager.findfont(font_family, fallback_to_default=False))
    if not resolved.is_file():
        raise RuntimeError(f"required Chinese font is unavailable: {CHINESE_FONT}")
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "axes.unicode_minus": False,
            "axes.titlesize": 8.0,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.6,
            "legend.fontsize": 6.2,
        }
    )


def generate() -> dict[str, object]:
    _configure_chinese_matplotlib()
    ensure_output_dirs()
    data = _load_and_validate()
    paired = data["paired"]
    methods = data["methods"]
    figure_data = {
        "schema": "figure03_cfm_structured_cuboid_contrast_data_v1",
        "figure_id": FIGURE_ID,
        "case_id": "five_body_cuboid_v1",
        "benchmark_semantics": {
            "description": data["q1_config"]["description"],
            "grid_shape": [64, 64, 64],
            "target_label": TARGET_LABEL,
            "fixed_body_count": 3,
            "hidden_body_count": 2,
            "hidden_candidate_indices": list(data["hidden_indices"]),
            "fixed_target_voxels": int(data["fixed_target"].sum()),
            "hidden_target_voxels": int(data["hidden_truth"].sum()),
        },
        "metrics": {
            key: {
                "method": method,
                "hard_correct_observation_rmse": float(paired[method]["hard_correct_observation_rmse"]),
                "hidden_target_iou": float(paired[method]["hidden_target_iou"]),
                "hidden_target_recall": float(paired[method]["hidden_target_recall"]),
                "hidden_target_precision": float(paired[method]["hidden_target_precision"]),
                "hard_attainment_relative_to_base_flow_endpoint": float(paired[method]["hard_attainment_relative_to_base_flow_endpoint"]),
            }
            for key, method in methods.items()
        },
        "d4_protocol_attainment": {
            "maximum": float(data["d4_run"]["best_hard_attainment"]),
            "final": float(data["d4_run"]["final_hard_attainment"]),
            "definition_note": "D4 per-step paired attainment; not the Stage7 fixed-BASE-endpoint column",
        },
        "selection": {
            "structured_is_cfm": False,
            "criterion": "minimum hard observed seismic RMSE only",
            "truth_used_for_selection": False,
            "truth_role": "retrospective display and evaluation only",
        },
        "rendering": {
            "camera_name": CAMERA_NAME,
            "camera": CAMERA_PRESETS[CAMERA_NAME],
            "coordinate_bounds": {"x": [0, 64], "y": [0, 64], "z": [0, 64]},
            "context_opacity": CONTEXT_OPACITY,
            "target_opacity": TARGET_OPACITY,
            "reference_outline_opacity": REFERENCE_OPACITY,
            "palette": {str(key): value for key, value in LABEL_COLORS.items()},
            "observed_reference_color": OBSERVATION_COLOR,
            "truth_outline_color": TRUTH_OUTLINE_COLOR,
        },
        "reproducibility": {
            "new_training": False,
            "hyperparameter_search": False,
            "sample_reselection": False,
            "truth_based_reranking": False,
            "saved_3d_tensor_missing": False,
            "unreproducible_panels": [],
        },
    }
    data_path = FIGURE_DATA_DIR / f"{FIGURE_ID}.json"
    write_json(data_path, figure_data)
    outputs = _make_figure(data)

    png_path = PROJECT_DIR / outputs["png"]
    with Image.open(png_path) as image:
        dpi = image.info.get("dpi")
        png_qc = {
            "pixel_size": list(image.size),
            "dpi": [float(value) for value in dpi] if dpi is not None else None,
        }
    if png_qc["dpi"] is None or any(abs(value - 600.0) > 1.0 for value in png_qc["dpi"]):
        raise ValueError(f"PNG is not 600 dpi: {png_qc}")
    if (PROJECT_DIR / outputs["pdf"]).read_bytes()[:4] != b"%PDF":
        raise ValueError("invalid PDF output")
    if "<svg" not in (PROJECT_DIR / outputs["svg"]).read_text(encoding="utf-8")[:2000]:
        raise ValueError("invalid SVG output")

    generation = generation_record(SCRIPT_PATH)
    generation["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    source_files = (
        (Q1_CONFIG, "authoritative Q0/Q1 resolved benchmark configuration"),
        (Q1_SUMMARY, "authoritative Q0/Q1 machine summary"),
        (Q1_REPORT, "authoritative Q0/Q1 report"),
        (TRUTH_LABELS, "saved analytic five-body truth labels"),
        (ANALYTIC_BASELINE, "saved analytic three-fixed-body baseline labels"),
        (CANDIDATE_MASKS, "saved frozen twelve-body candidate masks"),
        (FIXED_TARGET_MASK, "saved fixed-target-body mask"),
        (CONDITION_MASK, "saved hard-condition mask"),
        (D4_CONFIG, "authoritative D4 resolved configuration"),
        (D4_SUMMARY, "authoritative D4 machine summary"),
        (D4_REPORT, "authoritative D4 report"),
        (D4_TRACE, "authoritative D4 correct BASE_PLUS_PHYSICS trace"),
        (BASE_STATE, "saved frozen-CFM BASE endpoint state"),
        (GUIDED_STATE, "saved D4 BASE_PLUS_PHYSICS best-hard checkpoint state"),
        (GUIDED_FINAL_STATE, "saved D4 BASE_PLUS_PHYSICS final state"),
        (D7_REPORT, "authoritative D7 observation-specificity report"),
        (D7_VERDICT, "authoritative D7 mechanism verdict"),
        (STAGE7_CONFIG, "authoritative Stage7 structured-inference configuration"),
        (STAGE7_SUMMARY, "authoritative Stage7 machine summary"),
        (STAGE7_REPORT, "authoritative Stage7 report"),
        (PAIRED_CSV, "authoritative Stage7 paired comparison"),
        (STRUCTURED_LABELS, "saved Stage7 cuboid correct-arm selected hard labels"),
        (STRUCTURED_HISTORY, "saved Stage7 cuboid selected-event history"),
        (CHECKPOINT, "frozen CFM checkpoint used only for categorical state decoding"),
        (STYLE_PATH, "shared paper style and deterministic export helpers"),
        (data_path, "exact plotted values and rendering policy"),
    )
    manifest = {
        "schema_version": "paper_figure_manifest_v2",
        "figure_id": FIGURE_ID,
        "title_zh": TITLE,
        "git_head": generation["git_head"],
        "generation_timestamp_utc": generation["generated_at_utc"],
        "source_files": [source_record(path, role) for path, role in source_files],
        "source_tensor_hashes": {
            "truth": _tensor_record(TRUTH_LABELS, data["truth"], "analytic truth labels"),
            "analytic_baseline": _tensor_record(ANALYTIC_BASELINE, data["analytic_baseline"], "analytic fixed-body baseline"),
            "candidate_masks": _tensor_record(CANDIDATE_MASKS, data["candidate_masks"], "frozen candidate dictionary"),
            "fixed_target_mask": _tensor_record(FIXED_TARGET_MASK, data["fixed_target"].astype(np.uint8), "fixed-body mask"),
            "cfm_baseline_decoded": _tensor_record(
                BASE_STATE, data["baseline"], "decoded frozen-CFM BASE endpoint",
                decode_rule="cosine-normalized checkpoint embedding argmax minus one",
                checkpoint_sha256=sha256(CHECKPOINT),
            ),
            "cfm_guided_decoded": _tensor_record(
                GUIDED_STATE, data["guided"], "decoded D4 BASE_PLUS_PHYSICS best-hard checkpoint",
                decode_rule="cosine-normalized checkpoint embedding argmax minus one",
                checkpoint_sha256=sha256(CHECKPOINT),
            ),
            "structured_hard_labels": _tensor_record(
                STRUCTURED_LABELS, data["structured"], "Stage7 structured correct-arm hard labels",
                selection="minimum hard observed seismic RMSE only",
            ),
            "hidden_truth_mask": {
                "tensor_content_sha256": _array_sha256(data["hidden_truth"].astype(np.uint8)),
                "derivation": "candidate_masks[[4,6]].any(axis=0), from frozen config truth_candidate_indices",
                "truth_visibility": "retrospective display and evaluation only",
            },
        },
        "panel_sources": {
            "A": [str(path.relative_to(REPOSITORY_ROOT)) for path in (TRUTH_LABELS, FIXED_TARGET_MASK, CANDIDATE_MASKS, Q1_CONFIG)],
            "B": [str(path.relative_to(REPOSITORY_ROOT)) for path in (BASE_STATE, CHECKPOINT, PAIRED_CSV)],
            "C": [str(path.relative_to(REPOSITORY_ROOT)) for path in (GUIDED_STATE, CHECKPOINT, PAIRED_CSV, D4_SUMMARY, D4_TRACE)],
            "D": [str(path.relative_to(REPOSITORY_ROOT)) for path in (STRUCTURED_LABELS, STRUCTURED_HISTORY, STAGE7_SUMMARY, PAIRED_CSV)],
            "E": [str(PAIRED_CSV.relative_to(REPOSITORY_ROOT)), str(D4_SUMMARY.relative_to(REPOSITORY_ROOT))],
        },
        "metrics_source_paths": [
            {
                "metrics": "baseline/guided/structured hard RMSE, hidden IoU, hidden recall",
                "source_path": str(PAIRED_CSV.relative_to(REPOSITORY_ROOT)),
                "cross_check": str(STAGE7_SUMMARY.relative_to(REPOSITORY_ROOT)),
            },
            {
                "metrics": "D4 own-definition maximum/final hard attainment",
                "source_path": str(D4_SUMMARY.relative_to(REPOSITORY_ROOT)),
                "json_selector": "runs[control=correct,mode=BASE_PLUS_PHYSICS]",
                "cross_check": str(D4_TRACE.relative_to(REPOSITORY_ROOT)),
            },
        ],
        "benchmark_semantics": figure_data["benchmark_semantics"],
        "metrics": figure_data["metrics"],
        "d4_protocol_attainment": figure_data["d4_protocol_attainment"],
        "camera_and_rendering": figure_data["rendering"],
        "selection_and_truth_firewall": figure_data["selection"],
        "scientific_boundaries": {
            "training_run": False,
            "hyperparameter_search_run": False,
            "best_sample_reselected": False,
            "truth_based_reranking": False,
            "structured_result_is_cfm": False,
            "structured_result_is_cfm_posterior": False,
            "rendering_only_from_saved_states": True,
        },
        "panel_reproducibility": {
            "unreproducible_panels": [],
            "missing_required_saved_3d_tensors": [],
            "all_panels_directly_reproducible_from_frozen_files": True,
            "note": "Panels B/C decode saved continuous CFM states with the frozen checkpoint embedding; no sampler or forward model is executed.",
        },
        "generation": generation,
        "outputs": output_records(outputs),
        "quality_control": {
            "png": png_qc,
            "pdf_header_valid": True,
            "svg_header_valid": True,
            "shared_camera": True,
            "shared_coordinate_bounds": True,
            "shared_categorical_palette": True,
            "shared_opacity_policy": True,
            "condition_violations": {name: int(np.logical_and(data[name] != data["truth"], data["condition_mask"]).sum()) for name in ("baseline", "guided", "structured")},
            "computed_hidden_metrics_match_paired_csv": True,
            "paired_csv_matches_stage7_summary": True,
            "d4_summary_matches_trace": True,
            "reports_cross_checked": True,
        },
    }
    manifest_path = MANIFESTS_DIR / f"{FIGURE_ID}.json"
    write_json(manifest_path, manifest)
    return {
        "figure_id": FIGURE_ID,
        "outputs": outputs,
        "figure_data": str(data_path.relative_to(PROJECT_DIR)),
        "manifest": str(manifest_path.relative_to(PROJECT_DIR)),
        "quality_control": manifest["quality_control"],
    }


def main() -> None:
    result = generate()
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
