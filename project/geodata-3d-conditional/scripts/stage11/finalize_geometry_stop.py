#!/usr/bin/env python3
"""Freeze a Stage11 geometry-preparation stop without running geophysics."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for path in (PROJECT_DIR, STRUCTURALGEO_SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from guidance.stage11_diverse_geometry import build_stage11_diverse_case
from scripts.stage11.common import (
    DIVERSITY_SPEC_PATH,
    EXPERIMENT_DIR,
    PROTOCOL_PATH,
    REGISTRY_PATH,
    file_record,
    load_frozen_inputs,
    sha256,
    write_json,
)
from scripts.stage9.common import utc_now


def _load_truth(case_id: str) -> np.ndarray:
    value = torch.load(
        EXPERIMENT_DIR / "benchmark" / case_id / "truth_labels.pt",
        map_location="cpu",
        weights_only=False,
    )
    return (value[0, 0] == 9).numpy()


def _centroid(mask: np.ndarray) -> np.ndarray:
    return np.argwhere(mask).mean(axis=0)


def _components(mask: np.ndarray) -> int:
    return int(ndimage.label(mask, ndimage.generate_binary_structure(3, 1))[1])


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, object]:
    records = {}
    for suffix in ("pdf", "svg", "png"):
        path = stem.with_suffix(f".{suffix}")
        kwargs = {"bbox_inches": "tight"}
        if suffix == "png":
            kwargs.update(dpi=600, metadata={"Software": "Stage11 stopped geometry audit"})
        elif suffix == "pdf":
            kwargs["metadata"] = {"Creator": "Stage11 stopped geometry audit", "CreationDate": None, "ModDate": None}
        else:
            kwargs["metadata"] = {"Creator": "Stage11 stopped geometry audit", "Date": "2000-01-01T00:00:00Z"}
        fig.savefig(path, **kwargs)
        records[suffix] = file_record(path)
    plt.close(fig)
    return records


def _failure_figure(
    case_ids: list[str],
    masks: dict[str, np.ndarray],
    failed_case: str,
    failure_message: str,
) -> dict[str, object]:
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 8,
            "axes.titlesize": 8,
            "svg.fonttype": "none",
            "svg.hashsalt": "stage11-geometry-stop-v1",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(1, 5, figsize=(8.4, 2.75))
    fig.subplots_adjust(left=0.04, right=0.98, top=0.70, bottom=0.20, wspace=0.20)
    for index, (axis, case_id) in enumerate(zip(axes, case_ids)):
        if case_id in masks:
            projection = masks[case_id].max(axis=2)
            axis.imshow(projection.T, origin="lower", cmap="Oranges", vmin=0, vmax=1, interpolation="nearest")
            center = _centroid(masks[case_id])
            axis.plot(center[0], center[1], marker="+", color="#1F4E79", ms=7, mew=1.3)
            axis.set_title(f"C{index + 1:02d} built\nn={_components(masks[case_id])}")
            axis.set_xticks((0, 32, 63)); axis.set_yticks((0, 32, 63))
        else:
            axis.set_xlim(0, 1); axis.set_ylim(0, 1); axis.axis("off")
            status = "registered event empty\nSTOP triggered" if case_id == failed_case else "not attempted\nafter STOP"
            axis.text(0.5, 0.55, status, ha="center", va="center", color="#A33A36" if case_id == failed_case else "#666666")
            axis.set_title(f"C{index + 1:02d}")
    fig.suptitle(
        "Stage11 geometry preparation stopped before geophysics",
        fontsize=10,
        color="#A33A36",
        y=0.94,
    )
    fig.text(
        0.5,
        0.035,
        f"{failed_case}: {failure_message}; registered N=5 cohort is incomplete and cannot enter Stage11-A.",
        ha="center",
        va="bottom",
        fontsize=7.2,
    )
    return _save_figure(fig, EXPERIMENT_DIR / "figures/stage11_geometry_diversity")


def main() -> None:
    registry, diversity, protocol = load_frozen_inputs()
    audit_root = EXPERIMENT_DIR / "audit"
    report_root = EXPERIMENT_DIR / "reports"
    diagnostics_root = EXPERIMENT_DIR / "diagnostics"
    protected_outputs = (
        audit_root / "geometry_freeze.json",
        audit_root / "geometry_diversity.csv",
        audit_root / "borehole_layout.json",
        audit_root / "truth_firewall.json",
        report_root / "STAGE11A_DECISION.json",
    )
    completion_time = utc_now()
    if any(path.exists() for path in protected_outputs):
        existing = report_root / "STAGE11A_DECISION.json"
        if not existing.is_file():
            raise RuntimeError("partial Stage11 stop finalization cannot be refreshed")
        prior_decision = json.loads(existing.read_text(encoding="utf-8"))
        if prior_decision.get("machine_decision") != "STOP_BENCHMARK_NOT_DIVERSE":
            raise RuntimeError("refusing to overwrite a different Stage11 decision")
        completion_time = str(prior_decision["completed_at_utc"])
    report_root.mkdir(parents=True, exist_ok=True)
    diagnostics_root.mkdir(parents=True, exist_ok=True)

    cases = list(registry["cases"])
    case_ids = [str(case["case_id"]) for case in cases]
    built_ids = [
        case_id
        for case_id in case_ids
        if (EXPERIMENT_DIR / "benchmark" / case_id / "manifest.json").is_file()
    ]
    if built_ids != case_ids[:2]:
        raise RuntimeError(f"unexpected partial-preparation boundary: {built_ids}")
    failed_case = case_ids[len(built_ids)]
    failed_config = cases[len(built_ids)]
    try:
        build_stage11_diverse_case(
            failed_config,
            borehole_xy=registry["observation_layout"]["borehole_xy"],
        )
    except RuntimeError as error:
        failure_message = str(error)
    else:
        raise RuntimeError("registered failed case unexpectedly became buildable")
    if "empty" not in failure_message:
        raise RuntimeError(f"unexpected Stage11 preparation failure: {failure_message}")

    masks = {case_id: _load_truth(case_id) for case_id in built_ids}
    left, right = masks[built_ids[0]], masks[built_ids[1]]
    intersection = int((left & right).sum())
    union = int((left | right).sum())
    row = {
        "benchmark_status": "INCOMPLETE_STOPPED_BEFORE_GEOPHYSICS",
        "case_i": built_ids[0],
        "case_j": built_ids[1],
        "label9_iou": intersection / union if union else 0.0,
        "centroid_distance_voxels": float(np.linalg.norm(_centroid(left) - _centroid(right))),
        "volume_i": int(left.sum()),
        "volume_j": int(right.sum()),
        "volume_ratio_min_over_max": min(int(left.sum()), int(right.sum())) / max(int(left.sum()), int(right.sum())),
        "component_count_i": _components(left),
        "component_count_j": _components(right),
        "note": "partial pair only; the registered N=5 diversity gate was not evaluated",
    }
    with (audit_root / "geometry_diversity.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader(); writer.writerow(row)

    hit_records = []
    for case_id in built_ids:
        metadata = json.loads((EXPERIMENT_DIR / "benchmark" / case_id / "metadata.json").read_text(encoding="utf-8"))
        for hit in metadata["borehole_target_hits"]:
            hit_records.append({"case_id": case_id, **hit})
    write_json(
        audit_root / "borehole_layout.json",
        {
            "schema": "stage11_borehole_layout_audit_v1",
            "status": "registry_frozen_benchmark_preparation_stopped",
            "layout": registry["observation_layout"],
            "selection_used_hidden_truth": False,
            "built_case_hit_records": hit_records,
            "cases_excluded_due_to_hits": [],
            "unbuilt_cases": case_ids[2:],
        },
    )
    figures = _failure_figure(case_ids, masks, failed_case, failure_message)
    freeze = {
        "schema": "stage11_geometry_freeze_v1",
        "status": "STOPPED_BEFORE_GEOPHYSICAL_COMPUTATION",
        "machine_decision": "STOP_BENCHMARK_NOT_DIVERSE",
        "registered_benchmark_size": len(case_ids),
        "successfully_built_case_count": len(built_ids),
        "successfully_built_cases": built_ids,
        "failed_case": failed_case,
        "failure_type": "REGISTERED_STRUCTURALGEO_EVENT_EMPTY",
        "failure_message": failure_message,
        "unattempted_after_stop": case_ids[3:],
        "diversity_gate_evaluated": False,
        "reason_diversity_gate_not_evaluated": "registered N=5 cohort could not be constructed",
        "case_registry_modified_after_failure": False,
        "case_replacement_attempted": False,
        "case_registry": file_record(REGISTRY_PATH),
        "diversity_spec": file_record(DIVERSITY_SPEC_PATH),
        "protocol": file_record(PROTOCOL_PATH),
        "partial_geometry_csv": file_record(audit_root / "geometry_diversity.csv"),
        "figures": figures,
        "synthetic_seismic_forward_count": 0,
        "flow_prior_forward_count": 0,
        "flow_guidance_forward_count": 0,
        "property_inversion_count": 0,
        "bridge_construction_count": 0,
        "completed_at_utc": completion_time,
    }
    write_json(audit_root / "geometry_freeze.json", freeze)
    firewall = {
        "schema": "stage11_truth_firewall_audit_v1",
        "status": "STOPPED_AT_TRUTH_ONLY_GEOMETRY_PREPARATION",
        "truth_accessed_for_geometry_preparation": True,
        "truth_accessed_by_geophysical_inference": False,
        "observed_seismic_generated": False,
        "flow_prior_sampling_executed": False,
        "flow_guidance_executed": False,
        "property_inversion_executed": False,
        "probability_bridge_executed": False,
        "retrospective_stage11a_evaluation_executed": False,
        "old_stage10_cases_used": False,
        "stop_rule_enforced": True,
        "geometry_freeze": file_record(audit_root / "geometry_freeze.json"),
    }
    write_json(audit_root / "truth_firewall.json", firewall)
    not_executed = {
        "schema": "stage11a_not_executed_v1",
        "status": "NOT_EXECUTED",
        "reason": "STOP_BENCHMARK_NOT_DIVERSE at registered-cohort construction",
        "transfer_auprc": None,
        "transfer_brier": None,
        "transfer_roc_auc": None,
        "prior_vs_post": None,
        "shuffled_controls": None,
    }
    write_json(diagnostics_root / "STAGE11A_NOT_EXECUTED.json", not_executed)
    decision = {
        "schema": "stage11a_machine_decision_v1",
        "status": "COMPLETE_STOP",
        "machine_decision": "STOP_BENCHMARK_NOT_DIVERSE",
        "benchmark_geometry_diverse": False,
        "benchmark_diversity_evaluable": False,
        "stage11a_executed": False,
        "stage11b_authorized": False,
        "flow_guidance_executed": False,
        "failure_case": failed_case,
        "failure_message": failure_message,
        "original_stage10_machine_decision_unchanged": "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION",
        "stage10r_interpretation_unchanged": "CASE_GEOMETRY_CONFUNDED",
        "geometry_freeze": file_record(audit_root / "geometry_freeze.json"),
        "truth_firewall": file_record(audit_root / "truth_firewall.json"),
        "completed_at_utc": completion_time,
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPOSITORY_ROOT, text=True).strip(),
    }
    write_json(report_root / "STAGE11A_DECISION.json", decision)
    report = f"""# Stage 11 — Independent Diverse-Geometry Probability-Bridge Validation

## Machine decision

`STOP_BENCHMARK_NOT_DIVERSE`

The registered N=5 benchmark could not be constructed. StructuralGeo produced an
empty registered target event for `{failed_case}`. Cases 01 and 02 were built;
cases 04 and 05 were not attempted after the stop. The case registry was not
modified, N was not reduced, and no replacement case was selected.

## Geometry gate

The full pairwise diversity criterion was **not evaluable** because only 2/5
registered cases were successfully built. The single partial pair is retained in
`audit/geometry_diversity.csv` for forensic provenance and is not treated as a
benchmark result.

## Stage11-A status

Stage11-A was **NOT EXECUTED**. Consequently there is no N×N AUPRC, Brier, or
ROC-AUC transfer matrix; no prior-versus-post result; and no shuffled/constant
control result. Producing those artifacts after the failed geometry gate would
violate the pre-registered stop rule.

## Execution counts

- Synthetic seismic forwards: 0
- Flow prior forwards: 0
- Flow guidance forwards: 0
- Property inversions: 0
- Probability bridges: 0

## Frozen prior conclusions

The original Stage10 decision remains `STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION`.
The Stage10R interpretation remains `CASE_GEOMETRY_CONFUNDED`, with complementary
finding `SEISMIC_ADDS_INCREMENTAL_INFORMATION`. Neither is modified by this
benchmark-construction failure.

## Authorization

Stage11-B is **not authorized**. No next-stage computation was implemented.
"""
    (report_root / "STAGE11A_REPORT.md").write_text(report, encoding="utf-8")
    write_json(
        EXPERIMENT_DIR / "manifest.json",
        {
            "schema": "stage11_experiment_manifest_v1",
            "status": "COMPLETE_STOP",
            "machine_decision": "STOP_BENCHMARK_NOT_DIVERSE",
            "configs": {
                "case_registry": file_record(REGISTRY_PATH),
                "diversity_spec": file_record(DIVERSITY_SPEC_PATH),
                "protocol": file_record(PROTOCOL_PATH),
            },
            "audit": {
                "geometry_freeze": file_record(audit_root / "geometry_freeze.json"),
                "geometry_diversity": file_record(audit_root / "geometry_diversity.csv"),
                "borehole_layout": file_record(audit_root / "borehole_layout.json"),
                "truth_firewall": file_record(audit_root / "truth_firewall.json"),
            },
            "reports": {
                "decision": file_record(report_root / "STAGE11A_DECISION.json"),
                "report": file_record(report_root / "STAGE11A_REPORT.md"),
            },
            "figure": figures,
        },
    )
    print(json.dumps({"status": "COMPLETE_STOP", "machine_decision": "STOP_BENCHMARK_NOT_DIVERSE", "failed_case": failed_case}, sort_keys=True))


if __name__ == "__main__":
    main()
