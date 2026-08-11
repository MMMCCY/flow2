#!/usr/bin/env python3
"""Finalize the Stage 10 stop decision, report, and retrospective diagnostic."""

from __future__ import annotations

import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Mapping

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance.prior_ensemble import file_sha256
from scripts.stage10.common import (
    EXPERIMENT_DIR,
    inference_case_dir,
    load_frozen_config,
    load_stage10_inference_case,
    retrospective_case_dir,
    validate_bridge_collection,
)
from scripts.stage9.audit_prior_truth import load_retrospective_case
from scripts.stage9.common import file_record, read_json, utc_now, write_json_x


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _records_by_case_arm(rows: list[dict[str, str]]) -> dict[tuple[str, str], dict[str, str]]:
    return {(row["case_id"], row["arm"]): row for row in rows}


def _save_diagnostic_figure(
    config: Mapping[str, object],
    bridges: Mapping[str, tuple[dict[str, object], dict[str, torch.Tensor]]],
    metric_rows: list[dict[str, str]],
) -> dict[str, object]:
    metrics = _records_by_case_arm(metric_rows)
    case_data = []
    for case_id in config["case_ids"]:
        _, inference = load_stage10_inference_case(config, case_id)
        retrospective_manifest, retrospective = load_retrospective_case(
            retrospective_case_dir(config, case_id), expected_case_id=case_id
        )
        if retrospective_manifest["inference_manifest_sha256"] != file_sha256(
            inference_case_dir(config, case_id) / "manifest.json"
        ):
            raise ValueError("retrospective/inference case link mismatch")
        case_data.append(
            {
                "case_id": case_id,
                "observed": inference["observation_correct"].float().numpy(),
                "mean": bridges[case_id][1]["property_mean"].float().numpy(),
                "uncertainty": bridges[case_id][1]["property_uncertainty"].float().numpy(),
                "probability": bridges[case_id][1]["probability_label9"].float().numpy(),
                "truth": (retrospective["truth_labels"].long() == int(config["target_label"])).numpy(),
            }
        )
    seismic_values = np.concatenate([item["observed"].ravel() for item in case_data])
    seismic_limit = float(np.quantile(np.abs(seismic_values), 0.995))
    property_values = np.concatenate([item["mean"].ravel() for item in case_data])
    property_min, property_max = np.quantile(property_values, [0.01, 0.99])
    uncertainty_values = np.concatenate([item["uncertainty"].ravel() for item in case_data])
    uncertainty_max = float(np.quantile(uncertainty_values, 0.995))
    plt.rcParams.update(
        {
            "font.family": "STIXGeneral",
            "font.size": 7.0,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )
    fig, axes = plt.subplots(3, 5, figsize=(7.0, 5.45), constrained_layout=True)
    probability_cmap = LinearSegmentedColormap.from_list(
        "stage10_probability", ["#F5F5F5", "#F6C58F", "#D95F02"]
    )
    for row, item in enumerate(case_data):
        y_index = 42
        images = (
            axes[row, 0].imshow(
                item["observed"][0, 0, :, y_index, :].T,
                cmap="RdBu_r",
                vmin=-seismic_limit,
                vmax=seismic_limit,
                origin="upper",
                aspect="auto",
                interpolation="nearest",
            ),
            axes[row, 1].imshow(
                item["mean"][0, 0, :, y_index, :].T,
                cmap="viridis",
                vmin=property_min,
                vmax=property_max,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            ),
            axes[row, 2].imshow(
                item["uncertainty"][0, 0, :, y_index, :].T,
                cmap="magma",
                vmin=0.0,
                vmax=uncertainty_max,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            ),
            axes[row, 3].imshow(
                item["probability"][0, 0, :, y_index, :].T,
                cmap=probability_cmap,
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            ),
            axes[row, 4].imshow(
                item["truth"][0, 0, :, y_index, :].T,
                cmap=LinearSegmentedColormap.from_list("truth9", ["#F5F5F5", "#D95F02"]),
                vmin=0.0,
                vmax=1.0,
                origin="lower",
                aspect="equal",
                interpolation="nearest",
            ),
        )
        correct = metrics[(item["case_id"], "correct")]
        wrong = metrics[(item["case_id"], "wrong_case")]
        axes[row, 0].set_ylabel(
            f"{item['case_id'].replace('native_seed', 'seed ')}\n"
            f"AP correct/wrong {float(correct['auprc']):.3f}/{float(wrong['auprc']):.3f}"
        )
        for column, axis in enumerate(axes[row]):
            axis.set_xlabel("x index")
            axis.set_xticks((0, 32, 63))
            axis.set_yticks((0, 32, 63) if column else (0, 160, 319))
            if column == 0:
                axis.set_ylabel(axis.get_ylabel() + "\nTWT sample")
            elif column == 1:
                axis.set_ylabel("z index")
    titles = (
        "(a) Observed seismic\n(inference-visible)",
        "(b) Inverted log-impedance\nposterior mean",
        "(c) Posterior uncertainty\n(12-member spread)",
        "(d) $P_{bridge}$(label 9)\n(inference-visible)",
        "(e) Truth label 9\n(retrospective only)",
    )
    for axis, title in zip(axes[0], titles):
        axis.set_title(title)
    for column, (image, label) in enumerate(
        zip(
            images[:4],
            ("amplitude", "ln impedance", "population std", "probability"),
        )
    ):
        colorbar = fig.colorbar(image, ax=axes[:, column], orientation="horizontal", shrink=0.78, pad=0.03)
        colorbar.set_label(label)
    fig.suptitle(
        "Stage 10-A truth-blind bridge diagnostic — frozen before retrospective truth; gate failed 1/3",
        fontsize=8.5,
    )
    figures = EXPERIMENT_DIR / "figures"
    figures.mkdir(parents=True, exist_ok=False)
    outputs = {}
    metadata = {"Creator": "Stage10 deterministic diagnostic", "CreationDate": "D:20000101000000Z"}
    for extension in ("pdf", "svg", "png"):
        path = figures / f"bridge_diagnostic.{extension}"
        kwargs = {"bbox_inches": "tight"}
        if extension == "png":
            kwargs["dpi"] = 600
            kwargs["metadata"] = {"Software": "Stage10 deterministic diagnostic"}
        elif extension == "pdf":
            kwargs["metadata"] = {"Creator": metadata["Creator"], "CreationDate": None, "ModDate": None}
        elif extension == "svg":
            kwargs["metadata"] = {"Creator": metadata["Creator"], "Date": "2000-01-01T00:00:00Z"}
        fig.savefig(path, **kwargs)
        outputs[extension] = file_record(path, relative_to=REPOSITORY_ROOT)
    plt.close(fig)
    return outputs


def _metric_table(metric_rows: list[dict[str, str]]) -> list[str]:
    index = _records_by_case_arm(metric_rows)
    lines = [
        "| Case | Correct AP | Constant AP | Shuffled AP | Wrong-case AP | Correct Brier | Constant Brier | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|:---:|",
    ]
    controls = {
        row["case_id"]: row
        for row in _read_csv(EXPERIMENT_DIR / "diagnostics/bridge_controls.csv")
    }
    for case_id in ("native_seed20260901", "native_seed20260902", "native_seed20260903"):
        lines.append(
            "| {case} | {correct:.6f} | {constant:.6f} | {shuffled:.6f} | {wrong:.6f} | {brier:.6f} | {constant_brier:.6f} | {passed} |".format(
                case=case_id,
                correct=float(index[(case_id, "correct")]["auprc"]),
                constant=float(index[(case_id, "constant_prior")]["auprc"]),
                shuffled=float(index[(case_id, "shuffled_xy")]["auprc"]),
                wrong=float(index[(case_id, "wrong_case")]["auprc"]),
                brier=float(index[(case_id, "correct")]["brier"]),
                constant_brier=float(index[(case_id, "constant_prior")]["brier"]),
                passed=controls[case_id]["case_pass"],
            )
        )
    return lines


def main() -> None:
    config = load_frozen_config()
    decision_a = read_json(EXPERIMENT_DIR / "diagnostics/stage10a_decision.json")
    if decision_a.get("machine_action") != "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION":
        raise RuntimeError("this finalizer is only valid for the frozen Stage10-A stop")
    bridges = validate_bridge_collection(config)
    metric_rows = _read_csv(EXPERIMENT_DIR / "diagnostics/bridge_information_metrics.csv")
    reports = EXPERIMENT_DIR / "reports"
    if reports.exists():
        raise FileExistsError(f"refusing to reuse immutable reports: {reports}")
    for stage, reason in (
        ("pilot", "Stage10-B was forbidden because Stage10-A failed."),
        ("formal", "Stage10-C was forbidden because Stage10-A failed; Stage10-D was therefore also forbidden."),
    ):
        directory = EXPERIMENT_DIR / stage
        directory.mkdir(parents=True, exist_ok=False)
        write_json_x(
            directory / "NOT_EXECUTED.json",
            {
                "schema": f"stage10_{stage}_not_executed_v1",
                "executed": False,
                "reason": reason,
                "machine_action": decision_a["machine_action"],
            },
        )
    figure_outputs = _save_diagnostic_figure(config, bridges, metric_rows)
    reports.mkdir(parents=True)
    seed_bank_path = EXPERIMENT_DIR / "configs/flow_seed_bank.json"
    property_audit_path = EXPERIMENT_DIR / "audit/property_inversion_provenance.json"
    class_model_path = EXPERIMENT_DIR / "configs/petrophysical_class_model.json"
    machine = {
        "schema": "stage10_machine_decision_v1",
        "status": "complete",
        "machine_decision": "STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION",
        "stage10a_pass": False,
        "stage10a_case_passes": decision_a["case_passes"],
        "stage10a_passing_case_count": decision_a["passing_case_count"],
        "stage10b_executed": False,
        "stage10c_executed": False,
        "stage10d_executed": False,
        "flow_forward_count_stage10": 0,
        "guidance_parameter_sweep_count": 0,
        "checkpoint": config["checkpoint"],
        "checkpoint_sha256": config["checkpoint_sha256"],
        "git_head": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status_short": _git("status", "--short"),
        "registered_cases": config["case_ids"],
        "registered_but_unexecuted_flow_seed_bank": file_record(seed_bank_path, relative_to=REPOSITORY_ROOT),
        "property_inversion_provenance": file_record(property_audit_path, relative_to=REPOSITORY_ROOT),
        "petrophysical_class_model": file_record(class_model_path, relative_to=REPOSITORY_ROOT),
        "truth_firewall": file_record(EXPERIMENT_DIR / "audit/leakage_audit.json", relative_to=REPOSITORY_ROOT),
        "conference_paper_files": {
            "negative_result_diagnostic_only": figure_outputs,
            "new_main_paper_probability_bridge_figure_generated": False,
            "paired_delta_figure_generated": False,
            "reason": "Stage10-C was not authorized because Stage10-A failed."
        },
        "limitations": [
            "Correct bridges beat constant and shuffled controls in all three cases, but failed the strict wrong-case comparison in two cases.",
            "The result does not support case-specific geophysical-to-categorical discrimination under the registered gate.",
            "Inputs use noiseless synthetic inverse-crime seismic and a synthetic distinctive-label9 acoustic codebook.",
            "Twelve-member spread is prior sensitivity, not calibrated posterior uncertainty.",
            "No claim about Flow gain, hard categorical improvement, or deployable measured-geophysics performance is permitted."
        ],
        "forbidden_next_actions_without_reassessment": [
            "petrophysical variance tuning against truth",
            "probability smoothing or sharpening against truth",
            "Stage10-B/C/D Flow execution",
            "D-Flow, SMC, posterior ranking, more wells, or systematic retraining"
        ],
        "completed_at_utc": utc_now(),
    }
    write_json_x(reports / "STAGE10_MACHINE_DECISION.json", machine)
    report_lines = [
        "# Stage 10 final report: truth-blind geophysical probability bridge",
        "",
        "## Decision",
        "",
        "**STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION.** The strict Stage10-A gate passed in only 1/3 registered cases; at least 2/3 were required. Stage10-B, Stage10-C and Stage10-D were not executed, and no Flow forward pass was run by Stage 10.",
        "",
        "The negative classification is specifically caused by the wrong-case control: the correctly located bridge beats the constant and XY-shuffled controls and has lower Brier score in all three cases, but a cyclic wrong-case bridge has higher AUPRC in `native_seed20260901` and `native_seed20260903`. The frozen rule therefore does not permit a claim of case-specific geophysical discrimination.",
        "",
        "## Repository and frozen Flow",
        "",
        f"- Git branch: `{machine['git_branch']}`",
        f"- Git HEAD: `{machine['git_head']}`",
        "- The complete dirty state is recorded verbatim in `STAGE10_MACHINE_DECISION.json` and the pre-Stage10 state in `audit/repository_state.json`.",
        f"- Checkpoint: `{config['checkpoint']}`",
        f"- Checkpoint SHA-256: `{config['checkpoint_sha256']}`",
        "- Weight policy: EMA trainable parameters with raw frozen embedding; 32-step midpoint Euler, canonical condition projection and hard decode remained frozen.",
        "",
        "## Registered cases and seeds",
        "",
        "Cases: `native_seed20260901`, `native_seed20260902`, `native_seed20260903`. The pilot/formal/spatial Flow seeds were frozen in `configs/flow_seed_bank.json` before evaluation, but none were consumed because Stage10-A failed.",
        "",
        "## Property inversion and petrophysical model",
        "",
        "Each case reuses the Phase-5a linearized post-stack log-impedance Tikhonov inversion. The low-frequency prior comprises fixed unranked Stage9A candidates 100--111; no seismic or truth ranking was used. Inputs are observed synthetic seismic, the registered forward/operator configuration, sparse hard conditions and the registered acoustic codebook. Only inverted log acoustic impedance enters the bridge; fixed prior slowness and susceptibility do not.",
        "",
        "The bridge averages `P(k|q_s)` over all 12 posterior samples. Class means are registered codebook log impedances, every class uses the pre-registered half-median adjacent-codebook spacing, and the class prior is uniform. These choices were frozen before the retrospective evaluator opened truth.",
        "",
        "## Truth firewall proof",
        "",
        "The bridge builder and Flow-facing loader have no truth path/tensor argument. All bridge tensors and manifests were written and hashed first; constant, shuffled and wrong-case controls were then written and hashed. Only after both collections validated did `evaluate_bridge_information.py` load `retrospective/truth_labels.pt`. The inference builder used truth only indirectly to the permitted extent that synthetic observed seismic was originally generated from truth.",
        "",
        "## Stage10-A bridge discrimination",
        "",
        *_metric_table(metric_rows),
        "",
        "All metrics use the unconstrained subsurface (225,115 voxels per case). Constant-prior AUPRC equals prevalence by construction. The strict per-case pass requires correct AUPRC above constant, shuffled and wrong-case controls plus lower Brier than constant.",
        "",
        "## Later stages",
        "",
        "- Stage10-B paired Flow pilot: **not executed**.",
        "- Stage10-C formal paired experiment: **not executed**.",
        "- Stage10-D shuffled Flow control: **not executed**.",
        "- Consequently there are no paired IoU deltas, no hard-output gain claim, no paired-delta figure and no new main-paper probability-bridge figure.",
        "",
        "## Conference-paper artifacts",
        "",
        "`figures/bridge_diagnostic.{pdf,svg,png}` is suitable only as a retrospective negative-result or supplementary diagnostic. It marks observed seismic and bridge quantities as inference-visible and truth as retrospective-only. Existing main-paper Figure 1 and Figure 4 are not changed.",
        "",
        "## Limitations and prohibited claims",
        "",
        "This stage uses noiseless inverse-crime seismic, a deliberately distinctive label-9 synthetic acoustic codebook and an uncalibrated 12-member spread. It does not show measured-geophysics performance. It does not show that frozen Flow can exploit the bridge, because Flow execution was forbidden by the gate. Do not tune class variance, smoothing, thresholds or guidance against truth to overturn this result, and do not automatically proceed to D-Flow, SMC, posterior ranking, more wells or retraining without research-plan reassessment.",
        "",
        "## Authoritative files",
        "",
        "- `audit/property_inversion_provenance.json`",
        "- `audit/leakage_audit.json`",
        "- `bridge/manifest.json` and each case `bridge/<case_id>/manifest.json`",
        "- `controls/manifest.json`",
        "- `diagnostics/bridge_information_metrics.csv`",
        "- `diagnostics/bridge_controls.csv`",
        "- `diagnostics/stage10a_decision.json`",
        "- `reports/STAGE10_MACHINE_DECISION.json`",
        "",
    ]
    (reports / "STAGE10_REPORT.md").write_text("\n".join(report_lines), encoding="utf-8")
    print(json.dumps({"status": "COMPLETE", "machine_decision": machine["machine_decision"]}))


if __name__ == "__main__":
    main()
