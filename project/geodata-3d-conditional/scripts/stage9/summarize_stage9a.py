#!/usr/bin/env python3
"""Aggregate the immutable formal Stage9A case audits and stop."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import shlex
import subprocess
import sys
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_DIR.parents[1]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from guidance import prior_ensemble as ensemble
from scripts.stage9.audit_prior_ranking import RANKING_FILENAMES
from scripts.stage9.common import (
    create_staging_directory,
    file_record,
    publish_staging_directory,
    read_csv,
    read_json,
    utc_now,
    write_json_x,
)


def parse_args() -> argparse.Namespace:
    experiment = PROJECT_DIR / "experiments/stage9_flow_prior_posterior"
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=experiment / "configs/stage9a_prior_support_v1.json",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=experiment / "runs/stage9a_prior_support_v1/formal",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment / "reports/stage9a_prior_support_v1",
    )
    return parser.parse_args()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", *arguments], cwd=REPOSITORY_ROOT, text=True
    ).strip()


def _validate_file_record(root: Path, record: Mapping[str, object]) -> None:
    path = Path(root) / str(record["path"])
    if ensemble.file_sha256(path) != record.get("sha256"):
        raise ValueError(f"formal audit output hash mismatch: {path}")


def validate_formal_case(case_root: Path, case_id: str) -> dict[str, object]:
    pool_root = case_root / "pool"
    ranking_root = case_root / "ranking"
    audit_root = case_root / "audit"
    pool = read_json(pool_root / "manifest.json")
    ranking = read_json(ranking_root / "ranking_manifest.json")
    audit = read_json(audit_root / "audit_manifest.json")
    expected = {
        "pool": (pool, ensemble.STAGE9A_POOL_SCHEMA),
        "ranking": (ranking, ensemble.STAGE9A_RANKING_SCHEMA),
        "audit": (audit, ensemble.STAGE9A_AUDIT_SCHEMA),
    }
    for name, (manifest, schema) in expected.items():
        if manifest.get("schema") != schema or manifest.get("status") != "complete":
            raise RuntimeError(f"formal {name} is incomplete: {case_id}")
        if manifest.get("mode") != "formal" or manifest.get("scientific_evidence") is not True:
            raise RuntimeError(f"smoke/non-scientific {name} cannot enter formal summary")
        if manifest.get("case_id") != case_id:
            raise ValueError(f"formal {name} case ID mismatch")
        if int(manifest.get("candidate_count", -1)) != 1024:
            raise RuntimeError(f"formal {name} does not contain 1024 candidates")
    if audit["pool_manifest_sha256_before_truth_load"] != ensemble.file_sha256(
        pool_root / "manifest.json"
    ):
        raise ValueError("pool changed after retrospective validation")
    if audit["ranking_manifest_sha256_before_truth_load"] != ensemble.file_sha256(
        ranking_root / "ranking_manifest.json"
    ):
        raise ValueError("ranking changed after retrospective validation")
    for record in audit["outputs"].values():
        _validate_file_record(audit_root, record)
    truth_rows = read_csv(audit_root / "truth_metrics.csv")
    if len(truth_rows) != 1024:
        raise RuntimeError("formal truth-metric table is incomplete")
    metric_index = {row["candidate_id"]: row for row in truth_rows}
    correct_ranking = read_csv(ranking_root / RANKING_FILENAMES["correct"])
    seismic_top = correct_ranking[0]
    seismic_top_metrics = metric_index[seismic_top["candidate_id"]]
    oracle_label9 = max(
        truth_rows,
        key=lambda row: (float(row["label9_iou"]), row["candidate_id"]),
    )
    correlations = read_csv(audit_root / "correlations.csv")
    enrichment = read_csv(audit_root / "enrichment.csv")
    return {
        "case_id": case_id,
        "SUPPORT_PASS": bool(audit["case_support_pass"]),
        "DISCRIMINATION_PASS": bool(audit["case_discrimination_pass"]),
        "support_passing_candidate_count": len(audit["support_passing_candidate_ids"]),
        "support_passing_candidate_ids": audit["support_passing_candidate_ids"],
        "discrimination_checks": audit["discrimination_checks"],
        "ensemble": audit["ensemble"],
        "deployable_correct_seismic_top": {
            "candidate_id": seismic_top["candidate_id"],
            "hard_seismic_rmse": float(seismic_top["hard_seismic_rmse"]),
            "global_accuracy_retrospective": float(
                seismic_top_metrics["global_accuracy"]
            ),
            "truth_present_mean_iou_retrospective": float(
                seismic_top_metrics["truth_present_mean_iou"]
            ),
            "label9_iou_retrospective": float(seismic_top_metrics["label9_iou"]),
            "label9_recall_retrospective": float(
                seismic_top_metrics["label9_recall"]
            ),
            "major_component_mean_recall_retrospective": float(
                seismic_top_metrics["major_component_mean_recall"]
            ),
            "selection_used_truth": False,
        },
        "oracle_best_label9_iou": {
            "candidate_id": oracle_label9["candidate_id"],
            "label9_iou": float(oracle_label9["label9_iou"]),
            "label9_precision": float(oracle_label9["label9_precision"]),
            "label9_recall": float(oracle_label9["label9_recall"]),
            "major_component_min_recall": float(
                oracle_label9["major_component_min_recall"]
            ),
            "major_component_mean_recall": float(
                oracle_label9["major_component_mean_recall"]
            ),
            "deployable_selector": False,
        },
        "correlations": [
            {
                "observation": row["observation"],
                "metric": row["metric"],
                "spearman_rho": float(row["spearman_rho"]),
            }
            for row in correlations
        ],
        "enrichment": [
            {
                "observation": row["observation"],
                "metric": row["metric"],
                "subset": row["subset"],
                "count": int(row["count"]),
                "mean": float(row["mean"]),
                "enrichment": float(row["enrichment"]),
            }
            for row in enrichment
        ],
        "runtime_seconds": float(pool["runtime_seconds"]),
        "hard_seismic_forward_count": int(pool["hard_seismic_forward_count"]),
        "flow_velocity_forward_count": int(pool["flow_velocity_forward_count"]),
        "pool_manifest": file_record(pool_root / "manifest.json"),
        "ranking_manifest": file_record(ranking_root / "ranking_manifest.json"),
        "audit_manifest": file_record(audit_root / "audit_manifest.json"),
    }


def render_report(summary: Mapping[str, object]) -> str:
    lines = [
        "# Stage 9A report: frozen Flow-prior support and geophysical enrichment",
        "",
        f"Decision: **SUPPORT={'PASS' if summary['SUPPORT_PASS'] else 'FAIL'}; "
        f"DISCRIMINATION={'PASS' if summary['DISCRIMINATION_PASS'] else 'FAIL'}**",
        "",
        f"Machine next action: `{summary['NEXT_ACTION']}`.",
        "",
        "## Frozen execution",
        "",
        f"- Primary cases: 3 deterministic StructuralGeo-native cases; 1024 independent frozen-Flow samples per case.",
        f"- Hard seismic forwards: {summary['hard_seismic_forward_count_total']} candidate forwards.",
        f"- Flow velocity forwards: {summary['flow_velocity_forward_count_total']} (32 per candidate).",
        f"- Candidate-generation runtime: {summary['candidate_generation_runtime_seconds_total']:.1f} seconds total.",
        "- Frozen EMA policy, normal embedding, 32-step midpoint fixed Euler, exact condition projection, hard decode, and hard seismic forward were used throughout.",
        "- No training, structured correction, pCN, D-Flow, SMC, gravity, or truth-visible ranking was run.",
        "",
        "## Inference-visible evidence",
        "",
        "| Case | Correct seismic top | Hard RMSE | Unique models |",
        "|---|---|---:|---:|",
    ]
    for case in summary["cases"]:
        selected = case["deployable_correct_seismic_top"]
        lines.append(
            f"| `{case['case_id']}` | `{selected['candidate_id']}` | "
            f"{selected['hard_seismic_rmse']:.7g} | {case['ensemble']['unique_hard_model_count']}/1024 |"
        )
    lines.extend(
        [
            "",
            "All four rankings used the same cached float32 hard seismic prediction for every candidate. Rankings were frozen by ascending RMSE with candidate-ID ties before truth was loaded.",
            "",
            "## Retrospective truth evidence and oracle support ceiling",
            "",
            "| Case | Support | Passing candidates | Oracle best label-9 IoU | Discrimination |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for case in summary["cases"]:
        oracle = case["oracle_best_label9_iou"]
        lines.append(
            f"| `{case['case_id']}` | {case['SUPPORT_PASS']} | "
            f"{case['support_passing_candidate_count']} | {oracle['label9_iou']:.4f} | "
            f"{case['DISCRIMINATION_PASS']} |"
        )
    lines.extend(
        [
            "",
            f"Overall support passes in {summary['support_case_pass_count']}/3 cases; discrimination passes in {summary['discrimination_case_pass_count']}/3 cases.",
            "",
            "The best truth candidate is an oracle support ceiling and is not a deployable selector. The correct-observation seismic ranking is deployable in the synthetic protocol because it never receives truth.",
            "",
            "## Scope and stop",
            "",
            "A lower seismic loss alone is not project success. These three synthetic inverse-crime cases do not establish field generalization. Stage9A determines only frozen-prior support and hard-likelihood enrichment under this registered setup.",
            "",
            "Stage9A stops here. The machine next action is a recommendation only; Stage9B, Stage9C, posterior weighting, adaptive proposals, SMC, D-Flow, new likelihoods, and training were not implemented.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    config = read_json(args.config)
    ensemble.validate_protocol_config(config)
    cases = [
        validate_formal_case(
            args.runs_root / str(case_config["case_id"]),
            str(case_config["case_id"]),
        )
        for case_config in config["primary_cases"]
    ]
    support_count = sum(case["SUPPORT_PASS"] for case in cases)
    discrimination_count = sum(case["DISCRIMINATION_PASS"] for case in cases)
    required = int(config["overall_case_pass_minimum"])
    support_pass = support_count >= required
    discrimination_pass = discrimination_count >= required
    summary = {
        "schema": ensemble.STAGE9A_SUMMARY_SCHEMA,
        "status": "complete",
        "SUPPORT_PASS": support_pass,
        "DISCRIMINATION_PASS": discrimination_pass,
        "NEXT_ACTION": ensemble.next_action(support_pass, discrimination_pass),
        "support_case_pass_count": support_count,
        "discrimination_case_pass_count": discrimination_count,
        "overall_case_pass_minimum": required,
        "primary_case_count": 3,
        "formal_candidates_per_case": 1024,
        "formal_candidate_count_total": 3072,
        "hard_seismic_forward_count_total": sum(
            case["hard_seismic_forward_count"] for case in cases
        ),
        "flow_velocity_forward_count_total": sum(
            case["flow_velocity_forward_count"] for case in cases
        ),
        "candidate_generation_runtime_seconds_total": sum(
            case["runtime_seconds"] for case in cases
        ),
        "cases": cases,
        "truth_firewall": {
            "candidate_runner_received_truth": False,
            "ranking_runner_received_truth": False,
            "all_pool_and_ranking_hashes_frozen_before_truth_load": True,
            "truth_used_for_metrics_only": True,
        },
        "training_performed": False,
        "structured_search_modified_or_run": False,
        "posterior_chain_run": False,
        "stage9b_or_stage9c_implemented": False,
        "limitations": [
            "best truth candidate is an oracle ceiling, not a deployable selector",
            "lower seismic loss alone is not project success",
            "three synthetic inverse-crime cases do not establish field generalization",
            "Stage9A tests only frozen-prior support and likelihood enrichment",
        ],
        "completed_at_utc": utc_now(),
        "exact_command": shlex.join([sys.executable, *sys.argv]),
        "git_branch": _git("branch", "--show-current"),
        "git_sha": _git("rev-parse", "HEAD"),
        "git_status": _git("status", "--short"),
        "config": file_record(args.config),
        "spec": file_record(PROJECT_DIR / "docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md"),
        "runner": file_record(Path(__file__)),
        "prior_ensemble": file_record(Path(ensemble.__file__)),
    }
    staging = create_staging_directory(args.output_dir)
    write_json_x(staging / "summary.json", summary)
    with (staging / "STAGE9A_REPORT.md").open("x", encoding="utf-8") as stream:
        stream.write(render_report(summary))
    publish_staging_directory(staging, args.output_dir)
    print(
        json.dumps(
            {
                "SUPPORT_PASS": support_pass,
                "DISCRIMINATION_PASS": discrimination_pass,
                "NEXT_ACTION": summary["NEXT_ACTION"],
                "output_dir": str(args.output_dir),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
