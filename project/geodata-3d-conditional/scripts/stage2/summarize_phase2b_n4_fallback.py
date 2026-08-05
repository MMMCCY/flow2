#!/usr/bin/env python3
"""Audit the post-bracket Phase-2b paired_c100 seed-42 n=4 fallback."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Mapping


PROJECT_DIR = Path(__file__).resolve().parents[2]
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

import inference_runtime as runtime
from scripts.stage2.summarize_phase2a import read_json, write_csv, write_json
from scripts.stage2.summarize_phase2b_n4_bracket import audit_n4_level
from scripts.stage2.summarize_phase2b_screen import (
    _resolved_config_path,
    load_manifest,
)


FALLBACK_LEVEL = "paired_c100"
EXPECTED_BRACKET = {
    "paired_c025": "transition_region",
    "paired_c010": "confirmed_seed42_failure",
}


def parse_args() -> argparse.Namespace:
    experiment_root = PROJECT_DIR / "experiments/stage2_property"
    parser = argparse.ArgumentParser(
        description="Audit the frozen Phase-2b paired_c100 n=4 fallback.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=experiment_root
        / "configs/phase2b_codebook_ambiguity_v1/sweep_manifest.json",
    )
    parser.add_argument(
        "--runs-root",
        type=Path,
        default=experiment_root
        / "runs/cond_generation_0/phase2b_codebook_ambiguity_v1",
    )
    parser.add_argument(
        "--bracket-summary",
        type=Path,
        default=experiment_root
        / "reports/phase2b_codebook_ambiguity_v1_n4_bracket_seed42/summary.json",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--n-steps", type=int, default=32)
    parser.add_argument(
        "--run-name",
        default="seed42_n4_s32_a025_c025_followup",
    )
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--guided-name", default="alpha025")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=experiment_root
        / "reports/phase2b_codebook_ambiguity_v1_n4_fallback_c100_seed42",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_bracket_prerequisite(summary: Mapping[str, object]) -> dict[str, str]:
    """Require the completed frozen bracket that triggered this follow-up."""
    observed = {
        str(row["level_id"]): str(row["classification"])
        for row in summary["levels"]
    }
    if observed != EXPECTED_BRACKET:
        raise ValueError(
            "fallback requires the frozen c025-transition/c010-failure bracket; "
            f"observed {observed}"
        )
    selection = summary["selection"]
    if bool(selection["promote_to_multiseed"]):
        raise ValueError("fallback is invalid when the original bracket promotes a level")
    return observed


def fallback_decision(level_row: Mapping[str, object]) -> dict[str, object]:
    """Apply the pre-registered 4/4-plus-diversity promotion rule."""
    classification = str(level_row["classification"])
    promote = classification == "confirmed_seed42_pass"
    return {
        "status": (
            "seed42_fallback_confirmed"
            if promote
            else "no_seed42_fallback_candidate"
        ),
        "candidate_level": FALLBACK_LEVEL if promote else None,
        "candidate_classification": classification,
        "promote_to_multiseed": promote,
    }


def _report_markdown(summary: Mapping[str, object]) -> str:
    row = summary["level"]
    selection = summary["selection"]
    return "\n".join(
        [
            "# Phase-2b post-bracket paired_c100 seed-42 n=4 fallback",
            "",
            "## Decision",
            "",
            f"**{summary['decision']}**",
            "",
            "| Level | Pair gates | Diversity | Classification | Mean label-9 IoU / P / R |",
            "|---|---:|---|---|---|",
            f"| {row['level_id']} | {row['pair_gate_pass_count']}/4 | "
            f"{row['diversity_gate_pass']} | {row['classification']} | "
            f"{float(row['mean_guided_target_iou']):.4f} / "
            f"{float(row['mean_guided_target_precision']):.4f} / "
            f"{float(row['mean_guided_target_recall']):.4f} |",
            "",
            "## Promotion",
            "",
            f"- Candidate: `{selection.get('candidate_level')}`.",
            f"- Multi-seed promotion authorized: {selection['promote_to_multiseed']}.",
            "- The original c025/c010 bracket remains transition/failure and is not rewritten.",
            "- This remains a truth-derived full-resolution property experiment, not measured geophysics.",
            "",
        ]
    )


def main() -> None:
    args = parse_args()
    existing = list(args.output_dir.iterdir()) if args.output_dir.exists() else []
    if existing and not args.overwrite:
        raise FileExistsError(
            f"output directory is non-empty; pass --overwrite: {args.output_dir}"
        )

    bracket_summary = read_json(args.bracket_summary)
    bracket_outcomes = validate_bracket_prerequisite(bracket_summary)
    manifest = load_manifest(args.manifest)
    known = {str(level["id"]): level for level in manifest["levels"]}
    level = known[FALLBACK_LEVEL]
    config_path = _resolved_config_path(args.manifest, level)
    pair_root = args.runs_root / FALLBACK_LEVEL / args.run_name
    pair_rows, level_row = audit_n4_level(
        level_id=FALLBACK_LEVEL,
        level=level,
        config_path=config_path,
        pair_root=pair_root,
        seed=args.seed,
        n_steps=args.n_steps,
        baseline_name=args.baseline_name,
        guided_name=args.guided_name,
    )
    selection = fallback_decision(level_row)
    if selection["promote_to_multiseed"]:
        decision = "PASS: paired_c100 qualifies for unchanged multi-seed testing"
    else:
        decision = "NO PASS: paired_c100 does not qualify for multi-seed testing"
    summary = {
        "decision": decision,
        "scope": "Phase-2b post-bracket paired_c100 seed-42 n=4 fallback",
        "is_original_bracket": False,
        "is_measured_geophysics": False,
        "seed": args.seed,
        "n_samples": 4,
        "n_steps": args.n_steps,
        "alpha": 0.25,
        "max_guidance_ratio": 0.25,
        "source_bracket_summary_sha256": runtime.file_sha256(args.bracket_summary),
        "source_bracket_outcomes": bracket_outcomes,
        "level": level_row,
        "selection": selection,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "paired_samples.csv", pair_rows)
    write_csv(args.output_dir / "level_summary.csv", [level_row])
    write_json(args.output_dir / "summary.json", summary)
    (args.output_dir / "REPORT.md").write_text(
        _report_markdown(summary),
        encoding="utf-8",
    )
    print(args.output_dir / "REPORT.md")
    print(decision)


if __name__ == "__main__":
    main()
