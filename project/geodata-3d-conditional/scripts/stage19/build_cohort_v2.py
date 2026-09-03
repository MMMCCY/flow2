#!/usr/bin/env python3
"""Build the revised Stage19R cohort with complete availability provenance."""

from __future__ import annotations

import argparse
from collections import Counter
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
STRUCTURALGEO_SRC = REPOSITORY_ROOT / "StructuralGeo-main/src"
for root in (PROJECT_DIR, REPOSITORY_ROOT, STRUCTURALGEO_SRC):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

from guidance.full_structuralgeo_benchmark import prepare_candidate
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage19.audit_generator_reuse import run_audit
from scripts.stage19.build_cohort import _save_case
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, require_config, validate_asset, validate_stage19_cohort_contract

DEFAULT_CONFIG = CONFIG_DIR / "cohort_v2.json"
DEFAULT_REUSE_AUDIT = ROOT / "generator_reuse_audit/v1/summary.json"
DEFAULT_OUTPUT = ROOT / "cohort_v2"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--reuse-audit", type=Path, default=DEFAULT_REUSE_AUDIT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _trace_payload(split: str, frozen: dict[str, object], trace: list[dict[str, object]], accepted_count: int, status: str) -> dict[str, object]:
    counts = Counter(reason for row in trace for reason in row["rejection_reasons"])
    return {
        "schema": "stage19r_candidate_trace_v1",
        "run_status": status,
        "split": split,
        "seed_rule": frozen,
        "examined_count": len(trace),
        "accepted_count": accepted_count,
        "rejected_count": len(trace) - accepted_count,
        "rejection_reason_counts": dict(sorted(counts.items())),
        "candidate_trace": trace,
    }


def _write_trace(output: Path, split: str, frozen: dict[str, object], trace: list[dict[str, object]], accepted_count: int, status: str) -> Path:
    path = output / "audit" / f"{split}_candidate_trace.json"
    write_json(path, _trace_payload(split, frozen, trace, accepted_count, status))
    return path


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    config = require_config(args.config, "stage19_cohort_v2")
    audit = read_json(args.reuse_audit)
    if audit.get("run_status") != "completed" or audit.get("machine_decision") != "GENERATOR_REUSE_VALIDATED":
        raise RuntimeError("STOP_GENERATOR_REUSE_MISMATCH: reuse audit not passed")
    # Revalidate the live generator contract at build time, not only the saved audit.
    live_audit = run_audit(args.config)
    if live_audit["historical_replay"] != audit.get("historical_replay"):
        raise RuntimeError("STOP_GENERATOR_REUSE_MISMATCH: live replay differs from audit")
    reference_config = read_json(validate_asset(config["authoritative_reuse"]["benchmark_config"], "benchmark config"))
    validate_stage19_cohort_contract(config, reference_config)
    split_seed_sets = []
    for frozen in config["splits"].values():
        seeds = {int(frozen["start"]) + i * int(frozen["step"]) for i in range(int(frozen["max_candidates"]))}
        if any(seeds & previous for previous in split_seed_sets):
            raise ValueError("Stage19R seed ranges overlap")
        split_seed_sets.append(seeds)

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage19r_cohort_manifest_v2", Path(__file__), args.config)
    manifest.update({"run_status": "running", "revision_of": "stage19_cohort_v1", "generator_reuse_audit": asset(args.reuse_audit), "selection_uses_seismic_evidence_flow_or_downstream_metrics": False})
    write_json(args.output_dir / "cohort_manifest.json", manifest)
    registries, summaries = {}, {}
    active_split = None
    try:
        for split_name in ("train", "val", "test"):
            active_split = split_name
            frozen = config["splits"][split_name]
            accepted: list[dict[str, object]] = []
            trace: list[dict[str, object]] = []
            for offset in range(int(frozen["max_candidates"])):
                root_seed = int(frozen["start"]) + offset * int(frozen["step"])
                candidate = prepare_candidate(root_seed)
                row = {"root_seed": root_seed, "eligible": bool(candidate["eligible"]), "rejection_reasons": list(candidate["rejection_reasons"])}
                trace.append(row)
                _write_trace(args.output_dir, split_name, frozen, trace, len(accepted), "running")
                if candidate["eligible"]:
                    accepted.append(_save_case(args.output_dir, split_name, len(accepted) + 1, candidate))
                    _write_trace(args.output_dir, split_name, frozen, trace, len(accepted), "running")
                    print(f"Stage19R cohort accepted {split_name} {len(accepted)}/{frozen['accepted']} seed={root_seed}", flush=True)
                    if len(accepted) == int(frozen["accepted"]):
                        break
            status = "completed" if len(accepted) == int(frozen["accepted"]) else "failed"
            trace_path = _write_trace(args.output_dir, split_name, frozen, trace, len(accepted), status)
            summary = _trace_payload(split_name, frozen, trace, len(accepted), status)
            summaries[split_name] = {key: value for key, value in summary.items() if key != "candidate_trace"}
            registry = {"schema": "stage19_split_registry_v1", "run_status": status, "split": split_name, "accepted_count": len(accepted), "candidate_count": len(trace), "seed_rule": frozen, "cases": accepted, "candidate_trace_asset": asset(trace_path), "rejection_reason_counts": summary["rejection_reason_counts"]}
            registry_path = args.output_dir / f"{split_name}_registry.json"
            write_json(registry_path, registry)
            registries[split_name] = asset(registry_path)
            write_json(args.output_dir / "audit/rejection_summary.json", {"schema": "stage19r_rejection_summary_v1", "run_status": status, "splits": summaries})
            if status != "completed":
                raise RuntimeError(f"STOP_COHORT_V2_UNAVAILABLE: insufficient eligible {split_name} cases")
        manifest.update({"run_status": "completed", "machine_decision": "COHORT_V2_AVAILABLE", "registries": registries, "candidate_summary": summaries, "total_accepted": 84, "historical_cases_excluded": True, "training_overlap_wording": config["training_overlap_wording"]})
        write_json(args.output_dir / "cohort_manifest.json", manifest)
    except Exception as exc:
        # Trace/summary for the active split was already persisted before raise.
        manifest.update({"run_status": "failed", "machine_decision": "STOP_COHORT_V2_UNAVAILABLE", "active_split": active_split, "registries": registries, "candidate_summary": summaries, "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "cohort_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
