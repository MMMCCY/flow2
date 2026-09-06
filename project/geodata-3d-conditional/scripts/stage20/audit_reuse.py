#!/usr/bin/env python3
"""Freeze and validate every Stage20 authoritative reuse dependency."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT, REPOSITORY_ROOT / "StructuralGeo-main/src"):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from scripts.stage15.common import base_manifest, read_json, refuse_nonempty, write_json
from scripts.stage19.audit_generator_reuse import run_audit as run_generator_audit
from scripts.stage20.common import CONFIG_DIR, ROOT, SPEC_PATH, asset, require_config, resolve_project_path

DEFAULT_OUTPUT = ROOT / "reuse_audit"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cohort-config", type=Path, default=CONFIG_DIR / "cohort_v1.json")
    parser.add_argument("--evidence-config", type=Path, default=CONFIG_DIR / "evidence_v1.json")
    parser.add_argument("--training-config", type=Path, default=CONFIG_DIR / "training_v1.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def _record(path: Path) -> dict[str, object]:
    return {"path": str(path.resolve()), "sha256": runtime.file_sha256(path)}


def run_audit(cohort_config: Path, evidence_config: Path, training_config: Path) -> dict[str, object]:
    cohort = require_config(cohort_config, "stage20_cohort_v1")
    evidence = require_config(evidence_config, "stage20_continuous_impedance_evidence_v1")
    training = require_config(training_config, "stage20_continuous_impedance_adapter_training_v1")
    stage19_generator_config = PROJECT_DIR / "experiments/stage19_learned_evidence_adapter/configs/cohort_v2.json"
    generator = run_generator_audit(stage19_generator_config)
    if generator.get("machine_decision") != cohort["generator_reuse_required"]:
        raise RuntimeError("STOP_STAGE20_REUSE_MISMATCH: Stage19R generator audit")
    registries = {}
    for split, expected in (("train", 64), ("val", 8)):
        path = resolve_project_path(cohort[split]["reuse_registry"])
        payload = read_json(path)
        if payload.get("schema") != "stage19_split_registry_v1" or payload.get("run_status") != "completed" or payload.get("accepted_count") != expected or len(payload.get("cases", [])) != expected:
            raise RuntimeError(f"STOP_STAGE20_REUSE_MISMATCH: Stage19R {split} registry")
        registries[split] = _record(path)
    paths = {
        "phase4c_acoustic_config": resolve_project_path(evidence["phase4c"]["acoustic_config"]),
        "phase4c_seismic_config": resolve_project_path(evidence["phase4c"]["seismic_config"]),
        "phase4c_source": PROJECT_DIR / "guidance/seismic.py",
        "phase5a_inversion_config": resolve_project_path(evidence["phase5a"]["inversion_config"]),
        "phase5a_source": PROJECT_DIR / "guidance/seismic_inversion.py",
        "stage19r_adapter_source": PROJECT_DIR / "guidance/residual_velocity_adapter.py",
        "base_checkpoint": resolve_project_path(training["base_model"]["checkpoint"]),
        "stage20_spec": SPEC_PATH,
    }
    records = {name: _record(path) for name, path in paths.items()}
    if records["base_checkpoint"]["sha256"] != training["base_model"]["checkpoint_sha256"]:
        raise RuntimeError("STOP_STAGE20_REUSE_MISMATCH: base checkpoint hash")
    # Authoritative successful Phase4c/Phase5a records contain these exact files.
    authoritative = {
        "phase4c": PROJECT_DIR / "experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/manifest.json",
        "phase5a": PROJECT_DIR / "experiments/stage5_acoustic_inversion/outputs/cond_generation_0/model_based_fixed12_v1/manifest.json",
        "stage19r": PROJECT_DIR / "experiments/stage19_learned_evidence_adapter/checkpoints/formal_v2/training_manifest.json",
    }
    authoritative_records = {name: _record(path) for name, path in authoritative.items()}
    return {
        "schema": "stage20_reuse_audit_v1", "run_status": "completed",
        "machine_decision": "STAGE20_REUSE_VALIDATED", "generator_reuse": generator,
        "stage19r_registries": registries, "frozen_dependencies": records,
        "authoritative_manifests": authoritative_records,
        "missing_authoritative_hash_policy": "current_hash_frozen_by_this_preflight",
    }


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir); args.output_dir.mkdir(parents=True)
    try:
        result = run_audit(args.cohort_config, args.evidence_config, args.training_config)
        write_json(args.output_dir / "summary.json", result)
        print(result["machine_decision"])
    except Exception as exc:
        write_json(args.output_dir / "summary.json", {"schema": "stage20_reuse_audit_v1", "run_status": "failed", "machine_decision": "STOP_STAGE20_REUSE_MISMATCH", "error": f"{type(exc).__name__}: {exc}"})
        raise


if __name__ == "__main__":
    main()
