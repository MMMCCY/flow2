#!/usr/bin/env python3
"""Build deterministic well-IDW priors and Phase5a inverted logZ evidence."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.seismic import seismic_operator_from_config
from guidance.seismic_inversion import invert_acoustic_member, parse_inversion_config
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_json
from scripts.stage20.acoustic_semantics import enforce_stage20_acoustic_contract, stage20_labels_to_acoustic
from scripts.stage20.common import CONFIG_DIR, ROOT, SPEC_PATH, asset, assert_truth_firewall, build_low_frequency_acoustic, canonical_tensor_sha256, load_codebook, normalize_logz, require_config, resolve_project_path, validate_asset, validate_wells

DEFAULT_OUTPUT = ROOT / "evidence_fix2"


def build_test_inference_registry(records: list[dict[str, object]]) -> dict[str, object]:
    cases = []
    for record in records:
        if record["split"] != "test":
            continue
        cases.append({
            "case_id": record["case_id"], "split": "test", "root_seed": record["root_seed"],
            "condition_values": record["observation_assets"]["condition_values"],
            "condition_mask": record["observation_assets"]["condition_mask"],
            "subsurface_mask": record["observation_assets"]["subsurface_mask"],
            "observed_seismic": record["observation_assets"]["observed_seismic"],
            "normalized_low_frequency_logz": record["assets"]["normalized_low_frequency_logz"],
            "normalized_inverted_logz": record["assets"]["normalized_inverted_logz"],
        })
    registry = {"schema": "stage20_test_inference_registry_v1", "run_status": "completed", "case_count": len(cases), "cases": cases}
    assert_truth_firewall(registry, "formal TEST inference registry")
    return registry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=CONFIG_DIR / "evidence_v1.json")
    parser.add_argument("--input-registry", type=Path, default=ROOT / "inversion_input_registry_fix2/registry.json")
    parser.add_argument("--semantic-preflight", type=Path, default=ROOT / "acoustic_semantic_preflight/summary.json")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args(); refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage20_continuous_impedance_evidence_v1")
    raw_text = args.input_registry.read_text(encoding="utf-8")
    registry = json.loads(raw_text); assert_truth_firewall(registry, "inversion-only registry")
    if registry.get("schema") != "stage20_inversion_input_registry_v1" or registry.get("run_status") != "completed" or registry.get("case_count") != 84:
        raise ValueError("invalid Stage20 inversion-only registry")
    preflight = read_json(args.semantic_preflight)
    if preflight.get("machine_decision") != "STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT_VALIDATED" or preflight.get("case_count") != 84 or preflight.get("truth_loaded") is not False:
        raise RuntimeError("STOP_STAGE20_ACOUSTIC_SEMANTIC_PREFLIGHT")
    acoustic_path = resolve_project_path(cfg["phase4c"]["acoustic_config"])
    seismic_path = resolve_project_path(cfg["phase4c"]["seismic_config"])
    inversion_path = resolve_project_path(cfg["phase5a"]["inversion_config"])
    property_table, _ = load_codebook(acoustic_path); seismic_cfg = read_json(seismic_path)
    inversion_cfg = parse_inversion_config(read_json(inversion_path))
    if inversion_cfg.prior_relative_weight != 0.001 or inversion_cfg.vertical_smoothness_relative_weight != 0.01:
        raise RuntimeError("STOP_STAGE20_REUSE_MISMATCH: Phase5a regularization")
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable")
    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage20_continuous_evidence_run_v1", Path(__file__), args.config)
    provenance = {
        "revision_reason": "unified_stage20_subsurface_raw_air_acoustic_semantics",
        "previous_failed_evidence": asset(ROOT / "evidence_fix1/run_manifest.json"),
        "previous_failed_evidence_audit": asset(ROOT / "evidence_audit/summary.json"),
        "mapper_source": asset(Path(__file__).with_name("acoustic_semantics.py")),
        "acoustic_codebook": asset(acoustic_path),
        "observations_fix2": asset(ROOT / "observations_fix2/run_manifest.json"),
        "inversion_input_registry_fix2": asset(args.input_registry),
        "semantic_preflight": asset(args.semantic_preflight),
        "truth_loaded_by_evidence_builder": False,
        "parameter_sweep": False,
        "phase5a_inversion_parameters_unchanged": {
            "prior_relative_weight": inversion_cfg.prior_relative_weight,
            "vertical_smoothness_relative_weight": inversion_cfg.vertical_smoothness_relative_weight,
        },
        "low_frequency_prior_parameters_unchanged": {
            "method": cfg["low_frequency_prior"]["method"],
            "power": cfg["low_frequency_prior"]["power"],
            "empty_valid_well_depth_policy": cfg["low_frequency_prior"]["empty_valid_well_depth_policy"],
        },
    }
    manifest.update({"run_status": "running", "truth_loaded_by_runner": False, "posterior_ensemble_constructed": False, **provenance})
    write_json(args.output_dir / "run_manifest.json", manifest)
    records = []
    try:
        for case in registry["cases"]:
            case_id = str(case["case_id"]); obs = case["assets"]
            validate_wells(case["well_xy"])
            values = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/conditions")), "conditions").long()
            condition = normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool()
            support = normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/support")), "support").bool()
            observed = runtime.load_tensor(validate_asset(obs["observed_seismic"], f"{case_id}/seismic")).to(device=device, dtype=torch.float32)
            prior, fallback = build_low_frequency_acoustic(values, condition, support, case["well_xy"], property_table, power=float(cfg["low_frequency_prior"]["power"]))
            prior = prior.to(device)
            condition_target, condition_cleanup = stage20_labels_to_acoustic(values.to(device), support.to(device), property_table.to(device))
            operator, metadata = seismic_operator_from_config(seismic_cfg, grid_shape=support.shape[2:])
            prior_exact, inverted, fields, diagnostics = invert_acoustic_member(prior, observed_seismic=observed, subsurface_mask=support.to(device), condition_target=condition_target, condition_mask=condition.to(device), property_table=property_table.to(device), forward_operator=operator, config=inversion_cfg)
            enforce_stage20_acoustic_contract(prior_exact, support.to(device), property_table.to(device), context=f"{case_id}/LF")
            enforce_stage20_acoustic_contract(inverted, support.to(device), property_table.to(device), context=f"{case_id}/inverted")
            free = support.to(device) & ~condition.to(device)
            if not torch.equal(inverted[:, 1:2][free], prior_exact[:, 1:2][free]):
                raise RuntimeError(f"free-subsurface slowness changed: {case_id}")
            if int((inverted != condition_target)[condition.to(device).expand_as(inverted)].sum()):
                raise RuntimeError(f"condition acoustic violation: {case_id}")
            if not all(bool(torch.isfinite(t).all()) for t in (prior_exact, inverted, fields)):
                raise FloatingPointError(f"non-finite evidence: {case_id}")
            low_logz, inv_logz = prior_exact[:, 0:1].log(), inverted[:, 0:1].log()
            norm_low, lmin, lmax = normalize_logz(low_logz, property_table.to(device))
            norm_inv, lmin2, lmax2 = normalize_logz(inv_logz, property_table.to(device))
            if (lmin, lmax) != (lmin2, lmax2):
                raise RuntimeError("normalization bounds drifted")
            case_dir = args.output_dir / case_id; case_dir.mkdir()
            outputs = {"low_frequency_acoustic": prior_exact.cpu(), "inverted_acoustic": inverted.cpu(), "low_frequency_logz": low_logz.cpu(), "inverted_logz": inv_logz.cpu(), "normalized_low_frequency_logz": norm_low.cpu(), "normalized_inverted_logz": norm_inv.cpu(), "seismic_fields_prior_post": fields.cpu(), "delta_logz_depth": (inv_logz - low_logz).cpu()}
            for name, tensor in outputs.items():
                torch.save(tensor, case_dir / f"{name}.pt")
            case_manifest = {"schema": "stage20_continuous_evidence_case_v2", "run_status": "completed", "revision_reason": provenance["revision_reason"], "case_id": case_id, "split": case["split"], "root_seed": case["root_seed"], "truth_loaded_by_evidence_builder": False, "parameter_sweep": False, "posterior_member_count": 1, "condition_acoustic_cleanup": condition_cleanup, "low_frequency_fallback": fallback, "normalization": {"L_min": lmin, "L_max": lmax, "source_codebook_sha256": runtime.file_sha256(acoustic_path)}, "diagnostics": diagnostics, "operator_metadata": metadata, "assets": {name: asset(case_dir / f"{name}.pt") for name in outputs}, "tensor_content_hashes": {name: canonical_tensor_sha256(tensor) for name, tensor in outputs.items()}}
            write_json(case_dir / "manifest.json", case_manifest)
            records.append({"case_id": case_id, "split": case["split"], "root_seed": case["root_seed"], "manifest": asset(case_dir / "manifest.json"), "assets": case_manifest["assets"], "observation_assets": obs})
            print(f"Stage20 evidence {case_id} completed", flush=True)
        evidence_registry = {"schema": "stage20_evidence_registry_v1", "run_status": "completed", "case_count": len(records), "cases": records}
        inference_registry = build_test_inference_registry(records)
        write_json(args.output_dir / "evidence_registry.json", evidence_registry)
        write_json(args.output_dir / "test_inference_registry.json", inference_registry)
        manifest.update({"run_status": "completed", "case_count": len(records), "truth_loaded_by_runner": False, "posterior_ensemble_constructed": False, "evidence_registry": asset(args.output_dir / "evidence_registry.json"), "test_inference_registry": asset(args.output_dir / "test_inference_registry.json"), "input_registry": asset(args.input_registry), "acoustic_config": asset(acoustic_path), "seismic_config": asset(seismic_path), "inversion_config": asset(inversion_path), "stage20_spec": asset(SPEC_PATH)})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}"}); write_json(args.output_dir / "run_manifest.json", manifest); raise


if __name__ == "__main__":
    main()
