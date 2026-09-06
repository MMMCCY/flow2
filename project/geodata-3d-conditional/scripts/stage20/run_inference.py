#!/usr/bin/env python3
"""Truth-blind four-arm Stage20 TEST inference runner."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.generator_posterior import projected_fixed_euler_prior_sample
from guidance.residual_velocity_adapter import ResidualVelocityAdapter, fixed_euler_adapter_sample
from guidance.seismic import tensor_sha256
from scripts.stage15.common import base_manifest, git_value, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, asset, freeze_base_model, require_config, resolve_project_path, validate_asset, runtime_evidence, wrong_case_evidence
from scripts.stage20.formal_inference_provenance import freeze_formal_output_index
from scripts.stage20.train_adapter import EXPECTED_EVIDENCE_GATE_SHA256, EXPECTED_EVIDENCE_REGISTRY_SHA256, EXPECTED_TRAINING_CONFIG_SHA256

ARMS = ("FLOW_ONLY", "ADAPTER_PRIOR_ONLY", "ADAPTER_CORRECT", "ADAPTER_WRONG_CASE")
DEFAULT_CONFIG = CONFIG_DIR / "inference_v1.json"
DEFAULT_TRAINING_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_INFERENCE_REGISTRY = ROOT / "evidence_fix2/test_inference_registry.json"
DEFAULT_TRAINING_MANIFEST = ROOT / "checkpoints/formal_v1/training_manifest.json"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/formal_v1/adapter_checkpoint.pt"
DEFAULT_OUTPUT = ROOT / "formal/inference_v1"
PREVIOUS_RUNNER_SHA256 = "11e9f12c467ff27caae0bf1cdd17fe6805debbb73cb3722eca32773e9c9f28a6"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--training-config", type=Path, default=DEFAULT_TRAINING_CONFIG)
    parser.add_argument("--inference-registry", type=Path, default=DEFAULT_INFERENCE_REGISTRY)
    parser.add_argument("--training-manifest", type=Path, default=DEFAULT_TRAINING_MANIFEST)
    parser.add_argument("--adapter-checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode()); digest.update(tensor_sha256(parameter.detach().cpu()).encode())
    return digest.hexdigest()


def _case_inputs(case: dict[str, object]) -> dict[str, torch.Tensor]:
    case_id = str(case["case_id"])
    result = {
        "condition_values": normalize_volume(runtime.load_tensor(validate_asset(case["condition_values"], f"{case_id}/condition values")), "condition values").long(),
        "condition_mask": normalize_volume(runtime.load_tensor(validate_asset(case["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool(),
        "subsurface": normalize_volume(runtime.load_tensor(validate_asset(case["subsurface_mask"], f"{case_id}/subsurface")), "subsurface").bool(),
        "prior": normalize_volume(runtime.load_tensor(validate_asset(case["normalized_low_frequency_logz"], f"{case_id}/prior evidence")), "prior evidence", torch.float32),
        "correct": normalize_volume(runtime.load_tensor(validate_asset(case["normalized_inverted_logz"], f"{case_id}/inverted evidence")), "inverted evidence", torch.float32),
    }
    for name in ("prior", "correct"):
        if not bool(torch.isfinite(result[name]).all()):
            raise RuntimeError(f"non-finite Stage20 {name} evidence: {case_id}")
    return result


def load_validated_adapter_checkpoint(
    training_manifest_path: Path,
    checkpoint_path: Path,
    training_config_path: Path,
    training_config: dict[str, object],
    *,
    require_formal: bool,
    map_location: torch.device | str = "cpu",
) -> dict[str, object]:
    """Reject non-formal, stale or mismatched adapter checkpoints."""
    try:
        manifest = read_json(training_manifest_path)
        if manifest.get("run_status") != "completed":
            raise ValueError("training manifest is incomplete")
        if require_formal:
            expected = {
                "run_class": "formal_training",
                "smoke_subset": False,
                "optimizer_updates": 1024,
                "epochs": 4,
                "base_model_unchanged": True,
                "base_gradients_absent": True,
                "base_finetuning": False,
                "ema_applied": True,
                "checkpoint_epoch": 4,
                "checkpoint_selection": "epoch4_final",
                "adapter_parameter_count": 54003,
                "optimizer_contains_adapter_only": True,
                "training_gradients_finite": True,
                "training_gradients_nonzero": True,
                "all_losses_finite": True,
                "residual_ratio_cap_passed": True,
                "checkpoint_save_load_validated": True,
                "test_truth_loaded": False,
            }
            for field, value in expected.items():
                if manifest.get(field) != value:
                    raise ValueError(f"invalid formal training field: {field}")
            if manifest.get("base_model_hash_before") != manifest.get("base_model_hash_after"):
                raise ValueError("base parameter hash changed")
            if manifest.get("base_state_dict_hash_before") != manifest.get("base_state_dict_hash_after"):
                raise ValueError("base state_dict hash changed")
            validate_asset(manifest.get("evidence_registry"), "formal evidence registry")
            if manifest["evidence_registry"]["sha256"] != EXPECTED_EVIDENCE_REGISTRY_SHA256:
                raise ValueError("formal evidence-registry SHA mismatch")
            validate_asset(manifest.get("evidence_gate"), "formal evidence gate")
            if manifest["evidence_gate"]["sha256"] != EXPECTED_EVIDENCE_GATE_SHA256:
                raise ValueError("formal evidence-gate SHA mismatch")
            validate_asset(manifest.get("training_config"), "formal training config")
            if manifest["training_config"]["sha256"] != EXPECTED_TRAINING_CONFIG_SHA256:
                raise ValueError("formal training-config manifest SHA mismatch")
        checkpoint_record = manifest.get("adapter_checkpoint")
        recorded_path = validate_asset(checkpoint_record, "formal adapter checkpoint")
        if recorded_path.resolve() != checkpoint_path.resolve():
            raise ValueError("checkpoint path differs from training manifest")
        payload = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
        if payload.get("schema") != "stage20_adapter_checkpoint_v1":
            raise ValueError("adapter checkpoint schema mismatch")
        if require_formal and int(payload.get("epoch", -1)) != 4:
            raise ValueError("formal adapter checkpoint is not epoch 4")
        if payload.get("training_config_sha256") != runtime.file_sha256(training_config_path):
            raise ValueError("adapter training-config SHA mismatch")
        if payload.get("base_checkpoint_sha256") != training_config["base_model"]["checkpoint_sha256"]:
            raise ValueError("adapter base-checkpoint SHA mismatch")
        if int(payload.get("adapter_parameter_count", -1)) != int(training_config["adapter"]["expected_parameters"]):
            raise ValueError("adapter parameter count mismatch")
        if require_formal:
            payload_expected = {
                "run_class": "formal_training", "smoke_subset": False,
                "epoch": 4, "epochs": 4, "optimizer_updates": 1024,
                "checkpoint_selection": "epoch4_final",
                "evidence_registry_sha256": EXPECTED_EVIDENCE_REGISTRY_SHA256,
                "evidence_gate_sha256": EXPECTED_EVIDENCE_GATE_SHA256,
            }
            for field, value in payload_expected.items():
                if payload.get(field) != value:
                    raise ValueError(f"invalid formal checkpoint field: {field}")
        return payload
    except (KeyError, TypeError, ValueError, FileNotFoundError) as exc:
        raise RuntimeError(f"STOP_INVALID_FORMAL_ADAPTER_CHECKPOINT: {exc}") from exc


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    git_status_at_start = git_value("status", "--short")
    if git_value("status", "--short", "--untracked-files=no"):
        raise RuntimeError("STOP_STAGE20_SOURCE_PROVENANCE_UNRESOLVED: tracked worktree is dirty")
    cfg = require_config(args.config, "stage20_inference_v1")
    training_cfg = require_config(args.training_config, "stage20_continuous_impedance_adapter_training_v1")
    if runtime.file_sha256(args.training_config) != EXPECTED_TRAINING_CONFIG_SHA256:
        raise RuntimeError("STOP_STAGE20_TRAINING_UPSTREAM_MISMATCH: training config hash")
    if cfg.get("truth_loaded_by_runner") is not False or tuple(cfg["arms"]) != ARMS:
        raise ValueError("Stage20 inference truth firewall or arm order changed")
    registry_path = args.inference_registry
    registry = read_json(registry_path)
    serialized_registry = registry_path.read_text(encoding="utf-8").lower()
    if any(forbidden in serialized_registry for forbidden in ("truth", "true_model", "binary_truth")):
        raise RuntimeError("Stage19 inference registry contains a forbidden pointer")
    if registry.get("schema") != "stage20_test_inference_registry_v1" or registry.get("run_status") != "completed":
        raise ValueError("invalid stripped TEST inference registry")
    cases = list(registry["cases"])
    if len(cases) != 12:
        raise ValueError("Stage20 inference requires exactly 12 TEST cases")
    wrong = {str(case["case_id"]): str(cases[(index + 1) % len(cases)]["case_id"]) for index, case in enumerate(cases)}
    if any(key == value for key, value in wrong.items()):
        raise ValueError("wrong-case evidence mapping self-maps")
    executed_cases = cases[:1] if args.smoke else cases
    seeds = cfg["source_seeds"][:1] if args.smoke else cfg["source_seeds"]
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage20 inference requires CUDA")
    checkpoint = resolve_project_path(training_cfg["base_model"]["checkpoint"])
    if runtime.file_sha256(checkpoint) != training_cfg["base_model"]["checkpoint_sha256"]:
        raise RuntimeError("base checkpoint hash mismatch")
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, model_report = runtime.load_model_with_weight_policy(model_class=Geo3DStochInterp, checkpoint_path=checkpoint, map_location=device, weight_source="ema")
    model = model.to(device).eval(); freeze_base_model(model)
    if model_report.get("ema_applied") is not True:
        raise RuntimeError("EMA was not applied")
    base_hash_before = _parameter_hash(model)
    payload = load_validated_adapter_checkpoint(args.training_manifest, args.adapter_checkpoint, args.training_config, training_cfg, require_formal=not args.smoke, map_location=device)
    adapter = ResidualVelocityAdapter(model.embedding_dim, geophysics_channels=1, base_width=int(training_cfg["adapter"]["base_width"]), dilations=training_cfg["adapter"]["dilations"]).to(device)
    adapter.load_state_dict(payload["adapter_state_dict"]); adapter.eval()
    for parameter in adapter.parameters():
        parameter.requires_grad_(False)
    inputs = {str(case["case_id"]): _case_inputs(case) for case in cases}
    case_records = {str(case["case_id"]): case for case in cases}
    base_checkpoint_sha256 = runtime.file_sha256(checkpoint)
    adapter_checkpoint_sha256 = runtime.file_sha256(args.adapter_checkpoint)
    inference_config_sha256 = runtime.file_sha256(args.config)
    runner_source_sha256 = runtime.file_sha256(Path(__file__))

    args.output_dir.mkdir(parents=True)
    manifest = base_manifest("stage20_inference_run_v1", Path(__file__), args.config)
    manifest.update({"run_status": "running", "run_class": "engineering_smoke" if args.smoke else "formal_test_inference", "smoke_subset": args.smoke, "formal_inference_complete": False, "formal_output_count": 0, "truth_loaded_by_runner": False, "wrong_case_mapping": wrong, "git_status_at_start": git_status_at_start, "previous_runner_sha256": PREVIOUS_RUNNER_SHA256, "inference_sources": {"runner": asset(Path(__file__)), "stage20_common": asset(Path(__file__).with_name("common.py")), "residual_velocity_adapter": asset(PROJECT_DIR / "guidance/residual_velocity_adapter.py")}, "training_config": asset(args.training_config), "inference_config": asset(args.config)})
    write_json(args.output_dir / "run_manifest.json", manifest)
    rows, traces = [], []
    try:
        for case in executed_cases:
            case_id = str(case["case_id"]); tensors = inputs[case_id]
            values, mask, support = tensors["condition_values"], tensors["condition_mask"], tensors["subsurface"]
            with torch.no_grad():
                embedded = model.embed(values.to(device))
                conditioning = embedded * mask.to(device).expand_as(embedded)
            q_prior = runtime_evidence(tensors["prior"], support, mask).to(device)
            q_correct = runtime_evidence(tensors["correct"], support, mask).to(device)
            wrong_id = wrong[case_id]
            q_wrong = wrong_case_evidence(inputs[wrong_id]["correct"], support, mask).to(device)
            if not torch.equal(q_wrong.cpu(), runtime_evidence(inputs[wrong_id]["correct"], support, mask)):
                raise RuntimeError("wrong-case evidence did not use wrong values with current mask")
            for seed in seeds:
                initial_cpu = torch.randn((1, model.embedding_dim, *model.data_shape), generator=torch.Generator(device="cpu").manual_seed(int(seed)), dtype=embedded.dtype).contiguous()
                initial_sha = tensor_sha256(initial_cpu)
                for arm in ARMS:
                    if arm == "FLOW_ONLY":
                        final, trace = fixed_euler_adapter_sample(model=model, adapter=adapter, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_conditions=embedded, condition_mask=mask.to(device), geophysics=q_correct, n_steps=int(cfg["n_steps"]), adapter_scale=0.0, max_residual_ratio=float(cfg["max_residual_ratio"]))
                    else:
                        q = q_prior if arm == "ADAPTER_PRIOR_ONLY" else q_correct if arm == "ADAPTER_CORRECT" else q_wrong
                        final, trace = fixed_euler_adapter_sample(model=model, adapter=adapter, initial_state=initial_cpu.to(device), conditioning=conditioning, embedded_conditions=embedded, condition_mask=mask.to(device), geophysics=q, n_steps=int(cfg["n_steps"]), adapter_scale=float(cfg["adapter_scale"]), max_residual_ratio=float(cfg["max_residual_ratio"]))
                    with torch.no_grad():
                        decoded = (model.decode(final).detach().cpu() - 1).unsqueeze(1).long()
                    violations = int(((decoded != values) & mask).sum())
                    if violations:
                        raise RuntimeError(f"hard condition violation: {case_id}/{seed}/{arm}")
                    out = args.output_dir / case_id / f"seed_{seed}" / arm
                    out.mkdir(parents=True)
                    output_path = out / "decoded_geology.pt"
                    torch.save(decoded, output_path)
                    for item in trace:
                        traces.append({"case_id": case_id, "source_seed": int(seed), "arm": arm, **item})
                    input_q = q_wrong if arm == "ADAPTER_WRONG_CASE" else q_prior if arm == "ADAPTER_PRIOR_ONLY" else q_correct
                    evidence_source_case_id = wrong_id if arm == "ADAPTER_WRONG_CASE" else case_id
                    evidence_asset_record = case_records[evidence_source_case_id]["normalized_low_frequency_logz" if arm == "ADAPTER_PRIOR_ONLY" else "normalized_inverted_logz"]
                    output_relative = str(output_path.relative_to(args.output_dir))
                    output_tensor_sha256 = tensor_sha256(decoded)
                    rows.append({"case_id": case_id, "current_case_id": case_id, "source_seed": int(seed), "arm": arm, "initial_noise_sha256": initial_sha, "base_checkpoint_sha256": base_checkpoint_sha256, "adapter_checkpoint_sha256": adapter_checkpoint_sha256, "inference_config_sha256": inference_config_sha256, "runner_source_sha256": runner_source_sha256, "condition_asset_sha256": case_records[case_id]["condition_values"]["sha256"], "condition_mask_asset_sha256": case_records[case_id]["condition_mask"]["sha256"], "subsurface_mask_asset_sha256": case_records[case_id]["subsurface_mask"]["sha256"], "evidence_asset_sha256": evidence_asset_record["sha256"], "evidence_source_case_id": evidence_source_case_id, "mask_source_case_id": case_id, "n_steps": int(cfg["n_steps"]), "adapter_scale": 0.0 if arm == "FLOW_ONLY" else float(cfg["adapter_scale"]), "residual_cap": float(cfg["max_residual_ratio"]), "hard_condition_violation_count": violations, "output_tensor_sha256": output_tensor_sha256, "output_file_sha256": runtime.file_sha256(output_path), "output_path": output_relative, "input_evidence_case_id": evidence_source_case_id, "input_mask_case_id": case_id, "input_evidence_sha256": tensor_sha256(input_q.cpu()), "decoded_geology_sha256": output_tensor_sha256, "condition_violation_count": violations, "decoded_path": output_relative})
                    print(f"Stage20 inference {case_id} seed={seed} arm={arm}", flush=True)
                if args.smoke:
                    # Independent alpha-zero property path must match FLOW_ONLY exactly.
                    reference = projected_fixed_euler_prior_sample(model, initial_cpu.to(device), conditioning, embedded, mask.to(device), n_steps=int(cfg["n_steps"]))
                    flow_row = next(row for row in rows if row["case_id"] == case_id and row["source_seed"] == int(seed) and row["arm"] == "FLOW_ONLY")
                    ref_decoded = (model.decode(reference).detach().cpu() - 1).unsqueeze(1).long()
                    if tensor_sha256(ref_decoded) != flow_row["decoded_geology_sha256"]:
                        raise RuntimeError("adapter scale=0 does not reproduce FLOW_ONLY")
        expected = 4 if args.smoke else 144
        if len(rows) != expected:
            raise RuntimeError(f"unexpected formal output count: {len(rows)}")
        base_hash_after = _parameter_hash(model)
        if base_hash_before != base_hash_after:
            raise RuntimeError("base model changed during inference")
        write_csv(args.output_dir / "sample_manifest.csv", rows)
        write_csv(args.output_dir / "sampling_trace.csv", traces)
        index_asset = None
        if not args.smoke:
            index_asset = freeze_formal_output_index(
                rows, run_dir=args.output_dir,
                case_ids=[str(case["case_id"]) for case in cases], seeds=[int(seed) for seed in seeds], arms=ARMS,
                validation={"expected_base_checkpoint_sha256": base_checkpoint_sha256, "expected_adapter_checkpoint_sha256": adapter_checkpoint_sha256, "expected_inference_config_sha256": inference_config_sha256, "expected_runner_source_sha256": runner_source_sha256, "n_steps": int(cfg["n_steps"]), "residual_cap": float(cfg["max_residual_ratio"])},
            )
        manifest.update({"run_status": "completed", "output_count": len(rows), "formal_inference_complete": not args.smoke, "formal_output_count": len(rows) if not args.smoke else 0, "formal_output_index": index_asset, "formal_output_index_sha256": index_asset["sha256"] if index_asset else None, "executed_case_count": len(executed_cases), "source_seed_count": len(seeds), "all_condition_violations_zero": True, "base_model_unchanged": True, "base_model_hash_before": base_hash_before, "base_model_hash_after": base_hash_after, "model_load_report": model_report, "test_inference_registry": asset(registry_path), "training_manifest": asset(args.training_manifest), "base_checkpoint": asset(checkpoint), "adapter_checkpoint": asset(args.adapter_checkpoint), "truth_loaded_by_runner": False, "scale_zero_flow_equivalence_checked": bool(args.smoke), "wrong_case_value_source_validated": True, "current_case_mask_validated": True})
        write_json(args.output_dir / "run_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "formal_inference_complete": False, "formal_output_count": 0, "error": f"{type(exc).__name__}: {exc}"})
        write_json(args.output_dir / "run_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
