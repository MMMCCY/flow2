#!/usr/bin/env python3
"""Train only the frozen one-channel Stage20 residual velocity adapter."""

from __future__ import annotations

import argparse
import hashlib
import random
import shutil
import statistics
import sys
import time
from pathlib import Path

import torch

PROJECT_DIR = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = Path(__file__).resolve().parents[4]
for root in (PROJECT_DIR, REPOSITORY_ROOT):
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

import inference_runtime as runtime
from guidance.dflow import model_state_dict_hashes
from guidance.generator_posterior import project_conditions
from guidance.residual_velocity_adapter import ResidualVelocityAdapter, cap_residual_velocity, class_balancing_weights, residual_adapter_losses
from guidance.seismic import tensor_sha256
from scripts.stage15.common import base_manifest, git_value, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage20.common import CONFIG_DIR, ROOT, asset, freeze_base_model, require_config, resolve_project_path, validate_asset

DEFAULT_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence_fix2"
DEFAULT_GATE = ROOT / "evidence_audit_fix2/summary.json"
DEFAULT_OUTPUT = ROOT / "checkpoints/formal_v1"
EXPECTED_EVIDENCE_REGISTRY_SHA256 = "94cb0cd2e595be1a9d1ae90bfec0fd7fcdeff69bdbd7b00e3e9642cff41deb28"
EXPECTED_EVIDENCE_GATE_SHA256 = "475fe63a01146730cd7d092b6640a74ce808b5f3b434d33a736dbb16fd0647f0"
EXPECTED_TRAINING_CONFIG_SHA256 = "297083b9b14d935d5fae6f7833c357c569aa69b3d16fb8103bbe3e70227177d8"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--evidence-dir", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--evidence-gate", type=Path, default=DEFAULT_GATE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--smoke", action="store_true")
    return parser.parse_args()


def _model_parameter_hash(model) -> str:
    digest = hashlib.sha256()
    for name, parameter in model.named_parameters():
        digest.update(name.encode())
        digest.update(tensor_sha256(parameter.detach().cpu()).encode())
    return digest.hexdigest()


def _load(case: dict[str, object]) -> dict[str, torch.Tensor]:
    case_id = str(case["case_id"])
    obs = case["observation_assets"]
    result = {
        "truth": normalize_volume(runtime.load_tensor(validate_asset(case["truth_asset"], f"{case_id}/truth")), "truth").long(),
        "condition_values": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/conditions")), "conditions").long(),
        "condition_mask": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool(),
        "subsurface": normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/subsurface")), "subsurface").bool(),
        "score": normalize_volume(runtime.load_tensor(validate_asset(case["assets"]["normalized_inverted_logz"], f"{case_id}/evidence")), "score", torch.float32),
    }
    if not bool(torch.isfinite(result["score"]).all()):
        raise RuntimeError(f"non-finite Stage20 evidence: {case_id}")
    return result


def _objective(*, model, adapter, case: dict[str, object], time_value: float, initial_cpu: torch.Tensor, device: torch.device, cfg: dict[str, object], train: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    tensors = _load(case)
    truth = tensors["truth"].to(device)
    values = tensors["condition_values"].to(device)
    mask = tensors["condition_mask"].to(device)
    support = tensors["subsurface"].to(device)
    with torch.no_grad():
        x1 = model.embed(truth)
        embedded_conditions = model.embed(values)
        conditioning = embedded_conditions * mask.expand_as(embedded_conditions)
        time_tensor = torch.tensor([time_value], device=device, dtype=x1.dtype)
        state, target_velocity = model.interpolator.flow_objective(time_tensor, initial_cpu.to(device), x1)
        state = project_conditions(state, embedded_conditions, mask)
        base_velocity = model.net(state, conditioning, time_tensor).detach()
    q = tensors["score"].to(device=device, dtype=state.dtype) * (support & ~mask).float()
    raw = adapter(state, base_velocity, conditioning, mask, q, time_tensor)
    correction, used_ratio = cap_residual_velocity(raw, base_velocity, mask, max_ratio=float(cfg["adapter"]["max_residual_ratio"]))
    if int(torch.count_nonzero(raw[mask.expand_as(raw)])) or int(torch.count_nonzero(correction[mask.expand_as(correction)])):
        raise RuntimeError("adapter correction is nonzero at hard conditions")
    cap = float(cfg["adapter"]["max_residual_ratio"])
    if bool((used_ratio > cap + 8.0 * torch.finfo(used_ratio.dtype).eps).any()):
        raise RuntimeError("adapter residual ratio cap violated")
    weights = class_balancing_weights(truth, (~mask) & (truth != -1), model.num_categories).to(device)
    tr = cfg["training"]
    loss, diagnostics = residual_adapter_losses(
        state=state, target_velocity=target_velocity.detach(), base_velocity=base_velocity,
        correction=correction, truth=truth, condition_mask=mask,
        embedding_weight=model.embedding.weight.detach(), time=time_tensor, class_weights=weights,
        logit_temperature=float(tr["logit_temperature"]), flow_weight=float(tr["flow_weight"]),
        cross_entropy_weight=float(tr["cross_entropy_weight"]), dice_weight=float(tr["dice_weight"]),
        residual_regularizer_weight=float(tr["residual_regularizer_weight"]),
    )
    return loss, diagnostics, used_ratio


def main() -> None:
    args = parse_args()
    refuse_nonempty(args.output_dir)
    cfg = require_config(args.config, "stage20_continuous_impedance_adapter_training_v1")
    git_status_at_start = git_value("status", "--short")
    tracked_status_at_start = git_value("status", "--short", "--untracked-files=no")
    if tracked_status_at_start:
        raise RuntimeError("STOP_STAGE20_SOURCE_PROVENANCE_UNRESOLVED: tracked worktree is dirty")
    if runtime.file_sha256(args.config) != EXPECTED_TRAINING_CONFIG_SHA256:
        raise RuntimeError("STOP_STAGE20_TRAINING_UPSTREAM_MISMATCH: training config hash")
    gate = read_json(args.evidence_gate)
    if runtime.file_sha256(args.evidence_gate) != EXPECTED_EVIDENCE_GATE_SHA256 or gate.get("schema") != "stage20_evidence_audit_v1" or gate.get("machine_decision") != "CONTINUOUS_IMPEDANCE_EVIDENCE_VALIDATED" or gate.get("adapter_training_authorized") is not True:
        raise RuntimeError("STOP_STAGE20_TRAINING_UPSTREAM_MISMATCH: evidence authorization")
    registry_path = args.evidence_dir / "evidence_registry.json"
    registry = read_json(registry_path)
    if runtime.file_sha256(registry_path) != EXPECTED_EVIDENCE_REGISTRY_SHA256 or registry.get("schema") != "stage20_evidence_registry_v1" or registry.get("run_status") != "completed" or registry.get("case_count") != 84:
        raise RuntimeError("STOP_STAGE20_TRAINING_UPSTREAM_MISMATCH: evidence registry")
    cohort_cases = {}
    for split in ("train", "val"):
        cohort = read_json(ROOT / "cohort" / f"{split}_registry.json")
        cohort_cases.update({str(case["case_id"]): case for case in cohort["cases"]})
    cases = []
    for case in registry["cases"]:
        if case["split"] in {"train", "val"}:
            linked = dict(case); linked["truth_asset"] = cohort_cases[str(case["case_id"])]["assets"]["truth"]; cases.append(linked)
    train_cases = [case for case in cases if case["split"] == "train"]
    val_cases = [case for case in cases if case["split"] == "val"]
    if len(train_cases) != 64 or len(val_cases) != 8:
        raise ValueError("Stage20 training requires frozen 64/8 splits")
    if any(case["split"] == "test" for case in cases):
        raise RuntimeError("Stage20 supervised training registry contains TEST")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage20 CUDA training requires CUDA")
    checkpoint = resolve_project_path(cfg["base_model"]["checkpoint"])
    if runtime.file_sha256(checkpoint) != cfg["base_model"]["checkpoint_sha256"]:
        raise ValueError("base checkpoint hash mismatch")
    from model_train_sh_inference_cond import Geo3DStochInterp
    model, model_report = runtime.load_model_with_weight_policy(model_class=Geo3DStochInterp, checkpoint_path=checkpoint, map_location=device, weight_source="ema")
    model = model.to(device).eval()
    freeze_base_model(model)
    if model_report.get("ema_applied") is not True:
        raise RuntimeError("EMA was not applied")
    base_hash_before = _model_parameter_hash(model)
    state_hash_before, state_tensors_before = model_state_dict_hashes(model)
    torch.manual_seed(int(cfg["training"]["adapter_seed"]))
    torch.cuda.manual_seed_all(int(cfg["training"]["adapter_seed"]))
    adapter = ResidualVelocityAdapter(model.embedding_dim, geophysics_channels=1, base_width=int(cfg["adapter"]["base_width"]), dilations=cfg["adapter"]["dilations"]).to(device)
    if int(torch.count_nonzero(adapter.output_conv.weight)) or int(torch.count_nonzero(adapter.output_conv.bias)):
        raise RuntimeError("adapter output layer is not exactly zero initialized")
    parameter_count = adapter.parameter_count()
    if parameter_count != int(cfg["adapter"]["expected_parameters"]):
        raise RuntimeError("STOP_STAGE20_ADAPTER_ARCHITECTURE_CHANGED")
    if parameter_count >= int(cfg["adapter"]["max_parameters"]):
        raise RuntimeError("STOP_STAGE20_ADAPTER_ARCHITECTURE_CHANGED: parameter budget")
    optimizer = torch.optim.AdamW(adapter.parameters(), lr=float(cfg["training"]["learning_rate"]), weight_decay=float(cfg["training"]["weight_decay"]))
    if {id(p) for group in optimizer.param_groups for p in group["params"]} != {id(p) for p in adapter.parameters()}:
        raise RuntimeError("optimizer must contain only adapter parameters")
    generator = torch.Generator(device="cpu").manual_seed(int(cfg["training"]["state_generator_seed"]))
    times = [float(value) for value in cfg["training"]["times"]]
    epochs = 1 if args.smoke else int(cfg["training"]["epochs"])
    max_updates = 2 if args.smoke else None
    train_subset = train_cases[:2] if args.smoke else train_cases
    val_subset = val_cases[:1] if args.smoke else val_cases

    args.output_dir.mkdir(parents=True)
    shutil.copyfile(args.config, args.output_dir / "adapter_config.json")
    manifest = base_manifest("stage20_adapter_training_run_v1", Path(__file__), args.config)
    source_assets = {
        "training_runner": asset(Path(__file__)),
        "stage20_common": asset(Path(__file__).with_name("common.py")),
        "residual_velocity_adapter": asset(PROJECT_DIR / "guidance/residual_velocity_adapter.py"),
    }
    manifest.update({"run_status": "running", "run_class": "engineering_smoke" if args.smoke else "formal_training", "smoke_subset": args.smoke, "truth_used_for_supervised_training": True, "test_truth_loaded": False, "training_case_counts": {"train": len(train_cases), "val": len(val_cases), "test": 0}, "base_finetuning": False, "git_status_at_start": git_status_at_start, "git_tracked_status_at_start": tracked_status_at_start, "training_sources": source_assets, "adapter_output_zero_initialized": True, "adapter_parameter_count": parameter_count, "optimizer_contains_adapter_only": True, "evidence_q_shape": [1, 1, 64, 64, 64], "evidence_q_finite_required": True})
    write_json(args.output_dir / "training_manifest.json", manifest)
    trace, validation, state_rows = [], [], []
    start = time.monotonic()
    update_count = 0
    base_gradients_absent_all_updates = True
    training_gradients_finite = True
    training_gradients_nonzero = True
    all_losses_finite = True
    residual_ratio_cap_passed = True
    try:
        for epoch in range(epochs):
            order = list(train_subset)
            random.Random(int(cfg["training"]["shuffle_seed_base"]) + epoch).shuffle(order)
            stop = False
            for case in order:
                for time_value in times:
                    sample_shape = (1, model.embedding_dim, *model.data_shape)
                    initial_cpu = torch.randn(sample_shape, generator=generator, dtype=model.embedding.weight.dtype).contiguous()
                    noise_sha = tensor_sha256(initial_cpu)
                    optimizer.zero_grad(set_to_none=True)
                    loss, diagnostics, used_ratio = _objective(model=model, adapter=adapter, case=case, time_value=time_value, initial_cpu=initial_cpu, device=device, cfg=cfg, train=True)
                    if not bool(torch.isfinite(loss)):
                        all_losses_finite = False
                        raise RuntimeError("non-finite Stage20 training loss")
                    loss.backward()
                    base_grads_absent = all(p.grad is None for p in model.parameters())
                    if not base_grads_absent:
                        base_gradients_absent_all_updates = False
                        raise RuntimeError("frozen base received gradients")
                    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(cfg["training"]["gradient_clip_norm"]))
                    gradients_individually_finite = all(parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in adapter.parameters())
                    training_gradients_finite &= gradients_individually_finite and bool(torch.isfinite(grad_norm))
                    training_gradients_nonzero &= float(grad_norm) != 0.0
                    if not training_gradients_finite or not training_gradients_nonzero:
                        raise RuntimeError("adapter gradient is zero or non-finite")
                    optimizer.step()
                    update_count += 1
                    ratio_value = float(used_ratio.detach().max())
                    residual_ratio_cap_passed &= ratio_value <= float(cfg["adapter"]["max_residual_ratio"]) + 8.0 * torch.finfo(used_ratio.dtype).eps
                    trace.append({"epoch": epoch + 1, "update": update_count, "case_id": case["case_id"], "time": time_value, "initial_noise_sha256": noise_sha, "total_loss": float(loss.detach()), "flow_loss": float(diagnostics["flow_loss"].detach()), "cross_entropy_loss": float(diagnostics["cross_entropy_loss"].detach()), "dice_loss": float(diagnostics["dice_loss"].detach()), "residual_regularizer": float(diagnostics["residual_regularizer"].detach()), "endpoint_accuracy": float(diagnostics["endpoint_accuracy"].detach()), "used_residual_ratio": ratio_value, "gradient_norm": float(grad_norm), "gradients_finite": gradients_individually_finite, "base_gradients_absent": base_grads_absent})
                    state_rows.append({"epoch": epoch + 1, "case_id": case["case_id"], "time": time_value, "initial_noise_sha256": noise_sha})
                    if max_updates is not None and update_count >= max_updates:
                        stop = True
                        break
                if stop:
                    break
            # Validation X0 is fixed by case/time and epoch-independent.
            totals = []
            adapter.eval()
            with torch.no_grad():
                for case_index, case in enumerate(val_subset):
                    for time_index, time_value in enumerate(times):
                        val_generator = torch.Generator(device="cpu").manual_seed(7100 + case_index * len(times) + time_index)
                        initial_cpu = torch.randn((1, model.embedding_dim, *model.data_shape), generator=val_generator, dtype=model.embedding.weight.dtype)
                        loss, diagnostics, used_ratio = _objective(model=model, adapter=adapter, case=case, time_value=time_value, initial_cpu=initial_cpu, device=device, cfg=cfg, train=False)
                        row = {"epoch": epoch + 1, "case_id": case["case_id"], "time": time_value, "initial_noise_sha256": tensor_sha256(initial_cpu), "total_loss": float(loss), "flow_loss": float(diagnostics["flow_loss"]), "cross_entropy_loss": float(diagnostics["cross_entropy_loss"]), "dice_loss": float(diagnostics["dice_loss"]), "endpoint_accuracy": float(diagnostics["endpoint_accuracy"]), "used_residual_ratio": float(used_ratio.mean())}
                        validation.append(row); totals.append(row)
            adapter.train()
            print(f"Stage20 epoch {epoch + 1} updates={update_count} val_loss={statistics.mean(r['total_loss'] for r in totals):.6f}", flush=True)
        expected_updates = 2 if args.smoke else 1024
        if update_count != expected_updates:
            raise RuntimeError(f"STOP_STAGE20_FORMAL_UPDATE_COUNT_MISMATCH: {update_count}")
        base_hash_after = _model_parameter_hash(model)
        state_hash_after, state_tensors_after = model_state_dict_hashes(model)
        if base_hash_before != base_hash_after or state_hash_before != state_hash_after or state_tensors_before != state_tensors_after:
            raise RuntimeError("base model tensor hash changed")
        if not args.smoke and (cfg["training"]["checkpoint_selection"] != "epoch4_final" or epochs != 4):
            raise RuntimeError("formal checkpoint must be epoch4_final")
        checkpoint_payload = {"schema": "stage20_adapter_checkpoint_v1", "run_class": "engineering_smoke" if args.smoke else "formal_training", "smoke_subset": args.smoke, "epoch": 1 if args.smoke else 4, "epochs": epochs, "optimizer_updates": update_count, "checkpoint_selection": cfg["training"]["checkpoint_selection"], "adapter_state_dict": adapter.state_dict(), "adapter_parameter_count": parameter_count, "base_checkpoint_sha256": cfg["base_model"]["checkpoint_sha256"], "training_config_sha256": runtime.file_sha256(args.config), "evidence_registry_sha256": runtime.file_sha256(registry_path), "evidence_gate_sha256": runtime.file_sha256(args.evidence_gate), "git_head": git_value("rev-parse", "HEAD")}
        checkpoint_path = args.output_dir / "adapter_checkpoint.pt"
        torch.save(checkpoint_payload, checkpoint_path)
        replay = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        replay_adapter = ResidualVelocityAdapter(model.embedding_dim, geophysics_channels=1, base_width=int(cfg["adapter"]["base_width"]), dilations=cfg["adapter"]["dilations"])
        replay_adapter.load_state_dict(replay["adapter_state_dict"], strict=True)
        (args.output_dir / "adapter_checkpoint.sha256").write_text(runtime.file_sha256(checkpoint_path) + "\n", encoding="utf-8")
        write_csv(args.output_dir / "training_trace.csv", trace)
        write_csv(args.output_dir / "validation_trace.csv", validation)
        write_csv(args.output_dir / "training_case_manifest.csv", state_rows)
        manifest.update({"run_status": "completed", "optimizer_updates": update_count, "epochs": epochs, "checkpoint_epoch": checkpoint_payload["epoch"], "checkpoint_selection": cfg["training"]["checkpoint_selection"], "adapter_parameter_count": parameter_count, "adapter_checkpoint": asset(checkpoint_path), "checkpoint_save_load_validated": True, "base_checkpoint": asset(checkpoint), "base_model_hash_before": base_hash_before, "base_model_hash_after": base_hash_after, "base_state_dict_hash_before": state_hash_before, "base_state_dict_hash_after": state_hash_after, "base_state_tensor_count_before": len(state_tensors_before), "base_state_tensor_count_after": len(state_tensors_after), "base_model_unchanged": True, "base_gradients_absent": all(p.grad is None for p in model.parameters()) and base_gradients_absent_all_updates, "training_gradients_finite": training_gradients_finite, "training_gradients_nonzero": training_gradients_nonzero, "all_losses_finite": all_losses_finite, "residual_ratio_cap_passed": residual_ratio_cap_passed, "ema_applied": True, "evidence_registry": asset(registry_path), "evidence_gate": asset(args.evidence_gate), "training_config": asset(args.config), "elapsed_seconds": time.monotonic() - start})
        write_json(args.output_dir / "training_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}", "optimizer_updates": update_count})
        write_json(args.output_dir / "training_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
