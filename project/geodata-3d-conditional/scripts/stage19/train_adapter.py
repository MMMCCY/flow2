#!/usr/bin/env python3
"""Train only the small Stage19 evidence-to-velocity residual adapter."""

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
from scripts.stage15.common import base_manifest, normalize_volume, read_json, refuse_nonempty, write_csv, write_json
from scripts.stage19.common import CONFIG_DIR, ROOT, asset, freeze_base_model, require_config, resolve_project_path, validate_asset

DEFAULT_CONFIG = CONFIG_DIR / "training_v1.json"
DEFAULT_EVIDENCE = ROOT / "evidence"
DEFAULT_GATE = ROOT / "evidence_audit/summary.json"
DEFAULT_OUTPUT = ROOT / "checkpoints/formal_v1"


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
    return {
        "truth": normalize_volume(runtime.load_tensor(validate_asset(case["truth_assets"]["truth"], f"{case_id}/truth")), "truth").long(),
        "condition_values": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_values"], f"{case_id}/conditions")), "conditions").long(),
        "condition_mask": normalize_volume(runtime.load_tensor(validate_asset(obs["condition_mask"], f"{case_id}/condition mask")), "condition mask").bool(),
        "subsurface": normalize_volume(runtime.load_tensor(validate_asset(obs["subsurface_mask"], f"{case_id}/subsurface")), "subsurface").bool(),
        "score": normalize_volume(runtime.load_tensor(validate_asset(case["evidence"], f"{case_id}/evidence")), "score", torch.float32),
    }


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
    q = (tensors["score"].to(device=device, dtype=state.dtype) * support.float())
    raw = adapter(state, base_velocity, conditioning, mask, q, time_tensor)
    correction, used_ratio = cap_residual_velocity(raw, base_velocity, mask, max_ratio=float(cfg["adapter"]["max_residual_ratio"]))
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
    cfg = require_config(args.config, "stage19_learned_evidence_adapter_training_v1")
    gate = read_json(args.evidence_gate)
    if gate.get("adapter_training_authorized") is not True:
        raise RuntimeError("STOP_BEFORE_ADAPTER_TRAINING: evidence gate is not passed")
    registry_path = args.evidence_dir / "evidence_registry.json"
    registry = read_json(registry_path)
    train_cases = [case for case in registry["cases"] if case["split"] == "train"]
    val_cases = [case for case in registry["cases"] if case["split"] == "val"]
    if len(train_cases) != 64 or len(val_cases) != 8:
        raise ValueError("Stage19 training requires frozen 64/8 splits")
    device = torch.device(args.device)
    if device.type != "cuda" or not torch.cuda.is_available():
        raise RuntimeError("Stage19 CUDA training requires CUDA")
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
    if parameter_count >= int(cfg["adapter"]["max_parameters"]):
        raise RuntimeError("adapter parameter budget exceeded")
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
    manifest = base_manifest("stage19_adapter_training_run_v1", Path(__file__), args.config)
    manifest.update({"run_status": "running", "run_class": "engineering_smoke" if args.smoke else "formal_training", "smoke_subset": args.smoke, "truth_used_for_supervised_training": True, "base_finetuning": False})
    write_json(args.output_dir / "training_manifest.json", manifest)
    trace, validation, state_rows = [], [], []
    start = time.monotonic()
    update_count = 0
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
                        raise RuntimeError("non-finite Stage19 training loss")
                    loss.backward()
                    base_grads_absent = all(p.grad is None for p in model.parameters())
                    if not base_grads_absent:
                        raise RuntimeError("frozen base received gradients")
                    grad_norm = torch.nn.utils.clip_grad_norm_(adapter.parameters(), float(cfg["training"]["gradient_clip_norm"]))
                    if not bool(torch.isfinite(grad_norm)) or float(grad_norm) == 0.0:
                        raise RuntimeError("adapter gradient is zero or non-finite")
                    optimizer.step()
                    update_count += 1
                    trace.append({"epoch": epoch + 1, "update": update_count, "case_id": case["case_id"], "time": time_value, "initial_noise_sha256": noise_sha, "total_loss": float(loss.detach()), "flow_loss": float(diagnostics["flow_loss"]), "cross_entropy_loss": float(diagnostics["cross_entropy_loss"]), "dice_loss": float(diagnostics["dice_loss"]), "residual_regularizer": float(diagnostics["residual_regularizer"]), "endpoint_accuracy": float(diagnostics["endpoint_accuracy"]), "used_residual_ratio": float(used_ratio.mean()), "gradient_norm": float(grad_norm)})
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
            print(f"Stage19 epoch {epoch + 1} updates={update_count} val_loss={statistics.mean(r['total_loss'] for r in totals):.6f}", flush=True)
        expected_updates = 2 if args.smoke else 1024
        if update_count != expected_updates:
            raise RuntimeError(f"unexpected optimizer update count: {update_count}")
        base_hash_after = _model_parameter_hash(model)
        state_hash_after, state_tensors_after = model_state_dict_hashes(model)
        if base_hash_before != base_hash_after or state_hash_before != state_hash_after or state_tensors_before != state_tensors_after:
            raise RuntimeError("base model tensor hash changed")
        checkpoint_payload = {"schema": "stage19_adapter_checkpoint_v1", "epoch": 1 if args.smoke else 4, "adapter_state_dict": adapter.state_dict(), "adapter_parameter_count": parameter_count, "base_checkpoint_sha256": cfg["base_model"]["checkpoint_sha256"], "training_config_sha256": runtime.file_sha256(args.config)}
        checkpoint_path = args.output_dir / "adapter_checkpoint.pt"
        torch.save(checkpoint_payload, checkpoint_path)
        (args.output_dir / "adapter_checkpoint.sha256").write_text(runtime.file_sha256(checkpoint_path) + "\n", encoding="utf-8")
        write_csv(args.output_dir / "training_trace.csv", trace)
        write_csv(args.output_dir / "validation_trace.csv", validation)
        write_csv(args.output_dir / "training_case_manifest.csv", state_rows)
        manifest.update({"run_status": "completed", "optimizer_updates": update_count, "epochs": epochs, "adapter_parameter_count": parameter_count, "adapter_checkpoint": asset(checkpoint_path), "base_checkpoint": asset(checkpoint), "base_model_hash_before": base_hash_before, "base_model_hash_after": base_hash_after, "base_model_unchanged": True, "base_gradients_absent": all(p.grad is None for p in model.parameters()), "ema_applied": True, "evidence_registry": asset(registry_path), "evidence_gate": asset(args.evidence_gate), "training_config": asset(args.config), "elapsed_seconds": time.monotonic() - start})
        write_json(args.output_dir / "training_manifest.json", manifest)
    except Exception as exc:
        manifest.update({"run_status": "failed", "error": f"{type(exc).__name__}: {exc}", "optimizer_updates": update_count})
        write_json(args.output_dir / "training_manifest.json", manifest)
        raise


if __name__ == "__main__":
    main()
