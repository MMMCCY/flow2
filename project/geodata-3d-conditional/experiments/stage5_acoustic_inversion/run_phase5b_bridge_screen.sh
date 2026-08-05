#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
PROJECT="project/geodata-3d-conditional"
EXPERIMENT="$PROJECT/experiments/stage5_acoustic_inversion"
RUNNER="$PROJECT/scripts/stage2/run_property_guidance.py"
AUDITOR="$PROJECT/scripts/stage5/audit_inversion_property_bridge.py"
BRIDGE="$EXPERIMENT/bridge_observations/cond_generation_0/fixed12_log_impedance_v1"
PAIR_ROOT="$EXPERIMENT/runs/cond_generation_0/phase5b_inversion_property_bridge_v1/seed42_n1_s32_a025_c025"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/alpha025"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"

for asset in \
  "$RUNNER" \
  "$AUDITOR" \
  "$BRIDGE/manifest.json" \
  "$BRIDGE/property_config_resolved.json" \
  "$CKPT" \
  "$CASE/true_model.pt" \
  "$CASE/boreholes.pt"; do
  if [[ ! -f "$asset" ]]; then
    echo "Required asset not found: $asset" >&2
    exit 1
  fi
done

COMMON=(
  --ckpt-path "$CKPT"
  --model-weights ema
  --samples-dir "$CASE"
  --property-config "$BRIDGE/property_config_resolved.json"
  --experiment-stage phase5b_inversion_property_bridge_v1
  --confidence-mode external_posterior_spread_v1
  --external-property-dir "$BRIDGE"
  --property-sigma 0
  --property-scale-weight 1
  --target-label 9
  --target-roi-radius 6
  --n-samples 1
  --n-steps 32
  --max-guidance-ratio 0.25
  --tau-start 0.5
  --tau-end 0.1
  --tau-schedule cosine
  --guidance-start 0.25
  --guidance-schedule windowed_sine
  --guidance-scaling-mode reference_norm_relative_v2
  --grad-clip-norm 1.0
  --seed 42
  --device "$DEVICE"
)

echo "[1/3] Phase 5b alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/3] Phase 5b alpha=0.25 guided: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0.25 \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "[3/3] Phase 5b frozen gate audit"
PYTHONPATH=src "$PYTHON_BIN" "$AUDITOR" --pair-root "$PAIR_ROOT"
