#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Override PYTHON_BIN when the GPU-enabled
# environment is not the shell's default Python. This protocol-v4 experiment
# is intentionally isolated from all protocol-v1/v3 output directories.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
N_SAMPLES="${N_SAMPLES:-1}"
N_STEPS="${N_STEPS:-32}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}_n${N_SAMPLES}_s${N_STEPS}}"

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage1/run_probability_guidance.py"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"
PAIR_ROOT="$PROJECT/experiments/stage1_probability/runs/cond_generation_0/label9/all/phase1b_v4/calibrated_reference_windowed/$RUN_TAG"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/alpha025"

if [[ ! -f "$CKPT" ]]; then
  echo "Checkpoint not found: $CKPT" >&2
  exit 1
fi

COMMON=(
  --ckpt-path "$CKPT"
  --model-weights ema
  --samples-dir "$CASE"
  --target-label 9
  --component-mode all
  --target-sigma 0
  --target-sigma 1.5
  --target-scale-weight 0.5
  --target-scale-weight 0.5
  --roi-radius 6
  --n-samples "$N_SAMPLES"
  --n-steps "$N_STEPS"
  --max-guidance-ratio 0.25
  --tau-start 0.5
  --tau-end 0.1
  --tau-schedule cosine
  --guidance-start 0.25
  --guidance-schedule windowed_sine
  --guidance-scaling-mode reference_norm_relative_v2
  --grad-clip-norm 1.0
  --probability-loss-mode calibrated_soft_bce_hard_dice_v2
  --bce-weight 1.0
  --dice-weight 1.0
  --spatial-gradient-weight 0
  --seed "$SEED"
  --device "$DEVICE"
)

echo "[1/2] Phase 1b protocol-v4 alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/2] Phase 1b protocol-v4 calibrated alpha=0.25: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0.25 \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "Completed strict pair. Inspect:"
echo "  $GUIDED/paired_delta_summary.json"
echo "  $GUIDED/paired_soft_delta_summary.json"
echo "  $GUIDED/guidance_trace.csv"
