#!/usr/bin/env bash
set -euo pipefail

# Run from the repository root. Override PYTHON_BIN for the GPU environment.
# Every invocation uses a fresh RUN_TAG because the runner refuses overwrite.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
N_SAMPLES="${N_SAMPLES:-1}"
N_STEPS="${N_STEPS:-32}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}_n${N_SAMPLES}_s${N_STEPS}}"

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage2/run_property_guidance.py"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"
PROPERTY_CONFIG="${PROPERTY_CONFIG:-$PROJECT/experiments/stage2_property/configs/ideal_distinct_density_proxy_v1.json}"
PROPERTY_TAG="${PROPERTY_TAG:-ideal_distinct_density_proxy_v1}"
GUIDED_ALPHA="${GUIDED_ALPHA:-0.10}"
GUIDED_TAG="${GUIDED_TAG:-alpha010}"
MAX_GUIDANCE_RATIO="${MAX_GUIDANCE_RATIO:-0.10}"
PAIR_ROOT="$PROJECT/experiments/stage2_property/runs/cond_generation_0/$PROPERTY_TAG/phase2a_v1/$RUN_TAG"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/$GUIDED_TAG"

for asset in "$RUNNER" "$CKPT" "$CASE/true_model.pt" "$CASE/boreholes.pt" "$PROPERTY_CONFIG"; do
  if [[ ! -f "$asset" ]]; then
    echo "Required asset not found: $asset" >&2
    exit 1
  fi
done

COMMON=(
  --ckpt-path "$CKPT"
  --model-weights ema
  --samples-dir "$CASE"
  --property-config "$PROPERTY_CONFIG"
  --confidence-mode unconditioned_nonair_v1
  --property-sigma 0
  --property-sigma 1.5
  --property-sigma 3.0
  --property-scale-weight 0.50
  --property-scale-weight 0.30
  --property-scale-weight 0.20
  --target-label 9
  --target-roi-radius 6
  --n-samples "$N_SAMPLES"
  --n-steps "$N_STEPS"
  --max-guidance-ratio "$MAX_GUIDANCE_RATIO"
  --tau-start 0.5
  --tau-end 0.1
  --tau-schedule cosine
  --guidance-start 0.25
  --guidance-schedule windowed_sine
  --guidance-scaling-mode reference_norm_relative_v2
  --grad-clip-norm 1.0
  --seed "$SEED"
  --device "$DEVICE"
)

echo "[1/2] Phase 2a ideal-property alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/2] Phase 2a ideal-property alpha=$GUIDED_ALPHA: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha "$GUIDED_ALPHA" \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "Completed the Phase-2a one-sample strict pair. Inspect:"
echo "  $GUIDED/paired_delta_summary.json"
echo "  $GUIDED/paired_per_class_deltas.csv"
echo "  $GUIDED/guidance_trace.csv"
echo "  $GUIDED/config.json"
