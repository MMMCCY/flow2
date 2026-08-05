#!/usr/bin/env bash
set -euo pipefail

# Run one frozen Phase-3 level from the repository root. Inspect and audit each
# level before advancing; LEVEL=all is intentionally unsupported.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
LEVEL="${LEVEL:-identity_anchor_v1}"
N_SAMPLES="${N_SAMPLES:-1}"
N_STEPS="${N_STEPS:-32}"
SEED="${SEED:-42}"
RUN_TAG="${RUN_TAG:-seed${SEED}_n${N_SAMPLES}_s${N_STEPS}_a025_c025}"

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage3/run_spatial_property_guidance.py"
AUDITOR="$PROJECT/scripts/stage3/audit_spatial_screen.py"
CONFIG_ROOT="$PROJECT/experiments/stage3_spatial_property/configs"
PROPERTY_CONFIG="$PROJECT/experiments/stage2_property/configs/ideal_density_susceptibility_label9_contrast_v1.json"
RUNS_ROOT="$PROJECT/experiments/stage3_spatial_property/runs/cond_generation_0/phase3_spatial_property_v1"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"

case "$LEVEL" in
  identity_anchor_v1|gaussian_sigma1_v1|gaussian_sigma2_v1|gaussian_sigma4_v1)
    OBSERVATION_CONFIG="$CONFIG_ROOT/${LEVEL}.json"
    ;;
  *)
    echo "Unknown frozen Phase-3 LEVEL: $LEVEL" >&2
    exit 1
    ;;
esac

PAIR_ROOT="$RUNS_ROOT/$LEVEL/$RUN_TAG"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/alpha025"

for asset in \
  "$RUNNER" \
  "$AUDITOR" \
  "$CKPT" \
  "$CASE/true_model.pt" \
  "$CASE/boreholes.pt" \
  "$PROPERTY_CONFIG" \
  "$OBSERVATION_CONFIG"; do
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
  --observation-config "$OBSERVATION_CONFIG"
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
  --max-guidance-ratio 0.25
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

echo "[1/2] Phase 3 $LEVEL alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/2] Phase 3 $LEVEL alpha=0.25: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0.25 \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "Completed Phase-3 strict pair. Audit it with:"
echo "  PYTHONPATH=src $PYTHON_BIN $AUDITOR --level $LEVEL --run-name $RUN_TAG --seed $SEED --n-samples $N_SAMPLES --n-steps $N_STEPS --overwrite"

