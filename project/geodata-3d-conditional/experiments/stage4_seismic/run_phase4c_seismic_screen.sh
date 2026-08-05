#!/usr/bin/env bash
set -euo pipefail

# Frozen first Phase-4c strict pair. Run from the repository root.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
SEED="${SEED:-42}"
N_SAMPLES="${N_SAMPLES:-1}"
N_STEPS="${N_STEPS:-32}"
ALPHA="${ALPHA:-0.25}"
MAX_GUIDANCE_RATIO="${MAX_GUIDANCE_RATIO:-0.25}"

case "$ALPHA" in
  0.25) ALPHA_TAG="alpha025" ;;
  0.10|0.1) ALPHA_TAG="alpha010" ;;
  *)
    echo "Frozen Phase-4c screen permits ALPHA=0.25 or the conditional 0.10 diagnostic" >&2
    exit 1
    ;;
esac

if [[ "$MAX_GUIDANCE_RATIO" != "0.25" ]]; then
  echo "Frozen Phase-4c screen requires MAX_GUIDANCE_RATIO=0.25" >&2
  exit 1
fi

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage4/run_seismic_guidance.py"
AUDITOR="$PROJECT/scripts/stage4/audit_seismic_screen.py"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
OBSERVATION="$PROJECT/experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2"
CONTROLLER_MANIFEST="$PROJECT/experiments/stage4_seismic/configs/seismic_controller_manifest_v1.json"
RUNS_ROOT="$PROJECT/experiments/stage4_seismic/runs/cond_generation_0/phase4c_seismic_v1"
RUN_TAG="${RUN_TAG:-seed${SEED}_n${N_SAMPLES}_s${N_STEPS}_a${ALPHA_TAG#alpha}_c025}"
PAIR_ROOT="$RUNS_ROOT/$RUN_TAG"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/$ALPHA_TAG"

for asset in \
  "$RUNNER" \
  "$AUDITOR" \
  "$CKPT" \
  "$CASE/true_model.pt" \
  "$CASE/boreholes.pt" \
  "$OBSERVATION/manifest.json" \
  "$CONTROLLER_MANIFEST"; do
  if [[ ! -f "$asset" ]]; then
    echo "Required asset not found: $asset" >&2
    exit 1
  fi
done

COMMON=(
  --ckpt-path "$CKPT"
  --model-weights ema
  --samples-dir "$CASE"
  --observation-dir "$OBSERVATION"
  --controller-manifest "$CONTROLLER_MANIFEST"
  --controller-level "${ALPHA_TAG}_cap025"
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

echo "[1/2] Phase-4c alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/2] Phase-4c $ALPHA_TAG: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha "$ALPHA" \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "Completed Phase-4c strict pair. Audit with:"
echo "  PYTHONPATH=src $PYTHON_BIN $AUDITOR --run-name $RUN_TAG --seed $SEED --n-samples $N_SAMPLES --n-steps $N_STEPS --guided-name $ALPHA_TAG --overwrite"
