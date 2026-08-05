#!/usr/bin/env bash
set -euo pipefail

# Post-bracket Phase-2b seed-42 n=4 robustness fallback. The level and all
# sampler/controller settings are frozen in docs/PHASE2B_FOLLOWUP_SPEC.md.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
RUN_TAG="${RUN_TAG:-seed42_n4_s32_a025_c025_followup}"

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage2/run_property_guidance.py"
CONFIG_ROOT="$PROJECT/experiments/stage2_property/configs/phase2b_codebook_ambiguity_v1"
RUNS_ROOT="$PROJECT/experiments/stage2_property/runs/cond_generation_0/phase2b_codebook_ambiguity_v1"
BRACKET_SUMMARY="$PROJECT/experiments/stage2_property/reports/phase2b_codebook_ambiguity_v1_n4_bracket_seed42/summary.json"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"

LEVEL="paired_c100"
PROPERTY_CONFIG="$CONFIG_ROOT/paired_c100_v1.json"
PAIR_ROOT="$RUNS_ROOT/$LEVEL/$RUN_TAG"
BASELINE="$PAIR_ROOT/baseline"
GUIDED="$PAIR_ROOT/alpha025"

for asset in "$RUNNER" "$CKPT" "$CASE/true_model.pt" "$CASE/boreholes.pt" "$PROPERTY_CONFIG" "$BRACKET_SUMMARY"; do
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
  --experiment-stage phase2b_codebook_ambiguity_v1
  --confidence-mode unconditioned_nonair_v1
  --property-sigma 0
  --property-sigma 1.5
  --property-sigma 3.0
  --property-scale-weight 0.50
  --property-scale-weight 0.30
  --property-scale-weight 0.20
  --target-label 9
  --target-roi-radius 6
  --n-samples 4
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

echo "[1/2] Phase 2b fallback $LEVEL alpha-zero baseline: $BASELINE"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0 \
  --output-dir "$BASELINE"

echo "[2/2] Phase 2b fallback $LEVEL alpha=0.25: $GUIDED"
PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
  "${COMMON[@]}" \
  --alpha 0.25 \
  --baseline-dir "$BASELINE" \
  --output-dir "$GUIDED"

echo "Completed Phase-2b seed-42 n=4 fallback: $LEVEL"
echo "  $GUIDED/paired_deltas.csv"
echo "  $GUIDED/paired_per_class_deltas.csv"
echo "  $GUIDED/ensemble_summary.json"
echo "Audit the fallback with:"
echo "  PYTHONPATH=src $PYTHON_BIN $PROJECT/scripts/stage2/summarize_phase2b_n4_fallback.py --overwrite"
