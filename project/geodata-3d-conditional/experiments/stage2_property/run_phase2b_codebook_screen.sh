#!/usr/bin/env bash
set -euo pipefail

# Phase-2b predeclared seed-42 n=1 screen. Run from the repository root.
# LEVEL may be one level below or "all"; start with distinct_c100_anchor.
PYTHON_BIN="${PYTHON_BIN:-python}"
DEVICE="${DEVICE:-cuda}"
LEVEL="${LEVEL:-distinct_c100_anchor}"
RUN_TAG="${RUN_TAG:-seed42_n1_s32_a025_c025}"

PROJECT="project/geodata-3d-conditional"
RUNNER="$PROJECT/scripts/stage2/run_property_guidance.py"
CONFIG_ROOT="$PROJECT/experiments/stage2_property/configs"
RUNS_ROOT="$PROJECT/experiments/stage2_property/runs/cond_generation_0/phase2b_codebook_ambiguity_v1"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"

level_config() {
  case "$1" in
    distinct_c100_anchor)
      echo "$CONFIG_ROOT/ideal_density_susceptibility_label9_contrast_v1.json"
      ;;
    paired_c100|paired_c025|paired_c010|paired_c004_overlap)
      echo "$CONFIG_ROOT/phase2b_codebook_ambiguity_v1/${1}_v1.json"
      ;;
    *)
      echo "Unknown Phase-2b LEVEL: $1" >&2
      return 1
      ;;
  esac
}

run_level() {
  local level="$1"
  local property_config
  property_config="$(level_config "$level")"
  local pair_root="$RUNS_ROOT/$level/$RUN_TAG"
  local baseline="$pair_root/baseline"
  local guided="$pair_root/alpha025"

  for asset in "$RUNNER" "$CKPT" "$CASE/true_model.pt" "$CASE/boreholes.pt" "$property_config"; do
    if [[ ! -f "$asset" ]]; then
      echo "Required asset not found: $asset" >&2
      exit 1
    fi
  done

  local common=(
    --ckpt-path "$CKPT"
    --model-weights ema
    --samples-dir "$CASE"
    --property-config "$property_config"
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

  echo "[1/2] Phase 2b $level alpha-zero baseline: $baseline"
  PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
    "${common[@]}" \
    --alpha 0 \
    --output-dir "$baseline"

  echo "[2/2] Phase 2b $level alpha=0.25: $guided"
  PYTHONPATH=src "$PYTHON_BIN" "$RUNNER" \
    "${common[@]}" \
    --alpha 0.25 \
    --baseline-dir "$baseline" \
    --output-dir "$guided"

  echo "Completed Phase-2b screen level: $level"
  echo "  $guided/paired_deltas.csv"
  echo "  $guided/paired_per_class_deltas.csv"
  echo "  $guided/guidance_trace.csv"
}

if [[ "$LEVEL" == "all" ]]; then
  for level in \
    distinct_c100_anchor \
    paired_c100 \
    paired_c025 \
    paired_c010 \
    paired_c004_overlap; do
    run_level "$level"
  done
else
  run_level "$LEVEL"
fi

echo "Aggregate completed levels with:"
echo "  PYTHONPATH=src $PYTHON_BIN $PROJECT/scripts/stage2/summarize_phase2b_screen.py --overwrite"
