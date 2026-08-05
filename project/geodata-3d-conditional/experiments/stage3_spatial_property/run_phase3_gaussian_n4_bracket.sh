#!/usr/bin/env bash
set -euo pipefail

# Frozen post-screen seed-42 n=4 bracket. Run one LEVEL at a time from the
# repository root. Only the passing identity anchor and adjacent sigma-1 level
# are eligible.
PYTHON_BIN="${PYTHON_BIN:-python}"
LEVEL="${LEVEL:-identity_anchor_v1}"

case "$LEVEL" in
  identity_anchor_v1|gaussian_sigma1_v1)
    ;;
  *)
    echo "Phase-3 n=4 bracket only permits identity_anchor_v1 or gaussian_sigma1_v1" >&2
    exit 1
    ;;
esac

PROJECT="project/geodata-3d-conditional"
RUN_TAG="seed42_n4_s32_a025_c025"

N_SAMPLES=4 \
N_STEPS=32 \
SEED=42 \
RUN_TAG="$RUN_TAG" \
LEVEL="$LEVEL" \
PYTHON_BIN="$PYTHON_BIN" \
bash "$PROJECT/experiments/stage3_spatial_property/run_phase3_gaussian_screen.sh"

echo "Audit the completed n=4 level with:"
echo "  PYTHONPATH=src $PYTHON_BIN $PROJECT/scripts/stage3/audit_spatial_screen.py --level $LEVEL --run-name $RUN_TAG --seed 42 --n-samples 4 --n-steps 32 --overwrite"

