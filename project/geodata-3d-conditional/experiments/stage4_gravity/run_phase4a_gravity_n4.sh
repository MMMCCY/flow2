#!/usr/bin/env bash
set -euo pipefail

# Use only after the alpha025 seed-42 n=1 audit passes the complete gate.
PYTHON_BIN="${PYTHON_BIN:-python}"
PROJECT="project/geodata-3d-conditional"

SEED=42 \
N_SAMPLES=4 \
N_STEPS=32 \
ALPHA=0.25 \
MAX_GUIDANCE_RATIO=0.25 \
RUN_TAG=seed42_n4_s32_a025_c025 \
PYTHON_BIN="$PYTHON_BIN" \
bash "$PROJECT/experiments/stage4_gravity/run_phase4a_gravity_screen.sh"

echo "Before the final n=4 audit, run the mandatory post-hoc comparator:"
echo "  PYTHONPATH=src $PYTHON_BIN $PROJECT/scripts/stage4/rerank_gravity_ensemble.py --baseline-dir $PROJECT/experiments/stage4_gravity/runs/cond_generation_0/phase4a_gravity_v1/seed42_n4_s32_a025_c025/baseline --guided-dir $PROJECT/experiments/stage4_gravity/runs/cond_generation_0/phase4a_gravity_v1/seed42_n4_s32_a025_c025/alpha025 --output-dir $PROJECT/experiments/stage4_gravity/reports/seed42_n4_s32_a025_c025/reranking"
echo "Then audit with --reranking-summary pointing to reranking/summary.json."
