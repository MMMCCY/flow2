#!/usr/bin/env bash
set -euo pipefail

export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/matplotlib-geoflow}"

PY=/home/mcy/miniconda3/envs/geoflow/bin/python
ROOT=/home/mcy/Geoflow/flowtrain_stochastic_interpolation-main
PROJ="$ROOT/project/geodata-3d-conditional"
SCREEN="$PROJ/dike-demo-manual/candidate_screening"

echo "=== cond_generation_1_label10 full demo ==="
$PY "$PROJ/make_dike_guidance_demo.py" \
  --baseline-dir "$SCREEN/cond_generation_1_label10/baseline_alpha0" \
  --guided-dir "$SCREEN/cond_generation_1_label10/guided_alpha0_05" \
  --truth-model "$PROJ/samples/jupyter-demo/cond_generation_1/true_model.pt" \
  --boreholes "$PROJ/samples/jupyter-demo/cond_generation_1/boreholes.pt" \
  --density-config "$SCREEN/cond_generation_1_label10/density_config.json" \
  --observed-gravity "$SCREEN/cond_generation_1_label10/observed_gravity/observed_gravity.pt" \
  --baseline-metrics "$SCREEN/cond_generation_1_label10/screening/baseline_global_evaluation/metrics.csv" \
  --guided-metrics "$SCREEN/cond_generation_1_label10/screening/guided_global_evaluation/metrics.csv" \
  --target-label 10 \
  --kernel-size 9 \
  --output-dir "$SCREEN/cond_generation_1_label10/final_demo" \
  --device cuda

echo "=== cond_generation_1_label10 six-panel 3D, best target sample 12 ==="
$PY "$PROJ/visualize_dike_sample_triplet_3d.py" \
  --truth-model "$PROJ/samples/jupyter-demo/cond_generation_1/true_model.pt" \
  --baseline-sample "$SCREEN/cond_generation_1_label10/baseline_alpha0/sample_12.pt" \
  --guided-sample "$SCREEN/cond_generation_1_label10/guided_alpha0_05/sample_12.pt" \
  --target-label 10 \
  --sample-id 12 \
  --output-dir "$SCREEN/cond_generation_1_label10/final_demo/figures_triplet_3d" \
  --max-points-per-label 60000 \
  --max-target-points 200000 \
  --device cpu

echo "=== paper_cond_gen_0_label7 full demo ==="
$PY "$PROJ/make_dike_guidance_demo.py" \
  --baseline-dir "$SCREEN/paper_cond_gen_0_label7/baseline_alpha0" \
  --guided-dir "$SCREEN/paper_cond_gen_0_label7/guided_alpha0_05" \
  --truth-model "$PROJ/samples/jupyter-demo/paper_cond_gen_0/true_model.pt" \
  --boreholes "$PROJ/samples/jupyter-demo/paper_cond_gen_0/boreholes.pt" \
  --density-config "$SCREEN/paper_cond_gen_0_label7/density_config.json" \
  --observed-gravity "$SCREEN/paper_cond_gen_0_label7/observed_gravity/observed_gravity.pt" \
  --baseline-metrics "$SCREEN/paper_cond_gen_0_label7/screening/baseline_global_evaluation/metrics.csv" \
  --guided-metrics "$SCREEN/paper_cond_gen_0_label7/screening/guided_global_evaluation/metrics.csv" \
  --target-label 7 \
  --kernel-size 9 \
  --output-dir "$SCREEN/paper_cond_gen_0_label7/final_demo" \
  --device cuda

echo "=== paper_cond_gen_0_label7 six-panel 3D, best target sample 4 ==="
$PY "$PROJ/visualize_dike_sample_triplet_3d.py" \
  --truth-model "$PROJ/samples/jupyter-demo/paper_cond_gen_0/true_model.pt" \
  --baseline-sample "$SCREEN/paper_cond_gen_0_label7/baseline_alpha0/sample_4.pt" \
  --guided-sample "$SCREEN/paper_cond_gen_0_label7/guided_alpha0_05/sample_4.pt" \
  --target-label 7 \
  --sample-id 4 \
  --output-dir "$SCREEN/paper_cond_gen_0_label7/final_demo/figures_triplet_3d" \
  --max-points-per-label 60000 \
  --max-target-points 200000 \
  --device cpu

echo "Done. Outputs:"
echo "$SCREEN/cond_generation_1_label10/final_demo"
echo "$SCREEN/paper_cond_gen_0_label7/final_demo"
