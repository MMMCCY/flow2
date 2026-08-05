# Phase 1: oracle 3-D probability-volume guidance

This directory separates immutable experiment definitions from generated run
artifacts.  The Phase-1 runner refuses to overwrite a non-empty output
directory.  Keep generated tensors and traces under `runs/`; do not add them to
the source directories.

The probability target is derived from the truth model.  It is an upper-bound
mechanism experiment and must not be described as a measured geophysical
observation.

## Final Phase-1 decision

Protocol v4 completed 12 strict pairs over seeds 42, 142 and 242. Mean target
IoU/precision/recall reached 0.8099/0.8274/0.9747 and selected-ROI IoU reached
0.9392, with zero condition violations. The mechanism is validated and Phase 1
is closed with explicit topology and endpoint caveats. The raw component ratio
1.2812 exceeds the pre-registered 1.25 limit and therefore remains a failed
clause even though size-stratified topology shows four correct major ROI bodies
in every sample.

The final report is `docs/PHASE1_REPORT.md`; reproducible aggregate artifacts
are under `reports/phase1b_v4_12pair/`. Phase 2 is isolated under
`experiments/stage2_property/` and must not overwrite or reuse these runs.

## Case order

1. One-sample infrastructure smoke: `cond_generation_0`, label 9, all voxels.
2. Positive control: `cond_generation_1`, label 10, all target voxels.
3. Primary multi-seed case: `cond_generation_0`, label 9, all target voxels.
4. Sparse-component ablation: label 9 rank 0 (4,079 voxels, no borehole hit).
5. Observed-component ablation: label 9 rank 2 (2,043 voxels, 13 voxels in two boreholes).
6. Generalization case after the mechanism gate: `cond_generation_1`, label 7.

Component ranks use six-neighbour connectivity and decreasing component volume.
Every run saves the ranked component manifest and target tensor hashes.

## Current evidence

Protocol-v1 GPU runs established the standalone probability-volume mechanism:

- label-9 `all`, 12 strict pairs over seeds 42, 142, and 242: target IoU,
  precision, recall, and centroid distance improved in all 12 pairs at
  `alpha=0.1`; condition violations remained zero and 97.5% of paired changes
  were inside the ROI;
- label-9 rank 0 (4,079 voxels, zero conditioned voxels): selected ROI IoU
  improved in all four seed-42 samples, proving that a borehole hit is not
  required for this truth-derived oracle to affect the component, but the
  predicted component became substantially more fragmented;
- label-9 rank 2 (2,043 voxels, 13 conditioned voxels): selected ROI recall
  rose from 0.146 to 0.674 and its largest-component fraction rose from 0.761
  to 0.934;
- label-10 positive control: target IoU rose from 0.430 to 0.626 and absolute
  volume-error fraction fell from 0.254 to 0.069;
- label-7 generalization: target IoU rose from 0.078 to 0.154, but component
  count rose from 25.75 to 147 and largest-component fraction fell from 0.824
  to 0.729.

These results validate the gradient path and a reproducible directional
response, but they do not pass the material hard-geometry gate. Mean target
IoU reached only 0.0626, recall only 0.1027, only 0.3109% of hard voxels changed
per sample, and target components worsened from 52.17 to 78.25 on average.
Sparse/unobserved targets can improve by voxel metrics while fragmenting.
Phase 1 therefore remains open as Phase 1b.

## Protocol v2: spatial-gradient ablation

Protocol v2 adds an optional normalized 3-D target-probability gradient loss.
It penalizes local probability gradients that are absent from the oracle target
and is intended to suppress isolated fragments while retaining target
boundaries.  It is controlled by `--spatial-gradient-weight`, defaults to zero,
and is recorded separately in `guidance_trace.csv`.

Because source hashes and strict pairing fields changed, protocol-v2 runs must
create new alpha-zero baselines.  They must not use protocol-v1 baseline
directories.  The first v2 ablation target is label-9 selected rank 0, the
clearest failure case.  Compare:

1. BCE+Dice control with `--spatial-gradient-weight 0`;
2. spatial-gradient candidate with `--spatial-gradient-weight 0.1`.

Each candidate requires a baseline carrying the same spatial weight.  Alpha
zero still takes the explicit no-gradient branch; the duplicated baselines are
for strict protocol attribution, not because the regularizer is active at
alpha zero.

The seed-42 four-pair ablation has now completed for weights 0, 0.05 and 0.10.
Weight 0.05 reduced the mean selected-ROI component count from 83.75 to 70.75
and increased largest-component fraction from 0.5822 to 0.5935, while mean ROI
IoU declined from 0.0959 to 0.0914 and recall from 0.1177 to 0.1118. Weight
0.10 reduced components further to 61.75 but sacrificed more IoU/recall. Thus
0.05 is a provisional Pareto candidate, not a confirmed default; seeds 142 and
242 are optional confirmation work rather than evidence that fragmentation is
solved.

## Protocol v3: final soft-to-hard diagnostics and strong control

Protocol v3 does not change the fixed-Euler update. It saves the final target
probability, probability margin, tau-independent cosine-similarity margin and
target hard decision for every sample in `final_soft_fields.pt`. It also
writes region-separated statistics and paired boundary crossings:

- `final_soft_region_stats.csv` and `final_soft_summary.json`;
- `paired_soft_deltas.csv` and `paired_soft_delta_summary.json` for guided runs;
- tensor hashes and hard-decoder agreement checks in `config.json`.

The regions separate selected truth target, its unconditioned subset,
unselected truth-target components, ROI true background and outside-ROI true
background. This distinguishes "soft margin moved correctly but did not cross
the hard boundary" from true recovery, false addition and fragmentation.

Protocol-v1/v2 baselines cannot be reused. The first late-strong control-limit
experiment is retained for exact reproduction:

```bash
bash project/geodata-3d-conditional/experiments/stage1_probability/run_phase1b_late_strong.sh
```

The completed seed-42 single sample established that the soft/hard boundary is
not a structural blocker. Label-9 IoU rose from 0.0286 to 0.6653, recall from
0.0473 to 0.9977, and 13,846 hard voxels (5.28%) changed with zero condition
violations. However, predicted target volume exceeded truth by 49.7%, target
components rose from 37 to 209, and the last four Euler steps still changed
2,955, 3,673, 4,246, and 4,254 hard voxels. Do not run the four-sample
confirmation for this overdriven configuration.

The direct cause is now recorded: class-balanced BCE on the continuous
0.5-binary/0.5-Gaussian target used positive/negative scales 6.1485/0.5443.
For a halo target of 0.168, its per-voxel optimum is about 0.695 rather than
0.168. The unit-norm controller then discarded the small raw-gradient
magnitude and continued increasing its relative force near the endpoint.

## Protocol v4: calibrated target and convergence-aware control

Protocol v4 retains the complete v3 diagnostics and the legacy modes, while
adding two explicitly versioned alternatives:

- `calibrated_soft_bce_hard_dice_v2` uses unweighted proper BCE for the
  continuous multiscale probability and Dice against only the exact binary
  target core;
- `reference_norm_relative_v2` scales gradients relative to the first nonzero
  active gradient norm, so a shrinking gradient produces a shrinking update;
- `windowed_sine` rises after the configured start, peaks in the active
  interval, and decays to zero at the integration endpoint.

The trace records calibration error, core/background/halo probabilities,
legacy class scales, reference-gradient ratio, uncapped/capped guidance ratio,
and cap hits. These settings are strict pairing fields, so protocol-v3
baselines cannot be reused.

Run the new one-sample strict pair from the repository root:

```bash
bash project/geodata-3d-conditional/experiments/stage1_probability/run_phase1b_calibrated_control.sh
```

Set `PYTHON_BIN=/absolute/path/to/python` if the active GPU environment is not
the shell default. Every invocation creates a new alpha-zero baseline and
refuses to overwrite any non-empty result directory. Inspect the single sample
before increasing `N_SAMPLES`; only a setting that fixes overgrowth,
fragmentation, and endpoint churn proceeds to four and then twelve pairs.

## Paper-style 3-D visualization

The visualization runner consumes completed strict-pair artifacts and does
not rerun the network. It creates fixed-camera categorical cutaways,
target-geometry overlays, a paired hard-change audit, empirical ensemble
occurrence-probability surfaces, editable VTK image volumes, and a manifest.

Run from the repository root:

```bash
PROJECT=project/geodata-3d-conditional
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"
PAIR="$PROJECT/experiments/stage1_probability/runs/cond_generation_0/label9/all/screen_seed42_n4_s32"
OUT="$PROJECT/experiments/stage1_probability/figures/cond_generation_0_label9_all_seed42"

PYVISTA_OFF_SCREEN=true PYTHONPATH=src .venv/bin/python \
  "$PROJECT/scripts/stage1/visualize_probability_guidance.py" \
  --truth-model "$CASE/true_model.pt" \
  --boreholes "$CASE/boreholes.pt" \
  --baseline-dir "$PAIR/baseline" \
  --guided-dir "$PAIR/alpha010" \
  --target-label 9 \
  --sample-id 3 \
  --cut-fraction 0.52 \
  --probability-threshold 0.25 \
  --probability-threshold 0.5 \
  --probability-threshold 0.75 \
  --output-dir "$OUT" \
  --overwrite
```

Omit `--sample-id` to select the largest `delta_selected_roi_iou`; the rule is
recorded in `manifest.json`. For a paper, prefer an explicit sample ID and show
the ensemble/paired aggregate evidence beside it. The probability panels are
frequencies across saved hard-label realizations, not per-sample soft decoder
confidence. If the server's VTK lacks EGL/OSMesa support, prefix the same
command with `xvfb-run -a env PYVISTA_OFF_SCREEN=true`.

## GPU smoke commands

Run from the repository root.  The baseline and guided commands must use the
same device string and all the same non-alpha arguments.

```bash
PROJECT=project/geodata-3d-conditional
RUNS="$PROJECT/experiments/stage1_probability/runs"
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"

PYTHONPATH=src python "$PROJECT/scripts/stage1/run_probability_guidance.py" \
  --ckpt-path "$CKPT" \
  --model-weights ema \
  --samples-dir "$CASE" \
  --target-label 9 \
  --component-mode all \
  --n-samples 1 \
  --n-steps 16 \
  --alpha 0 \
  --seed 42 \
  --device cuda \
  --output-dir "$RUNS/cond_generation_0/label9/all/smoke_baseline"

PYTHONPATH=src python "$PROJECT/scripts/stage1/run_probability_guidance.py" \
  --ckpt-path "$CKPT" \
  --model-weights ema \
  --samples-dir "$CASE" \
  --target-label 9 \
  --component-mode all \
  --n-samples 1 \
  --n-steps 16 \
  --alpha 0.05 \
  --seed 42 \
  --device cuda \
  --baseline-dir "$RUNS/cond_generation_0/label9/all/smoke_baseline" \
  --output-dir "$RUNS/cond_generation_0/label9/all/smoke_alpha005"
```

The guided command stops before sampling when target hashes, EMA policy,
initial-noise policy, solver settings, or any other strict pairing field differ.

## Required artifacts

Each completed run contains:

- `config.json` with protocol fields, source/asset hashes, initial-noise hashes,
  and completion status;
- `model_load_report.json` and `input_validation.json`;
- `target_mask.pt`, `target_probability.pt`, `target_roi_mask.pt`, and
  `target_manifest.json`;
- `sample_*.pt` and their stable tensor hashes;
- `guidance_trace.csv` with loss, raw/used gradients, velocity ratios,
  optional spatial-gradient loss/diagnostics, hard-label changes, and
  pre/post-projection violations;
- `sample_metrics.csv` and `metrics_summary.json`;
- `final_soft_fields.pt`, region statistics and final soft summaries;
- `ensemble_summary.json` with uniqueness, target coverage, variance, and
  pairwise hard-label disagreement;
- for a guided run, paired baseline metrics, `paired_deltas.csv`, a sparse
  class-transition matrix, `paired_delta_summary.json`, and paired soft/hard
  boundary-crossing diagnostics.

Phase 1 passes only when paired hard-label and geometric metrics improve across
seeds while post-projection condition violations remain exactly zero.  Loss
reduction by itself is not a pass.
