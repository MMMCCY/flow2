# Phase 1 report: truth-derived 3-D probability-volume guidance

Updated: 2026-07-29 after the completed protocol-v4 12-pair confirmation.

## Decision

Phase 1 is closed as **mechanism validated with topology and endpoint-stability
caveats**. The frozen inference path can use a three-dimensional oracle to
materially and consistently change the decoded hard-label geometry while
preserving the original surface and borehole conditions. This is not an
unqualified pass of every pre-registered clause, and it is not evidence that
measured geophysics can yet reconstruct the model.

The authoritative machine-generated aggregate is
`experiments/stage1_probability/reports/phase1b_v4_12pair/summary.json`. It is
reproducible from immutable saved tensors and CSV files with:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage1/summarize_phase1b.py \
  --overwrite
```

## Experiment identity

- Case: `cond_generation_0`
- Target: all truth label-9 voxels (8,968 voxels)
- Seeds: 42, 142 and 242
- Samples: four per seed, 12 strict pairs in total
- Sampler: 32-step fixed-Euler midpoint path
- Weights: normal frozen embedding plus EMA for all 411 trainable entries
- Probability loss: `calibrated_soft_bce_hard_dice_v2`
- Controller: `reference_norm_relative_v2`
- Schedule: `windowed_sine`
- Guided alpha/cap: 0.25/0.25
- Condition projection: before sampling and after every Euler step

All 12 guided samples share their checkpoint, truth, boreholes, source hashes,
target/ROI hashes, initial-noise hashes, time grid and non-alpha arguments with
their alpha-zero baselines. All saved runs report `completed`; all condition
violations and soft/hard decoder mismatches are zero.

## Primary hard-label evidence

| Metric | Baseline mean | Guided mean | Result |
|---|---:|---:|---|
| Global voxel accuracy | 0.5972 | **0.6432** | improved 12/12 |
| Global mean IoU | 0.1804 | **0.2178** | improved 12/12 |
| Label-9 IoU | 0.0314 | **0.8099** | improved 12/12 |
| Label-9 precision | 0.0788 | **0.8274** | improved 12/12 |
| Label-9 recall | 0.0520 | **0.9747** | improved 12/12 |
| Label-9 centroid distance | 16.9518 | **3.3798** | improved 12/12 |
| Selected-ROI IoU | 0.0351 | **0.9392** | improved 12/12 |
| Selected-ROI precision | 0.1050 | **0.9627** | improved 12/12 |
| Selected-ROI absolute volume-error fraction | 0.5101 | **0.0139** | improved 12/12 |
| Outside-ROI voxel accuracy | 0.639290 | **0.639312** | effectively preserved |

The guided result changed 5.1811% of the full categorical volume on average;
98.7588% of all changes were inside the predeclared ROI. Each seed retained
four unique decoded samples. Outside-ROI pairwise ensemble disagreement was
also preserved, so the result is not an identical-sample collapse.

## Pre-registered gate audit

The absolute IoU/precision/recall, IoU delta, centroid reduction, hard-change
range, ROI concentration, zero-condition, outside-ROI and diversity clauses
pass. The strict gate nevertheless remains `not_full_pass`:

- mean full-volume target-component ratio is 1.2812 versus the limit 1.25;
- the diagnostic ROI component ratio is 1.2665 versus 1.25;
- raw largest-component fraction changes from 0.7239 to 0.3856, so the
  conservative no-loss reading fails;
- a protocol-v4 fixed-camera rendering still needs to be generated and
  visually reviewed; older Phase-1a/v3 figures do not satisfy that item.

These failures are retained as failures. No threshold has been rewritten
after seeing the result.

## Truth-relative topology interpretation

Raw component counts treat a one-voxel fragment like a major geological body.
The size-stratified audit therefore provides context without replacing the
pre-registered metric:

- truth label 9 has seven six-connected components with sizes
  4,079, 2,192, 2,043, 627, 22, 4 and 1;
- every one of the 12 guided samples contains exactly four ROI components with
  at least 20 voxels;
- mean guided ROI top-four sizes are approximately
  4,074, 2,203, 2,129 and 600;
- the guided ROI largest-component fraction is 0.4487 versus truth 0.4548;
- components of at most five voxels contain only 0.66% of guided ROI target
  mass on average;
- most remaining false target mass lies outside the selected ROI and is a
  baseline remnant rather than target-oracle growth.

The principal truth bodies are therefore recovered well. The raw-count failure
is dominated by tiny fragments, but it remains an honest limitation of the
current decoder/sampler endpoint.

## What Phase 1 proved

- The inference-only gradient path is active and capable of crossing the real
  soft-to-hard decoder boundary at useful scale.
- A truth-aligned 3-D oracle can reconstruct the sparsely conditioned label-9
  geometry consistently across independent paired samples.
- Surface and borehole conditions can remain exact under guidance by projecting
  them after every integration step.
- Strong target recovery need not eliminate ensemble diversity or damage the
  model outside the target region.

## What Phase 1 did not prove

- The probability target is derived directly from truth and is not a measured,
  inverted or simulated geophysical observation.
- Only one target class was guided; improvement of complete multi-lithology
  geometry has not been established.
- It does not show that geophysics can replace surface or borehole conditions.
- It does not establish density, susceptibility or seismic-property mappings.
- It does not establish that a 2-D gravity/magnetic observation contains enough
  depth information for the same reconstruction.
- Final-step hard-label churn remains above the paired baseline and should stay
  visible in subsequent trace audits.

## Transition to Phase 2

Phase 2 replaces the one-class probability oracle with a complete
three-dimensional property volume. It first uses a full-resolution, noiseless,
truth-derived property target and all soft categories. Both the predicted and
target properties must pass through identical multi-scale 3-D operators before
comparison. Blur, resolution loss, noise, missing regions and real/2-D forward
physics remain later stages.

