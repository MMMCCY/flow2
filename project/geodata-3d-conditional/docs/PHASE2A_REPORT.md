# Phase 2a report: ideal 3-D full-lithology property guidance

Date: 2026-07-29

Decision: **PASS with explicit caveats.**

Phase 2a validates an inference-time upper bound in which a frozen conditional
flow model receives a truth-derived, full-resolution, noiseless two-channel 3-D
property volume. It does not validate measured geophysics, a realistic
petrophysical inversion, removal of surface/borehole conditions, or a 2-D
gravity/magnetic forward loss.

## Experimental path

All experiments preserve the Phase-0/Phase-1 invariants:

- normal frozen category embedding and EMA values for all 411 trainable entries;
- fixed-Euler midpoint integration with strictly paired CPU-generated noise;
- identical checkpoint, inputs, target/property hashes, temperature and time grid;
- alpha zero uses the explicit no-gradient baseline path;
- surface and borehole embeddings are projected before sampling and after every step;
- no training, U-Net or checkpoint modification, and no historical 2-D field loss.

Three single-sample steps established the operating point:

1. A scalar density proxy reduced property residual and improved global metrics,
   but label-9 recall and volume error worsened. A single soft expectation was
   underdetermined and did not make the intrusion observable.
2. Adding a complete synthetic susceptibility channel with a distinctive
   label-9 contrast improved label-9 direction at alpha/cap 0.10, but recovery
   remained too sparse and fragmented.
3. Keeping the property target fixed and increasing only alpha/cap to 0.25
   materially crossed hard-label boundaries and recovered portions of all four
   major target bodies. This justified the pre-registered multi-seed run.

The confirmation contains 12 strict pairs: seeds 42/142/242, four sequential
samples per seed, 32 fixed-Euler steps, alpha/cap 0.25.

## Strict 12-pair result

The authoritative generated evidence is:

- `experiments/stage2_property/reports/phase2a_v1_12pair/REPORT.md`;
- the adjacent `summary.json` and `paired_samples.csv`;
- `class_summary.csv`, `component_summary.csv`, and `seed_summary.csv`.

Regenerate it without modifying immutable runs:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2a.py \
  --overwrite
```

The aggregator revalidates strict baseline/guided configs, source/property
hashes, run completion, EMA, paired initial noise, finite 32-step traces, exact
conditions, confidence locality, topology, component recovery and diversity.

All 12 per-pair frozen gates and all three seed-level diversity gates pass.
Aggregate means are:

- global voxel accuracy: `0.5972 -> 0.6381`;
- truth-present fixed-set mIoU: `0.2771 -> 0.3443`;
- historical dynamic-union mIoU: `0.1804 -> 0.1980`;
- hard-property loss: `1.4781 -> 0.5187`;
- label-9 IoU: `0.0314 -> 0.4808`;
- label-9 precision: `0.0788 -> 0.9032`;
- label-9 recall: `0.0520 -> 0.5075`;
- label-9 centroid distance: `16.9518 -> 3.9825` voxels.

Global accuracy, fixed-set mIoU, hard-property residual, label-9 IoU,
precision and recall improve in every pair. Dynamic-union mIoU improves in
11/12 pairs; its single negative delta is caused by the discontinuous entry of
a tiny absent-class prediction and is retained rather than hidden.

## Full-model and geometry findings

At least five of the eight truth-present non-air classes improve IoU in every
pair. Class-level outcomes across 12 pairs are:

- labels 1/3/4/5/9 improve in 12/12 pairs;
- label 6 improves in 10/12 and decreases in 2/12, at very low absolute IoU;
- label 2 decreases in 12/12 pairs, with mean IoU delta `-0.0129`;
- label 13 remains unrecovered with IoU zero.

Label 9 is accurately located but incomplete. The four major truth components
contain 4,079/2,192/2,043/627 voxels. Their mean guided recalls are
0.4222/0.6047/0.6138/0.3955, with minima
0.3685/0.5360/0.5091/0.3158 across all pairs. Tiny components of at most five
voxels contain 3.70%-8.06% of guided target mass; the eight largest components
contain 86.57%-94.36%. Final-step hard churn is 0.79%-0.99% of the full volume.

All conditions remain exact, all paired hard changes remain inside property
confidence, every seed retains four unique guided samples, and outside-ROI
pairwise disagreement remains 0.1443-0.1526.

## What Phase 2a proves

Phase 2a proves that:

- a frozen conditional generative prior can accept differentiable, complete
  multi-lithology 3-D property evidence during inference;
- sufficient property observability produces consistent soft-to-hard crossing,
  rather than only lowering a continuous loss;
- the result improves the complete hard model as well as the sparse label-9
  pressure test while preserving exact hard conditions and ensemble diversity;
- the effect is robust across 12 strict pairs at the tested ideal operating point.

## What it does not prove

Phase 2a does not prove that:

- realistic density/susceptibility values uniquely determine all lithologies;
- blurred, noisy, sparse or depth-degraded observations retain this performance;
- 2-D gravity or magnetic fields can recover the same 3-D geometry;
- boreholes and surface conditions can be removed;
- label-9 geometry is complete or topologically faithful;
- CUDA guidance is bitwise identical across independent processes. One repeated
  sample differed at 10 of 262,144 hard voxels (0.0038%), while alpha-zero was
  byte-identical.

## Decision and next work

The ideal full-resolution property upper bound is validated with geometry,
class-tradeoff and determinism caveats. Phase 2a is complete.

Phase 2b should now reduce information privilege without changing the sampler:

1. introduce overlapping or shared property values between lithologies;
2. reduce the artificial label-9 susceptibility contrast in predeclared steps;
3. test property-codebook sensitivity and identify the observability threshold;
4. retain the same 12-pair hard, class, component, condition and diversity gates.

Only after the ambiguity study should Phase 3 degrade spatial observation by
blur, downsampling, missing regions, depth-dependent confidence and noise.
Two-dimensional gravity/magnetic forward losses remain later work and must not
be merged into these attribution experiments.
