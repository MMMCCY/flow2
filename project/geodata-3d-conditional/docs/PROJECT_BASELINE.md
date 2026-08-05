# Project baseline

Last updated: 2026-07-30 after the completed Phase-2b seed-42 n=4 bracket and
post-bracket `paired_c100` robustness fallback.

## Goal and scope

The current goal is to improve inference-time conditional 3-D geological
generation under sparse surface and borehole constraints by adding spatially
broad synthetic geophysical information. The first phase is an in-distribution
synthetic oracle experiment. Training, the U-Net architecture, and the current
checkpoint remain frozen.

The scientific target is a meaningful improvement in decoded geology:
location, extent, volume, connectivity, and ensemble uncertainty. A reduction
in continuous geophysical proxy loss alone is not success.

## Authoritative code and checkpoint

- Development repository: `flow2`
- Starting code revision: `5b717ec`
- Checkpoint SHA-256:
  `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`
- Checkpoint size: `849415038` bytes
- Epoch/global step: `1651 / 8376`
- `state_dict`: 412 tensors, 53,049,574 elements
- `ema_shadow`: 411 tensors, 53,049,349 elements

The only state entry absent from EMA is `embedding.weight`. This is expected:
the embedding is frozen and the EMA callback only tracks trainable parameters.
Canonical inference therefore uses the raw checkpoint embedding and EMA values
for all trainable network parameters.

## `cond_generation_0` input facts

Both uploaded tensors are finite, integer-valued float32 volumes with shape
`[1, 1, 64, 64, 64]`.

- `true_model.pt` SHA-256:
  `a14f5a740a3ea6af8c2eec8507d79fe3a94f1b704626f0478a9e526f33b71cb3`
- `boreholes.pt` SHA-256:
  `1e2592ce0e820569d0b3fa13eaa011c3bc70c312a19e5eab878c428facdbdb41`
- Truth label range: `-1..13`
- Every non-air saved borehole voxel exactly equals the truth.
- Full vertical boreholes: 9
- Effective air/surface/borehole condition: 69,107 voxels (26.3622%)
- Unconstrained subsurface: 193,037 voxels (73.6378%)

Truth label counts:

| Label | Voxels |
|---:|---:|
| -1 | 64,589 |
| 1 | 45,273 |
| 2 | 24,520 |
| 3 | 14,533 |
| 4 | 73,801 |
| 5 | 18,902 |
| 6 | 11,543 |
| 9 | 8,968 |
| 13 | 15 |

For the current label-9 demonstration:

- Label-9 fraction: 3.4210%
- Borehole-conditioned label-9 voxels: 13 (0.1450%)
- Boreholes intersecting label 9: 2 of 9
- Label-9 components: 7 with 6-connectivity; 4 with 26-connectivity
- 26-connected component sizes: 4,079; 2,197; 2,043; 649

This is a suitable sparse-condition stress case. “Label 9” remains a case-level
target class, not a universal dataset-wide synonym for one dike geometry.

## Development sequence

1. Freeze and audit model weights, inputs, random noise, and baseline semantics.
2. Run a target-probability-volume upper-bound experiment.
3. Run a multiscale 3-D property-volume experiment.
4. Degrade the property volume by blur, resolution loss, missing regions, and
   noise.
5. Compare against the historical 2-D proxy and perform paired ablations.

Current decision: steps 1 and 2 are complete. The Phase-1 mechanism was
validated across 12 strict pairs, with documented raw topology and endpoint
caveats; see `docs/PHASE1_REPORT.md`. Step 3 now has an isolated property loss,
fixed-Euler sampler, evaluator, strict runner and alpha-zero CPU regression.
A first strict Phase-2 GPU pair ran with the controlled scalar density
proxy. The path was active and valid: conditions remained exact, global voxel
accuracy and mean IoU increased, and hard-property loss fell by 17.94%.
However, label-9 recall and volume error worsened, so the run failed the
geological success gate. A second strict pair added a complete synthetic
susceptibility channel with a distinct label-9 contrast. Label-9 IoU, precision,
recall and true-positive count then improved, confirming that property
observability matters, but predicted target volume remained too small and
components increased from 37 to 109. A controlled `alpha=cap=0.25` upper-bound
pair then produced label-9 IoU/precision/recall of
0.4816/0.9005/0.5087, centroid distance 3.42 voxels, and improvements in global
accuracy and fixed truth-present mIoU. All four major truth bodies are partly
recovered, though still split. The pre-registered seed-42 n=4 confirmation
then passed all eight frozen gates: every pair improves
global accuracy, fixed truth-present mIoU and label-9 IoU/precision/recall;
size-stratified topology, endpoint churn, conditions and diversity also pass.
Seeds 142 and 242 also pass, giving 12/12 successful strict pairs. Phase 2a's
ideal two-channel full-resolution 3-D property upper bound is therefore
validated with explicit fragmentation, label-2 tradeoff, label-13 failure and
CUDA repeatability caveats; see `docs/PHASE2A_REPORT.md`. Phase 2b must now test
overlapping/less-distinctive property codebooks before spatial degradation.
Its protocol, five predeclared levels, isolated launcher, codebook diagnostics
and screen summarizer are implemented. The seed-42 `distinct_c100_anchor` GPU
pair passes strict pairing, the full per-pair gate and the Phase-2a regression;
the ambiguous levels `paired_c100` and `paired_c025` also pass, with the latter
close to component/top-eight thresholds. `paired_c010` then fails the absolute
target, major-component and size-stratified topology gates despite improved
global/continuous directions. The exact label-6/label-9 collision also fails
and removes useful target recovery. The frozen screen selects passing
`paired_c025` and adjacent failing `paired_c010` for a seed-42 n=4 bracket.
That bracket is now complete: `paired_c025` lands in the frozen transition
region at 3/4 pair gates, while `paired_c010` is a confirmed 0/4 seed-42
failure. Both ensembles remain diverse, but neither level is authorized for
multi-seed confirmation. The separately frozen `paired_c100` seed-42 n=4
fallback also lands at 3/4: sample 2 misses only the four-major-body minimum
recall gate (`0.2313 < 0.25`). Phase 2b therefore closes without a
multi-seed-confirmed ambiguous-codebook operating point. The distinct Phase-2a
property upper bound remains the only 12/12 validated starting point for a
separately frozen Phase-3 spatial-degradation study; see
`docs/PHASE2B_REPORT.md`.
