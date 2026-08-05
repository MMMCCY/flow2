# Phase 2: ideal full-lithology 3-D property-volume guidance

## Objective

Determine whether a frozen conditional flow model can use complete
three-dimensional physical-property information to improve the full decoded
geological model, including the sparsely conditioned label-9 intrusion, rather
than only matching one categorical oracle.

Phase 2a uses a full-resolution, noiseless property volume derived from truth.
It is an idealized property oracle and not measured geophysics. Its purpose is
to test the property mapping, all-category gradient and hard-label response
before adding inverse-resolution effects or a 2-D forward operator.

## Mathematical path

For raw geological label `k` and property channel `j`, an explicit complete
table supplies `q[j,k]`. The current continuous state is soft-decoded with the
unchanged frozen category embedding:

```text
p_t(k,r) = softmax_k(cos(x_t(r), e_k) / tau_t)
q_t(j,r) = sum_k p_t(k,r) q(j,k)
```

At every configured three-dimensional scale `s`, the same operator is applied
to the current expected property and truth-derived target:

```text
L_property = sum_s lambda_s || W_s * [S_s(q_t) - S_s(q_true)] ||^2
```

The implementation normalizes each property channel by its weighted target
standard deviation before combining channels. Density, susceptibility or
velocity units therefore cannot dominate only because of numerical magnitude.

## Phase 2a scope

- Explicit complete mapping for every model class, raw labels `-1..13`.
- One or more named property channels with recorded units and weights.
- Full soft-class expectation, not a target-label-only probability.
- Matched Gaussian operators on predicted and target volumes.
- Full-resolution and coarse-scale property residuals in one audited loss.
- Confidence masks capable of excluding hard-conditioned cells and later
  representing missing/uncertain regions.
- Reuse of the protocol-v4 reference-norm controller and endpoint-decaying
  schedule for the first controlled smoke.
- Hard global, per-class, target-label, geometry, condition and ensemble metrics.

## Out of scope for Phase 2a

- Retraining, U-Net changes or checkpoint replacement.
- Calling the controlled property values calibrated petrophysics.
- Blur, downsampling, depth-dependent resolution, noise or missing data on the
  supplied target; those are Phase 3 degradation experiments.
- A 2-D gravity, magnetic or seismic forward loss.
- Combining property-volume and historical 2-D gravity guidance.
- Treating a lower property loss as evidence of geological success.

## Property configuration rules

The schema is `full_lithology_property_channels_v1`. Every channel must contain
finite values for every raw label `-1..13`; silent defaults are prohibited.
Property tables, target tensors, confidence masks and implementation sources
must be hashed in every run.

The first controlled configuration is
`experiments/stage2_property/configs/ideal_distinct_density_proxy_v1.json`.
Its scalar values are an injective synthetic proxy inherited from the existing
screening code, not measured densities. This is intentionally the easiest
property-volume bridge at the hard-label table level, although a single soft
expectation remains non-unique for 15 categorical probabilities. The second
configuration,
`ideal_density_susceptibility_label9_contrast_v1.json`, is a controlled
observability ablation: it adds a complete synthetic susceptibility channel
with a distinctive label-9 response. A subsequent ambiguity ablation must
merge or overlap selected class properties to test whether the geological
prior can resolve non-unique property values.

## Experimental invariants

- Normal checkpoint embedding and EMA values for every trainable parameter.
- Same checkpoint, property config/target/confidence hashes, input tensors,
  seed, initial noise, temperature, fixed-Euler grid and source hashes.
- Alpha zero takes an explicit no-gradient branch and reproduces its paired
  baseline exactly.
- Conditions are projected before the first step and after every step.
- Guided and baseline run directories are immutable and never overwritten.
- Phase-1 probability loss and historical 2-D field loss are inactive.

## Required evaluation

- Property loss and per-scale loss; property MAE per channel.
- Raw/used gradient norm, reference-gradient ratio, cap hits and endpoint churn.
- Full hard-label voxel accuracy, historical dynamic-union mean IoU, and
  fixed-denominator truth-present-class mean IoU.
- Per-present-class IoU, precision, recall and volume error.
- Label-9 IoU, precision, recall, centroid, volume and size-stratified topology.
- Paired class-transition matrix and changes inside/outside property-error ROIs.
- Surface/borehole condition violations.
- Ensemble uniqueness and inside/outside disagreement.
- Property residual slices/volumes and fixed-camera categorical 3-D figures.

## Development gate before GPU sampling

1. Exact hard-label-to-property mapping passes for raw labels `-1..13`.
2. Soft one-hot probabilities equal hard property mapping.
3. The matched-scale loss is zero for identical volumes and positive for a
   controlled perturbation.
4. The loss has finite non-zero gradients through the real soft decoder.
5. Zero-confidence cells do not contribute at full resolution.
6. Phase-1 and Phase-0 lightweight tests remain green.
7. A new Phase-2 runner records strict pairing fields and refuses incomplete or
   non-empty outputs before a GPU command is issued.

All seven development-gate items are now implemented. The alpha-zero sampler
matches a reference projected fixed-Euler trajectory exactly in CPU tests; the
positive-alpha path has finite non-zero gradients and preserves conditioned
embeddings; strict pairing includes all property/source/target/confidence
hashes; and the runner refuses non-empty outputs.

The first GPU smoke with the scalar configuration completed as a valid strict
pair. It improved global voxel accuracy (`+0.00948`), global mean IoU
(`+0.00236`) and hard-property loss (`-17.94%`), but failed the geological
gate: label-9 recall decreased, predicted label-9 volume moved farther from
truth, and 2,623 baseline label-9 voxels changed to other classes while only
31 other voxels changed to label 9. This is evidence that the implemented
objective is active, but that the low-contrast scalar expectation does not
make the target intrusion sufficiently observable. It must not be promoted to
the multi-sample confirmation.

The one-sample two-channel observability ablation also completed as a valid
strict pair. It improved label-9 IoU, precision, recall and true-positive count,
as well as global accuracy, truth-present fixed-set mIoU and six of eight
truth-present class IoUs. It nevertheless under-predicted label-9 volume and
increased its component count from 37 to 109. The path is informative but has
not yet passed the geometry gate. Details are frozen in
`docs/PHASE2_PROGRESS.md`.

At cap 0.10 the controller was capped in 18 of 24 guidance-active steps. The
next predeclared single-sample experiment therefore keeps the two-channel
property target fixed and tests the already audited Phase-1 upper controller
setting `alpha=cap=0.25`.

That upper-bound pair improves label-9 IoU/precision/recall to
0.4816/0.9005/0.5087 and recovers portions of all four major truth bodies, while
global accuracy and truth-present mIoU also improve. Its raw topology remains
fragmented. The pre-registered seed-42 n=4 confirmation subsequently passed
all eight validity, hard-geology, size-stratified topology, endpoint and
diversity gates. Seeds 142 and 242 subsequently pass as well: all 12 strict
pairs meet the frozen per-pair gates and all three seed ensembles meet the
diversity gate. Phase 2a is complete with caveats; details are in
`docs/PHASE2A_REPORT.md` and generated evidence under
`experiments/stage2_property/reports/phase2a_v1_12pair/`.

## Phase 2a scientific success gate

The first full-resolution property experiment succeeds only if, across strict
multi-seed pairs:

- fixed truth-present mean IoU and voxel accuracy improve consistently, not
  only label 9; the historical dynamic-union mIoU and absent-class
  hallucinations must also remain reported;
- label 9 retains a material hard-geometry improvement;
- a majority of truth-present non-air classes improve IoU without one class
  absorbing implausible volume;
- property loss improvement survives hard decoding as lower hard-property
  residual and better categorical metrics;
- conditions remain exact and outside-error regions are preserved;
- ensemble diversity remains measurable;
- topology and endpoint churn are reported even when voxel metrics improve.

Numeric thresholds will be pre-registered after the alpha-zero regression and
one-sample smoke reveal the natural scale of this new, weaker information
source. They must be fixed before the four- and twelve-pair confirmation.

This requirement was satisfied: thresholds were frozen before n=4 and the
12-pair confirmation passed them without post-hoc changes. The next experiment
is the separately frozen Phase-2b ambiguity study in `docs/PHASE2B_SPEC.md`, not
a modification of the completed Phase-2a runs. Its CPU development gate is
complete. The first Phase-2b GPU anchor subsequently passed; the ambiguity
levels remain a separate in-progress screen.
