# Phase 4: acquisition-domain geophysical guidance

Date frozen: 2026-07-31, before implementation of the new gravity operator or
any Phase-4 GPU run.

## Objective

Replace truth-derived 3-D oracle guidance with differentiable acquisition-domain
physics while keeping the trained flow2 model frozen. Geophysics should add
global information where sparse boreholes are weak; the learned geological
prior plus exact surface/borehole conditions should, in return, reduce the
large family of 3-D property models compatible with the observations.

The required output is not merely a lower field residual. It is an ensemble of
condition-exact, geologically plausible hard-label models that improves the
complete 3-D geology and target geometry while matching independently fixed
synthetic observations.

## Stage decomposition

Phase 4 is not one experiment:

- **Phase 4a — gravity-only upper bound:** full-support differentiable gravity
  forward modeling from categorical density contrast to a 2-D surface survey.
- **Phase 4b — reciprocal joint constraint:** combine the frozen Phase-3 smooth
  3-D operating point with gravity and compare baseline, 3-D-only,
  gravity-only and joint guidance under strict pairing.
- **Phase 4c — complementary physics:** add convolutional seismic response,
  then test gravity-plus-seismic. Magnetics/electrical methods remain optional
  later extensions and require separate calibrated assumptions.

Phase 3 remains an independent attribution experiment. Phase 4a operator and
CPU validation may be developed before Phase 3 finishes, but Phase 4b cannot
claim a joint result until the Phase-3 operating point is frozen.

## Phase-4a observation model

For raw label `k`, an explicit synthetic density contrast `rho(k)` in
`kg m^-3` is required for every category `-1..13`. Values must be justified as
a synthetic contrast scenario rather than claimed as site-calibrated
petrophysics. At least one lower-contrast/collision control is mandatory.

The predicted density volume is the soft expectation over all categories. At
surface, air and borehole condition voxels, it is overwritten by the exact
hard density before forward modeling. The gravity observation is

```text
d_pred = M F_g(rho_pred_known)
d_obs  = M F_g(rho_true) + epsilon
L_g    = || C_d^(-1/2) (d_pred - d_obs) ||^2 / valid_observations
```

where `F_g` is the vertical gravitational acceleration of rectangular prisms,
`M` is a fixed survey mask and `C_d` defines explicit observation uncertainty.
Output units are mGal. Coordinate origin, axis order, cell dimensions,
topography/observation height and reference-density convention are immutable
manifest fields.

The first Phase-4a gate is a noiseless, regular-grid synthetic inverse-crime
upper bound. It must be named as such. Noise, sparse stations, irregular
topography and uncertain petrophysics are later controlled degradations.

## Forward-operator requirements

The historical `SimpleGravityForward` is retained only as an audited negative
baseline. Its truncated 9x9 point-mass-style convolution, unit cells, `G=1`,
relative output and mean removal are not accepted as the new physics path.

The new operator must:

- use the exact rectangular-prism `g_z` response or a numerically verified
  full-support equivalent;
- include every cell in every observation (no silent finite convolution
  support and no circular wraparound);
- use SI coordinates internally and report mGal;
- define the vertical tensor axis explicitly (current data: final spatial
  axis, larger index upward);
- support a fixed observation grid/mask and optional diagonal uncertainty;
- be differentiable with respect to density;
- cache geometry kernels without detaching the density path;
- preserve dtype/device deliberately and fail on invalid or non-finite inputs.

CPU acceptance tests before any GPU guidance:

1. comparison against a direct small-grid prism reference;
2. linearity and sign tests;
3. horizontal symmetry and translation/no-wraparound tests;
4. deeper identical masses have weaker surface response;
5. full-volume support, including far-corner cells;
6. finite-difference directional derivative against autograd;
7. adjoint dot-product test;
8. observation mask and uncertainty normalization;
9. hard-label and one-hot soft density mappings agree;
10. exact-condition overwrite gives zero condition-voxel gradient;
11. alpha-zero equals the paired projected fixed-Euler baseline.

## Immutable synthetic observations

Observations are built in a separate command before sampling. The builder
writes a manifest containing:

- truth, density-codebook and source hashes;
- grid shape, cell size, origin and axis convention;
- station coordinates, topography rule, height, mask and uncertainty hashes;
- raw and processed response hashes and units;
- noise seed/model and whether the experiment is inverse crime;
- an explicit `truth_derived=true`, `measured_geophysics=false` statement.

The guidance runner consumes these files read-only and refuses an asset hash
mismatch. It must not generate a new observation per sample or per Euler step.

## Strict paired experiments

For each frozen observation, run:

1. `alpha=0` projected fixed-Euler baseline;
2. gravity-only guidance;
3. post-hoc baseline ensemble reranking by gravity residual;
4. Phase-2a ideal 3-D property ceiling as context, not as a competing physical
   method;
5. the historical proxy as an explicitly non-physical audit only.

Use the same checkpoint/EMA policy, conditioning, CPU noise, fixed time grid,
step count, decoder schedule and sample IDs within each comparison. Positive
alpha always requires a matching alpha-zero directory. Surface/borehole
violations must remain zero after every step.

Start with seed 42, one sample, then a frozen four-sample bracket. Multi-seed
claims require seeds 42/142/242 and the same 12-pair/3-ensemble discipline as
Phases 1 and 2. Controller strength may be screened only in a predeclared
order; failed settings remain reported.

## Phase-4a success and failure

A valid Phase-4a positive result requires all of the following:

- gravity residual improves against the strict paired baseline and against
  simple post-hoc reranking;
- global accuracy and truth-present mIoU improve consistently;
- most truth-present classes do not degrade through one-class volume capture;
- label-9 IoU/precision/recall, volume and major-component recovery improve
  materially;
- final topology, fragmentation and endpoint churn remain acceptable;
- all conditions remain exact and the guided ensemble retains non-zero
  diversity;
- observation-consistent ensemble spread is reported, especially in depth
  regions weakly resolved by gravity.

If gravity loss falls but hard geology remains baseline-like, Phase 4a has
demonstrated field fitting without geological recovery and is a scientific
failure at that operating point. Do not repair that conclusion by reporting
only soft fields or label-9 voxel-count multipliers.

## Phase-4b joint evidence and double-counting rule

The complete joint ablation is:

| Arm | Frozen prior/conditions | 3-D smooth observation | Gravity |
|---|---|---:|---:|
| Baseline | yes | no | no |
| Phase-3 only | yes | yes | no |
| Gravity only | yes | no | yes |
| Joint | yes | yes | yes |

If the 3-D smooth product is derived by inversion from the same gravity survey,
it is not independent evidence. Such a joint loss must either use a documented
shared likelihood/regularization formulation or be labeled a same-data
algorithmic ablation; it cannot be advertised as two independent observations.
The preferred first joint experiment uses separately generated synthetic 3-D
property information and gravity so the information sources are explicit.

## Phase-4c rationale

Gravity integrates density over depth and is strongly non-unique. A
convolutional seismic response retains lateral and time/depth localization and
is therefore the first complementary physics candidate. It requires a complete
velocity/impedance codebook, wavelet, sampling interval, boundary convention,
noise model and differentiable forward validation. Gravity and seismic may
have complementary sensitivities, but similar lithology properties can still
collide; multi-physics reduces rather than eliminates non-uniqueness.

## Scope limits

- No training, U-Net, embedding or checkpoint changes.
- No claim of measured-data validation from truth-derived synthetic fields.
- No removal of borehole/surface constraints in Phase 4.
- No physical claim for uncalibrated property tables.
- No success claim from continuous field loss alone.
- No conflation of Phase-3 spatial degradation with Phase-4 acquisition-domain
  physics.

## Implementation status

The Phase-4a forward/operator CPU gate was implemented on 2026-07-31:

- `guidance/gravity.py` contains the analytic downward-positive rectangular-
  prism vertical response, full-support zero-padded FFT linear convolution,
  complete density mappings, diagonal-uncertainty field loss, exact-condition
  overwrite and deterministic observation construction;
- `guidance/gravity_sampling.py` injects that loss into the already validated
  projected fixed-Euler sampler without copying or changing the solver;
- `scripts/stage4/build_gravity_observation.py` writes immutable tensor assets,
  source/config hashes, physical-coordinate metadata and field statistics;
- `experiments/stage4_gravity/configs/` contains the distinct label-9 synthetic
  upper bound, an exact label-6/9 density-collision control and the first
  full-grid noiseless survey;
- `tests/test_phase4_gravity.py` covers direct prism agreement, linearity,
  sign, symmetry, translation/no-wraparound, depth decay, far support, finite
  differences, the adjoint identity, mask/uncertainty normalization, hard/soft
  mapping, zero condition gradients, deterministic assets and alpha-zero
  projected-Euler equality.
- `scripts/stage4/run_gravity_guidance.py` validates every observation/source
  hash, loads EMA, preserves the sequential CPU noise policy, runs the existing
  fixed-Euler solver and writes complete hard geology/gravity evidence;
- `scripts/stage4/audit_gravity_screen.py` enforces Phase-2a alpha-zero
  regression and the complete geology-plus-gravity gate;
- `scripts/stage4/rerank_gravity_ensemble.py` provides the mandatory n=4
  post-hoc baseline comparator without changing any geology;
- `configs/gravity_controller_manifest_v1.json` freezes alpha/cap 0.25 first
  and a conditional alpha 0.10 harm diagnostic. Its hash and level ID are
  strict pair fields.

The Phase-4-focused CPU gate passes 18 tests and the complete local suite
passes 103 tests. Analytic geometry is always constructed in CPU float64 before
conversion to the inference dtype, avoiding float32 far-field corner-term
cancellation while leaving the density gradient path differentiable.
The canonical cond_generation_0 64-cubed observation is
`experiments/stage4_gravity/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
Its field is finite, spans -1.0706 to 1.5260 mGal and has standard deviation
0.7621 mGal. The current runner also completes a real-checkpoint one-step CPU
alpha-zero smoke test with EMA and zero condition violations.

The frozen seed-42 n=1 alpha/cap 0.25 GPU strict pair has now completed and
fails the complete gate (`0/1`). Pairing, immutable hashes, alpha-zero hard
regression and all conditions pass. Hard gravity RMSE improves by
`0.06561 mGal`, but label-9 IoU/precision/recall degrade from
`0.0286/0.0675/0.0473` to `0.0159/0.0638/0.0207`; final prediction volume
falls from `6283` to `2914`, and none of the four major truth components meets
the recovery gate. The final paired outputs differ at `6562` hard voxels,
including `3392` label-9-to-other and only `23` other-to-label-9 transitions.
This is field fitting through density/lithology non-uniqueness, not geological
recovery. Because the strong controller both lowers the hard gravity residual
and harms hard geology, the pre-registered alpha 0.10 diagnostic is now
permitted. The alpha-0.25 result is not eligible for n=4.

The alpha-0.10 diagnostic has also completed and fails the complete gate
(`0/1`). Its hard gravity RMSE is lower (`0.87032 mGal` versus baseline
`0.95848 mGal`), but label-9 IoU/precision/recall still degrade to
`0.02111/0.06490/0.03033`; `2101` hard voxels leave label 9 while only `9`
enter it, and major-component minimum recall remains zero. Both pre-registered
controllers are now exhausted. Phase 4a is closed without n=4 promotion; the
authoritative interpretation is `docs/PHASE4A_REPORT.md`.

Phase 4c convolutional-seismic implementation is now active under the separately
frozen `docs/PHASE4C_SPEC.md`. Its CPU operator, immutable observation builder,
fixed-Euler adapter, strict runner, audit and pre-registered controller are
implemented. The canonical full-lateral 320-sample observation is
`experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
Fifteen focused tests and the complete 118-test suite pass. Real-checkpoint CPU
alpha-zero and positive-alpha one-step smokes pass with EMA, strict initial
noise pairing and zero condition violations. No Phase-4c GPU scientific result
exists yet.
