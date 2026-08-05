# Development handoff

Updated: 2026-08-02 after completing the Phase-6P frozen inference-limit audit
and extreme seismic-guidance ladder on an RTX 4090 D. Phase 6A remains the
learned-adapter mechanism anchor; formal Phase-6B training has not started.

## Read this first in a new conversation

For a paste-ready continuation prompt and a compact statement of the active
Phase-6B task, first open `docs/NEXT_CONVERSATION_PROMPT.md`. This handoff
remains the authoritative detailed history.

The active implementation is `flow2` on branch `main`, starting from commit
`5b717ec`. The working tree intentionally contains uncommitted user assets,
Phase-0/Phase-1 code and runs, documentation, and the new Phase-2 source. Never
discard, reset, upload-overwrite or silently reorganize those files. Inspect
`git status` and overlapping diffs before every edit.

Required reading:

1. `docs/AGENTS.md`
2. `docs/PROJECT_BASELINE.md`
3. `docs/EXPERIMENT_PROTOCOL.md`
4. this file
5. `docs/STAGE1_SPEC.md`
6. `docs/PHASE1_REPORT.md`
7. `docs/PHASE2_SPEC.md`
8. `docs/PHASE2A_REPORT.md`
9. `docs/PHASE2B_SPEC.md`
10. `docs/PHASE2B_FOLLOWUP_SPEC.md`
11. `docs/PHASE2B_REPORT.md`
12. `docs/RESEARCH_GOAL.md`
13. `experiments/stage1_probability/README.md`
14. `experiments/stage2_property/README.md`
15. `docs/PHASE3_SPEC.md`
16. `docs/PHASE4_SPEC.md`
17. `experiments/stage3_spatial_property/README.md`
18. `docs/PHASE3_REPORT.md`
19. `docs/PHASE4A_REPORT.md`
20. `docs/PHASE4C_SPEC.md`
21. `docs/PHASE4C_REPORT.md`
22. `docs/PHASE4D_SPEC.md`
23. `docs/PHASE4D_REPORT.md`
24. `docs/PHASE5A_SPEC.md`
25. `experiments/stage5_acoustic_inversion/README.md`
26. `docs/PHASE5A_REPORT.md`
27. `docs/PHASE5B_SPEC.md`
28. `docs/PHASE5B_REPORT.md`
29. `docs/PHASE5C_SPEC.md`
30. `experiments/stage5_generator_posterior/README.md`
31. `docs/PHASE5C_REPORT.md`
32. `docs/PHASE6_ADAPTER_SPEC.md`
33. `experiments/stage6_geo_adapter/README.md`
34. `docs/PHASE6A_REPORT.md`
35. `docs/NEXT_CONVERSATION_PROMPT.md`
36. `docs/PHASE6P_INFERENCE_LIMIT_SPEC.md`
37. `docs/PHASE6P_REPORT.md`

## Global goal and immutable boundary

The research goal is to improve decoded three-dimensional geology when surface
and sparse boreholes under-constrain deep intrusive structures, by introducing
spatially broad 3-D/property/geophysical evidence during inference. Success is
defined in hard labels, complete-model accuracy, geometry and ensemble
uncertainty. Continuous proxy loss alone is not success.

The final system is reciprocal: geophysical data should add global constraints
beyond sparse wells, while the frozen learned geological prior and exact
surface/borehole conditions should reduce the non-unique set of geophysical
solutions. The intended output is a condition-exact, observation-consistent,
geologically plausible ensemble, not a geophysics-only deterministic inverse.

Do not modify training, the 3-D U-Net, checkpoint or EMA convention. The normal
frozen `embedding.weight` is used with EMA values for all 411 trainable model
entries. Attribution experiments use identical inputs/noise/time grids and a
strict fixed-Euler alpha-zero pair. Conditions are projected before sampling
and after every integration step. Do not combine the current work with the
historical 2-D gravity path yet.

## Stage status

- Phase 0: complete.
- Phase 1: closed as mechanism validated with topology and endpoint caveats.
- Phase 2a: complete. The ideal full-resolution density-plus-susceptibility
  property upper bound passes all frozen gates in 12/12 strict pairs across
  seeds 42/142/242, with the documented class, geometry and CUDA-repeatability
  caveats.
- Phase 2b property-codebook ambiguity/contrast sensitivity: protocol,
  codebooks, stage-aware runner, single-sample launcher, report gate and CPU
  tests are complete. The distinct-codebook GPU anchor, `paired_c100` and the
  narrow `paired_c025` pass; `paired_c010` fails the target/component/topology
  gates, and the final exact-collision control also fails. The five-level
  screen is complete. The n=4 bracket is also complete: `paired_c025` is a 3/4
  transition and `paired_c010` is a 0/4 confirmed seed-42 failure. Neither is
  eligible for multi-seed promotion. The separately pre-registered
  `paired_c100` seed-42 n=4 fallback is also a 3/4 transition, so Phase 2b
  closes with no multi-seed-confirmed ambiguous-codebook operating point.
- Phase 3 spatial degradation: closed. Identity passes, but every nonzero
  Gaussian level fails; sigma 1 also fails its n=4 confirmation 0/4.
- Phase 4 acquisition-domain physics: the full-support gravity (4a),
  convolutional-seismic guidance (4c) and fixed-12 identifiability audit (4d)
  are implemented and closed negative. Each can improve a physical residual,
  but none improves the required hard label-9/major-body geology.
- Phase 5a no-training acoustic inversion bridge: complete. The continuous
  property gate passes, but direct nearest-codebook hard projections worsen;
  it authorized the now-completed single-pair Phase-5b flow bridge gate.
- Phase 5b inversion-posterior flow bridge: closed negative. The formal CUDA
  seed-42 n=1 pair is valid and lowers hard bridge loss, but fails global,
  class-majority, absolute label-9 and major-body gates. No n=4 or tuning is
  authorized.
- Phase 5c direct generator-posterior search: implemented and closed negative.
  Full-dimensional pCN over the Gaussian initial noise is technically active,
  condition-exact and inexpensive enough for a short chain, but the frozen
  8-proposal CUDA pilot lowers hard seismic loss while worsening label-9 IoU,
  recall and four-major-body recovery. No longer chain or pCN tuning is
  authorized on the repeatedly audited legacy case.
- Phase 6A frozen-flow residual adapter: engineering/oracle mechanism pass.
  The 54,327-parameter external adapter trains while the original model remains
  bitwise unchanged, preserves exact conditions and greatly improves the
  same-case truth-derived acoustic-oracle hard geology. This is not held-out or
  truth-blind geophysics; hard seismic loss worsens. Phase 6 is started but not
  complete. The next task is deterministic grouped data and a held-out oracle
  pilot.
- Phase 6P frozen inference-limit audit: complete. A predeclared trajectory
  ladder raises alpha/cap from 0.25 to 4.0; maximum hard-seismic RMSE
  attainment is only 9.88% at ratio 1.0, and stronger interference damages
  geology while physical fit regresses. A separate 200-step endpoint optimizer
  reaches only 1.69%; its soft response improves while the hard response
  diverges. All engineering, hash, condition and historical-regression gates
  pass. Same-case inference-strength tuning is closed.

## Phase-0 result

- Checkpoint/input validation passed.
- EMA loading passed: 411 trainable entries from EMA; the frozen embedding kept
  its normal checkpoint value.
- Borehole/truth mismatches are zero.
- Adaptive Dopri5 and 32-step fixed Euler differ by about 4.8645% of decoded
  voxels and cannot be mixed in attribution.
- Historical 2-D gravity guidance at relative alpha 0.05 changed only 31.25
  hard voxels per sample on average. This proved an active path, not improved
  label-9 geometry.

## Phase-1 final decision

The authoritative evidence is:

- human-readable decision: `docs/PHASE1_REPORT.md`;
- generated report:
  `experiments/stage1_probability/reports/phase1b_v4_12pair/REPORT.md`;
- machine summary: the adjacent `summary.json`;
- per-pair and topology tables: `paired_samples.csv` and
  `topology_samples.csv`.

Rebuild those derived artifacts, without changing immutable runs, from the
repository root:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage1/summarize_phase1b.py \
  --overwrite
```

The aggregation revalidates each baseline/guided config, completion state,
EMA policy, initial-noise hash identity, saved pairing verdict, zero condition
violations and zero decoder mismatches. It then recomputes size-stratified
six-connected topology directly from the saved tensors.

### Protocol-v4 12-pair evidence

Seeds 42/142/242, four samples each, alpha/cap 0.25, 32 fixed-Euler steps,
`calibrated_soft_bce_hard_dice_v2`, `reference_norm_relative_v2` and
`windowed_sine`:

- target IoU `0.0314 -> 0.8099`, precision `0.0788 -> 0.8274`, recall
  `0.0520 -> 0.9747`;
- selected-ROI IoU `0.0351 -> 0.9392`, precision `0.1050 -> 0.9627`, absolute
  volume-error fraction `0.5101 -> 0.0139`;
- target-centroid distance `16.9518 -> 3.3798`;
- global voxel accuracy `0.5972 -> 0.6432`, mean IoU `0.1804 -> 0.2178`;
- all primary target metrics, global metrics and centroid distance improve in
  all 12 pairs;
- 5.1811% mean full-volume hard changes, 98.7588% inside ROI;
- outside-ROI accuracy change about `+0.000022` and all condition violations
  zero;
- all three guided ensembles retain four unique samples and outside-ROI
  disagreement.

### Honest caveats

The pre-registered strong gate is `not_full_pass`:

- full target component ratio is 1.2812 versus limit 1.25;
- ROI diagnostic component ratio is 1.2665;
- raw largest-component fraction falls `0.7239 -> 0.3856`;
- the Phase-2 handoff occurred before a protocol-v4 fixed-camera render was
  generated and visually reviewed;
- last-step hard churn remains 802-1,398 voxels rather than baseline-level
  stability.

Truth-relative size stratification shows why raw topology is misleading but
does not change the failed gate: truth component sizes are
4,079/2,192/2,043/627/22/4/1; all 12 guided samples have exactly four ROI
components of at least 20 voxels; mean guided top-four sizes are approximately
4,074/2,203/2,129/600; fragments of at most five voxels contain only 0.66% of
guided ROI target mass.

Phase 1 therefore proves inference-time 3-D control and soft-to-hard crossing.
It does not prove real geophysics, complete-model multi-lithology recovery, or
that boreholes/surface conditions can be removed.

## Phase-1 implementation paths

- `guidance/probability_volume.py`
- `guidance/probability_sampling.py`
- `guidance/probability_evaluation.py`
- `scripts/stage1/run_probability_guidance.py`
- `scripts/stage1/visualize_probability_guidance.py`
- `scripts/stage1/summarize_phase1b.py`
- `tests/test_stage1_probability_guidance.py`
- `tests/test_stage1_probability_visualization.py`
- `tests/test_stage1_phase_report.py`
- `experiments/stage1_probability/`

## Phase-2 design and current code

Phase 2a replaces target-label probability with complete expected properties:

```text
p_t(k,r) = softmax_k(cos(x_t(r), e_k) / tau_t)
q_t(j,r) = sum_k p_t(k,r) q(j,k)
L = sum_s lambda_s || W_s * [S_s(q_t) - S_s(q_true)] ||^2
```

The same 3-D Gaussian `S_s` is applied to predicted and truth-derived target
properties. Each property channel is divided by its weighted target standard
deviation before channel weighting. This implements the operator-matching rule
and prevents units with large numerical values from dominating.

Implemented in `guidance/property_volume.py`:

- strict `full_lithology_property_channels_v1` parser;
- mandatory explicit raw-label coverage `-1..13` with no silent default;
- hard-label to multi-channel property mapping;
- all-class soft probability to expected property mapping;
- depthwise matched multi-scale 3-D Gaussian operators;
- normalized multi-channel MSE and confidence/missing-region weights;
- end-to-end differentiable loss through the unchanged soft decoder;
- property/entropy diagnostics.

The first controlled configuration is
`experiments/stage2_property/configs/ideal_distinct_density_proxy_v1.json`.
It maps all classes to distinct scalar proxy values in controlled relative
units. It is deliberately an easy Phase-2a bridge and not calibrated density.
Because a single soft expected scalar remains non-unique across 15 category
probabilities, `ideal_density_susceptibility_label9_contrast_v1.json` adds a
second complete controlled channel with an explicit label-9 contrast. It tests
information identifiability, not measured magnetic inversion. Later Phase-2
ablation must introduce overlapping/ambiguous properties before Phase 3 adds
spatial degradation and noise.

CPU tests in `tests/test_phase2_property_volume.py` cover config completeness,
raw-label offset, hard/soft mapping equality, channel-isolated Gaussian blur,
matched-loss zero/positive behavior, unit invariance, finite soft-decoder
gradients and zero-confidence behavior.

## Implemented Phase-2 execution path

- `guidance/property_sampling.py`: isolated fixed-Euler sampler. It reuses the
  proven Phase-1 temperature, reference-norm controller and `windowed_sine`
  definitions without changing Phase-1 source or activating its probability
  loss. Alpha zero is an explicit no-gradient branch.
- `guidance/property_evaluation.py`: global and label-9 geometry, hard-property
  loss, all raw non-air class metrics, paired per-class deltas, and a stable
  truth-present-class mIoU alongside the historical dynamic-union mIoU.
- `scripts/stage2/run_property_guidance.py`: EMA loading, truth/borehole
  validation, explicit property parsing, target/confidence construction,
  immutable outputs, strict config comparison, sequential CPU noise hashes,
  sampling, hard evaluation and ensemble artifacts.
- `experiments/stage2_property/run_phase2a_ideal_property_smoke.sh`: isolated
  one-sample alpha-zero/alpha-0.10 GPU launcher.

The property confidence is the unconditioned non-air subsurface, so exact air,
surface and borehole conditions do not consume the full-resolution property
loss. They are still re-projected after every Euler step.

## First Phase-2 GPU result

The seed-42, one-sample, 32-step scalar-property pair under
`ideal_distinct_density_proxy_v1/phase2a_v1/seed42_n1_s32` is valid:

- strict pairing passed, EMA and identical initial noise were confirmed;
- all trace values were finite and all condition violations were zero;
- global accuracy changed `0.58737 -> 0.59684`, mean IoU
  `0.19291 -> 0.19527`, and hard-property loss
  `1.19629 -> 0.98169`;
- label-9 recall changed `0.04728 -> 0.04472`, predicted volume
  `6283 -> 3691` versus truth `8968`, and 2,623 label-9 voxels were converted
  away while only 31 converted into label 9.

This is a functioning sampler and controller but a failed scientific result
for the scalar observation. Do not raise alpha and do not run its n=4 batch.

## Second Phase-2 GPU result

The two-channel observability run under
`ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n1_s32`
also passed strict runtime validation. Global accuracy increased by `0.01129`,
truth-present fixed-set mIoU increased by `0.00606`, six of eight
truth-present classes improved IoU, and hard-property loss fell by 30.06%.
Label-9 IoU changed `0.02860 -> 0.05795`, precision
`0.06748 -> 0.22429`, recall `0.04728 -> 0.07248`, and true positives
`424 -> 650`.

This remains a partial result: label-9 predicted volume fell to 2,898 versus
truth 8,968, components increased `37 -> 109`, and the largest-component
fraction fell. The historical dynamic-union mIoU also fell by 0.01204 because
four erroneous label-8 voxels introduced an additional zero-IoU class into its
per-sample denominator. Both that metric and the new fixed truth-present mIoU
are retained. See `docs/PHASE2_PROGRESS.md`.

## Third Phase-2 GPU result

The same two-channel target at `alpha=cap=0.25` materially improves the hard
geology: label-9 IoU/precision/recall are 0.4816/0.9005/0.5087, centroid
distance falls to 3.42 voxels, global accuracy rises by 0.0410, and fixed-set
mIoU rises by 0.0667. Six of eight truth-present classes improve IoU.

Raw label-9 components increase to 202, but top-eight mass is 87.2% and tiny
mass is 5.37%. The four major truth components have
48.3%/53.6%/57.9%/37.2% recall. The result passes the single-sample directional
gate while retaining an explicit fragmentation caveat.

The Phase-2 evaluator now writes size-stratified topology fields and separate
truth-component recovery CSVs. Numeric n=4 gates are frozen in
`docs/PHASE2_PROGRESS.md`.

## Seed-42 n=4 confirmation

All eight pre-registered gates pass. Across four pairs, mean global accuracy
changes `0.59245 -> 0.63412`, fixed truth-present mIoU changes
`0.26969 -> 0.33847`, dynamic-union mIoU changes `0.17692 -> 0.19341`, and
hard-property loss changes `1.48448 -> 0.51833`. Mean guided label-9
IoU/precision/recall are 0.4905/0.9116/0.5152. Every pair improves six of eight
truth-present classes, all four major truth bodies pass the recovery gate,
topology/endpoint gates pass, conditions remain exact, and four guided samples
remain unique.

The two independent guided sample-0 invocations differ at 10 hard voxels
(0.0038%) despite identical inputs/noise; alpha zero is byte-identical. Strict
paired conclusions are stable, but cross-process bitwise CUDA determinism is
not claimed.

## Seed-142 n=4 confirmation

All eight gates pass again. Mean global accuracy changes
`0.60014 -> 0.64139`, fixed-set mIoU `0.28212 -> 0.34832`, hard-property loss
`1.51606 -> 0.52133`, and guided label-9 IoU/precision/recall are
0.4735/0.8940/0.5028. Topology, endpoint, exact conditions and diversity pass.
Sample 3 improves exactly five of eight truth-present class IoUs; label 2 and
the tiny label-6 prediction decline. Label 2 remains the consistent class-level
tradeoff.

## Seed-242 and Phase-2a decision

Seed 242 also passes all eight gates. The complete 12-pair decision is:

**PASS: Phase-2a ideal 3-D property upper bound validated with caveats.**

The authoritative human report is `docs/PHASE2A_REPORT.md`; generated evidence
is under `experiments/stage2_property/reports/phase2a_v1_12pair/`. Rebuild it:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2a.py \
  --overwrite
```

All 12 pairs improve global accuracy, fixed-set mIoU, hard-property loss and
label-9 IoU/precision/recall. Label 2 declines in all 12, label 13 remains
unrecovered, target bodies remain incomplete/fragmented, and cross-process
CUDA bitwise determinism is not claimed.

## Phase-2b code and frozen screen

The authoritative protocol is `docs/PHASE2B_SPEC.md`. The code paths are:

- manifest/configs:
  `experiments/stage2_property/configs/phase2b_codebook_ambiguity_v1/`;
- GPU launcher:
  `experiments/stage2_property/run_phase2b_codebook_screen.sh`;
- completed-level auditor:
  `scripts/stage2/summarize_phase2b_screen.py`;
- seed-42 n=4 bracket launcher:
  `experiments/stage2_property/run_phase2b_codebook_n4_bracket.sh`;
- seed-42 n=4 bracket auditor:
  `scripts/stage2/summarize_phase2b_n4_bracket.py`;
- post-bracket fallback protocol:
  `docs/PHASE2B_FOLLOWUP_SPEC.md`;
- `paired_c100` seed-42 n=4 fallback launcher/auditor:
  `experiments/stage2_property/run_phase2b_codebook_n4_fallback.sh` and
  `scripts/stage2/summarize_phase2b_n4_fallback.py`;
- strict Phase-2 fixed-camera renderer and VTK exporter:
  `scripts/stage2/visualize_property_guidance.py`;
- codebook and promotion regression:
  `tests/test_phase2b_codebook_ambiguity.py`.

The five levels are `distinct_c100_anchor`, `paired_c100`, `paired_c025`,
`paired_c010` and `paired_c004_overlap`. The final level makes truth-present
labels 6 and 9 exactly property-equivalent. All levels keep alpha/cap 0.25,
seed 42, n=1, 32 fixed-Euler steps and a separate alpha-zero strict baseline.

## Phase-3 implementation and current evidence

The frozen protocol is `docs/PHASE3_SPEC.md`. It starts from the distinct
two-channel Phase-2a codebook because this is the only 12/12 validated
operating point. The Phase-2b c100/c025 transition levels are not promoted.

Implemented paths:

- `guidance/spatial_property.py`: identity, Gaussian and average-pool response,
  exact known-property overwrite, depth/missing confidence, deterministic
  observation-only noise, soft loss and hard-observation loss;
- `guidance/property_sampling.py`: optional loss injection with the original
  Phase-2 property loss as its default, preserving the existing path;
- `scripts/stage3/run_spatial_property_guidance.py`: EMA/fixed-Euler runner,
  immutable observation tensors and hashes, strict pairs and full hard metrics;
- `scripts/stage3/audit_spatial_screen.py`: strict source/asset audit, bytewise
  Phase-2a alpha-zero regression and the frozen hard gate;
- `experiments/stage3_spatial_property/configs/`: frozen identity and Gaussian
  sigma 1/2/4 configs plus ordered manifest;
- `experiments/stage3_spatial_property/run_phase3_gaussian_screen.sh`: one
  level per invocation; no automatic all-level run;
- `scripts/stage3/summarize_gaussian_screen.py`: validates all four n=1
  reports/hashes and freezes the n=4 bracket decision;
- `experiments/stage3_spatial_property/run_phase3_gaussian_n4_bracket.sh`:
  restricted n=4 launcher for identity and sigma 1 only;
- `tests/test_phase3_spatial_property.py`: observation and sampler CPU gate.

The complete seed-42 n=1 screen is now available under
`experiments/stage3_spatial_property/reports/gaussian_screen_seed42_n1/`.
Identity passed the complete gate. Gaussian sigma 1, 2 and 4 all failed despite
lower observation loss. Label-9 IoU declined monotonically
`0.4881 -> 0.3357 -> 0.2064 -> 0.1026`, and major-body mean recall declined
`0.4984 -> 0.3981 -> 0.2584 -> 0.1305`.

The aggregate decision is **no nonzero Gaussian blur passed the n=1 gate**.
The identity side of the seed-42 n=4 bracket then passed 4/4 complete gates
and diversity. Mean accuracy and truth-present mIoU improve by 0.0420 and
0.0696; guided label-9 IoU/precision/recall average
0.4966/0.9144/0.5211. Conditions, hashes and Phase-2a alpha-zero regression
all pass. This confirms the undegraded anchor only.

The sigma-1 side then failed 0/4. All four pairs failed the label-9 target
thresholds because guided precision remained below 0.75; two also failed the
major-body recovery gate. Its mean guided label-9 IoU/precision/recall are
0.3482/0.6938/0.4120. Pairing, conditions, alpha-zero regression and diversity
remain valid. Phase 3 is therefore closed without a promoted nonzero-blur
working point. Do not run sigma 2/4 n=4 or seeds 142/242. Read
`docs/PHASE3_REPORT.md` for the authoritative conclusion.

Phase 4 is frozen separately in `docs/PHASE4_SPEC.md`. Phase-4a full-support
gravity development proceeds independently. The originally planned Phase-4b
joint guidance cannot assume a Phase-3 Gaussian working point because none was
promoted; it requires a new frozen design or an identity-only upper-bound arm.

## Phase-4a forward/operator implementation

Phase-4a CPU development is now active and isolated from the historical
`SimpleGravityForward` proxy:

- `guidance/gravity.py`: exact rectangular-prism downward `g_z`, SI geometry,
  mGal output, all-cell/all-station support through zero-padded FFT linear
  convolution, density mapping, field loss and immutable observations;
- `guidance/gravity_sampling.py`: gravity-loss injection into the proven
  projected fixed-Euler solver; alpha zero remains its explicit no-gradient
  branch;
- `scripts/stage4/build_gravity_observation.py`: immutable builder with source,
  config and tensor hashes;
- `experiments/stage4_gravity/configs/`: distinct synthetic upper bound,
  exact label-6/9 collision control and full-grid noiseless inverse-crime
  survey;
- `tests/test_phase4_gravity.py`: 18 operator, mapping, gradient, sampler,
  immutable-asset, pairing, controller and reranking gates.

The second implementation unit is also complete:

- `scripts/stage4/run_gravity_guidance.py`: immutable observation/source hash
  validation, EMA, CPU noise, strict fixed-Euler pairs and complete evaluation;
- `scripts/stage4/audit_gravity_screen.py`: alpha-zero regression and frozen
  geology-plus-gravity gate;
- `scripts/stage4/rerank_gravity_ensemble.py`: n=4 post-hoc baseline comparator;
- `experiments/stage4_gravity/run_phase4a_gravity_screen.sh`: first n=1 GPU
  launcher;
- `configs/gravity_controller_manifest_v1.json`: pre-registered alpha/cap 0.25
  and conditional alpha 0.10 diagnostic, included in strict pair hashes.

The Phase-4-focused CPU gate is 18/18 and the complete suite is 103 passed. The canonical
64-cubed observation is under
`experiments/stage4_gravity/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
It is finite with range -1.0706 to 1.5260 mGal and standard deviation 0.7621
mGal. A real-checkpoint CPU one-step alpha-zero runner smoke test completes with
EMA and zero condition violations.

The seed-42 n=1, 32-step, alpha/cap 0.25 GPU strict pair and audit are complete.
All pair, immutable-hash, alpha-zero regression and hard-condition checks pass,
but the complete geology-plus-gravity gate is `0/1`. Hard gravity RMSE decreases
by `0.06561 mGal` and global accuracy rises by `0.00907`; nevertheless label-9
IoU/precision/recall fall from `0.0286/0.0675/0.0473` to
`0.0159/0.0638/0.0207`. Prediction volume falls from `6283` to `2914`, major-
component minimum recall is `0`, and the `6562` hard changes contain `3392`
label-9-to-other versus only `23` other-to-label-9 transitions. Treat this as
field fitting under gravity non-uniqueness, not a partial Phase-4 success. It
must not advance to n=4. The manifest's condition for the single alpha=0.10
harm diagnostic was met, and that strict pair is also complete. It lowers hard
gravity RMSE to `0.87032 mGal` but degrades label-9 IoU/precision/recall to
`0.02111/0.06490/0.03033`; `2101` hard voxels leave label 9 and only `9` enter
it, with major-component minimum recall still zero. Phase 4a is closed without
n=4 or further alpha search. Read `docs/PHASE4A_REPORT.md`. The next independent
upper bound is convolutional seismic response; Phase-4b joint 3-D property plus
gravity remains unavailable because Phase 3 promoted no nonzero Gaussian level.

## Phase-4c convolutional-seismic implementation

The first two implementation units are complete under the frozen
`docs/PHASE4C_SPEC.md`:

- `guidance/seismic.py`: complete acoustic codebook, expected impedance and
  slowness, known-subsurface non-air renormalization, exact condition overwrite,
  local-surface TWT, linear deposition, zero-phase Ricker convolution, finite
  prediction-window cropping and normalized trace loss;
- `guidance/seismic_sampling.py`: seismic loss injection into the proven
  projected fixed-Euler solver;
- `scripts/stage4/build_seismic_observation.py`: immutable observation builder;
- `scripts/stage4/run_seismic_guidance.py`: EMA/CPU-noise strict-pair runner and
  complete hard geology/acoustic/seismic evaluator;
- `scripts/stage4/audit_seismic_screen.py`: pairing, tensor/source hashes,
  alpha-zero Phase-2a hard regression and complete geology-plus-seismic gate;
- `experiments/stage4_seismic/configs/`: complete distinct label-9 synthetic
  acoustic upper bound, full 320-sample noiseless observation and frozen
  alpha/cap controller order;
- `experiments/stage4_seismic/run_phase4c_seismic_screen.sh`: seed-42 n=1 GPU
  launcher;
- `tests/test_phase4_seismic.py`: 15 codebook, mapping, orientation, timing,
  deposition, boundary, gradient, condition, immutable-asset, controller,
  pairing and alpha-zero gates.

The canonical immutable observation is
`experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
It has shape `1 x 1 x 64 x 64 x 320`, amplitude range `-0.4716..0.4931`,
193459 valid interfaces and maximum truth TWT `1428.43 ms` within the fixed
`2552 ms` window. The original asset predates subsurface soft-air handling;
`fix1` predates explicit finite-record prediction cropping. Both remain
immutable historical assets and are rejected by current source-hash validation.

Real-checkpoint CPU one-step alpha-zero and alpha-0.25 strict-pair smokes pass.
They load all 411 EMA entries, share the exact initial-noise hash, produce a
finite nonzero seismic gradient, preserve strict pairing and end with zero hard
condition violations. These smokes are not scientific evidence.

The pre-registered seed-42 n=1, 32-step alpha/cap 0.25 GPU strict pair and
audit are complete. Pairing, immutable hashes, EMA, alpha-zero Phase-2a
regression and exact conditions pass, but the complete gate is `0/1`. Hard
seismic loss falls `17.86063 -> 15.24736` and RMSE falls
`0.042262 -> 0.039048`; global accuracy rises by only `0.002354`. Label-9
IoU/precision/recall change from `0.02860/0.06748/0.04728` to
`0.02593/0.07027/0.03947`, predicted volume falls `6283 -> 5038`, and the four
major truth-body recalls are `0.0306/0/0.1121/0`, all no better than baseline.
Of `3737` final paired hard changes, `1438` leave label 9 and only `193` enter.
The total paired change fraction is `1.4256%`; the distinct last-step churn is
`0.2678%`. Both pass their limits, so this is not the pre-registered excessive-
change case and alpha 0.10 is not authorized. Phase 4c is closed without n=4,
extra alpha search or gravity fusion. Read `docs/PHASE4C_REPORT.md`.

## Phase-4d seismic identifiability and posterior selection

Phase 4d is implemented and complete. It reads exactly the 12 existing
Phase-2a alpha-zero baselines from seeds 42/142/242, validates their EMA,
fixed-Euler, checkpoint/input/sample hashes and exact-condition invariants, and
requires seed-42 sample 0 to match the Phase-4c alpha-zero anchor. It generates
no samples. Candidate ranking uses only the immutable Phase-4c hard seismic
loss; complete truth geology is evaluated after ranking. A separate whole-
class truth substitution matrix measures operator sensitivity without entering
the selector.

Both frozen gates fail. No candidate meets the absolute label-9 and
major-component support thresholds; the oracle-best label-9 IoU/recall and
major-body mean recall are only `0.0672/0.0959/0.0963`. The seismic-selected
top candidate is `seed42_sample1` with label-9 IoU/precision/recall
`0.0245/0.0876/0.0329`. The selected top-three means are below ensemble means
for label-9 IoU (`0.02359 < 0.03144`), recall (`0.03702 < 0.05196`) and
major-body mean recall (`0.03905 < 0.05344`). Loss correlations with those
metrics have the wrong positive sign: `+0.552/+0.587/+0.580`.

Replacing all `8955` unconditioned truth label-9 voxels is visible to the
operator (least-visible replacement RMSE `0.017692`), but this target signal is
smaller than the full candidate RMSE of roughly `0.041..0.046` and is dominated
by other interface/timing errors. Phase 4d is closed without a larger pool,
posterior-selection promotion or ranking tuning. See `docs/PHASE4D_REPORT.md`.
The next main direction is a separately authorized geophysics-aware training or
fine-tuning protocol; do not modify training, U-Net or checkpoint without that
explicit authorization.

## Phase-5a no-training acoustic inversion bridge

The user authorized a bounded no-large-training route before any conditioning
adapter. `docs/PHASE5A_SPEC.md` freezes a truth-blind model-based log-impedance
inversion around the exact fixed 12-member Phase-4d prior pool. The builder
never loads truth geology or truth acoustics, and does not load the model or
checkpoint weights. It uses the Phase-4c full-cube noiseless observation, each
prior's fixed slowness/time-depth map, one pre-registered Tikhonov operating
point and exact surface/borehole overwrites. A separate audit opens truth only
after output/source hashes and the completed anti-leakage manifest pass.

All six Phase-5a bridge-eligibility checks pass. Exact nonlinear seismic RMSE,
unconstrained log-impedance RMSE and target-region log-impedance MAE improve in
12/12 members. Ensemble-mean log-impedance RMSE improves
`0.2514293 -> 0.2433104`; target-region MAE improves only
`0.6485778 -> 0.6430638`. Conditions have zero violations and posterior spread
is finite/nonzero.

This is not a geological success. Direct nearest-codebook projection reduces
12-member mean global accuracy `0.5972 -> 0.5163`, label-9 IoU
`0.03144 -> 0.01912` and recall `0.05196 -> 0.02496`. Read
`docs/PHASE5A_REPORT.md`. The only authorized next no-training step is a
single-seed/sample Phase-5b flow bridge with strict alpha-zero pairing and the
existing full hard-label/major-body gate. Stop if continuous loss improves
without hard geology; do not tune the inversion on truth, expand seeds, fuse
gravity, or claim calibrated uncertainty.

## Phase-5b completed negative evidence

`docs/PHASE5B_SPEC.md` freezes the only authorized next screen. The derived
bridge directory is
`experiments/stage5_acoustic_inversion/bridge_observations/cond_generation_0/fixed12_log_impedance_v1/`.
It contains the posterior mean log-impedance target, the complete log-impedance
category table and truth-blind spread confidence. Confidence is active over
193037 unconstrained subsurface voxels, has mean `0.51683`, and is exactly zero
at every hard condition. The builder only reads the Phase-5a pass bit as a stop
gate; no truth metric enters asset construction.

`scripts/stage2/run_property_guidance.py` now has an additive, validated
`phase5b_inversion_property_bridge_v1` external-target mode; historical
Phase-2a/2b behavior remains the default and their tests pass. The formal
launcher runs one alpha-zero/alpha-0.25 seed-42, n=1, 32-step EMA/fixed-Euler
pair and then `scripts/stage5/audit_inversion_property_bridge.py`. The audit
requires bytewise Phase-2a alpha-zero hard regression, lower hard bridge loss,
and the full existing global/per-class/label-9/major-body/topology gate.

The formal CUDA seed-42 n=1, 32-step pair and frozen audit are now complete.
Pairing, all hashes, EMA, exact conditions and bytewise Phase-2a alpha-zero
hard regression pass. Hard inversion-observation loss decreases by only
`0.000683`, while global accuracy decreases by `0.000213`. Guided label-9
IoU/precision/recall are `0.02890/0.06870/0.04750`, effectively unchanged and
far below the absolute gate. Only 4/8 truth-present classes improve. The four
major-body recalls are `0.04266/0.00046/0.12286/0`, giving minimum/mean
`0/0.04149` instead of the required `0.25/0.40`.

The path is active: 24 guidance steps are nonzero, soft loss reaches `0.5484`,
and 1212 hard voxels change. But 194 label-9 voxels leave the class and only
112 enter, with predicted label-9 volume decreasing `6283 -> 6201` versus
truth 8968. This is information/likelihood-to-lithology misalignment, not an
inactive soft-hard mechanism. Read `docs/PHASE5B_REPORT.md`.

Phase 5b is closed. Do not run n=4, more seeds, alternate alpha/confidence,
sharpening/blur or gravity fusion. The next distinct option is a lightweight
learned geophysics-conditioning adapter with the original U-Net/checkpoint
frozen. It needs a new split/anti-leakage/training protocol and explicit user
authorization before any training code is changed.

## Phase-5c completed direct-posterior evidence

The two newly supplied generator-prior seismic inversion papers motivated a
scientifically distinct no-training screen. `docs/PHASE5C_SPEC.md` freezes pCN
over the complete conditional flow's Gaussian initial noise. Each proposal is
generated by the canonical 32-step EMA fixed-Euler path, projected to exact
surface/borehole conditions after every step, hard decoded and scored by the
immutable Phase-4c hard convolutional-seismic likelihood. There is no
inversion-property bridge and no soft likelihood gradient. The proposal
preserves the Gaussian prior, so the Metropolis decision uses only the hard
seismic energy. Sampling and truth audit are separate programs.

The RTX 4090 D mechanism run is computationally feasible: one proposal takes
about 4 seconds after load and peak allocated memory is about 1.66 GB. The
primary chain accepts 7/8 proposals and retains eight unique condition-exact
hard models. The historical seed-42 initial-noise hash and decoded sample
exactly reproduce the Phase-4c alpha-zero baseline.

The minimum-seismic-loss retained state lowers hard loss
`17.860632 -> 17.653818`, raises global accuracy
`0.587366 -> 0.594143` and truth-present mIoU
`0.265249 -> 0.269580`, but label-9 IoU falls
`0.028596 -> 0.025298`, recall falls `0.047279 -> 0.040700`, and
four-major-body mean recall falls `0.041423 -> 0.034587`. This is another
physics-better/target-geology-worse result, now without a soft-hard gradient
bridge. Read `docs/PHASE5C_REPORT.md`.

The incomplete `performance_smoke_v1` directory records the first shape-audit
stop and must not be reused. The fixed successful smoke is
`performance_smoke_v1_fix1`; authoritative primary evidence is under
`experiments/stage5_generator_posterior/{runs,reports}/cond_generation_0/primary_pilot_v1/`.
Do not lengthen the chain, sweep beta/likelihood weight, select on truth or add
seeds for this exposed protocol. Proceed to the separately frozen Phase-6
adapter path.

## Phase-6A residual-adapter oracle mechanism

`docs/PHASE6_ADAPTER_SPEC.md` freezes an external residual-velocity adapter
while leaving the original EMA U-Net, embedding and checkpoint immutable. The
base velocity is evaluated under `no_grad`; AdamW contains only 54,327 adapter
parameters. The adapter receives state, detached base velocity, sparse
conditioning, condition mask, a two-channel 3-D acoustic feature and time. It
uses all-class endpoint CE/Dice in addition to normalized flow loss, and its
output is zero at hard conditions. `adapter_scale=0` takes an explicit branch
that exactly reproduces the paired fixed-Euler baseline.

The completed `oracle_tiny_overfit_v1` run is deliberately a legacy same-case
learnability screen. Its acoustic input is a normalized, full-resolution
truth-derived property volume; four cached flow states are repeated for 80
adapter-only steps. On the RTX 4090 D, the cached training section takes about
2.99 seconds and peaks at roughly 1.97 GB allocated CUDA memory. The original
model tensor hash is identical before/after and all base gradients remain
absent. First/last ten-step mean loss changes `0.205246 -> 0.084735`; cached
endpoint accuracy changes `0.941159 -> 0.971063`.

The strict seed-42 sample changes global accuracy `0.587366 -> 0.745053`,
truth-present mIoU `0.265249 -> 0.473283`, label-9 IoU
`0.028596 -> 0.511914`, precision `0.067484 -> 0.833687`, recall
`0.047279 -> 0.570138`, and four-major-body mean recall
`0.041423 -> 0.507007`. Conditions remain exact. Hard seismic loss worsens
`17.860632 -> 19.828848`, so the result proves categorical adapter capacity,
not the final reciprocal geophysics/geology objective. Read
`docs/PHASE6A_REPORT.md`.

Do not tune or promote this same-case checkpoint. Before formal Phase-6
training, materialize deterministic cases; split by complete geological
history with all sibling well/noise/property variants in one split; exclude
all current demos from formal test; freeze normalization, test manifest and
evaluator hashes. Then run a held-out oracle/degraded-feature pilot before a
truth-blind seismic encoder. Correct/zero/shuffled observation controls and a
small pre-registered physical auxiliary loss are mandatory, and both hard
geology and hard seismic must improve.

## Phase-6P frozen inference-limit result

Before formal training, Phase 6P directly tested the concern that the Phase-4C
guidance cap might simply be too weak. The truth-blind physical runner and the
independent truth auditor are specified in
`docs/PHASE6P_INFERENCE_LIMIT_SPEC.md`; the decision is in
`docs/PHASE6P_REPORT.md`.

The first arm exactly reproduces the historical alpha-zero baseline and the
historical alpha/cap-0.25 Phase-4C sample, then reports all predeclared
alpha=cap levels 0.25/0.5/1/2/4. Hard-seismic RMSE attainment is
7.60%/9.36%/9.88%/6.20%/4.43%. Ratio 1 is physically best, while ratio 4
changes 29.90% of hard voxels, lowers global mIoU `0.19291 -> 0.10201` and
lowers label-9 IoU `0.02860 -> 0.01334`. Increasing the cap beyond 1 therefore
amplifies off-direction updates rather than approaching the observation.

The second arm bypasses the trajectory and applies 200 Adam steps to the
continuous baseline endpoint with exact condition projection. Its best hard
state occurs at step 15 and reaches only 1.69% hard-seismic attainment. Soft
seismic RMSE nevertheless falls from `0.03177` at step 1 to `0.01682` at step
200 while final hard RMSE rises to `0.09732`. This is direct evidence of a
large soft-acoustic/hard-decode relaxation gap.

Both runs keep the base tensor hash unchanged, leave all base gradients absent,
have zero condition violations and distinguish file hashes, raw tensor hashes
and canonical int64 categorical-content hashes. The runner never selects on
truth geology; each frozen output is evaluated by a separate auditor.

The correct interpretation is low physical reachability under the tested
frozen generator/decoder/controller, not a mathematical impossibility result.
Geophysical non-uniqueness remains expected, but it cannot be the primary
explanation for the current notebook visualization because the hard seismic
response has not first reached the observation manifold. Further same-case
guidance-strength tuning is closed. Formal training still requires explicit
user confirmation and must use held-out grouped data, hard-aware categorical
alignment, a pre-registered physical term and correct/zero/shuffled controls.

## Validation state at handoff

- Phase-1 report regeneration succeeded locally with `.venv/bin/python`.
- Phase-2/Phase-3 focused property and sampler tests: `24 passed`.
- Complete local lightweight suite after Phase-6P implementation:
  `152 passed`,
  with 13 existing Matplotlib/pyparsing deprecation warnings.
- The default system Python lacks PyTorch; use `.venv/bin/python` locally or the
  user's CUDA environment on the remote terminal.
- Static compilation and shell syntax pass; `git diff --check` is part of the
  final handoff check.
- Real 64-cubed observation construction succeeds for identity and Gaussian
  sigma 1/2/4; all outputs are finite and have stable distinct hashes.
- The first identity run tag `seed42_n1_s32_a025_c025` is incomplete and
  invalid: baseline sampling wrote `sample_0.pt`, then hard-observation
  evaluation rejected its unnormalized 3-D shape. The bug is fixed and covered
  by a regression test; use a new `..._fix1` run tag and never modify/reuse the
  partial directory.
- Phase 2a is complete: 12/12 strict pairs and 3/3 diversity gates pass with
  the documented caveats. The Phase-2b anchor, `paired_c100` and `paired_c025`
  pass; `paired_c010` and `paired_c004_overlap` fail. The n=1 screen is complete
  and the final n=4 bracket is c025 3/4 transition plus c010 0/4 failure. The
  c100 fallback is also a 3/4 transition. Phase 2b is closed. Phase-3 n=1 is
  complete: identity passes and every nonzero Gaussian blur fails. Identity
  also passes its seed-42 n=4 confirmation 4/4, while sigma 1 fails 0/4. Phase
  3 is closed without a spatially degraded primary working point. Phase-4a
  forward/operator, observation assets, GPU runner, strict audit and reranking
  comparator are implemented. Its seed-42 n=1 alpha/cap 0.25 and alpha=0.10
  strict pairs both fail the complete gate despite lower hard gravity residual;
  n=4 and further alpha tuning are blocked and Phase 4a is closed. The frozen
  convolutional-seismic upper bound is implemented and its seed-42 n=1
  alpha/cap 0.25 strict pair is complete. It lowers hard seismic residual but
  fails the geological gate `0/1`, with worse label-9 and major-body recovery.
  Phase 4c is closed; alpha 0.10, n=4 and gravity fusion are blocked by the
  frozen protocol. Phase-4b joint guidance also cannot assume a promoted
  Phase-3 Gaussian level. Phase 4d's fixed 12-sample gradient-free seismic
  selector also fails both support and ranking gates, showing that the current
  frozen route has both proposal-support and likelihood-alignment limitations.
  Phase 5a then converts the same seismic input to a fixed-12 uncertain 3-D
  impedance posterior without training. Its continuous bridge gate passes,
  but nearest-codebook hard geology worsens. The formal single-pair Phase-5b
  flow bridge is now complete and fails the frozen hard gate despite lower
  bridge loss. Phase 5c then directly samples the frozen generator posterior
  with hard seismic pCN. Its 8-proposal chain is active and physically better,
  but label-9 and major-body recovery worsen. The tested no-training routes are
  closed. Phase 6A now shows that a tiny learned adapter can strongly recover
  same-case oracle hard geology without changing the base model, but it does
  not yet generalize or improve physical fit. Phase 6P then shows that raising
  inference guidance to 4x or directly fitting the endpoint cannot close more
  than 9.88%/1.69% of the hard-seismic RMSE and progressively damages geology.
  Same-case inference-strength tuning is closed; Phase 6 remains the active
  path, but formal Phase-6B training has not started.
