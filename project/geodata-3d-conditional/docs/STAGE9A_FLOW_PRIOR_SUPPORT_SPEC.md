# Stage 9A specification: frozen Flow-prior support and geophysical enrichment

Date frozen: 2026-08-10, after standalone Stage 8A-v4 terminal closure and
before any Stage9A CUDA evidence.

## Decision questions

Stage9A answers only:

1. Under fixed surface and borehole conditions, does the frozen conditional
   Flow prior produce hard-label models with useful concealed/deep geological
   support at non-negligible frequency?
2. Without geological truth, does the validated hard seismic likelihood
   systematically enrich those better realizations within the same fixed
   Flow ensemble?

Stage9A does not train, fine-tune, use LoRA/adapters, alter the U-Net,
checkpoint, EMA convention, embedding, decoder, structured proposal search,
or run D-Flow, pCN, SMC, gravity, STE, or a posterior chain.

## Frozen model and sampling protocol

- Checkpoint:
  `demo_model/conditional-weights.ckpt`, expected SHA-256
  `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`.
- Weight policy: retain the normal frozen checkpoint value of
  `embedding.weight`; apply checkpoint EMA values to every trainable model
  parameter. Missing trainable EMA entries or shape mismatches are an
  engineering failure.
- Integrator: `fixed_euler_midpoint_v1`, exactly 32 equal Euler updates. The
  U-Net time for step `s=0..31` is `(s+0.5)/32` and `dt=1/32`.
- Guidance: none. No likelihood, structured correction, proposal, or truth
  information enters a Flow trajectory.
- Conditioning: clean categorical embeddings are projected before the first
  update and after every Euler update. Hard condition labels are projected
  again after hard decoding.
- Source points: independent standard-Gaussian CPU tensors, one generator per
  candidate. For case index `j=0,1,2` and candidate index `i=0..1023`, the
  formal source seed is `9301000 + 10000*j + i`. Engineering smoke uses the
  disjoint offset `+500000`.
- Candidate ID: `candidate_000000` through `candidate_001023`. IDs, source
  seeds, and hashes are independent of batching. The tie break is lexical
  candidate ID, which is identical to numeric candidate order here.
- Formal population: exactly 1024 candidates per primary case. The frozen
  implementation batch size is four; batching is an engineering detail and
  never changes the seed schedule.

The sampler must be regression-tested against the existing Phase5c projected
alpha-zero/no-guidance implementation. Hard decoding must use
`Geo3DStochInterp.decode(...)-1` followed by exact categorical condition
projection.

## Frozen primary cases

The three primary cases use the existing
`build_structuralgeo_native_case` logic without modification:

| Case ID | StructuralGeo seed | Wrong-case seed | Shuffle seed |
|---|---:|---:|---:|
| `native_seed20260901` | 20260901 | 20261901 | 20266943 |
| `native_seed20260902` | 20260902 | 20261902 | 20266944 |
| `native_seed20260903` | 20260903 | 20261903 | 20266945 |

The shuffle schedule exactly reuses Stage7's `6042 + native_seed` rule. These
seeds were fixed without rendering truth, generating Flow candidates, or
calculating seismic rankings. There is no truth-based case replacement.

The existing native builder's eligibility checks are reused exactly: canonical
64-cubed grid and air boundary, all five events nonempty, no event-mask
overlap, and neither hidden event intersecting a hard condition. Failure of a
fixed seed to satisfy those construction invariants is an engineering failure;
there is no adaptive fallback seed.

`cond_generation_0` and native seeds 20260807/20260808/20260809 remain
legacy/regression assets and cannot count toward a Stage9A gate.

## Case construction and hard physics

Each case is materialized once by a case-construction program. That program is
the only Stage9A construction path allowed to open synthetic truth before the
retrospective auditor. It writes two physically separated asset roots:

- `inference/`: condition values, condition mask, subsurface mask, acoustic
  property table, correct/zero/shuffled-XY/wrong-case observations, and a
  manifest that contains no truth tensor path;
- `retrospective/`: truth labels, native body masks, and a truth manifest.

The hard petrophysical mapping is the complete Phase4c
`acoustic_distinct_label9_upper_bound_v1` density/velocity codebook converted
to impedance and slowness. The hard forward is the canonical Stage7-compatible
Phase4c normal-incidence local-surface TWT deposition plus fixed zero-phase
25-Hz Ricker convolution on all 64x64 traces and all 320 samples, using
`full_cube_noiseless_inverse_crime_v1`.

Controls are constructed once from case truth before inference:

- `correct`: hard forward of the fixed case truth;
- `zero`: an all-zero field;
- `shuffled_xy`: one fixed lateral trace permutation preserving every trace;
- `wrong_case`: hard forward of the independently generated fixed wrong case.

The same cached candidate predictions are evaluated against all four
observations. Controls never regenerate Flow candidates.

## Truth firewall

The execution paths are separate programs and APIs.

### Inference-visible path

The candidate runner accepts only a checkpoint, frozen config, and an
`inference/` case directory. Its API has no truth argument. It:

1. loads the frozen EMA Flow;
2. creates the registered independent source tensors;
3. samples, hard decodes, and re-projects exact conditions;
4. performs the hard petrophysical/seismic forward;
5. caches hard models and predicted observations;
6. writes candidate/source/model/prediction hashes and a completed pool
   manifest.

The ranking runner accepts only a completed pool and `inference/` assets. Its
API has no truth argument. It validates every pool artifact/hash, computes
plain full-field hard seismic RMSE from the same cached float32 prediction for
each observation, and ranks by `(RMSE ascending, candidate_id ascending)`.
It freezes all four ranking CSV files and a completed ranking manifest.

### Retrospective path

The truth auditor accepts the retrospective asset directory only after it has
validated that the pool and all four rankings are completed, immutable, and
hash-consistent. Only then may it load truth. Truth computes metrics and gates
only; it cannot create, remove, reorder, select, or stop candidates.

Tests must inspect public function signatures, exercise the validation-before-
truth-load ordering, reject incomplete/tampered pools and rankings, and prove
that changing truth cannot alter a frozen ranking.

## Lossless prediction cache

Because the server volume is nearly full, hard models and float32 predicted
observations are stored in deterministic, lossless gzip-compressed PyTorch
chunks of four candidates. Compression changes neither dtype, shape, bytes,
RMSE, nor tensor SHA-256 after decompression. Every chunk records compressed
file SHA-256 and uncompressed tensor SHA-256; every candidate records its own
uncompressed prediction hash. Ranking and later stages consume the chunks via
the shared validated decompressor. Float16 quantization is forbidden.

The runner writes into a unique incomplete staging directory and atomically
renames it only after all artifacts and hashes validate. Existing final output
directories are never reused or overwritten. Incomplete directories are
preserved as engineering evidence and are rejected as formal inputs.

## Retrospective metrics

For every candidate, after ranking freeze, compute:

- global hard-label accuracy;
- mIoU over truth-present non-air classes;
- IoU, precision, and recall for every truth-present non-air class;
- label-9 IoU, precision, and recall;
- six-connected truth-component recovery and recalls for the four largest
  truth label-9 components;
- label-9 volume, centroid, size-stratified connected-component, and
  retrospective body-match diagnostics;
- ensemble unique hard-model count, voxelwise disagreement, and expected
  pairwise hard-label disagreement.

Label 9 is an audit focus only. It cannot affect generation, filtering,
ranking, stopping, or seed choice.

## Prior-support gate

The unchanged Phase4d support thresholds apply simultaneously to one
candidate:

- label-9 IoU >= 0.30;
- label-9 precision >= 0.75;
- label-9 recall >= 0.30;
- minimum recall across the four largest truth components >= 0.25;
- mean recall across those four components >= 0.40;
- hard condition violations == 0.

A case is `SUPPORT_PASS` if at least one of its 1024 candidates satisfies every
clause. Overall `SUPPORT_PASS` requires at least two of three primary cases.

For ordered prefixes N=1,4,16,64,256,1024, report per-metric oracle maxima,
the count of simultaneous support-pass candidates, and whether the prefix
contains any support-pass candidate. This best-of-N curve is a retrospective
prior-support ceiling, never an inference selector.

## Geophysical-enrichment gate

For each of correct/zero/shuffled/wrong rankings, compute Spearman correlation
between hard seismic RMSE and:

- global accuracy;
- truth-present mIoU;
- label-9 IoU;
- label-9 recall;
- four-major-component mean recall.

Ties use deterministic average ranks. Also compute the full-ensemble mean and
top 10%, 5%, and 1% means for label-9 IoU, label-9 recall, and major-component
mean recall. Top counts use `ceil(fraction*N)` and at least one member.
Enrichment is `top_mean - full_mean`.

A case is `DISCRIMINATION_PASS` iff:

1. under the correct observation, all three target-metric Spearman values are
   strictly negative;
2. correct-observation top-5% means are strictly above full-prior means for all
   three targets;
3. correct-observation top-5% enrichment is strictly greater than the zero,
   shuffled, and wrong-case top-5% enrichment for each target.

No p-value or additional truth-tuned magnitude threshold enters the gate.
Overall `DISCRIMINATION_PASS` requires at least two of three primary cases.

## Machine decision tree

The aggregate `summary.json` must emit booleans `SUPPORT_PASS` and
`DISCRIMINATION_PASS` plus exactly one `NEXT_ACTION`:

| Support | Discrimination | `NEXT_ACTION` |
|---|---|---|
| pass | pass | `STAGE9B_POSTERIOR_WEIGHTING` |
| fail | pass | `STAGE9C_ADAPTIVE_PROPOSAL_FEASIBILITY` |
| pass | fail | `STOP_REDESIGN_LIKELIHOOD_OR_PETROPHYSICS` |
| fail | fail | `STOP_REASSESS_FROZEN_INFERENCE_ROUTE` |

Truth leakage, condition violations, checkpoint/EMA mismatch,
non-deterministic replay, or corrupt/incomplete artifacts produce
`ENGINEERING_FAIL`. Only the implementation defect may be fixed; the same
frozen algorithm, cases, seeds, sample count, ranking, and thresholds must be
rerun.

## Required outputs

For every formal case:

- pool `manifest.json`, `candidate_pool.csv`, lossless model/prediction chunks;
- `ranking_correct.csv`, `ranking_zero.csv`, `ranking_shuffled.csv`,
  `ranking_wrong_case.csv`, and `ranking_manifest.json`;
- `truth_metrics.csv`, `per_class_metrics.csv`, `component_metrics.csv`,
  `best_of_n.csv`, `enrichment.csv`, `correlations.csv`, and audit manifest.

Aggregate outputs are `summary.json` and `STAGE9A_REPORT.md`. They must separate
inference-visible evidence, retrospective truth evidence, oracle support
ceiling, and deployable seismic ranking. The report must state that the best
truth candidate is not deployable, lower seismic loss alone is not project
success, three synthetic cases are not field generalization, and Stage9A tests
only frozen-prior support and likelihood enrichment.

## Stop rule

After the formal Stage9A gate and report, stop. A Stage9B/9C recommendation is
not authorization to implement posterior weighting, SMC, D-Flow, adaptive
proposals, a new likelihood, or training. A failed Stage9A cannot be rescued by
10k samples, new cases/controls, truth-favorable seeds, ranking changes,
petrophysical/seismic tuning, decoder changes, or parameter sweeps.
