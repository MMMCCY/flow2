# Phase 4d specification: seismic identifiability and posterior selection

Date frozen: 2026-07-31, after Phase 4c closure and before evaluating the
fixed 12-sample baseline pool with the Phase-4c seismic observation.

## Decision context

Phase 4c showed that differentiable seismic guidance can lower the hard trace
residual and cross hard-label boundaries, but its changes do not recover label
9 or its major three-dimensional bodies. Additional alpha tuning is blocked.
The next question is therefore not controller strength. It is whether the
frozen conditional flow prior already proposes truth-like candidates and
whether the seismic likelihood can identify them without access to truth.

Phase 4d is an offline diagnostic and posterior-selection upper bound. It does
not alter a geological sample, run a new model trajectory, promote Phase 4c,
or combine gravity and seismic.

## Frozen candidate population

Use exactly the existing Phase-2a alpha-zero fixed-Euler baselines from the
validated distinct-property experiment:

- seed 42, sample IDs 0-3;
- seed 142, sample IDs 0-3;
- seed 242, sample IDs 0-3.

This gives 12 candidates. Every source run must be completed with four samples,
32 midpoint fixed-Euler steps, alpha zero, EMA for all trainable parameters,
the normal frozen embedding, exact conditions, the canonical checkpoint,
truth and boreholes, and the sequential CPU-noise policy. Saved sample hashes
must match their configs. The seed-42 sample-0 tensor must also equal the
completed Phase-4c alpha-zero anchor, linking the historical pool to the
current baseline semantics.

No candidate may be discarded, regenerated or added after its seismic score is
known. Candidate identity is the ordered `(seed, local_sample_id)` pair.

## Frozen observation and selection rule

Use the immutable Phase-4c `distinct_upper_bound_v1_fix2` observation,
acoustic codebook, masks, uncertainty and forward operator. For each hard
candidate:

1. overwrite the hard acoustic properties at known surface and borehole
   voxels with the exact truth properties;
2. compute its full-cube hard seismic response;
3. compute the normalized hard seismic loss, RMSE and MAE;
4. rank candidates only by ascending hard seismic loss, breaking an exact tie
   by `(seed, local_sample_id)`.

Truth labels, target masks and geological metrics are forbidden in the
selection score. They are used only after the ranking has been frozen, to test
whether lower seismic mismatch actually selects better geology.

## Geological support audit

For every candidate, recompute complete hard-label, per-class and label-9
component metrics with current evaluation code. The fixed candidate pool has
adequate target support only if at least one candidate simultaneously has:

- zero hard-condition violations;
- label-9 IoU at least `0.30`;
- label-9 precision at least `0.75`;
- label-9 recall at least `0.30`;
- minimum recall across the four largest truth components at least `0.25`;
- mean recall across those components at least `0.40`.

The oracle-best candidate is reported only as a support ceiling. It is never a
deployable selector because choosing it uses truth.

## Seismic ranking audit

Report Spearman rank correlation between hard seismic loss and each of:

- global voxel accuracy;
- truth-present mean IoU;
- label-9 IoU;
- label-9 recall;
- four-major-body mean recall.

Because lower loss should correspond to higher quality, useful ordering has a
negative coefficient. Report deterministic one-sided permutation p-values
with 10,000 permutations and seed 0; p-values are descriptive and do not
replace effect-size or hard-geology requirements.

The ranking gate requires all of the following:

1. loss correlation is at most `-0.50` for label-9 IoU, label-9 recall and
   four-major-body mean recall;
2. the top-three seismic-selected mean strictly exceeds the full-ensemble mean
   for those same three metrics;
3. the top-one seismic-selected candidate passes the complete geological
   support thresholds above.

Promotion requires both the population support gate and the ranking gate. A
good correlation without a geologically adequate candidate is not recovery;
an adequate oracle candidate that seismic cannot select is not a usable
posterior method.

## Truth substitution sensitivity

As a separate mechanistic diagnostic, start from the hard truth and, for each
truth-present non-air source label, replace all of its *unconditioned* voxels
with each alternative rock label `0..13`. Known surface and borehole voxels
remain exact. Record changed voxel count, hard seismic loss, RMSE and MAE.

This matrix asks whether a whole-class lithology substitution is visible to the
frozen forward operator. It is a truth-derived oracle perturbation and is not a
generation result, an inversion, or a selection score. Strong substitution
sensitivity can coexist with poor local identifiability because compensating
boundary, timing and lithology changes remain possible.

## Decision tree

- Support fail: the 12-sample prior pool does not contain a sufficiently good
  target model; reranking cannot recover geology that was not proposed.
- Support pass, ranking fail: the candidate pool contains useful geology but
  this seismic likelihood does not identify it; enrich the observation model.
- Support and ranking pass: freeze a larger held-out ensemble protocol before
  testing posterior selection or particle-based inference.
- In every failure case, do not tune the ranking rule on truth, reopen Phase-4c
  alpha search, or combine gravity to rescue the score.

If inference-time proposal support remains inadequate even under this favorable
inverse-crime diagnostic, the next scientifically distinct direction is joint
geophysics-aware training or fine-tuning. That work requires a new protocol and
explicit authorization because it changes the frozen-model boundary.

## Required outputs

- immutable input/source/hash manifest;
- one row per candidate with seismic and complete hard-geology metrics;
- seismic ranking with fixed candidate IDs;
- per-class and truth-component rows;
- rank correlations, top-k enrichment, oracle ceilings and gate verdicts;
- truth-substitution sensitivity matrix;
- machine-readable summary and a concise report;
- tests for source-run validation, deterministic ranking/ties, correlations,
  support/ranking gates, exact conditions and non-empty-output refusal.

## Completion status

The fixed 12-sample CPU audit is complete. Source invariants, hashes, the
Phase-4c alpha-zero anchor and exact conditions all pass. No candidate passes
the support gate, the seismic-selected top-three target metrics are below the
ensemble means, and seismic loss correlates with label-9 IoU/recall and
major-body mean recall in the wrong direction (`rho=+0.552/+0.587/+0.580`).
Phase 4d is closed without promotion. See `docs/PHASE4D_REPORT.md`.
