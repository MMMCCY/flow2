# Phase 2b report: property-codebook ambiguity and contrast sensitivity

Updated: 2026-07-30 after the completed seed-42 n=4 bracket and post-bracket
`paired_c100` robustness fallback.

## Decision

Phase 2b is complete as an **observability-sensitivity study with no
multi-seed-confirmed ambiguous-codebook operating point** under the frozen
4/4 rule.

This is not a claim that all property guidance failed. The ambiguous
`paired_c100` and `paired_c025` levels produce strong mean hard-label gains,
but each passes only 3/4 complete seed-42 pair gates. `paired_c010` passes 0/4.
The pre-registered rule therefore authorizes no seeds 142/242 run and no
post-hoc alpha or threshold adjustment.

The authoritative machine reports are:

- single-sample screen:
  `experiments/stage2_property/reports/phase2b_codebook_ambiguity_v1_screen_seed42/`;
- frozen c025/c010 n=4 bracket:
  `experiments/stage2_property/reports/phase2b_codebook_ambiguity_v1_n4_bracket_seed42/`;
- post-bracket c100 n=4 fallback:
  `experiments/stage2_property/reports/phase2b_codebook_ambiguity_v1_n4_fallback_c100_seed42/`.

## Frozen experiment identity

- Case: `cond_generation_0`; target pressure test: raw label 9.
- Full-resolution, noiseless truth-derived two-channel property target.
- Frozen training, 3-D U-Net, checkpoint and categorical embedding.
- Normal checkpoint embedding plus EMA for all 411 trainable entries.
- Strict alpha-zero/guided pairs with identical sequential CPU noise.
- Seed 42, 32-step fixed-Euler midpoint integration.
- Guided alpha/cap 0.25/0.25 with the unchanged Phase-2a controller, schedule,
  confidence mask and three-scale property loss.
- Surface, air and borehole conditions projected before sampling and after
  every integration step.
- Phase-1 probability and historical 2-D field guidance inactive.

## Single-sample screen

| Level | Screen gate | Label-9 IoU / P / R | Interpretation |
|---|---|---|---|
| `distinct_c100_anchor` | pass | 0.4818 / 0.9004 / 0.5089 | Phase-2a implementation anchor |
| `paired_c100` | pass | 0.4260 / 0.8699 / 0.4550 | overlap with strong target contrast |
| `paired_c025` | pass | 0.3951 / 0.8632 / 0.4215 | narrow pass |
| `paired_c010` | fail | 0.2370 / 0.7569 / 0.2566 | target/body/topology failure |
| `paired_c004_overlap` | fail | 0.0355 / 0.1391 / 0.0455 | exact label-6/9 collision control |

The screen selected `paired_c025` and adjacent `paired_c010` for the frozen
four-sample bracket. A one-sample pass was never treated as confirmation.

## Seed-42 n=4 evidence

| Level | Pair gates | Class | Accuracy delta | Fixed mIoU delta | Hard-property delta | Mean label-9 IoU / P / R |
|---|---:|---|---:|---:|---:|---|
| `paired_c100` | 3/4 | transition | +0.04325 | +0.06552 | -0.81585 | 0.4412 / 0.8816 / 0.4695 |
| `paired_c025` | 3/4 | transition | +0.04110 | +0.06033 | -0.79273 | 0.4040 / 0.8762 / 0.4290 |
| `paired_c010` | 0/4 | failure | +0.03244 | +0.03785 | -0.61635 | 0.2376 / 0.7756 / 0.2556 |

All three ensembles contain four unique decoded samples and retain non-zero
outside-ROI disagreement (`0.1532`, `0.1532` and `0.1536`, respectively).
Strict pairing, EMA coverage, finite traces, zero condition violations and
confidence locality pass.

At `paired_c100`, samples 0/1/3 pass every complete pair gate. Sample 2 passes
the global directions, six-of-eight class improvement, target thresholds,
size-stratified topology and endpoint churn gates. Its four-major-body mean
recall is 0.4473, but the minimum is 0.2313 versus the frozen 0.25 threshold.
The 3/4 classification is therefore a genuine geometric robustness miss, not a
pairing or numerical failure.

## Scientific interpretation

Phase 2b establishes that:

- overlapping lithology property vectors weaken the consistency of hard
  geometric recovery even when mean global, target and hard-property metrics
  remain strongly favorable;
- the exact label-6/9 collision removes useful target identifiability at this
  operating point;
- continuous property loss and favorable ensemble means cannot replace
  per-sample major-body recovery;
- under the conservative 4/4 rule, the only multi-seed-validated operating
  point remains the distinct Phase-2a property upper bound.

It does not establish that calibrated or measured geophysics will fail. The
codebooks are controlled relative values, the target is truth-derived and
full-resolution, and no 2-D gravity/magnetic forward operator is present. It
also does not justify removing surface or borehole conditions.

## Transition to Phase 3

Phase 3 should isolate spatial information loss using the already
multi-seed-validated distinct Phase-2a two-channel property codebook. Starting
from a 3/4 ambiguous transition level would confound spatial degradation with
an unconfirmed categorical observability failure.

Before any Phase-3 GPU run, freeze a separate protocol for blur, downsampling,
missing regions, depth-dependent confidence and noise. Each degradation family
must first change only one observation property, retain a new alpha-zero strict
pair and preserve the same hard-label, geometry, condition and diversity
evaluation. The Phase-2b ambiguous levels remain reported sensitivity results
and are not silently promoted to accepted Phase-3 baselines.
