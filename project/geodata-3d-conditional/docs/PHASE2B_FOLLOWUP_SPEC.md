# Phase 2b post-bracket robustness fallback

Date frozen: 2026-07-30, after the completed `paired_c025`/`paired_c010`
seed-42 n=4 bracket and before any `paired_c100` n=4 GPU run.

## Why this follow-up exists

The frozen single-sample screen selected `paired_c025` and its adjacent lower
level `paired_c010`. Their four-sample results do not produce a multi-seed
candidate:

- `paired_c025` passes 3/4 complete pair gates and is a transition region;
- `paired_c010` passes 0/4 and is a confirmed seed-42 failure;
- both ensembles pass the diversity requirement;
- the original bracket therefore authorizes no multi-seed run.

This result is final and is not reinterpreted. The next scientific question is
whether the next higher-contrast ambiguous level from the original predeclared
screen, `paired_c100`, is robust across four samples. This is an explicitly
post-bracket fallback, not part of the original selected bracket.

## Frozen fallback experiment

Run only `paired_c100` with seed 42 and four samples. Keep every Phase-2b
setting unchanged:

- frozen model, checkpoint, EMA convention and categorical embedding;
- a new alpha-zero strict baseline for the `paired_c100` property config;
- identical CPU initial noise within each baseline/guided pair;
- 32-step fixed-Euler midpoint integration;
- `alpha=0.25` and maximum guidance ratio `0.25`;
- unchanged temperature, controller, schedule, confidence and three-scale loss;
- full-resolution, noiseless truth-derived target properties;
- surface, air and borehole projection before sampling and after every step;
- inactive Phase-1 probability and 2-D gravity/magnetic guidance.

No alpha, cap, threshold, property value or topology rule may be adjusted after
seeing the result.

## Frozen decision rule

Apply the same complete per-pair gate and ensemble diversity gate used by the
original Phase-2b bracket:

- 4/4 pair gates plus four unique decoded samples and non-zero outside-ROI
  disagreement: confirmed seed-42 fallback pass;
- 4/4 pair gates without diversity: diversity failure;
- 1/4 to 3/4: transition region, not a pass;
- 0/4: confirmed seed-42 failure.

Only a confirmed 4/4 pass authorizes unchanged `paired_c100` runs at seeds 142
and 242. Any other result closes Phase 2b without a multi-seed-confirmed
ambiguous-codebook operating point at the tested levels. It does not authorize
post-hoc controller tuning.

## Interpretation boundary

Even a pass remains a full-resolution truth-derived property-oracle result. It
does not establish calibrated petrophysics, realistic inversion resolution,
2-D field identifiability, or independence from borehole/surface conditions.
Spatial degradation remains a later experiment and must not be merged into
this fallback.

## Completed result

`paired_c100` passes 3/4 complete pair gates with a valid diversity gate and is
therefore a transition region, not a confirmed fallback pass. Sample 2 misses
only the major-component minimum-recall requirement (`0.2313 < 0.25`). No
multi-seed run is authorized. The final interpretation is in
`docs/PHASE2B_REPORT.md`.
