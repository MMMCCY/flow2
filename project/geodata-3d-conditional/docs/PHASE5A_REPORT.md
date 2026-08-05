# Phase 5a final report: no-training acoustic inversion bridge

Date closed: 2026-07-31.

## Decision

**PASS FOR THE NEXT CONTROLLED BRIDGE TEST, NOT A GEOLOGICAL-RECOVERY PASS.**

The frozen fixed-12 model-based inversion satisfies all six Phase-5a property
and integrity checks. It is therefore eligible to be tested as an uncertain
three-dimensional observation in a later strictly paired flow-guidance
experiment. Phase 5a alone does not show that hard lithology or label-9
geometry improved; its nearest-codebook diagnostics in fact become worse.

## Controlled design

The builder reads the immutable Phase-4c noiseless inverse-crime seismic
observation, acoustic codebook, surface mask, boreholes and exactly the 12
Phase-2a EMA/fixed-Euler alpha-zero samples already frozen in Phase 4d. It does
not load `true_model.pt`, `truth_acoustic.pt`, the U-Net or checkpoint weights.
The checkpoint is only hashed to validate source provenance.

For every prior, a linearized post-stack log-impedance correction is solved on
the regular-time grid with the one pre-registered Tikhonov operating point.
The correction is mapped to depth using that prior's slowness; slowness itself
is not inverted in v1. Surface and borehole acoustics are exact before and
after inversion. The separate auditor opens truth only after the completed
posterior manifest and all output hashes have been validated.

## Frozen-gate result

All frozen checks pass:

- exact-condition violations: `0` for all 12 priors, all 12 inverted members
  and the posterior mean;
- exact nonlinear seismic RMSE improves in `12/12` members, with the member
  mean changing `0.0426463 -> 0.0293196` (`-31.25%`);
- unconstrained log-impedance RMSE improves in `12/12` members, with the member
  mean changing `0.3072620 -> 0.2852502` (`-7.16%`);
- prior/posterior ensemble-mean log-impedance RMSE changes
  `0.2514293 -> 0.2433104`;
- prior/posterior ensemble-mean label-9-region log-impedance MAE changes
  `0.6485778 -> 0.6430638` (`-0.85%`);
- posterior log-impedance spread is finite, non-negative and nonzero over the
  unconstrained subsurface (mean `0.11404`, maximum `0.51034`).

The label-9 property improvement is real under the fixed metric but small.
It must not be described as recovered label-9 volume.

## Hard-projection warning

Projecting each continuous acoustic member independently to its nearest
normalized acoustic code makes the hard result worse:

| 12-member mean | Prior projection | Inverted projection |
|---|---:|---:|
| Global voxel accuracy | 0.5972 | 0.5163 |
| Label-9 IoU | 0.03144 | 0.01912 |
| Label-9 recall | 0.05196 | 0.02496 |

The prior/posterior mean projections also have near-zero label-9 recall
(`0.00401/0.00346`). This projection is not the planned flow decoder, but it
is a strong warning that better trace and continuous-property fit does not
resolve the acoustic-to-lithology non-uniqueness. It reinforces rather than
relaxes the requirement for hard-label and three-dimensional geometry gates.

## What Phase 5a proves

- A truth-blind, no-training model-based inversion can use the frozen flow
  ensemble as low-frequency/time-depth priors and produce a non-degenerate 3-D
  impedance posterior with lower synthetic trace and property errors.
- The surface/borehole conditions can remain exact through inversion.
- The inversion-to-3-D-property bridge is technically valid enough to test
  through the frozen geological decoder/prior.

## What Phase 5a does not prove

- It does not recover hard geology, label 9 or its major bodies.
- It does not solve density/velocity separation; v1 inverts impedance while
  holding each prior's slowness fixed.
- Ensemble spread from 12 flow samples is not calibrated posterior confidence.
- The Phase-4c input is synthetic, noiseless, truth-derived and inverse-crime;
  this is not measured seismic or a field-ready inversion.
- It does not justify sharpening, blurring or tuning regularization on truth.

## Next controlled experiment

Freeze a Phase-5b bridge that maps posterior mean and spread to a full 3-D
acoustic observation/confidence pair and runs the existing EMA/fixed-Euler
flow sampler against an identical alpha-zero baseline. Alpha zero must be
bytewise equivalent to the validated baseline, all conditions must remain
exact, and promotion requires hard global, per-class, label-9 and major-body
geometry improvement. Because the direct hard projection worsens, Phase 5b
must start with a single seed/sample gate and stop immediately if continuous
loss falls without hard-geology improvement.

Authoritative evidence:

- `experiments/stage5_acoustic_inversion/outputs/cond_generation_0/model_based_fixed12_v1/`;
- its `manifest.json` and `member_inversion_metrics.csv`;
- the adjacent `audit/summary.json`, hard-projection tables and `REPORT.md`.

The complete local lightweight suite passes `131` tests with 13 existing
Matplotlib/pyparsing deprecation warnings.

