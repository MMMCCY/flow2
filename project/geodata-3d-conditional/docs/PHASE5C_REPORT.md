# Phase 5c final report: direct conditional-flow posterior search

Date closed: 2026-08-01.

## Decision

**CLOSED NEGATIVE LEGACY SCREEN: full-dimensional pCN is active and can lower
the hard seismic likelihood, but the frozen 8-proposal chain again trades
target-body geology for a better global fit.  Do not expand or tune this
implementation on `cond_generation_0`; proceed to Phase 6.**

This result closes the current implementation, not every latent-posterior
algorithm.  The case is repeatedly truth-audited and is not held-out evidence.

## What was implemented

Phase 5c is the previously missing direct-generator analogue of published GAN
prior inversion.  It treats the Gaussian initial noise of the complete frozen
conditional flow as the latent variable.  A pCN proposal is passed through the
canonical EMA U-Net using 32-step midpoint fixed Euler, with surface and
borehole embeddings projected before the first step and after every step.  The
proposal is hard decoded, converted through the immutable Phase-4c acoustic
codebook, and accepted or rejected using the hard convolutional-seismic
likelihood.

Unlike Phase 5b, there is no intermediate impedance-posterior target and no
soft property guidance.  Unlike Phase 4c, the physical residual does not
locally modify Euler velocities.  Every proposal is a complete sample from the
frozen conditional generator.  pCN preserves the Gaussian prior, so the
Metropolis ratio contains only the frozen hard seismic energy.

Sampling and truth evaluation are separated.  The runner records immutable
likelihoods, decisions, tensors and hashes without calculating geological
truth metrics.  A separate auditor validates completion and hashes before
opening truth.

## Integrity and performance

The historical seed-42 initial noise hash is
`5b3e3cc13e70f1477679559ea82781e02fc58130469ff39d5c71b5db647f9eea`.
Its decoded initial state exactly matches the completed Phase-4c alpha-zero
baseline.  EMA coverage, observation/source hashes and condition projection
all pass.  Maximum hard-condition violations are zero.

On the RTX 4090 D:

- each 32-step proposal evaluation takes about `4.0` seconds after model load;
- peak allocated CUDA memory is about `1.66 GB`;
- the primary chain accepts `7/8` proposals and retains eight unique hard
  models including the initial state;
- retained models move as far as `5.20%` of all hard voxels from the initial
  baseline.

The first attempted smoke directory `performance_smoke_v1` is incomplete.  It
finished the initial generator evaluation but correctly stopped because the
new batched decoded shape and historical unbatched saved sample were compared
without normalization.  The comparison is fixed, regression-tested, and the
immutable successful rerun is `performance_smoke_v1_fix1`.  Do not reuse or
overwrite the incomplete directory.

## Frozen primary result

The one-proposal performance smoke happened to improve every mechanism-screen
direction, which authorized the already frozen 8-proposal pilot.  It was not a
scientific conclusion.

The primary pilot's minimum-seismic-loss retained state is index 7:

| Metric | Initial baseline | Min-likelihood retained | Delta |
|---|---:|---:|---:|
| Hard seismic loss | 17.860632 | 17.653818 | -0.206814 |
| Global voxel accuracy | 0.587366 | 0.594143 | +0.006778 |
| Truth-present mIoU | 0.265249 | 0.269580 | +0.004330 |
| Label-9 IoU | 0.028596 | 0.025298 | -0.003298 |
| Label-9 precision | 0.067484 | 0.062661 | -0.004823 |
| Label-9 recall | 0.047279 | 0.040700 | -0.006579 |
| Four-major-body mean recall | 0.041423 | 0.034587 | -0.006837 |

The final retained state has truth-present mIoU `0.269806`, label-9 IoU
`0.025827` and label-9 recall `0.042596`.  Thus the global/common-class result
can improve while the sparse intrusion and its principal bodies become worse.

Integrity, chain activity and physical-improvement gates pass.  The joint hard
geology gate fails because label-9 IoU, recall and four-major-body recall all
move in the wrong direction.  A lower hard physical loss is not sufficient.

## Interpretation

The literature-inspired algorithm is technically feasible on the current
three-dimensional flow despite its roughly four-million-dimensional initial
noise.  pCN creates diverse condition-exact hard models, and the hard
likelihood entirely avoids the previous soft-hard gradient explanation.

The negative result is instead consistent with the Phase-4d identifiability
diagnosis.  The global seismic likelihood rewards changes to common interfaces
that improve complete-cube fit while allowing the weakly observed label-9
bodies to shrink or move.  Sampling the generator posterior directly does not
create lithology identity information absent from the current observation and
petrophysical mapping.

The high `0.875` acceptance and short chain do not establish calibrated
posterior mixing.  Longer chains could explore farther, but the pre-registered
promotion gate failed on a truth-audited case.  Longer chains, beta/temperature
sweeps, truth-informed selection and additional seeds are therefore not
authorized for this implementation.

## Next work

Proceed to Phase 6: freeze the original EMA U-Net, embedding and checkpoint,
and train only a small geophysics-conditioned residual-velocity adapter.  Its
loss must explicitly include all-class hard-alignment objectives in addition
to flow and physical consistency.  Phase 6 first needs deterministic data
manifests, geology-grouped splits, adapter-off exact baseline regression and a
legacy tiny-overfit engineering screen before any formal training claim.

Authoritative evidence:

- `docs/PHASE5C_SPEC.md`;
- `experiments/stage5_generator_posterior/runs/cond_generation_0/primary_pilot_v1/`;
- `experiments/stage5_generator_posterior/reports/cond_generation_0/primary_pilot_v1/`;
- its `summary.json`, `REPORT.md`, retained metrics, class transitions and
  component-recovery tables.

