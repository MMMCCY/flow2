# Stage 13A — Binary label-9 geophysical probability bridge

## Final machine decision

`STOP_PHYSICS_BINARY_BRIDGE_NOT_IDENTIFIABLE`

Pre-implementation classification:
`PHYSICS_BINARY_BRIDGE_REQUIRES_LEARNED_MAPPER`.

The current synthetic post-stack seismic operator and registered acoustic
codebook are insufficient to define a physically grounded voxelwise
`P(label9 | d,x)` without learning the missing waveform-context/geology
distribution. Stage13-A therefore stopped at the required design audit. No
probability map, retrospective Stage13 metric, diagnostic figure, Flow sample,
or Flow forward was produced.

## Repository and execution provenance

- Remote execution host: `172.27.231.254`, user `xmj`, repository
  `/home/xmj/mcy/flowtrain_stochastic_interpolation-main`.
- Runtime environment: repository `.venv`, prompt `flowtrain`, Python 3.10.12.
- Branch and HEAD: `main @ 72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f`.
- The worktree was already dirty with the existing StructuralGeo and Stage9–12
  artifacts. They were preserved; Stage13 only adds its own experiment tree.
- The protocol was written with status
  `frozen_before_stage13_truth_evaluation`. No Stage13 truth tensor or
  hidden-label9 tensor was loaded.

## Frozen historical boundaries

- Stage10 remains `STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION`.
- Stage10R remains a diagnostic interpretation:
  `CASE_GEOMETRY_CONFUNDED` with complementary
  `SEISMIC_ADDS_INCREMENTAL_INFORMATION`; it does not reopen Stage10.
- Stage12A remains `FULL_STRUCTURALGEO_BENCHMARK_READY`.
- Stage12B remains `STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC`.
- The five Stage12A cases are unchanged and no case was selected or replaced.

## Why the physics-only likelihood is not identifiable

The forward operator produces a band-limited convolution of vertical interface
reflection coefficients at velocity-dependent travel times. It does not
measure absolute impedance, and a homogeneous label-9 interior has no local
reflection response. A waveform near a boundary depends jointly on label 9,
the neighboring nuisance lithology, vertical ordering, thickness, tuning,
wavelet, and time-depth mapping. Non-label9 interfaces can reproduce the same
amplitude, energy, or multiscale context.

Thus the acoustic codebook is not enough to specify
`p(feature | label9)` or the required non-label9 nuisance mixture. Estimating
those distributions from independently generated simulations would be a learned
pattern mapper. It may be a valid next study, but it is not a pure-physics
formula licensed by Stage13.

The Phase5a inverted-impedance anomaly is not an alternative here: its missing
low-frequency/time-depth information comes from geological prior members, and
mapping that scalar inversion to categorical probabilities is the Stage12B
bridge that this stage explicitly forbids repackaging.

## Stage12B versus the proposed binary bridge

Stage12B computed a 15-class posterior from scalar inverted log impedance,

`P(k|Z) ∝ prior_k N(log Z; mu_k, sigma_k^2)`,

then selected the label-9 channel. A genuine Stage13 bridge would instead model
a target-specific observation likelihood directly,

`log p(seismic context|label9) - log p(seismic context|non-label9 mixture)`,

and calibrate that evidence to a voxelwise binary probability. The second
representation can use waveform pattern and spatial context, whereas the first
collapses the observation to one scalar property. The present audit finds that
the second representation requires learned context distributions; they are not
defined by the existing physics assets.

## Required final questions

1. **What is the essential distinction from Stage12B?** Stage12B is a
   multiclass scalar-impedance posterior followed by channel selection. Stage13
   requires a direct binary waveform/context likelihood with non-label9 as a
   nuisance mixture.
2. **Is current seismic physics sufficient for target-specific voxel
   probability?** No. It supplies interface reflectivity and travel time, not a
   unique class/interior likelihood.
3. **Does the binary bridge have case specificity?** Not evaluated: no
   defensible bridge volume was constructed.
4. **Does correct beat shuffled, constant, or wrong-case?** Not evaluated; all
   evaluation files are explicit `NOT_EXECUTED` manifests.
5. **Is the Phase1-style frozen-Flow probability-guidance pilot authorized?**
   No. Stage13-B is not authorized and Flow forward count is zero.
6. **Where is the failure most likely?** Primarily seismic observability and
   feature representation, with geology/petrophysical nonuniqueness as the
   mechanism. Probability calibration is downstream and was never reached.

## Exact implementation paths reviewed

The complete path list is recorded in
`audit/physics_identifiability_audit.md`. It includes the Phase1 probability
construction/guidance/evaluation modules, `guidance/seismic.py`,
`guidance/seismic_inversion.py`, the Phase5a builder and reports, the acoustic
codebook, `guidance/geophysical_probability_bridge.py`, and all required
Stage12A/12B reports and machine decisions.

## Recommended next stage — not implemented

Prospectively register a learned `seismic -> P(label9)` encoder or likelihood
estimator using independently generated calibration/training simulations that
exclude all five Stage12A cases. Freeze generator seeds, data splits, features,
architecture, calibration, controls, and the same 5x5 case-specificity gate
before opening Stage12A truth evaluation. Do not run Flow until that bridge
passes. This recommendation awaits manual approval.
