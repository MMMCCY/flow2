# Stage15-B2 — Frozen-Flow binary-seismic pCN pilot

Status: **truth-blind fixed pilot completed; STOP before retrospective evaluation or Flow guidance**.

## Frozen design

- Four independent chains, 32 proposals per chain, burn-in 8, thinning 1.
- 32-step fixed-Euler frozen conditional Flow with EMA weights and exact original categorical conditions.
- pCN beta `0.1`; likelihood weight `1.0`; no parameter sweep.
- Likelihood: numerically hard decoded label9 versus every other subsurface lithology, mapped through the Stage15 binary acoustic upper bound and unchanged `ConvolutionalSeismic` operator.
- Occupancy contains all 96 post-burn-in current chain states, including rejection duplicates. It is an ensemble occupancy frequency, not a calibrated Bayesian posterior probability.
- No hidden truth, Stage15-B1 result, soft Flow probability, STE, voxel optimization, morphology, filtering, or threshold tuning was used.

Phase5c used a 15-class petrophysical seismic likelihood. Stage15-B2 instead uses the target-specific `binary_label9_hard_seismic_likelihood` after hard 15-class Flow decoding.

## Sampler

| Chain | Accepted | Rate | Rejections | Initial loss | Post-burn-in median | Post-burn-in minimum |
|---:|---:|---:|---:|---:|---:|---:|
| 0 | 30/32 | 0.9375 | 2 | 16.727596 | 14.278574 | 13.697871 |
| 1 | 27/32 | 0.84375 | 5 | 18.697872 | 15.690130 | 14.115924 |
| 2 | 31/32 | 0.96875 | 1 | 14.489023 | 13.431053 | 12.587500 |
| 3 | 30/32 | 0.9375 | 2 | 16.671556 | 16.365367 | 14.600726 |

- Overall acceptance: `118/128 = 0.921875`.
- Retained states: exactly 96.
- Unique retained categorical models: 89.
- Unique retained binary models: 89.
- Repeated states from rejection accounting: 7.
- Global post-burn-in hard-loss median/range: `14.635796`, `12.587500–18.521929`.
- Total CUDA runtime: `555.77 s`.

Independent validation confirmed that every rejected iteration retained the preceding categorical and binary state hashes, all output tensor hashes match the manifest, and the 96-state occupancy exactly equals the saved binary-state mean.

## Label9 volume

| Chain | Initial fraction | Post-burn-in median | Post-burn-in range |
|---:|---:|---:|---:|
| 0 | 0.022692 | 0.021002 | 0.019352–0.022399 |
| 1 | 0.029723 | 0.022986 | 0.016978–0.026960 |
| 2 | 0.014269 | 0.014583 | 0.010762–0.018334 |
| 3 | 0.027997 | 0.025770 | 0.020217–0.028650 |

Across all retained states, target fraction has median `0.021255` and range `0.010762–0.028650`; target voxel count has median 4,199 and range 2,126–5,660. All initial/proposed/current condition violations are zero.

The Stage15-B1 free-voxel failure mode is absent: B1 produced roughly 58% label9, whereas every B2 retained Flow realization remains below 2.87%. The likelihood generally lowers seismic loss without escaping the frozen Flow geological support. It also slightly reduces mean target fraction relative to the four-sample initial diagnostic; this is reported without reinterpretation.

## Occupancy and fixed 0.8/0.2 consensus

- P9 min/max/mean within subsurface: `0 / 1 / 0.020828`.
- Four-initial-model diagnostic mean frequency: `0.023671`.
- Positive voxels: 114.
- Negative voxels: 188,772.
- Unknown voxels: 8,669.
- Confidence coverage: `0.956119`.
- Unconditioned guidance ROI voxels: 184,368.

Positive geometry is spatially localized without cleanup:

- six-connected components: 2;
- components with at least 20 voxels: 2;
- largest component: 63 voxels (`55.26%` of positive mass);
- bounding box: `[36,56,2]` to `[46,63,25]`.

## Stop decision

The narrow B2 mechanism question passes: the existing frozen Flow prior prevents the Stage15-B1 label9 volume explosion while pCN produces diverse, condition-exact realizations and lowers binary hard-seismic loss in the aggregate. This does not establish truth alignment, calibrated posterior probability, realistic field inversion, or benefit to Flow guidance.

Per protocol, do not extend chains, run N=100, tune beta/likelihood/thresholds, open truth, or run Phase1 guidance without a new manual decision.
