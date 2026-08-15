# Stage15-B2 frozen-Flow binary-seismic pCN pilot

Status: truth-blind fixed 4×32 pilot complete. No retrospective evaluation or Flow guidance was run.

## Sampler

- Per-chain acceptance: [0.9375, 0.84375, 0.96875, 0.9375]
- Overall acceptance: 0.921875
- Post-burn-in states: 96 (includes rejection duplicates)
- Unique categorical / binary states: 89 / 89

## Seismic and geology

- Initial hard seismic losses: [16.727596282958984, 18.697872161865234, 14.489023208618164, 16.67155647277832]
- Minimum hard seismic loss: 12.5875
- Median post-burn-in loss: 14.635796
- Initial target fractions: [0.02269241477057022, 0.029723368175951, 0.014269443952317076, 0.02799726658398927]
- Post-burn-in target-fraction range: 0.010762–0.028650
- Condition violations: 0

## Occupancy consensus

- P9 min / max / mean: 0.000000 / 1.000000 / 0.020828
- Positive / negative / unknown: 114 / 188772 / 8669
- Confidence coverage: 0.956119
- Guidance ROI voxels: 184368
- Positive components / >=20 / largest: 2 / 2 / 63

This is an ensemble occupancy frequency from frozen-Flow pCN chain states, not a calibrated Bayesian posterior probability. Phase5c used a 15-class petrophysical likelihood; Stage15-B2 uses a target-specific hard binary label9-vs-all-other-subsurface likelihood.
