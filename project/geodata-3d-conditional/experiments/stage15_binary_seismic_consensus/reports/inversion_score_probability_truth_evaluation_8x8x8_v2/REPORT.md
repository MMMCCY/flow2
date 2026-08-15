# Stage15-F — 8^3 inversion score to empirical P(label9)

This experiment uses one deterministic unregularized 8^3 inversion per synthetic case and one 64-bin empirical lookup. It uses no Flow, learned model, class balancing, or parameter sweep. The inversion score is not treated as a probability until after calibration.

## Calibration and validation

- Cases: 128 (96 train / 32 validation)
- Train label9 prevalence: 0.011732678
- Validation label9-positive cases: 5/32
- Validation prevalence / AUPRC: 0.0080414098 / 0.067740438
- Validation truth/background mean P9: 0.071841292 / 0.010199734
- Positive-case median AUPRC: 0.043167431
- Positive cases with AUPRC above own prevalence: 3/5
- Positive cases with truth mean P9 above background: 3/5
- Empty quantile bins caused by tied inversion scores: 58/64

## Retrospective cond_generation_0

- P9 AUPRC: 0.089872368
- Truth mean/median P9: 0.084640637 / 0.099212527
- Background mean/median P9: 0.038820338 / 0.0047967033
- P9 range: 0.0047967033 to 0.11641598
- Positive >=0.8 voxels: 0
- Positive precision / recall / IoU: None / 0.0 / 0.0
- Truth positive/unknown/negative: 0 / 0 / 8968

Raw held-out coarse score localization: Pearson 0.16796768, Spearman 0.20251804, target/background mean 0.50837985/0.50122123, top-k 80/129, centroid distance 1.4338608 coarse cells.

Historical AUPRC: Stage15-B2 = 0.057284505; Stage15-C = 0.045394953.
