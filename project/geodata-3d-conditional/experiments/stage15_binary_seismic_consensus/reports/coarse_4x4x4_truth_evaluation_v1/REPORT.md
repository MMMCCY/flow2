# Stage15-E — 4^3 coarse binary seismic inversion identifiability

Machine decision: **COARSE_LABEL9_LOCALIZATION_FAILS**

The truth-blind runner optimized only 64 sigmoid coarse variables against raw mean-squared seismic misfit. It used no Flow, checkpoint, prior, regularizer, threshold, or observation regeneration. A 4^3 grid over 64^3 necessarily uses 16^3 fine voxels per coarse cell.

## Eight inversions

| Run | Seed | Final MSE | Pearson | Spearman | Target q | Background q | Top-k overlap | Centroid distance |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 151700 | 0.00070509239 | -0.02639529 | 0.00037563653 | 0.57689792 | 0.5437714 | 18/33 (0.54545455) | 0.87639733 |
| 1 | 151701 | 0.00070500764 | 0.052223639 | 0.089956804 | 0.46942999 | 0.42986422 | 21/33 (0.63636364) | 0.9750968 |
| 2 | 151702 | 0.00070497644 | -0.027823421 | -0.04143077 | 0.50605099 | 0.56523801 | 17/33 (0.51515152) | 1.0081314 |
| 3 | 151703 | 0.00070501195 | -0.16141913 | -0.073362364 | 0.43610714 | 0.44814916 | 19/33 (0.57575758) | 0.91025617 |
| 4 | 151704 | 0.00070493761 | -0.076273586 | 0.022513345 | 0.61657682 | 0.59241551 | 21/33 (0.63636364) | 0.94813286 |
| 5 | 151705 | 0.0007049159 | -0.080628992 | 0.005425287 | 0.51353114 | 0.50548372 | 18/33 (0.54545455) | 0.89905624 |
| 6 | 151706 | 0.00070488453 | 0.069628349 | 0.014718414 | 0.56237713 | 0.5684532 | 17/33 (0.51515152) | 0.94095912 |
| 7 | 151707 | 0.00070476474 | -0.098101416 | -0.15346201 | 0.46159371 | 0.48691244 | 17/33 (0.51515152) | 1.0119678 |

## Eight-run mean coarse occupancy

- Pearson / Spearman: -0.11286014 / -0.10096198
- Supplementary coarse-presence AUPRC: 0.58938215
- Target-containing / background-only mean q: 0.5178206 / 0.51753595
- Top-k overlap: 19/33 (0.57575758)
- Centroid distance: 0.93652026 coarse cells

## Interpretation

The frozen inversions do not consistently concentrate high coarse occupancy at true label9 coarse locations; seismic-loss reduction alone is insufficient.

This decision uses the complete frozen localization diagnostics, not seismic-loss reduction alone and not a newly introduced numerical gate.
