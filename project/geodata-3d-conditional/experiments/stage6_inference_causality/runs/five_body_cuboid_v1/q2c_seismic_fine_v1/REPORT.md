# Phase 6Q Q2b monotone hard-coordinate report

Completed: 2026-08-04T09:49:53.448085+00:00

Every accepted update was selected by hard physics RMSE; no soft endpoint, shape/count prior, regularizer, flow checkpoint or training was used.

| Mode | Control | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Accepted/total |
|---|---|---:|---|---|---:|---:|
| seismic | correct | 0.039993 | 0.1250/1.0000/0.1250 | 0.1500/0.1000 | 160 | 5/6 |
| seismic | zero | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0/1 |
| seismic | shuffled_xy | 0.016352 | 0.0013/0.0078/0.0016 | 0.0000/0.0031 | 256 | 8/9 |

Geometry metrics are post-run audits and were not used for proposal selection.
