# Phase 6Q Q2b monotone hard-coordinate report

Completed: 2026-08-04T09:48:28.067310+00:00

Every accepted update was selected by hard physics RMSE; no soft endpoint, shape/count prior, regularizer, flow checkpoint or training was used.

| Mode | Control | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Accepted/total |
|---|---|---:|---|---|---:|---:|
| property | correct | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 3/4 |
| blurred_property | correct | 0.968465 | 0.9542/0.9766/0.9766 | 0.9906/0.9625 | 1280 | 6/7 |
| reflectivity_spikes | correct | 0.044139 | 0.1000/1.0000/0.1000 | 0.1000/0.1000 | 128 | 1/2 |
| seismic | correct | 0.156380 | 0.7000/1.0000/0.7000 | 0.7000/0.7000 | 896 | 3/4 |
| seismic | zero | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0/1 |
| seismic | shuffled_xy | 0.016352 | 0.0013/0.0078/0.0016 | 0.0000/0.0031 | 256 | 2/3 |
| gravity | correct | 0.857794 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 772 | 13/14 |

Geometry metrics are post-run audits and were not used for proposal selection.
