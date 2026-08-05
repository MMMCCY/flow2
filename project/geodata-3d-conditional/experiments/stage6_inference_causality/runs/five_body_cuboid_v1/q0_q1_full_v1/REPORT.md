# Phase 6Q Q0/Q1 machine report

Completed: 2026-08-04T09:31:43.608966+00:00

This is a generator-free analytic inverse-crime diagnostic. No flow checkpoint was loaded and no training was performed.

## Q0 hard enumeration

| Mode | Truth rank | Truth RMSE | Second nontruth RMSE | Zero-count | Baseline RMSE |
|---|---:|---:|---:|---:|---:|
| property | 1 | 0 | 0.069877125 | 1 | 0.069877125 |
| blurred_property | 1 | 0 | 0.050540738 | 1 | 0.050581846 |
| reflectivity_spikes | 1 | 0 | 0.0070976806 | 1 | 0.007097689 |
| seismic | 1 | 0 | 0.0088858148 | 1 | 0.0088858241 |
| gravity | 1 | 0 | 0.089346223 | 1 | 0.40612191 |

## Q1 best hard checkpoints

| Mode | Control | Method | Hard attainment | Selected | Body P/R | Best step |
|---|---|---|---:|---|---|---:|
| property | correct | soft | 1.000000 | [4, 6] | 1.000/1.000 | 14 |
| property | correct | ste_top2 | 1.000000 | [4, 6] | 1.000/1.000 | 1 |
| blurred_property | correct | soft | 1.000000 | [4, 6] | 1.000/1.000 | 14 |
| blurred_property | correct | ste_top2 | 1.000000 | [4, 6] | 1.000/1.000 | 1 |
| reflectivity_spikes | correct | soft | 1.000000 | [4, 6] | 1.000/1.000 | 24 |
| reflectivity_spikes | correct | ste_top2 | 1.000000 | [4, 6] | 1.000/1.000 | 1 |
| seismic | correct | soft | 1.000000 | [4, 6] | 1.000/1.000 | 24 |
| seismic | correct | ste_top2 | 1.000000 | [4, 6] | 1.000/1.000 | 1 |
| seismic | zero | soft | 0.000000 | [] | 0.000/0.000 | 0 |
| seismic | zero | ste_top2 | 0.000000 | [] | 0.000/0.000 | 0 |
| seismic | shuffled_xy | soft | 0.000000 | [] | 0.000/0.000 | 0 |
| seismic | shuffled_xy | ste_top2 | 0.000000 | [] | 0.000/0.000 | 0 |
| gravity | correct | soft | 1.000000 | [4, 6] | 1.000/1.000 | 56 |
| gravity | correct | ste_top2 | 1.000000 | [4, 6] | 1.000/1.000 | 23 |

See `summary.json`, full enumeration CSVs and optimization traces for authoritative values.
