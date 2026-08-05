# Phase 6Q Q3 frozen embedding endpoint report

Completed: 2026-08-04T10:02:38.125434+00:00

Only the checkpoint embedding matrix was loaded. The flow U-Net was not instantiated, and no training was performed.

| Mode | Control | Method | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Air voxels | Best step |
|---|---|---|---:|---|---|---:|---:|---:|
| property | correct | soft_embedding | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |
| property | correct | ste_embedding_rock | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |
| blurred_property | correct | soft_embedding | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 157 | 180 |
| blurred_property | correct | ste_embedding_rock | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 65 |
| reflectivity_spikes | correct | soft_embedding | 0.127138 | 0.0500/1.0000/0.0500 | 0.0000/0.1000 | 64 | 0 | 165 |
| reflectivity_spikes | correct | ste_embedding_rock | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | correct | soft_embedding | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | correct | ste_embedding_rock | 0.378008 | 0.2000/1.0000/0.2000 | 0.2000/0.2000 | 256 | 0 | 200 |
| seismic | zero | soft_embedding | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | zero | ste_embedding_rock | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | shuffled_xy | soft_embedding | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | shuffled_xy | ste_embedding_rock | 0.161162 | 0.0022/0.0071/0.0031 | 0.0000/0.0063 | 560 | 0 | 190 |
| gravity | correct | soft_embedding | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| gravity | correct | ste_embedding_rock | 0.706745 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 190 |

Best-state selection used hard physics only. All geology metrics are post-run audits.
