# Phase 6Q Q3 frozen embedding endpoint report

Completed: 2026-08-04T10:05:45.538097+00:00

Only the checkpoint embedding matrix was loaded. The flow U-Net was not instantiated, and no training was performed.

| Mode | Control | Method | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Air voxels | Best step |
|---|---|---|---:|---|---|---:|---:|---:|
| property | correct | soft_embedding_binary | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |
| property | correct | ste_embedding_binary | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |
| blurred_property | correct | soft_embedding_binary | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 85 |
| blurred_property | correct | ste_embedding_binary | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 65 |
| reflectivity_spikes | correct | soft_embedding_binary | 0.044139 | 0.1000/1.0000/0.1000 | 0.1000/0.1000 | 128 | 0 | 115 |
| reflectivity_spikes | correct | ste_embedding_binary | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | correct | soft_embedding_binary | 0.032980 | 0.1000/1.0000/0.1000 | 0.1000/0.1000 | 128 | 0 | 110 |
| seismic | correct | ste_embedding_binary | 0.102570 | 0.4000/1.0000/0.4000 | 0.4000/0.4000 | 512 | 0 | 30 |
| seismic | zero | soft_embedding_binary | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | zero | ste_embedding_binary | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| seismic | shuffled_xy | soft_embedding_binary | 0.012434 | 0.0014/0.0104/0.0016 | 0.0000/0.0031 | 192 | 0 | 180 |
| seismic | shuffled_xy | ste_embedding_binary | 0.045844 | 0.0036/0.0086/0.0063 | 0.0000/0.0125 | 928 | 0 | 30 |
| gravity | correct | soft_embedding_binary | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 | 0 |
| gravity | correct | ste_embedding_binary | 0.436240 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 470 | 0 | 110 |

Best-state selection used hard physics only. All geology metrics are post-run audits.
