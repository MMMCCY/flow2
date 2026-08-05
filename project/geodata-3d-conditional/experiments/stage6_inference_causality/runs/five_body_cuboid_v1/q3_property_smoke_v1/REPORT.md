# Phase 6Q Q3 frozen embedding endpoint report

Completed: 2026-08-04T09:59:33.936173+00:00

Only the checkpoint embedding matrix was loaded. The flow U-Net was not instantiated, and no training was performed.

| Mode | Control | Method | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Air voxels | Best step |
|---|---|---|---:|---|---|---:|---:|---:|
| property | correct | soft_embedding | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |
| property | correct | ste_embedding_rock | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 0 | 30 |

Best-state selection used hard physics only. All geology metrics are post-run audits.
