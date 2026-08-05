# Phase 6Q Q2 free-voxel machine report

Completed: 2026-08-04T09:41:17.131611+00:00

No candidate shape/count/location prior, flow checkpoint or training was used.

| Mode | Control | Method | Hard attainment | Hidden IoU/P/R | Body recalls | Hard voxels | Best step |
|---|---|---|---:|---|---|---:|---:|
| property | correct | soft_voxel | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 40 |
| property | correct | ste_voxel | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 40 |
| blurred_property | correct | soft_voxel | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 135 |
| blurred_property | correct | ste_voxel | 1.000000 | 1.0000/1.0000/1.0000 | 1.0000/1.0000 | 1280 | 95 |
| reflectivity_spikes | correct | soft_voxel | 0.044139 | 0.1000/1.0000/0.1000 | 0.1000/0.1000 | 128 | 110 |
| reflectivity_spikes | correct | ste_voxel | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 |
| seismic | correct | soft_voxel | 0.032980 | 0.1000/1.0000/0.1000 | 0.1000/0.1000 | 128 | 110 |
| seismic | correct | ste_voxel | 0.102570 | 0.4000/1.0000/0.4000 | 0.4000/0.4000 | 512 | 40 |
| seismic | zero | soft_voxel | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 |
| seismic | zero | ste_voxel | 0.000000 | 0.0000/0.0000/0.0000 | 0.0000/0.0000 | 0 | 0 |
| seismic | shuffled_xy | soft_voxel | 0.015044 | 0.0013/0.0086/0.0016 | 0.0000/0.0031 | 232 | 110 |
| seismic | shuffled_xy | ste_voxel | 0.045844 | 0.0036/0.0086/0.0063 | 0.0000/0.0125 | 928 | 40 |
| gravity | correct | soft_voxel | 0.938491 | 0.0953/0.1778/0.1703 | 0.0000/0.3406 | 1226 | 300 |
| gravity | correct | ste_voxel | 0.737762 | 0.0654/0.1411/0.1086 | 0.0000/0.2172 | 985 | 255 |

Model selection used hard physics only. Geometry metrics above are post-run audits.
