# Stage15-C — Direct binary seismic attribute to label9 probability

The truth-blind runner calibrated 64 fixed quantile bins from 128 newly seeded full StructuralGeo cases, then mapped the frozen `cond_generation_0` seismic observation without opening held-out truth. This separate evaluator opened truth retrospectively only after the probability volume was frozen.

## Fixed attribute

Centered local trace energy is the mean of squared seismic amplitudes over the 17-sample (128 ms) Ricker-scale window. Fixed zero padding retains the same denominator at record edges. Each trace is linearly resampled to 64 voxel centers using local surface datum, 50 m cells and constant 2500 m/s background velocity. No truth velocity, lateral filtering, second attribute, neural network or parameter sweep is used.

## Held-out probability metrics

- Voxelwise AUPRC: 0.045394953
- Truth-label9 mean / median P9: 0.0098096961 / 0.0098096961
- Background mean / median P9: 0.0098096961 / 0.0098096961
- P9>=0.8 positive voxels / TP / FP: 0 / 0 / 0
- Positive precision / recall / IoU: nan / 0 / 0
- Truth label9 in positive / unknown / negative: 0 / 0 / 8968
- Truth fractions in positive / unknown / negative: 0 / 0 / 1

## Spatial localization

- Probability-weighted centroid to truth centroid distance: 10.139801 voxels
- Probability mass inside the truth-label9 bounding box: 0.70396596
- Maximum P9: 0.0098096961

## Frozen B2 comparison

- B2 occupancy AUPRC: 0.057284505
- Direct attribute AUPRC: 0.045394953
- Direct minus B2: -0.011889552
- Direct attribute is better by voxelwise AUPRC: **False**

The 0.8/0.2 masks are diagnostics only; no threshold sweep or downstream Flow guidance was run.
