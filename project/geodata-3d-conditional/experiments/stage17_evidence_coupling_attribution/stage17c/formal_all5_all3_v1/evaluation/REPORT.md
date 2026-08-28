# Stage17C same-evidence coupling report

Decision: **TRAJECTORY_COUPLING_POSITIVE**

Primary case-level result: 5/5 cases have positive median paired target-IoU delta; group median is 0.12505113.

The three source seeds are within-case stochastic replicates. The pooled 15-pair table is auxiliary only.

The D-Flow geophysical arm was not run because the independent Stage17B property gate failed.

| Case | Median delta target IoU | Positive seeds | Median delta precision | Median delta recall | Median delta abs. volume-error fraction | Median delta centroid distance |
|---|---:|---:|---:|---:|---:|---:|
| fullgeo_case01 | 0.007168 | 3/3 | 0.007142 | 0.766667 | 17.691667 | -6.129555 |
| fullgeo_case02 | 0.133220 | 3/3 | 0.101845 | 0.472362 | 1.216253 | -6.909861 |
| fullgeo_case03 | 0.125051 | 3/3 | 0.124530 | 0.863479 | 4.175691 | -21.252445 |
| fullgeo_case04 | 0.104195 | 3/3 | 0.060214 | 0.362597 | 0.818325 | -1.395920 |
| fullgeo_case05 | 0.219219 | 3/3 | 0.201201 | 0.795184 | 2.226222 | -9.361650 |

Negative centroid-distance deltas mean localization improved. Positive
absolute-volume-error deltas mean calibration worsened: the trajectory arm
systematically overpredicts target volume while sharply increasing recall.
As an auxiliary pooled diagnostic only, median target IoU rises from 0.029990
to 0.237611, recall from 0.220419 to 0.857719, and centroid distance falls from
16.5940 to 7.5265 voxels; median absolute volume-error fraction worsens from
0.564282 to 2.319874. This is consistent target localization, not final-goal
success: volume calibration and topology remain insufficient.
