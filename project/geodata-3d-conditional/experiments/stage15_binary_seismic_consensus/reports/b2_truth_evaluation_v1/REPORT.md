# Stage15-B3 — Frozen B2 retrospective truth evaluation

Machine decision: **POSITIVE_CORE_NOT_VALID**

This evaluation opened `cond_generation_0` truth only after all B2 pCN, occupancy, and 0.8/0.2 masks were frozen. It did not rerun or modify B2 and performed no threshold sweep.

## Positive core

- Positive / TP / FP: 114 / 56 / 58
- Precision / recall / IoU: 0.49122807 / 0.0062444246 / 0.0062042987
- Positive components: 2; truth components: 7
- Positive centroid nearest-truth-component distance: 13.074258 voxels

## Truth-label9 partition

- Positive: 56 (0.0062444246)
- Unknown: 541 (0.060325602)
- Negative: 8371 (0.93342997)
- Negative mask voxels: 188772
- Fraction of truth label9 marked negative: 0.93342997

## Frozen ensemble

- Initial median IoU / precision / recall / centroid distance: 0.029703357 / 0.080157843 / 0.043543711 / 11.236162
- Post-burn-in median: 0.025446498 / 0.079259094 / 0.036128457 / 12.983566
- Best retained IoU / precision / recall / centroid distance: 0.037179992 / 0.11240772 / 0.052631579 / 17.298244
- Post-burn-in IoU range: 0.018483159–0.037179992
- Post-burn-in precision range: 0.054726368–0.11887291
- Post-burn-in recall range: 0.022970562–0.054192685
- Post-burn-in centroid-distance range: 6.7801309–17.67478
- Overall truth alignment improved: False. No overall improvement: retained medians have lower IoU, precision, and recall and a larger centroid distance than the four initial models; the best retained IoU is higher, but that is not an ensemble-wide shift.

## Continuous occupancy

- Truth-label9 mean / median P9: 0.037322517 / 0
- Background mean / median P9: 0.020043856 / 0
- Voxelwise AUPRC: 0.057284505

The decision is an explicit manual interpretation of these frozen metrics, not the output of a newly introduced numerical gate.
