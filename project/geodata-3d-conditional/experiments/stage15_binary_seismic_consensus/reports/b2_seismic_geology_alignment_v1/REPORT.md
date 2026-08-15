# Stage15-B4 — Frozen B2 seismic-loss / geology alignment audit

Machine decision: **SEISMIC_LOSS_NOT_ALIGNED_WITH_GEOLOGY**

This audit joined all 96 frozen retained states by retained index and by dtype-aware binary SHA256. It consumed B3 truth metrics without opening truth itself. It did not rerun sampling or inversion and changed no B2/B3 input, threshold, or likelihood weight.

## Spearman correlations

- Loss vs IoU: rho=0.43426327, p=9.8449439e-06
- Loss vs precision: rho=-0.42585244, p=1.5208718e-05
- Loss vs recall: rho=0.63222765, p=4.8944775e-12
- Loss vs centroid distance: rho=-0.12604711, p=0.22106017

For alignment, lower loss should accompany higher IoU/precision/recall (negative rho) and smaller centroid distance (positive rho when loss increases).

## Loss quartiles

- Lowest-loss 24 median IoU / precision / recall / centroid distance: 0.022414038 / 0.090388783 / 0.029605263 / 12.182395
- Highest-loss 24 median IoU / precision / recall / centroid distance: 0.027546059 / 0.072006198 / 0.042038359 / 9.3213844

## Representative states

- Minimum-loss state: retained 49, chain 2, iteration 10; loss / IoU / precision / recall / centroid distance: 12.5875 / 0.023242944 / 0.11853246 / 0.028099911 / 10.970178
- Maximum-IoU state: retained 39, chain 1, iteration 24; loss / IoU / precision / recall / centroid distance: 15.023236 / 0.037179992 / 0.11240772 / 0.052631579 / 17.298244

## Interpretation

Frozen hard-binary seismic loss does not consistently rank better truth-label9 geometry; the relationships are weak, unrelated, or directionally conflicting.

The conclusion is a manual interpretation of the frozen audit metrics, not a new numerical pass/fail threshold.
