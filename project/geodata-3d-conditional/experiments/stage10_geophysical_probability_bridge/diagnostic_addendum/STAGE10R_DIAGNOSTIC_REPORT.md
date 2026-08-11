# Stage 10R — Geophysical Probability-Bridge Mechanism Diagnostic Addendum

## Frozen boundary

Stage10 is not rerun or reinterpreted. Its machine decision remains `STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION` (Stage10-A pass 1/3; Stage10-B/C/D not executed; Stage10 Flow forwards 0). Stage10R is a retrospective mechanism diagnostic and cannot authorize Stage10-B.

No Flow sampling, probability guidance, property/seismic inversion, training, new cases, parameter sweep, smoothing, or sharpening was executed. The fixed 0.5 probability threshold is used only for retrospective Dice/IoU summaries.

## Diagnostic interpretation

- Primary: `CASE_GEOMETRY_CONFUNDED`.
- Complementary mechanism finding: `SEISMIC_ADDS_INCREMENTAL_INFORMATION`.
- Not assigned: `PRIOR_TEMPLATE_DOMINATED`, `INCONCLUSIVE`.

The frozen Stage10 result remains FAIL. These findings diagnose why the wrong-case control was difficult and whether seismic altered the categorical probability map; they do not retroactively change the gate.

## Diagnostic 1 — all-by-all post-seismic bridge transfer

| bridge \ truth | T01 | T02 | T03 |
|---|---:|---:|---:|
| B01 | 0.2522 | 0.2222 | 0.2188 |
| B02 | 0.2851 | 0.2812 | 0.2701 |
| B03 | 0.2008 | 0.1826 | 0.1654 |

Diagonal mean AUPRC = 0.2329; off-diagonal mean = 0.2299; difference = 0.0030. The diagonal is the row maximum in 33% and the column maximum in 33% of cases.

Full Brier, ROC-AUC, and fixed-threshold Dice/IoU values are in the CSV outputs.

## Truth geometry across cases

| pair | label-9 IoU | centroid distance [voxels] | volume ratio | components | matched-body centroid mean/max |
|---|---:|---:|---:|---:|---:|
| T01–T02 | 0.6294 | 1.9764 | 0.7388 | 5/5 | 0.2508/0.3647 |
| T01–T03 | 0.6193 | 3.4928 | 0.7975 | 5/5 | 0.2333/0.4158 |
| T02–T03 | 0.7292 | 2.3711 | 0.9264 | 5/5 | 0.2451/0.6547 |

Pairwise truth IoU spans 0.6193–0.7292. Every truth has five connected components, and matched native-body centroids differ by less than 0.66 voxel at worst. Together with the nearly equal diagonal/off-diagonal transfer AUPRC, this is direct evidence that the wrong-case gate is confounded by a shared target-location/geometry template.

## Diagnostic 2 — prior-only versus post-seismic bridge

| case | AP prior | AP post | ΔAP seismic | Brier prior | Brier post | ΔBrier seismic | Pearson prior/post | Spearman prior/post | MAD |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 01 | 0.2504 | 0.2522 | +0.0019 | 0.0303 | 0.0101 | +0.0202 | 0.7585 | 0.9388 | 0.0571 |
| 02 | 0.1854 | 0.2812 | +0.0958 | 0.0284 | 0.0104 | +0.0180 | 0.7420 | 0.9335 | 0.0556 |
| 03 | 0.1410 | 0.1654 | +0.0243 | 0.0294 | 0.0115 | +0.0179 | 0.7519 | 0.9449 | 0.0535 |

AP and Brier improve in all three cases. The AP gain is heterogeneous and almost zero in case 01, while ROC-AUC decreases in all three cases; therefore the evidence is incremental, not a claim of uniformly better ranking. Probability changes are nontrivial (MAD 0.0535–0.0571; RMS 0.1207–0.1237), despite high rank similarity.

## Diagnostic 3 — cross-case bridge-map similarity

Prior-only pairwise Pearson spans 0.7923–0.8261; post-seismic Pearson spans 0.6938–0.7342. Prior-only cosine spans 0.8342–0.8602; post-seismic cosine spans 0.7185–0.7548. Post-seismic maps are therefore less cross-case correlated than the priors, consistent with observation-dependent changes. Their smaller pairwise MAD reflects lower/sparser probability mass and is not interpreted alone.

## Answers to the two mechanism questions

**Q1.** Yes. There is strong evidence that the Stage10 wrong-case control is confounded by the three StructuralGeo truths sharing almost the same five-body location template. This diagnoses the control but does not invalidate or replace its frozen FAIL result.

**Q2.** The prior already contains a strong shared spatial template, but the correct seismic assimilation adds incremental label-9 information: ΔAP and ΔBrier are positive in 3/3 cases, and post maps show observation-dependent changes. The evidence is not consistent with a pure `PRIOR_TEMPLATE_DOMINATED` interpretation.

## Recommended next research branch (not executed)

After manual approval, use an independently pre-registered benchmark whose target anchors, locations, orientations, and body geometries are deliberately diverse, with no outcome-based case selection. Re-evaluate the probability-bridge concept only on that independent benchmark. Do not proceed directly to Stage10-B on the current three cases.
