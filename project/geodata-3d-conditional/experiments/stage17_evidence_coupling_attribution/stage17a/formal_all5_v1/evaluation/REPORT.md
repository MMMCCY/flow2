# Stage17A multi-case evidence report

Decision: **EVIDENCE_CASE_SPECIFIC**

Primary specificity uses prevalence-corrected AP skill; raw AUPRC is auxiliary.

Positive diagonal advantages: 5/5; group median Delta skill: 0.558335134.

This report is retrospective. The inversion runner did not load truth or run Flow.

| Case | Raw AUPRC | Prevalence | Within-case AP skill | Delta AP skill |
|---|---:|---:|---:|---:|
| fullgeo_case01 | 0.652938 | 0.000783 | 0.652666 | 0.666678 |
| fullgeo_case02 | 0.470053 | 0.077445 | 0.425566 | 0.378922 |
| fullgeo_case03 | 0.590627 | 0.011525 | 0.585854 | 0.589557 |
| fullgeo_case04 | 0.571510 | 0.077164 | 0.535681 | 0.506017 |
| fullgeo_case05 | 0.562901 | 0.018586 | 0.554623 | 0.558335 |

The primary specificity mask is the 124,350-voxel intersection of all five
unconditioned original-nonair domains. `ap_skill_specificity_matrix.csv` is
primary; `raw_auprc_specificity_matrix.csv` is mandatory auxiliary output.
The failed observation-v1 directory is preserved. Observation v2 froze a
column-contiguous seismic support rule before successful generation; the
evidence config was created only after all observation files were hashed.
