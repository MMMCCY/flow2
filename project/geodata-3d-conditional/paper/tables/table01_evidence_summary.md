# Table 1. Quantitative evidence summary

Compact quantitative evidence summary for flow2. Results summarize distinct diagnostic protocols and are not a single shared-case leaderboard.

| Evidence / inference | Observation type | Truth used at inference? | Target-geology result | Physical result | Primary interpretation |
| --- | --- | --- | --- | --- | --- |
| Sparse CFM | Surface + 9 boreholes | No | Label-9 IoU 0.031 | Not used | Sparse conditions under-constrain target geometry. |
| Oracle probability guidance | Truth-derived 3-D P(label 9) | Yes (oracle) | Label-9 IoU 0.031 -> 0.810 | Not acquisition-domain | Strong controllability upper bound. |
| Ideal property guidance | Truth-derived noiseless 3-D properties | Yes (upper bound) | Label-9 IoU 0.031 -> 0.481 | Hard-property loss 1.478 -> 0.519 | Ideal property controllability is demonstrated. |
| Structured hard-seismic inference | Synthetic hard seismic | No | Hidden IoU 0.914-0.987 | Correct ranks first 3/3 | Seismic discriminates bounded hypotheses. |
| Unrestricted frozen-Flow ranking | Synthetic hard seismic | No; oracle audit after ranking | Support 0/3; discrimination 0/3 | 3072 unique models ranked | Frozen-prior support is limiting. |
| FullGeo probability bridge | Synthetic seismic -> P(label 9) | No | Diagonal/off-diagonal AP 0.0418/0.0419 | Diagonal row maximum 1/5 | Current bridge lacks case specificity. |
| Seismic-probability-guided Flow | Synthetic post-seismic P(label 9) | No | Median change in hidden IoU -0.0315; positive 1/5 | 0 hard-condition violations | Current end-to-end bridge is insufficient. |

Notes:
1. Results summarize distinct diagnostic protocols and are not a single shared-case leaderboard.
2. Oracle metrics are privileged controllability upper bounds, not measured-geophysics results.
3. Direct gradient guidance is summarized in text: soft seismic attainment reached 0.1909, while maximum/final hard attainment was 0.0244/-0.0080; the first reproducible soft-hard divergence occurred at reflectivity/TWT.
4. Stage-9 oracle-best candidates are retrospective support ceilings and are not deployable selectors.
5. All seismic experiments are synthetic; no field-data validation is claimed.
