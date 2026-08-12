# flow2 paper result facts

> Machine-grounded writing aid generated from frozen reports. It is not manuscript prose and does not make cross-protocol leaderboard claims.

## SAFE CLAIMS

- Frozen conditional Flow is strongly controllable under truth-derived 3-D probability evidence: mean label-9 IoU changed from 0.0314 to 0.8099 across 12 strict pairs.
- Structured hard-geophysics inference ranks the correct observation first in all three registered StructuralGeo-native replicas.
- Stage 9A demonstrates limited unrestricted frozen-Flow prior support for the three registered hidden-target tasks: support and discrimination both pass in 0/3 cases.
- Current seismic-derived probability guidance through frozen Flow does not improve hidden label-9 recovery on the five Full StructuralGeo cases: overall median paired delta is -0.031508, with one positive case median.
- All tested hard surface and borehole conditions remain exact in the reported Phase 1, Phase 2a, Stage 7, and Stage 14 evaluations.

## QUALIFIED CLAIMS

- Synthetic seismic contains sufficient information to discriminate hidden structures inside the bounded Stage-7 hypothesis family (hidden IoU range 0.9142-0.9874); this is structured search, not a CFM posterior.
- Acquisition-domain geophysical constraints remain limited by observability and representation in the current synthetic setup.
- Ideal, truth-derived 3-D properties provide a physical controllability upper bound (mean label-9 IoU 0.0314 to 0.4808), not evidence for realistic property inversion.
- Stage 12B shows small constructive probability changes in some metrics, but fails the prospective case-specificity gate and therefore does not validate the bridge.
- Full StructuralGeo cases are independently generated and prospectively registered from the same recipe, but historical sample-level non-overlap with the streaming training run cannot be certified.

## PROHIBITED CLAIMS

- CFM seismic inversion recovers hidden geology with IoU 0.987.
- Phase 1 IoU 0.81 is a measured-geophysics result.
- The method provides a calibrated Bayesian posterior.
- Stage 7 truth was used for candidate selection.
- Current results demonstrate field-data generalization.
- Results from Phase 1, Phase 2, Stage 7, Stage 9, Stage 12, and Stage 14 form a shared-case head-to-head leaderboard.

## A. Abstract-ready metrics

1. Oracle 3-D probability guidance: label-9 IoU 0.031 to 0.810 over 12 paired runs.
2. Ideal property guidance: label-9 IoU 0.031 to 0.481 over 12 paired runs.
3. Structured seismic: hidden IoU 0.914-0.987; correct observation ranks first 3/3.
4. Frozen-Flow prior audit: 3072 unique models, support 0/3 and discrimination 0/3.
5. Probability-guided Flow: median paired hidden-label9 IoU delta -0.0315; positive case medians 1/5.

## B. Introduction-ready problem statements

- Sparse surface and borehole observations leave substantial ambiguity in hidden 3-D target geometry.
- Controllability under privileged geological evidence does not imply observability from acquisition-domain geophysics.
- Geophysical consistency must be assessed after hard categorical decoding, because soft physical improvement can fail to transfer across categorical boundaries.
- Inference parameterization and proposal support determine whether an informative observation can affect the frozen generator.

## C. Methods facts

- Model resolution: 64 x 64 x 64 categorical voxels.
- Geological conditioning: the categorical surface plus nine prospectively fixed vertical boreholes.
- Inference uses frozen Flow weights; no checkpoint update is performed in the reported guidance experiments.
- Continuous embeddings are integrated and then hard-decoded to categorical geology before geological and physical evaluation.
- Label 9 is the registered pressure-test target; raw labels 10-13 are not merged into it in the Full StructuralGeo cohort.

## D. Results facts

### 3.1 Privileged-evidence controllability

- Phase 1: global accuracy 0.5972 to 0.6432; global mIoU 0.1804 to 0.2178; label-9 IoU 0.0314 to 0.8099.
- Phase 2a: truth-present mIoU 0.2771 to 0.3443; hard-property loss 1.4781 to 0.5187.

### 3.2 Acquisition-domain observability and structured inference

- Direct seismic guidance: maximum/final soft attainment 0.1909/0.1909, but maximum/final hard attainment 0.0244/-0.0080.
- Stage 7: mean correct hard attainment 0.4270; hidden recall range 0.9160-0.9958; correct rank first 3/3.

### 3.3 Prior support and probability bridge

- Stage 9A: 1024 frozen-Flow samples per case and 98304 Flow velocity forwards; support/discrimination 0/3.
- Stage 12B: diagonal/off-diagonal mean AUPRC 0.041767/0.041871; diagonal row maximum 1/5.
- Stage 14 per-case median hidden-IoU deltas: fullgeo_case01 -0.003025, fullgeo_case02 -0.093144, fullgeo_case03 +0.001386, fullgeo_case04 -0.166759, fullgeo_case05 -0.031508.

## E. Limitations

- All geophysical observations are synthetic; no field validation has been performed.
- The seismic studies use an inverse-crime forward configuration where stated.
- The synthetic acoustic codebook deliberately gives label 9 distinctive impedance and is not site-calibrated petrophysics.
- Phase-1 probability and Phase-2a property inputs are truth-derived oracle/upper-bound evidence.
- Stage-7 structured search is a bounded inference mechanism, not a posterior sampled by CFM.
- Protocols, cases, observations, and inference spaces differ across stages; their metrics are not directly exchangeable leaderboard scores.
- Full StructuralGeo training-sample non-overlap cannot be certified because the historical streaming run retained no seed/sample manifest.

## F. Recommended future work

1. Establish a more identifiable observation model and realistic petrophysical likelihood under a new prospective protocol.
2. Develop and validate a learned seismic-to-geology evidence representation before reconnecting it to Flow guidance.
3. Consider D-Flow or source optimization only after the observation representation passes an independent case-specificity gate.
4. Add noise, survey incompleteness, petrophysical ambiguity, and external/field validation without altering the frozen diagnostic conclusions.

## Source policy

Every number above is generated from `paper/figure_data/paper_evidence_summary.json`; consult each metric's `source.path` and `source.json_key` before reuse.
