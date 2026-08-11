# Stage 12B — Full-StructuralGeo Geophysical Probability-Bridge Validation

## Final machine decision

`STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC`

Stage12B-A failed the prospectively specified bridge-only gate. Stage12B-B was therefore not authorized and was not executed. Stage12B Flow forwards: **0**.

The five prospectively registered Stage12A cases were retained unchanged. No case was replaced using seismic, bridge, Flow, or truth metrics. The frozen Stage10 machine decision was not modified or reinterpreted.

## Repository and execution provenance

- Branch: `main`
- Git HEAD before and after the run: `72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f`
- Worktree before the run: dirty, containing pre-existing Stage9–Stage12A work and StructuralGeo changes; these changes were preserved.
- Runtime: project `.venv`; fixed property inversion was run through SSH on `LabRS` with the available CUDA GPU.
- Protocol: `configs/frozen_protocol.json`, frozen before Stage12B seismic generation.

The 12-member property-prior bank was frozen before any new Stage12B observation was generated. It is the common, unranked Stage10 candidate range 100–111 from the registered `native_seed20260901` Stage9A formal pool. No new Flow sampling was performed. This bank is deliberately recorded as a sensitivity prior: its source hard models were originally conditioned on a different registered synthetic case, after which each Stage12B case's exact surface and borehole acoustics were overwritten before inversion.

## 1. Pre-geophysics descriptive geometry audit

Raw-label9 geometry was measured before synthetic seismic generation and was never used to change the cohort.

| case | label9 voxels | centroid (x, y, z) | CC-6 | CC-26 |
|---|---:|---|---:|---:|
| fullgeo_case01 | 120 | (6.73, 48.99, 2.55) | 6 | 4 |
| fullgeo_case02 | 16,083 | (30.84, 32.98, 20.94) | 71 | 16 |
| fullgeo_case03 | 1,736 | (16.58, 19.19, 15.05) | 82 | 2 |
| fullgeo_case04 | 13,293 | (44.04, 35.29, 22.15) | 3 | 2 |
| fullgeo_case05 | 4,111 | (40.61, 36.02, 7.57) | 34 | 4 |

Across the 20 directed off-diagonal pairs, mean raw-mask IoU was **0.00926**, maximum IoU was **0.03890**, mean centroid distance was **27.79 voxels**, and the minimum centroid distance was **13.45 voxels**. The mean symmetric volume ratio was **0.2167**. These cases are substantially more diverse in position, size, and component geometry than a shared-template cohort.

Full pairwise results are in `geometry/truth_similarity.csv`, with separate IoU, centroid-distance, and volume-ratio matrices in the same directory.

## 2. Synthetic seismic and property inversion

For each frozen truth, the unchanged Stage4 upper-bound codebook and forward operator generated a noiseless full-64×64 observation. The truth-generation process then closed; downstream inversion and bridge builders loaded only the copied inference-visible conditions, support mask, codebook, and observed seismic. Every observation and tensor has both file and content hashes in its case manifest under `observations/<case>/manifest.json`.

The unchanged Phase5a linearized post-stack log-impedance Tikhonov inversion processed all 12 frozen prior members per case. It preserved exact surface/borehole properties with zero violations. Mean seismic RMSE across the 12 members changed as follows:

| case | prior RMSE | inverted RMSE |
|---|---:|---:|
| fullgeo_case01 | 0.043741 | 0.028920 |
| fullgeo_case02 | 0.064597 | 0.050700 |
| fullgeo_case03 | 0.055094 | 0.042411 |
| fullgeo_case04 | 0.058973 | 0.045834 |
| fullgeo_case05 | 0.054374 | 0.041653 |

Prior/post log-impedance samples, means, population spreads, all-class probabilities, label9 probabilities, and entropies are frozen under `bridge/<case>/`. No nearest-codebook conversion, temperature, smoothing, sharpening, or truth-tuned threshold was used.

## 3. Complete 5×5 transfer matrix

Primary statistic: per-case AUPRC on each truth case's unconstrained subsurface; no pooled-voxel primary statistic was used.

| bridge \ truth | case01 | case02 | case03 | case04 | case05 |
|---|---:|---:|---:|---:|---:|
| case01 | **0.000692** | 0.105367 | 0.012473 | 0.073079 | 0.019607 |
| case02 | 0.000683 | **0.102298** | 0.012395 | 0.073049 | 0.018541 |
| case03 | 0.000686 | 0.100692 | **0.012312** | 0.074351 | 0.019701 |
| case04 | 0.000688 | 0.104475 | 0.012307 | **0.075046** | 0.019110 |
| case05 | 0.000683 | 0.103712 | 0.012217 | 0.073597 | **0.018486** |

- Diagonal mean AUPRC: **0.0417669**
- Off-diagonal mean AUPRC: **0.0418706**
- Diagonal minus off-diagonal: **−0.0001038**
- Diagonal row maximum: **1/5**
- Diagonal column maximum (reported, not gated): **2/5**

The Brier and secondary ROC-AUC transfer matrices are stored as `evaluation/stage12b_a/transfer_brier_matrix.csv` and `transfer_roc_auc_matrix.csv`.

## 4. Correct bridge, controls, and prior-only comparison

| case | post AP | prior AP | ΔAP | shuffled AP | constant AP | post Brier | prior Brier | prior−post Brier |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| case01 | 0.000692 | 0.000676 | +0.000016 | 0.000662 | 0.000783 | 0.010063 | 0.042931 | +0.032868 |
| case02 | 0.102298 | 0.103452 | −0.001154 | 0.103332 | 0.077445 | 0.077761 | 0.090711 | +0.012950 |
| case03 | 0.012312 | 0.012221 | +0.000092 | 0.011210 | 0.011525 | 0.019949 | 0.053797 | +0.033848 |
| case04 | 0.075046 | 0.074467 | +0.000579 | 0.076395 | 0.077164 | 0.081194 | 0.101250 | +0.020056 |
| case05 | 0.018486 | 0.017921 | +0.000564 | 0.018548 | 0.018586 | 0.024370 | 0.048227 | +0.023857 |

Macro ΔAP was **+0.0000194**. Post-seismic AP exceeded prior-only AP in **4/5** cases, but the increments were very small and case02 decreased. Brier improved in all five cases, with macro prior-minus-post Brier **+0.024716** and no catastrophic degradation. Much of that Brier benefit coincides with a broad reduction of mean label9 probability toward these sparse prevalences; it does not establish case-specific localization by itself.

## 5. Prospective Stage12B-A gate

| clause | result |
|---|---|
| diagonal mean AP > off-diagonal mean AP | FAIL |
| at least 4/5 diagonal row maxima | FAIL (1/5) |
| at least 4/5 correct > shuffled | FAIL (2/5) |
| 5/5 correct > constant | FAIL (2/5) |
| at least 4/5 post AP > prior AP | PASS (4/5) |
| no catastrophic Brier degradation | PASS (5/5) |

Column maximum is reported above but is not a machine-gate clause, matching the user protocol. A serialization note documents that an unused column threshold was accidentally placed under `success_gate` in the pre-seismic JSON; the official evaluator did not apply it, and it has no decision impact because four explicitly authorized clauses already failed.

## 6. Scientific interpretation

The Stage12B cohort does **not** support shared hidden-body geometry as the explanation for wrong-case failure: cross-case label9 overlap is low, centroids are separated, and volumes/component counts vary strongly. Nevertheless, the post-seismic maps are not reliably matched to their own truth; diagonal and off-diagonal macro AP are essentially equal, only one diagonal is a row maximum, and correct maps beat shuffled and constant controls in only two cases each.

There is limited evidence that seismic assimilation changes categorical probabilities constructively: 4/5 AP deltas are positive and all Brier scores improve. The magnitude and specificity evidence are insufficient, however, to validate a label9-specific probability bridge. The formal result is therefore not a Flow-interface failure; the bridge itself failed before Flow was authorized.

## 7. Stop and recommended next branch

Stage12B stops here. No paired Flow pilot, additional seeds, parameter sweep, D-Flow, SMC, retraining, adapter training, additional wells, or replacement benchmark cases were run. Because Stage12A supplied diverse geometry while Stage12B-A still failed case specificity, the recommended next research branch is to **close the current frozen scalar log-impedance probability-bridge route** and, only under a new prospective protocol, investigate a more identifiable observation model or richer petrophysical likelihood before any further Flow-guidance experiment.

The pass-conditional geophysical diagnostic and Flow paper main figure were intentionally not generated.
