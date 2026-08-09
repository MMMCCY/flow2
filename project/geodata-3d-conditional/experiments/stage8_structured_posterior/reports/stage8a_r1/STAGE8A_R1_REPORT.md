# Stage 8A-R1 — Birth Basin / Reachability Audit

**Primary category: `RANDOM_BIRTH_BASIN_MISS`**

Stage8A v1 remains frozen as `FAIL_STAGE8A_STOP_BEFORE_STAGE8B`. This audit did not rerun search, call the forward model, train, change the 961-call budget, tune a search hyperparameter, or run Stage8B.

## Scope and trace integrity

All four correct-arm histories contain 961 forward-evaluated states: one empty reference plus 960 noninitial proposals. Because proposal states and hard losses were complete, R1 analyzed the frozen records and performed no replay. Delta distributions below use the 960 noninitial proposals; the machine summary also records the 961-state distribution including the zero-delta initial state.

## Frozen Stage8A proposal audit

| case | empty RMSE | min Δ | p01 | p05 | median | p95 | Δ<0 | zero-edit | best nonempty Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| analytic_five_body | 0.008885824122 | 0 | 0 | 0 | 0.00193716865 | 0.004527129047 | 0 | 80 | 0.0009091431275 |
| native_seed20260807 | 0.01467432547 | 0 | 0 | 0 | 0.001237764955 | 0.002957485244 | 0 | 80 | 0.0006267968565 |
| native_seed20260808 | 0.01435015257 | 0 | 0 | 0 | 0.001191792544 | 0.002993235271 | 0 | 80 | 0.0005114926025 |
| native_seed20260809 | 0.01621644013 | 0 | 0 | 0 | 0.001086978242 | 0.002735350747 | 0 | 80 | 0.0005304124206 |

No correct arm contained a loss-improving proposal. In every arm the best proposal regardless of acceptance was a zero-edit death state exactly tied with empty; the best nonempty state was strictly worse. Therefore an improving state was not lost by acceptance/retention logic.

### Changed-voxel and best-state audit

| case | changed min | p01 | p05 | median | p95 | max | mean |
|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_five_body | 0 | 0 | 0 | 406 | 1151.05 | 1612 | 487.6864583 |
| native_seed20260807 | 0 | 0 | 0 | 398 | 1124.35 | 1722 | 493.6677083 |
| native_seed20260808 | 0 | 0 | 0 | 396 | 1108 | 1743 | 483.8197917 |
| native_seed20260809 | 0 | 0 | 0 | 400 | 1114.15 | 1631 | 489.5833333 |

| case | best proposal trace/state | move | bodies | hard RMSE | Δ | best nonempty state | best nonempty Δ |
|---|---|---|---:|---:|---:|---|---:|
| analytic_five_body | 98 / `g002_p000001_g001_p000055_empty` | death | 0 | 0.008885824122 | 0 | `g010_p000003_g009_p000068_g008_p000031_g007_p000045_g006_p000019_g005_p000009_g004_p000001_g003_p000016_g002_p000007_g001_p000055_empty` | 0.0009091431275 |
| native_seed20260807 | 98 / `g002_p000001_g001_p000078_empty` | death | 0 | 0.01467432547 | 0 | `g009_p000001_g008_p000001_g007_p000046_g006_p000019_g005_p000052_g004_p000025_g003_p000067_g002_p000031_g001_p000086_empty` | 0.0006267968565 |
| native_seed20260808 | 98 / `g002_p000001_g001_p000070_empty` | death | 0 | 0.01435015257 | 0 | `g003_p000072_g002_p000037_g001_p000057_empty` | 0.0005114926025 |
| native_seed20260809 | 98 / `g002_p000001_g001_p000094_empty` | death | 0 | 0.01621644013 | 0 | `g001_p000094_empty` | 0.0005304124206 |

### Move and body-parameter distributions

Each arm recorded 560 births and 80 each of death, translate, resize, rotate, and change-shape. All 80 zero-edit proposals per arm were death moves.

| case | body instances | dike_hemisphere | ellipsoid | material 9 |
|---|---:|---:|---:|---:|
| analytic_five_body | 960 | 526 | 434 | 960 |
| native_seed20260807 | 960 | 475 | 485 | 960 |
| native_seed20260808 | 960 | 485 | 475 | 960 |
| native_seed20260809 | 960 | 485 | 475 | 960 |

The numeric body distributions below count every body instance in every noninitial proposal state (including inherited bodies in two-body states).

| case | parameter | min | p01 | p05 | median | p95 | max | mean |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| analytic_five_body | center_x | 4.054388655 | 4.407166229 | 6.216055109 | 30.92570024 | 56.43097296 | 59 | 31.62123463 |
| analytic_five_body | center_y | 4 | 4.229812381 | 6.720781981 | 30.64588817 | 57.24072297 | 59 | 30.67971656 |
| analytic_five_body | center_z | 6 | 6.150413713 | 6.746660323 | 23.665628 | 47.15084183 | 49.79886434 | 25.97250845 |
| analytic_five_body | size_x | 6 | 6.065834007 | 6.20576636 | 9.003162443 | 14.97035231 | 15.96877813 | 9.635955964 |
| analytic_five_body | size_y | 6 | 6.083901728 | 6.122992295 | 8.888854795 | 15.03698629 | 15.9898951 | 9.599693544 |
| analytic_five_body | size_z | 6 | 6.1048047 | 6.403338043 | 9.64331524 | 13.53773837 | 13.97950988 | 9.826901794 |
| analytic_five_body | orientation_deg | 0 | 3.03484427 | 10.4351562 | 85.83696912 | 173.2447732 | 180 | 88.78408707 |
| native_seed20260807 | center_x | 4.180141585 | 4.67225745 | 6.906820953 | 31.97662136 | 56.40488595 | 59 | 31.96340762 |
| native_seed20260807 | center_y | 4 | 4.54925379 | 6.032887648 | 30.59531978 | 57.29422221 | 58.93823113 | 30.57060373 |
| native_seed20260807 | center_z | 6 | 6.456674743 | 8.654894848 | 29.58716391 | 47.6701828 | 49.7263749 | 28.42762666 |
| native_seed20260807 | size_x | 6 | 6.02123489 | 6.174814643 | 9.088840975 | 15.08088027 | 15.99427779 | 9.790906672 |
| native_seed20260807 | size_y | 6 | 6.120485381 | 6.283576581 | 8.897404307 | 15.02602892 | 15.95863381 | 9.6360472 |
| native_seed20260807 | size_z | 6.000857113 | 6.150714255 | 6.589276808 | 9.536621116 | 13.34796268 | 13.98684276 | 9.706961955 |
| native_seed20260807 | orientation_deg | 0 | 1.891716516 | 7.324686151 | 98.86056398 | 170.0943514 | 179.6100803 | 92.15231233 |
| native_seed20260808 | center_x | 4 | 4.740119921 | 7.1296866 | 30.06241928 | 56.28420393 | 58.92885547 | 29.80640031 |
| native_seed20260808 | center_y | 4.098977662 | 5.433729696 | 7.953217645 | 33.12621608 | 55.34463192 | 59 | 32.37117126 |
| native_seed20260808 | center_z | 6.131793799 | 6.753405291 | 9.809671458 | 25.56017525 | 48.13941147 | 49.9371639 | 26.88287532 |
| native_seed20260808 | size_x | 6 | 6.009886696 | 6.187586306 | 8.174402382 | 15.14466669 | 15.96772485 | 9.487398254 |
| native_seed20260808 | size_y | 6 | 6.043456498 | 6.147021348 | 7.993371045 | 15.0645974 | 15.97143919 | 9.347410345 |
| native_seed20260808 | size_z | 6 | 6.164295168 | 6.664393319 | 10.2265406 | 13.55753755 | 13.97062158 | 10.13282627 |
| native_seed20260808 | orientation_deg | 0 | 0.05915251824 | 1.71809448 | 92.26178891 | 170.9593195 | 180 | 89.9710405 |
| native_seed20260809 | center_x | 4 | 4.158985836 | 7.942772765 | 28.3787107 | 56.04699538 | 59 | 30.29597192 |
| native_seed20260809 | center_y | 4.014202181 | 4.157227608 | 7.0992924 | 32.74880005 | 57.2909844 | 59 | 32.52676211 |
| native_seed20260809 | center_z | 6 | 6.07846827 | 7.469761797 | 28.11213389 | 47.78018875 | 49.98999101 | 27.73364123 |
| native_seed20260809 | size_x | 6 | 6.099584582 | 6.213857721 | 8.428949847 | 15.15751179 | 15.91846362 | 9.42787161 |
| native_seed20260809 | size_y | 6 | 6.039683077 | 6.235002155 | 9.057812303 | 15.10828296 | 15.99883942 | 9.811704209 |
| native_seed20260809 | size_z | 6 | 6.058780976 | 6.374639563 | 9.387380368 | 13.43609728 | 13.98290016 | 9.808599644 |
| native_seed20260809 | orientation_deg | 0 | 2.927864306 | 9.586362959 | 95.86180991 | 168.4426617 | 180 | 94.86405041 |

Move-conditioned loss and changed-voxel distributions, plus every serialized proposal body, are retained in the machine-readable tables and JSON.

## Frozen Stage7 12-candidate barrier audit

The loss-only table was frozen and hashed before the truth-index configuration was opened. It contains 79 unique states: one empty, 12 singletons, and all 66 unordered pairs (the source trace has 132 directed pair evaluations). The loss-selected pair is `candidate_04 + candidate_06` with hard RMSE 0; retrospective truth annotation subsequently confirmed that this is the benchmark truth pair.

| path | empty | singleton | pair | strictly improving |
|---|---:|---:|---:|:---:|
| empty -> candidate_04 -> candidate_04+candidate_06 | 0.008885824122 | 0.006283232477 | 0 | True |
| empty -> candidate_06 -> candidate_04+candidate_06 | 0.008885824122 | 0.006283220369 | 0 | True |

Thus the exact solution is reachable by either of two strictly monotonically improving single-body-addition paths. It does not require crossing a single-birth energy barrier or proposing both bodies simultaneously.

## ORACLE_POSTMORTEM_ONLY local basin-width diagnostic

This read-only diagnostic used the 62 already-recorded direct local mutations of the loss-selected Stage7 zero-loss pair. It made no new forward calls and is not inference success, gate evidence, proposal selection, or hyperparameter tuning. All 62/62 variants remained better than empty; 34 were still exactly zero after voxelization. Even the worst recorded local variant had Δ versus empty -0.0004790229723. The recorded perturbations include center/size changes of ±0.25 and ±1 voxel, rotations of ±5° and ±15°, removals, shape changes, and material changes. This evidence rejects a too-narrow hard-loss basin for the analytic solution at the tested local widths.

## Classification

The primary category is `RANDOM_BIRTH_BASIN_MISS`. The implementation did not discard an improving proposal (none existed), Stage7 shows no greedy birth barrier, and its local basin is not narrow at the recorded perturbations. Stage8A's uniform global births simply never entered an improving basin in any correct arm.

The analytic Stage8A shape list omitted `cuboid`, so a parameterization mismatch is a real secondary limitation for exact analytic representation. It is not the primary cross-case explanation: all three native fixtures are `DikeHemisphere` cases within the configured family, yet they show the same no-improving-birth pattern.

## Exactly one minimal truth-blind recommendation

Replace uniform global **birth-center selection** with a deterministic residual/sensitivity-ranked birth-center initializer, while replacing (not adding to) the existing birth proposals so the 961-call budget and all other frozen search settings remain unchanged. The initializer may use only the observed seismic residual, acquisition/domain geometry, and condition masks; it must not use truth geometry, truth candidate indices, retrospective metrics, or seed selection.

This is a recommendation only. No Stage8A v2 code or sweep was implemented.

## Machine-readable artifacts

- `stage8a_r1_summary.json`
- `stage8a_correct_proposals.csv`
- `stage8a_correct_arm_summary.csv`
- `stage8a_move_summary.csv`
- `stage8a_body_parameters.csv`
- `stage7_library_losses_frozen.csv`
- `stage7_library_truth_posthoc.csv`
- `oracle_postmortem_local_basin.csv`
