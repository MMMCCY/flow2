# Stage20 Continuous-Impedance Adapter Formal TEST Report

## Final machine decision

`CONTINUOUS_EVIDENCE_COUPLING_NOT_VALIDATED`

The frozen decision order stops scientifically at Gate A. The continuous-impedance adapter did not improve label-9 IoU over the frozen Flow prior under the preregistered case-first TEST12 criteria. No Stage20 v1 parameter, asset, output, threshold, or cohort was changed after unblinding.

## Scientific question

This retrospective TEST asks whether a learned residual-velocity adapter driven by continuous Phase5a inverted impedance improves hard categorical geology over the frozen Flow prior, whether any improvement is incremental to the nine-well low-frequency prior and case-specific, and whether it is supported by the frozen acquisition-domain seismic operator.

## Provenance and pre-truth firewall

The complete truth-blind audit passed before the first TEST truth dereference, establishing `retrospective_truth_access_authorized = true`.

| Frozen asset | SHA256 |
|---|---|
| Source git head used by inference | `24abcd29224976c44cfebe72f7fa7f66dbcc32b0` |
| Formal inference runner | `1bebc207108e6850a97784ca01f26c33ff64413b2c555ab4b3b6c48eae15fe3d` |
| Retrospective evaluator | `c5a95d944ca3a6bf93162dedce8769430fc2d84bde9fe7d8045e0b0ed96557f7` |
| Formal run manifest | `12bbde73d548de8ff5bf46a58a47bdb63bfcb9c9651b3138981347faa3eb401d` |
| Formal output index | `0f1f6534423a0fc600a23fd4d84fba4bd10ecbecc60f2fe33cdf6ebf408d6c7e` |
| Base-state provenance attestation | `32caffc3f038a33a4491249ad89a650563b8754262837c100b7586cfb3ea999a` |
| Formal inference freeze report | `570a271adb55377ce7fc43b871e8f565e6eeca43658264075c381546f2da0422` |
| Base Flow checkpoint | `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c` |
| Formal adapter checkpoint | `f8b71d0889ec8f76f033583be3b94f3041da888c0b910e92c8f686e24a093213` |
| Formal training manifest | `0add2f3ce80b1ea370e9dbf9f3c3a3ce8f9b383ff9ffd3dce4978aaeebc9665a` |
| Frozen inference config | `41c38912b1d49335e3e4ac85a39d86d6ee1d5e0064adbd924a9862d8ae0b88d8` |
| Stripped TEST inference registry | `11c201720328eab972ecebed11b0c31b2348733bd2cce86d280a6e7f2649cba3` |
| Full evidence/evaluation registry | `94cb0cd2e595be1a9d1ae90bfec0fd7fcdeff69bdbd7b00e3e9642cff41deb28` |
| Full observation registry | `de5c22327d305d58f387c77cb3c8385402d97a891bf2cc4dba29b3581f6eaec9` |
| Evidence authorization gate | `475fe63a01146730cd7d092b6640a74ce808b5f3b434d33a736dbb16fd0647f0` |
| Phase4c acoustic codebook | `8980e30c59624af9f1bfa59447b7e22c85a2975f8a556fe6e98f5346f95d10ae` |
| Phase4c seismic configuration | `7ee49c3c1c685126cf9b727e05a895463734b4335300ccbad6d6dd6a865bfd02` |
| Stage20 unified acoustic mapper | `cd7d6ff8860ede78d52c56cc070f6766b71df7d09eb9775761d7f14b6abb1506` |

The firewall revalidated exactly 144 unique records (`12 cases × 3 seeds × 4 arms`), all output file and tensor hashes, 36/36 paired-noise groups, the frozen cyclic wrong-case mapping, current-case masks, sampler/checkpoint/config/runner provenance, and zero hard-condition violations. The base-state attestation decision was `STAGE20_FORMAL_INFERENCE_FROZEN`; `truth_loaded_by_runner` remained `false`.

Pre-truth evaluation initially stopped because the human authorization listed an incorrect freeze-report SHA256 (`3b641...`). Independent file hashing established the actual immutable report SHA256 as `570a271a...0422`. No formal output, inference asset, report content, TEST truth, or scientific metric was modified or inspected during this correction.

## First truth-access declaration

TEST truth was first dereferenced only after the corrected full provenance firewall passed. From that point Stage20 v1 became permanently scientifically unblinded. The evaluator completed once; formal inference was not rerun.

## Formal inventory and arms

- TEST cases: 12, with no exclusion or filtering.
- Source seeds per case: `[42, 142, 242]`.
- Formal outputs: 144.
- `FLOW_ONLY`: frozen Flow runner with adapter scale 0.
- `ADAPTER_PRIOR_ONLY`: current-case normalized low-frequency log-impedance with adapter scale 1.
- `ADAPTER_CORRECT`: current-case normalized Phase5a inverted log-impedance with adapter scale 1.
- `ADAPTER_WRONG_CASE`: cyclic next-case inverted evidence values with current-case masks and adapter scale 1.
- All arms used 32 fixed midpoint-Euler steps and residual cap 0.25.

`PROPERTY_ONLY_NEAREST_LOGZ` is an evaluation-only, one-output-per-case diagnostic. It is not a formal arm and does not enter Gate A–G.

## Case-first methodology

For every formal case/arm pair, each metric was first reduced by the median over its three paired seeds. Gates A–F use only the resulting 12 independent case-level values. The 36 case-seed comparisons are reported separately as secondary diagnostics and never replace the formal case-first gates.

Hard-seismic metrics map each final categorical volume through the unified Stage20 acoustic semantics, frozen full multiclass Phase4c codebook, and frozen Phase4c forward operator. They use hard labels only; no inversion or soft-category probability is used during evaluation.

## Cross-case medians

Each entry is the median across the 12 case-level medians for formal arms; PROPERTY_ONLY is the median across its 12 deterministic case outputs.

| Arm | Label9 IoU | Precision | Recall | Abs volume-error fraction | Centroid distance | Global accuracy | Truth-present mIoU | Components | Largest fraction | Top4 mass | Top8 mass | Hard RMSE | Hard MAE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| FLOW_ONLY | 0.125774 | 0.203456 | 0.226124 | 0.624432 | 9.984104 | 0.671472 | 0.364487 | 102.5 | 0.647620 | 0.877966 | 0.942079 | 0.049867 | 0.015906 |
| ADAPTER_PRIOR_ONLY | 0.088459 | 0.095629 | 0.399830 | 3.336991 | 15.143244 | 0.546736 | 0.290109 | 265.0 | 0.940975 | 0.962094 | 0.968504 | 0.069835 | 0.026187 |
| ADAPTER_CORRECT | 0.088826 | 0.095984 | 0.399631 | 3.336428 | 15.131005 | 0.546817 | 0.290163 | 263.5 | 0.942490 | 0.961897 | 0.968004 | 0.069795 | 0.026183 |
| ADAPTER_WRONG_CASE | 0.088377 | 0.095224 | 0.380779 | 3.431279 | 16.314785 | 0.549452 | 0.293127 | 278.5 | 0.948579 | 0.967265 | 0.974799 | 0.069747 | 0.026435 |
| PROPERTY_ONLY_NEAREST_LOGZ | 0.099038 | 0.820078 | 0.111245 | 0.854064 | 9.110557 | 0.541290 | 0.277220 | 85.0 | 0.448371 | 0.773381 | 0.868883 | 0.033588 | 0.011283 |

## Per-case transparency

Arm cells contain `label9 IoU / truth-present mIoU / absolute volume-error fraction / hard-seismic RMSE`. Deltas are label9 IoU differences using each case's three-seed medians.

| Case | Flow | PriorOnly | Correct | WrongCase | Correct−Flow | Correct−Prior | Correct−Wrong |
|---|---|---|---|---|---:|---:|---:|
| case001 | 0.0000 / 0.1567 / 28.1771 / 0.0390 | 0.0002 / 0.1496 / 267.9062 / 0.0580 | 0.0002 / 0.1496 / 267.8229 / 0.0580 | 0.0002 / 0.1508 / 265.0521 / 0.0568 | +0.000193 | +0.000000 | −0.000041 |
| case002 | 0.0002 / 0.3473 / 0.9616 / 0.0389 | 0.0326 / 0.2461 / 1.8482 / 0.0753 | 0.0328 / 0.2460 / 1.8591 / 0.0753 | 0.0379 / 0.2393 / 2.6988 / 0.0795 | +0.032613 | +0.000234 | −0.005036 |
| case003 | 0.0412 / 0.2998 / 0.5037 / 0.0544 | 0.0222 / 0.2880 / 7.5162 / 0.0721 | 0.0223 / 0.2881 / 7.5250 / 0.0721 | 0.0234 / 0.2913 / 8.0857 / 0.0740 | −0.018953 | +0.000017 | −0.001171 |
| case004 | 0.3690 / 0.4199 / 0.0511 / 0.0424 | 0.1476 / 0.3309 / 2.4589 / 0.0648 | 0.1474 / 0.3309 / 2.4590 / 0.0648 | 0.1340 / 0.3298 / 2.5468 / 0.0662 | −0.221649 | −0.000223 | +0.013374 |
| case005 | 0.2913 / 0.2741 / 0.9507 / 0.0482 | 0.1952 / 0.2361 / 2.2805 / 0.0627 | 0.1952 / 0.2361 / 2.2827 / 0.0626 | 0.1919 / 0.2358 / 2.3008 / 0.0628 | −0.096163 | −0.000013 | +0.003240 |
| case006 | 0.5798 / 0.3816 / 0.3171 / 0.0515 | 0.3643 / 0.2922 / 1.1664 / 0.0688 | 0.3644 / 0.2923 / 1.1668 / 0.0687 | 0.3657 / 0.2949 / 1.1607 / 0.0681 | −0.215418 | +0.000154 | −0.001312 |
| case007 | 0.2557 / 0.2057 / 0.9053 / 0.0619 | 0.1404 / 0.1934 / 2.1843 / 0.0746 | 0.1404 / 0.1934 / 2.1847 / 0.0746 | 0.1321 / 0.1926 / 2.2736 / 0.0760 | −0.115371 | +0.000011 | +0.008309 |
| case008 | 0.1318 / 0.4252 / 1.1656 / 0.0402 | 0.0914 / 0.3680 / 4.2151 / 0.0568 | 0.0914 / 0.3680 / 4.2139 / 0.0568 | 0.0923 / 0.3743 / 4.1638 / 0.0555 | −0.040391 | +0.000043 | −0.000855 |
| case009 | 0.1207 / 0.4117 / 0.1809 / 0.0411 | 0.1082 / 0.3721 / 0.8346 / 0.0570 | 0.1083 / 0.3721 / 0.8340 / 0.0570 | 0.1105 / 0.3764 / 0.7840 / 0.0573 | −0.012451 | +0.000089 | −0.002286 |
| case010 | 0.1308 / 0.2695 / 0.1587 / 0.0539 | 0.0855 / 0.1977 / 5.5997 / 0.0788 | 0.0862 / 0.1978 / 5.6072 / 0.0787 | 0.0845 / 0.1939 / 6.0156 / 0.0790 | −0.044603 | +0.000691 | +0.001753 |
| case011 | 0.0870 / 0.3855 / 0.2868 / 0.0569 | 0.0484 / 0.3168 / 6.8658 / 0.0763 | 0.0483 / 0.3168 / 6.8658 / 0.0764 | 0.0329 / 0.2971 / 9.2177 / 0.0790 | −0.038709 | −0.000149 | +0.015334 |
| case012 | 0.0860 / 0.4085 / 0.7451 / 0.0605 | 0.0347 / 0.3498 / 5.0536 / 0.0709 | 0.0349 / 0.3497 / 5.0626 / 0.0709 | 0.0326 / 0.3460 / 5.3018 / 0.0714 | −0.051065 | +0.000198 | +0.002246 |

No failed or unfavorable case was omitted. Full-precision values remain in `reports/formal_v1/case_arm_primary_summary.csv`.

## Frozen Gate A–G results

| Gate | Formal observation | Frozen requirement | Result |
|---|---|---|---|
| G — hard conditions | 0 violations across 144 outputs | exactly 0 | PASS |
| A — Correct vs Flow IoU | 2/12 improved; median delta −0.042497 | ≥9/12 and median delta > +0.05 | **FAIL** |
| B — Correct vs PriorOnly IoU | 9/12 improved; median delta +0.000030 | ≥8/12 and median delta > +0.02 | FAIL |
| C — Correct vs WrongCase IoU | 6/12 improved; median delta +0.000856 | ≥8/12 and median delta > +0.02 | FAIL |
| D — Correct vs Flow hard RMSE | 0/12 improved; median delta +0.017483 | ≥8/12 and median delta < 0 | FAIL |
| E — volume calibration | Correct 3.336428 vs Flow 0.624432 | Correct cross-case median < Flow | FAIL |
| F — global geology preservation | median Correct−Flow mIoU = −0.058005 | ≥ −0.01 | FAIL |

Gate G passes. Gate A is the first failure in the frozen order, so later failures are diagnostic and do not change the machine decision.

## Secondary 36-sample audit

These paired case-seed comparisons are secondary diagnostics only:

| Comparison | Improved samples |
|---|---:|
| Correct IoU > Flow | 6/36 |
| Correct IoU > PriorOnly | 27/36 |
| Correct IoU > WrongCase | 19/36 |
| Correct hard-seismic RMSE < Flow | 0/36 |

They do not replace the 12-case formal gates.

## PROPERTY_ONLY_NEAREST_LOGZ diagnostic

The evaluation-only baseline was run exactly once per TEST case after truth authorization. Its cross-case medians were label9 IoU 0.099038, precision 0.820078, recall 0.111245, absolute volume-error fraction 0.854064, truth-present mIoU 0.277220, and hard-seismic RMSE 0.033588. It is not one of the 144 formal outputs and enters no gate.

The high precision paired with low recall indicates conservative label-9 classification. Its lower hard-seismic RMSE than all formal generative arms is a diagnostic observation under the favorable inverse-crime setting, not evidence that it satisfies the registered generative geology objective.

## Morphology diagnostics

The Correct arm had a median 263.5 predicted label-9 components versus 102.5 for Flow, while its largest-component fraction was 0.942490 versus 0.647620 and its top-eight mass fraction was 0.968004 versus 0.942079. This combination indicates many additional small components around a dominant predicted mass. Morphology remains diagnostic only and does not alter the decision tree.

## Scientific interpretation

The learned adapter did not validate continuous-evidence coupling on frozen TEST12. Correct evidence improved case-median label9 IoU in only 2 of 12 cases relative to Flow, and the median effect was negative. Correct and PriorOnly results were nearly identical: although 9 cases had a numerically positive Correct−Prior delta, the median increment was only +0.000030, far below +0.02. Correct-versus-Wrong differences were likewise small and not consistently positive, so case specificity was not demonstrated.

The categorical changes were not supported by the registered hard-seismic criterion: all 12 case medians had worse Correct RMSE than Flow, with median delta +0.017483. Correct also had worse volume calibration and truth-present mIoU. These observations are reported without post-unblinding remediation; any revised research protocol must be a separately defined Stage21 or later study.

## Evaluation artifacts

| Artifact | SHA256 |
|---|---|
| `reports/formal_v1/summary.json` | `ab2f50197968cf77e17484ed79fbf79a0a02329b4942245a3ece830f9327970d` |
| `reports/formal_v1/sample_metrics.csv` | `456944538840c57f5a7d14953beee4ae035cf14946fd36f99e9723b24bb6de38` |
| `reports/formal_v1/per_class_iou.csv` | `2501297715d43714ff8a6dcb7effd3e6cc23dc5dc7df6bc8d0a73f26b7a83ab8` |
| `reports/formal_v1/case_arm_primary_summary.csv` | `666fe5536b26cc68acbff1943b5026b064873cf4c39296e5a5478965ebc304b6` |
| `reports/formal_v1/cross_case_summary.csv` | `00c7f34d8482f4a279615e0f937acab70909b21a158fb2438de8b21aa12a61c3` |
| `reports/formal_v1/STAGE20_REPORT.md` | `81c0f6cb34103f1fdd1c2a97d07182d6d228fedc674e7dae7036e66d14df2d69` |

The per-output file contains 144 formal metric rows plus 12 PROPERTY_ONLY rows. The per-class artifact contains the class-wise IoU/precision/recall and truth/predicted volume records.

## Limitations

- Results are restricted to the StructuralGeo synthetic distribution.
- Petrophysics use a synthetic frozen multiclass acoustic codebook.
- Observations are noiseless.
- The evaluation is an inverse-crime setting.
- The controlled petrophysical mapping is favorable relative to field conditions.
- The low-frequency borehole prior is deterministic.
- Training and TEST cases come from the same synthetic generator family.
- There is no realistic acquisition mismatch.
- Petrophysical uncertainty is not calibrated.
- No field-generalization or field-readiness claim is made.
- No exact Bayesian posterior claim is made.

## Final declaration

Stage20 v1 is complete and permanently unblinded. No formal inference was rerun, no frozen output was modified, no case was removed, and no post-TEST training, tuning, threshold change, or alternative Stage20 v1 result was created. Work stops here; Stage21 is not started automatically.
