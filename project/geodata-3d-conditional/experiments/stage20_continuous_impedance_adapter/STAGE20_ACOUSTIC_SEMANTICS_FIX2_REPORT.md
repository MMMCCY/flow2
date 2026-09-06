# Stage20 acoustic semantic contract completion report

## Decision

`ADAPTER_TRAINING_AUTHORIZED`

This is a Stage20 pre-training acoustic semantic contract completion, not a new scientific experiment result. CUDA smoke, adapter training, formal inference, and TEST-metric evaluation were not executed.

Repository state used: branch `main`, commit `2284d7235cd62295d741c3982eb5063fe6577ac6`.

## Audit and root cause

The repository matched the frozen prompt assumptions, so the authorized unique repair was applicable. The geological condition tensors legitimately contain raw label `-1` at some frozen subsurface condition voxels. The old LF exact-condition overwrite and Phase5a condition-target path mapped those labels through the Phase4c mapper directly, interpreting them as real air impedance `416.5`. Free-subsurface impedance was IDW/inversion reconstructed and bounded, while condition voxels were exactly overwritten; this is why `416.5` appeared only in `condition & subsurface`.

The independent truth-blind diagnostic exactly matched the preregistered hypothesis:

- `stage19_val_case001`: 15 geological bad-mask voxels, 15 LF out-of-bound voxels, 15 inverted out-of-bound voxels; coordinate sets identical.
- `stage19_val_case005`: 58, 58, and 58 respectively; coordinate sets identical.
- The other six VAL cases: 0 in all three sets.

No truth, IoU, target label, or TEST metric was used by the diagnostic.

## Implementation

The unique `stage20_labels_to_acoustic(labels, subsurface_mask, property_table)` mapper was added in `scripts/stage20/acoustic_semantics.py`. It does not mutate labels; maps support-exterior raw `-1` to Phase4c real air; maps support-interior raw `-1` through `neutral_rock_category(property_table)`; preserves raw labels `0..13`; and strictly checks finite values, positive slowness, and frozen rock impedance bounds.

The same mapper is now used by:

1. truth geology to synthetic observation;
2. geological conditions to exact acoustic conditions and LF exact overwrites;
3. hard generated geology to predicted seismic;
4. retrospective truth geology to truth acoustic/logZ evaluation.

The frozen 9-well same-depth IDW (`power=2`) and nearest-valid-depth zero-order fallback are unchanged. Phase5a inversion weights remain `0.001` and `0.01`; recording window, wavelet, codebook, and scientific gate thresholds are unchanged. Training/inference defaults only point forward to the successful fix2 gate/output and were not executed.

## Validation results

- Focused Stage20 tests: `28 passed`.
- Phase4 + Phase5a + Stage19 + Stage20 regression selection: `82 passed`.
- 84-case CPU semantic preflight: passed for 64 TRAIN, 8 VAL, and 12 TEST cases.
- Preflight truth/Flow/adapter loads: false; selection and parameter sweep: false.
- LF fallback: 114 total fallback depths across the cohort, maximum per-case distance 19; no case was filtered by this statistic.
- `evidence_fix2/`: completed 84/84, beginning at `stage19_train_case001`; 759 files, with per-asset hashes in the evidence/case registries.
- Generated TEST inference registry: 12 cases and truth-firewall passed.

The unchanged VAL gate recomputed from `evidence_fix2` produced:

| Gate item | Result |
|---|---:|
| Integrity | 8/8 passed |
| Condition acoustic violations | 0 for every VAL case |
| LF and inverted impedance strict bounds | passed for every VAL case |
| Free-subsurface slowness unchanged | passed for every VAL case |
| Forward closure | passed for every VAL case |
| Seismic improved | 8/8 |
| Median seismic RMSE delta | -0.008719999343156815 |
| Continuous logZ improved | 6/8 |
| Median logZ RMSE delta | -0.008417539298534393 |

Final machine decision: `CONTINUOUS_IMPEDANCE_EVIDENCE_VALIDATED`; `adapter_training_authorized=true`.

## Preserved immutable artifacts

No files were overwritten or continued in: `observations/`, `observations_fix1/`, `observations_fix2/`, `inversion_input_registry_fix2/`, `evidence/`, `evidence_fix1/`, `evidence_audit/`, or `evidence_bound_diagnostic/`. The new outputs are `acoustic_condition_diagnostic/`, `acoustic_semantic_preflight/`, `evidence_fix2/`, and `evidence_audit_fix2/`.

## SHA256 provenance

Core frozen inputs:

| Asset | SHA256 |
|---|---|
| Stage20 specification | `bb4ed2c005fb6155680480eb0cea9de77988d3cfd4cab3faf3164848d40e60dd` |
| Phase4c acoustic codebook | `8980e30c59624af9f1bfa59447b7e22c85a2975f8a556fe6e98f5346f95d10ae` |
| Phase4c seismic config | `7ee49c3c1c685126cf9b727e05a895463734b4335300ccbad6d6dd6a865bfd02` |
| Phase5a inversion config | `6b690aec3978b48bc6922e7387fc9fca55566e2b54d5f1e2dcb2669143f79088` |
| observations_fix2 run manifest | `0bcc74d508bdc167f8947bc402e77ed2ba9542ca36848501e445a691542e0c24` |
| inversion_input_registry_fix2 | `a1e72def5f44a9e5f055e6d8de0c4028d8d4687f15ea5b402bd9ff637e03284b` |
| previous evidence_fix1 run manifest | `ff4cf92bd22635b3d26424f66db3558a768ce6d2c007dcfb744691c64a2d6665` |
| previous failed evidence audit | `9165d25b54790b69add60a92c7dd7b0584a8d57bf60d693cc487bf9ae1c22f74` |

Relevant source/config hashes:

| Asset | SHA256 |
|---|---|
| `acoustic_semantics.py` | `cd7d6ff8860ede78d52c56cc070f6766b71df7d09eb9775761d7f14b6abb1506` |
| `common.py` | `92f562d1725347d5483a72f6f21d04afcec993a22bd15a9c855532f0fecac7c6` |
| `build_observations.py` | `1c63e7286f3a27fa64f88c871888c0a868c2e25619c3b5c90914fdf584ee5bf7` |
| `build_continuous_evidence.py` | `551a614ff48397bbab38b80f5454f7e54a2c5461079e218f5de7e6ebfcbf2827` |
| `audit_evidence.py` | `70993349f7931552d475133b3353ba1fd1716038d5823ae8836e4536328654a8` |
| `evaluate.py` | `a27b1b400a368b67c42e70f3d294488c2c08d27571466fa350aaeef748a4c8f3` |
| `diagnose_acoustic_conditions.py` | `698496ab7b5592d6fcce41cba77e982b0f94be033890ef0c2e3bab11f036c6d6` |
| `preflight_acoustic_semantics.py` | `8aa658f5cf6b8172217217ed0aa65e54fa5a464a9ecdc5932dca27a4b8c25dc4` |
| Stage20 evidence config | `b96fbbd7cbd04e9557b3e37805cd07ac59d566a442050ec1086dd53eae05ec7b` |
| Focused tests | `50675551aae50b38c6ed0f27688f6747456f9c46127343c18a9e8cccc8a760b0` |

Output index hashes:

| Asset | SHA256 |
|---|---|
| Acoustic diagnosis summary | `2376cacbcf9361b359ff0f688ed489c296ec485824cda155e0c5d20124cd367c` |
| CPU preflight summary | `16d036fa5ac0033ec16b3a7515ac7bcdeaa3d2aee54550f8caaa6ef4c36d33dd` |
| CPU preflight per-case CSV | `1be3f6200f3349cc1c1e6e836055a7a9a7051bcc10c06431fa1e52c27db51279` |
| evidence_fix2 run manifest | `432cb6dc522d007d4f04a152a9f4b6867f99c1db5805eb5a2a0b614270a84581` |
| evidence_fix2 evidence registry | `94cb0cd2e595be1a9d1ae90bfec0fd7fcdeff69bdbd7b00e3e9642cff41deb28` |
| evidence_fix2 TEST inference registry | `11c201720328eab972ecebed11b0c31b2348733bd2cce86d280a6e7f2649cba3` |
| fix2 VAL audit summary | `475fe63a01146730cd7d092b6640a74ce808b5f3b434d33a736dbb16fd0647f0` |
| fix2 VAL per-case CSV | `8dfd907a2a8ab924153035d35cee6daf8e255afe0bcbdfab1f632ec8d2ac58b7` |

The evidence registry, each of the 84 case manifests, and the TEST inference registry contain the SHA256 records for all tensor outputs; the run manifest also records source/config/input provenance.

## Execution boundary

No CUDA smoke, adapter optimization update, formal adapter training, formal inference, TEST metric, TEST tuning, parameter sweep, or case filtering was performed. Work stops at the successful evidence gate pending human review.
