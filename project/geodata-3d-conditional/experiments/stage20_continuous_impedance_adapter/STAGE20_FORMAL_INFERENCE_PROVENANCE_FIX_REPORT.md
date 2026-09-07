# Stage20 Formal-Inference Provenance Contract Fix Report

## Decision

`STAGE20_FORMAL_INFERENCE_PROVENANCE_READY`

This task completed the engineering/provenance protocol before formal TEST unblinding. No formal inference, TEST truth access, or TEST scientific metric was executed.

## Why the previous contract was incomplete

The previous runner produced the correct frozen four-arm sampling outputs, but its per-output CSV did not contain every preregistered checkpoint/config/source/input/output provenance field. It also used `run_status=completed` without a separately hashed deterministic 144-output master index. The previous evaluator could dereference TEST truth after checking only run status and row count, and its remaining decision branch used machine-decision names that had not been frozen in the scientific protocol.

## Source changes

- `scripts/stage20/run_inference.py`
  - Records complete per-output provenance, including base/adapter/config/runner hashes, condition/evidence source hashes, current/wrong/mask case IDs, sampler settings, hard-condition count, and both tensor/file SHA256.
  - Keeps FLOW_ONLY scale at 0 and all adapter-arm scales at 1.
  - Builds a deterministic case/seed/frozen-arm ordered master index only after all 144 unique outputs and hashes validate.
  - Sets `formal_inference_complete=true` only after master-index creation succeeds; failure preserves false.
- `scripts/stage20/formal_inference_provenance.py`
  - Adds the shared truth-blind inventory validator, index freezer, and retrospective firewall.
  - Rejects missing/duplicate/unexpected combinations, wrong cyclic sources, current-mask mismatch, sampler drift, hard-condition violations, path traversal/missing files, tensor/file hash changes, and master-index hash/order drift.
- `scripts/stage20/evaluate.py`
  - Runs the complete freeze audit before the first TEST truth dereference.
  - Uses only the frozen master index for formal records.
  - Implements the frozen Gate G → A → B → C → D → E/F → success decision order.
  - Removes `ENGINEERING_FAIL` and `STAGE20_PARTIAL_OR_NEGATIVE` as scientific decisions.
- `tests/test_stage20_continuous_impedance_adapter.py`
  - Adds tiny synthetic, truth-free tests for all provenance, inventory, hash, firewall, wrong-case, sampler, hard-condition, and decision-order requirements.

No sampling equation, network, metric, or gate threshold was changed.

## Git provenance

- Previous git head: `6b5a4cab86f064ff5325f0ab5427e5c3b32fda35`.
- New source provenance commit: `24abcd29224976c44cfebe72f7fa7f66dbcc32b0`.
- Commit subject: `stage20 formal inference provenance contract`.
- Commit scope: four small source/test files only; no generated tensors or scientific assets.
- Tracked worktree was clean after the commit.
- `git diff --check`: passed.

Source SHA256 transition:

| Source | Previous SHA256 | New SHA256 |
|---|---|---|
| `run_inference.py` | `11e9f12c467ff27caae0bf1cdd17fe6805debbb73cb3722eca32773e9c9f28a6` | `1bebc207108e6850a97784ca01f26c33ff64413b2c555ab4b3b6c48eae15fe3d` |
| `evaluate.py` | `a27b1b400a368b67c42e70f3d294488c2c08d27571466fa350aaeef748a4c8f3` | `c5a95d944ca3a6bf93162dedce8769430fc2d84bde9fe7d8045e0b0ed96557f7` |

Additional new source hashes:

- `formal_inference_provenance.py`: `55d7931a95f59b631076fb737f89d73758a9925aa42c1d9f3e9b061f75be69b3`.
- Focused test source: `9adf8066bc424d932b70d817785f81a0a62e798b119077fdc9be39437a96cc1f`.

## Unchanged scientific assets

| Asset | SHA256 |
|---|---|
| Frozen inference config | `41c38912b1d49335e3e4ac85a39d86d6ee1d5e0064adbd924a9862d8ae0b88d8` |
| Formal adapter checkpoint | `f8b71d0889ec8f76f033583be3b94f3041da888c0b910e92c8f686e24a093213` |
| evidence_fix2 evidence registry | `94cb0cd2e595be1a9d1ae90bfec0fd7fcdeff69bdbd7b00e3e9642cff41deb28` |
| Stripped TEST inference registry | `11c201720328eab972ecebed11b0c31b2348733bd2cce86d280a6e7f2649cba3` |
| Evidence authorization gate | `475fe63a01146730cd7d092b6640a74ce808b5f3b434d33a736dbb16fd0647f0` |

The base Flow checkpoint, adapter checkpoint, evidence, observations, cohort, TEST12, source seeds, arm definitions, sampler, adapter scale, residual cap, normalization, acoustic semantics, evaluation metrics, and Gate A–G thresholds were not modified. Training code was not modified and no retraining occurred.

## Verification

- Stage20 focused plus Stage19R/Phase4/Phase5a regression selection: 95 passed.
- Python compilation: passed.
- Synthetic inventory size: exact 12 × 3 × 4 = 144.
- Missing and duplicate inventory records: rejected.
- Altered output file and master-index hashes: rejected before truth authorization.
- Output file/tensor hashes: independently recomputed and validated.
- Wrong-case cyclic source and current-case mask provenance: validated.
- FLOW_ONLY scale 0; adapter-arm scale 1; steps 32; residual cap 0.25: validated.
- Hard-condition violation: prevents formal freeze.
- Gate G priority and all A/B/C/D/E/F decision branches: validated.
- No real TEST metric is needed or used by these tests.

## Execution boundary

`formal/inference_v1/` remains absent. No real formal TEST case was run, no 144-output collection was produced, no TEST truth was loaded, and no TEST IoU, mIoU, seismic, volume, component, centroid, PROPERTY_ONLY, or other scientific metric was inspected.

The new runner/evaluator/helper hashes and provenance commit are ready to be frozen by human review before the separately authorized formal TEST run.
