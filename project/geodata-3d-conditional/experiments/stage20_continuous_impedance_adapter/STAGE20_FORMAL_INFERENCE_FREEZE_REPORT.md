# Stage20 Formal Inference Freeze Report

## Machine decision

`STAGE20_FORMAL_INFERENCE_FROZEN`

The unique blind formal inference is accepted via **pre-unblinding post-run provenance attestation**. This report makes no Stage20 scientific conclusion.

## Immutable formal inference

| Item | Frozen value |
|---|---|
| Git head | `24abcd29224976c44cfebe72f7fa7f66dbcc32b0` |
| Runner SHA256 | `1bebc207108e6850a97784ca01f26c33ff64413b2c555ab4b3b6c48eae15fe3d` |
| Original run manifest SHA256 | `12bbde73d548de8ff5bf46a58a47bdb63bfcb9c9651b3138981347faa3eb401d` |
| Original formal output index SHA256 | `0f1f6534423a0fc600a23fd4d84fba4bd10ecbecc60f2fe33cdf6ebf408d6c7e` |
| Formal output count | `144` |
| Base checkpoint SHA256 | `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c` |
| Adapter checkpoint SHA256 | `f8b71d0889ec8f76f033583be3b94f3041da888c0b910e92c8f686e24a093213` |
| Inference config SHA256 | `41c38912b1d49335e3e4ac85a39d86d6ee1d5e0064adbd924a9862d8ae0b88d8` |
| Stripped TEST registry SHA256 | `11c201720328eab972ecebed11b0c31b2348733bd2cce86d280a6e7f2649cba3` |

The original run manifest and formal output index retained their authorized hashes throughout this attestation. No original output or formal-inference provenance asset was modified.

## Base-state provenance closure

The original inference runner measured:

- `base_model_hash_before = e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244`
- `base_model_hash_after = e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244`
- `base_model_unchanged = true`

Historical formal-training provenance records:

- Base state_dict hash: `10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68`
- State_dict tensor count: `412`
- Model parameter tensor count: `412`

An independent reconstruction used the same `Geo3DStochInterp` architecture, frozen base checkpoint, EMA loading logic, and eval mode. It did not load TEST evidence, conditions, or truth and did not invoke a sampler. It produced:

- Reconstructed base-model parameter hash: `e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244`
- Reconstructed base state_dict hash: `10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68`
- Parameter tensor count: `412`
- State_dict tensor count: `412`
- Persistent non-parameter state keys: `[]`
- Parameter keys missing from state_dict: `[]`
- Parameter/state_dict exact key equivalence: `true`

The frozen architecture therefore has no persistent buffers or other state outside `named_parameters()`. The inference-time parameter hash covered all persistent model tensors. The absent `base_state_dict_hash_before/after` fields are classified as a **redundant logging omission**, not an unverified model mutation.

The missing runtime fields were not fabricated: the original run still does not claim that state_dict hashes were measured during inference.

## Truth-blind output re-audit

The existing provenance helper re-audited the original collection without regenerating any output:

- Exact unique inventory: passed, `144/144`.
- Output file hashes: passed.
- Output tensor hashes: passed.
- Hard-condition violations: `0`.
- Paired initial-noise groups: passed, `36/36` case/seed groups.
- Frozen wrong-case cyclic mapping: passed.
- Current-case mask provenance: passed.
- Sampler/checkpoint/config/runner provenance: passed.
- `truth_loaded_by_runner = false`.
- Truth firewall: passed.

Independent attestation asset:

- Path: `formal/inference_v1_audit/base_state_provenance_attestation.json`
- SHA256: `32caffc3f038a33a4491249ad89a650563b8754262837c100b7586cfb3ea999a`

## Declarations

- The original formal outputs were not modified.
- Formal inference was not rerun.
- The original `run_manifest.json` was not modified.
- The original `formal_output_index.json` was not modified.
- Missing runtime fields were not fabricated.
- No TEST truth was loaded.
- No TEST scientific metric was inspected.
- The retrospective evaluator was not run.
- No retraining, tuning, retry, or parameter modification occurred.

The frozen inference now awaits human review before any separately authorized retrospective TEST evaluation.
