# Stage20 Continuous-Impedance Adapter CUDA Smoke + Formal Training Report

## Final decision

`STAGE20_FORMAL_ADAPTER_TRAINING_VALIDATED`

This decision validates only the engineering, provenance, and frozen training contract. It does not validate the Stage20 scientific hypothesis or claim improvement on formal TEST geology.

## Git and upstream provenance

- Starting branch/commit: `main` / `2284d7235cd62295d741c3982eb5063fe6577ac6`.
- Training provenance commit: `ae5e206b4bc463c0b5271e7da9b09d72ccb4499c` (`stage20 acoustic evidence validated`).
- The provenance commit contains 23 small source/config/test/spec/report files and no `.pt`, `.ckpt`, generated cube, cache, diagnostic tensor, smoke output, or formal output.
- Tracked worktree status at smoke and formal training start: clean. Full `git status --short` was recorded in each run manifest; it contains only preserved untracked experiment artifacts.
- `git diff --check`: passed.

Authoritative upstream hashes were checked exactly before any optimizer update:

| Asset | SHA256 |
|---|---|
| Acoustic semantics fix2 report | `db9eb622a4b7d06b41f318b2dcf00edcf35e5d7fd42b655d40770bb0e14696ad` |
| evidence_fix2 run manifest | `432cb6dc522d007d4f04a152a9f4b6867f99c1db5805eb5a2a0b614270a84581` |
| evidence_fix2 evidence registry | `94cb0cd2e595be1a9d1ae90bfec0fd7fcdeff69bdbd7b00e3e9642cff41deb28` |
| evidence_fix2 stripped TEST inference registry | `11c201720328eab972ecebed11b0c31b2348733bd2cce86d280a6e7f2649cba3` |
| evidence_audit_fix2 summary | `475fe63a01146730cd7d092b6640a74ce808b5f3b434d33a736dbb16fd0647f0` |
| Base Flow checkpoint | `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c` |
| Frozen training config | `297083b9b14d935d5fae6f7833c357c569aa69b3d16fb8103bbe3e70227177d8` |
| Frozen inference config | `41c38912b1d49335e3e4ac85a39d86d6ee1d5e0064adbd924a9862d8ae0b88d8` |

The evidence decision was exactly `CONTINUOUS_IMPEDANCE_EVIDENCE_VALIDATED`, with `adapter_training_authorized=true`, 84 evidence cases, frozen TRAIN64/VAL8, and a 12-case stripped TEST registry that passed the truth firewall.

## Frozen adapter and training contract

- Implementation: existing `ResidualVelocityAdapter` from Stage19R.
- Geophysics channels: 1.
- Base width: 12.
- Dilations: `[1, 2, 4, 1]`.
- Maximum residual ratio: 0.25.
- Final output convolution: exactly zero initialized.
- Exact parameter count: 54,003 (`<100,000`).
- Base Flow: EMA weights, `eval()`, all parameters `requires_grad=False`.
- Optimizer: AdamW over adapter parameters only.
- Seeds: adapter 6200; state generator 6100; shuffle base 6201.
- Epochs/times/batch: 4 / `[0.2, 0.4, 0.6, 0.8]` / 1.
- Learning rate/weight decay/gradient clip: 0.002 / 0.0001 / 1.0.
- Loss weights: flow 1.0, CE 0.25, Dice 0.25, residual regularizer 0.0001; temperature 0.1.
- Scheduler and early stopping: none/false.
- Checkpoint selection: epoch 4 final only.
- No physics, topology, seismic, property reconstruction, or uncertainty loss; no base fine-tuning.

Training consumed only TRAIN64 and VAL8 records from `evidence_fix2`. Supervised TRAIN/VAL truth was used under the frozen Stage19R training role. No TEST truth was loaded and TEST records were not added to the supervised training case list.

## CUDA engineering smoke

Training smoke output: `smoke/training_v1/`.

- Device: NVIDIA GeForce RTX 4090 D, 24,564 MiB.
- TRAIN subset: first 2 cases; VAL subset: first 1 case.
- Optimizer updates: exactly 2.
- Adapter output zero-init, q shape/finite checks, finite/nonzero gradients, optimizer isolation, residual cap, hard-condition correction zero, finite losses, EMA load, base-gradient absence, and checkpoint save/load: all passed.
- Base parameter hash before/after: `e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244` / identical.
- Base state_dict hash before/after: `10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68` / identical.
- Smoke training manifest SHA256: `c4f9ba8db453131c25c7d5aa5d8af734125827d2f2ca85643f21bd6858e25f53`.
- Smoke checkpoint SHA256: `edfc5c2547788ae92bda4b69899a6785f62cf6fb0842c55d7b081688852396fe`.

Truth-blind inference wiring smoke output: `smoke/inference_v1/`.

- Exactly 1 stripped TEST case × 1 source seed × 4 frozen arms = 4 outputs.
- Arms: `FLOW_ONLY`, `ADAPTER_PRIOR_ONLY`, `ADAPTER_CORRECT`, `ADAPTER_WRONG_CASE`.
- All hard-condition violations: zero.
- Independent FLOW_ONLY scale-zero equivalence: passed.
- Base parameter hash before/after: identical.
- Wrong-case value source: case002 for current case001; current case001 mask/support retained.
- Runner truth load: false. No TEST metric or retrospective evaluator was invoked.
- Smoke inference manifest SHA256: `68b85ebd7ffc5dd388c88121e044fa0a7ffd1096dbe03667585f6b4f3b782e17`.

## Formal training and checkpoint audit

Formal output: `checkpoints/formal_v1/`.

The single authorized run completed exactly 4 epochs and 1,024 optimizer updates. VAL loss was diagnostic only and did not select a checkpoint:

| Epoch | Updates complete | Mean VAL diagnostic loss |
|---:|---:|---:|
| 1 | 256 | 0.14529898212640546 |
| 2 | 512 | 0.14130921286414377 |
| 3 | 768 | 0.14090346090961248 |
| 4 | 1024 | 0.14052530541084707 |

Across all 1,024 recorded updates:

- Losses were finite.
- Adapter gradients were individually finite and nonzero; recorded norm range was `0.009560002014040947` to `1.0923529863357544` before clipping.
- Base gradients were absent at every update.
- Maximum used residual ratio was exactly 0.25, within the frozen cap.
- Base parameter hash before/after was identical: `e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244`.
- Base state_dict hash before/after was identical: `10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68`.
- EMA applied: true; base fine-tuning: false.

Formal checkpoint:

- Path: `experiments/stage20_continuous_impedance_adapter/checkpoints/formal_v1/adapter_checkpoint.pt`.
- Schema: `stage20_adapter_checkpoint_v1`.
- Run class/smoke subset: `formal_training` / false.
- Epoch/checkpoint selection: 4 / `epoch4_final`.
- Adapter parameter count: 54,003.
- SHA256: `f8b71d0889ec8f76f033583be3b94f3041da888c0b910e92c8f686e24a093213`.

Formal checkpoint guard output: `checkpoints/formal_v1_audit/summary.json`.

- It reloaded the checkpoint, strictly checked manifest and payload fields, verified exact config/evidence/gate/base hashes, validated all 1,024 trace rows and finite checkpoint tensors, and confirmed the pre/post base hashes.
- Audit summary SHA256: `92920f04b5f0e6e954e0dad725a4e47e46b7d933176f27aa9be93e2c37d853f2`.
- Machine decision: `STAGE20_FORMAL_ADAPTER_TRAINING_VALIDATED`.

Additional formal artifact hashes:

| Asset | SHA256 |
|---|---|
| Formal training manifest | `0add2f3ce80b1ea370e9dbf9f3c3a3ce8f9b383ff9ffd3dce4978aaeebc9665a` |
| Formal training trace | `23be076152fbbe43bfaf6ff6a698ba1f2baa64fa8bee7d7158d910df309f32e9` |
| Formal validation trace | `c5af358cdbe5f24c89dd3ec499681fe93904958aeed82ca46056116ebee6e978` |
| Training runner source | `7f25ffd9345b23986fbd745980fd24ffc08f5f0e7c379caf1bc8335649152967` |
| Inference runner source | `11e9f12c467ff27caae0bf1cdd17fe6805debbb73cb3722eca32773e9c9f28a6` |
| Formal checkpoint auditor source | `d8df2fb5681848ab286c78af560eafb95b3cf12760610966645935b68e876e8b` |

## Tests and execution boundary

- Stage20 focused plus Stage19R/Phase4/Phase5a regression selection: 85 passed.
- Stage20 Python compilation: passed.
- `git diff --check`: passed.

No formal TEST inference was executed. No target IoU, truth mIoU, hard-seismic TEST comparison, volume/component statistic, nearest-code TEST baseline, TEST visualization, or other formal TEST scientific metric was inspected. No TEST tuning, evidence tuning, adapter hyperparameter tuning, architecture search, retry, or parameter sweep was performed.

The frozen inference config and stripped TEST inference registry are ready for a separately authorized next stage. Execution stops here for human review.
