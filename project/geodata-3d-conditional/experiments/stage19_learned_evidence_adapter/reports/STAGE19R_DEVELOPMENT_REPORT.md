# Stage19R cohort repair and learned-adapter development report

## Scope and repository audit

- Remote: `xmj@172.27.231.254:22`
- Repository: `/home/xmj/mcy/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional`
- Starting branch/commit: `main @ f97c8ca1713421fe31ff250a62671e0cbe06fb41`
- Initial status: only the user's untracked
  `project/FLOW2_STAGE19R_COHORT_REPAIR_CODEX_SPEC.md`
- Stage19 v1 was audited as immutable. Its stopped cohort, failed manifest,
  report, configs, and handoff text were not edited or overwritten.

Stage19 v1 remains a cohort-availability stop (33/256 eligible TRAIN
candidates), not a scientific adapter result. Stage19R repairs only the
prospectively frozen candidate search envelope and adds provenance,
truth-firewall, and formal-checkpoint enforcement. Eligibility, accepted
counts, seeds, generator recipe, observations, inversion, adapter, training,
inference, metrics, and gates are unchanged.

## Generator reuse and cohort repair

`scripts/stage19/audit_generator_reuse.py` compared 12 generator-critical
assets with the authoritative Full StructuralGeo benchmark. All source and
default Markov-matrix SHA-256 values matched. Exact replay of root seed
`120260003` reproduced:

- truth tensor: `d1a18ebeedc0c25a4cbc35132b39630e84b1ec362927a318fb487ceaa3f31df7`
- condition values: `6f396bcf4cea81703e7ddc167e8d2200aac3360957b9c355aaa24808cc18d097`
- condition mask: `3a9374f12b526d0fc7979a2081317e44a15ce198a41be034aa8ccfc2610245fb`
- exact historical Markov event sequence and eligibility

The audit decision is `GENERATOR_REUSE_VALIDATED`. The authoritative Stage19R
cohort config SHA-256 is
`301d381628d25f5c49e13ccac39f14c090682cfbaabdaae54e62788d74dab903`.
Relative to cohort v1, the only scientific field changes are:

| split | accepted/start/step | v1 max candidates | v2 max candidates |
| --- | --- | ---: | ---: |
| TRAIN | 64 / 220260001 / 1 | 256 | 1024 |
| VAL | 8 / 220270001 / 1 | 128 | 256 |
| TEST | 12 / 220280001 / 1 | 128 | 256 |

Candidate traces are written after every candidate and retained on failure.
The completed run produced:

| split | examined | accepted | rejected | final accepted seed |
| --- | ---: | ---: | ---: | ---: |
| TRAIN | 431 | 64 | 367 | 220260431 |
| VAL | 60 | 8 | 52 | 220270060 |
| TEST | 63 | 12 | 51 | 220280063 |

Rejection counts are non-exclusive because a candidate may fail multiple
eligibility clauses. TRAIN/VAL/TEST respectively recorded 349/48/47 missing
label9-producing intrusion, 55/12/10 missing fold-or-fault, 11/3/2 absent
final raw label9, and 11/3/2 no hidden raw label9. No seismic, evidence, Flow,
or downstream metric participated in selection. Historical benchmark case IDs
were excluded. Sample-level pretraining overlap cannot be certified because
the old streaming CFM training retained no seed/sample manifest.

## Evidence provenance and validation gate

Stage19R validates the authoritative Stage17A formal config SHA-256
`2e6cfd2e1ea7999f587454db3ed68564b39599592f375a43fdae6a150b9058b0`
and the frozen binary acoustic, seismic, and trace-inversion config hashes
before reuse. Observation forward closure passed at `<= 1e-7` for all 84
cases. The unchanged Stage17A evidence runner completed all 84 cases without
loading truth.

The VAL-only gate passed with 8/8 positive AP-skill cases, median AP skill
`0.5492527938962282`, and 8/8 positive correct-case specificity cases. Its
decision is `EVIDENCE_REUSE_VALIDATED`; no inversion parameter was tuned.

The generated TEST inference registry contains 12 cases and no truth key,
truth asset, target tensor, or truth path. Formal inference reads this stripped
registry only; the retrospective evaluator separately reads the full registry.

## Checkpoint guard, smoke, and formal training

Formal inference now requires a completed `formal_training` manifest, a
non-smoke run, exactly 4 epochs and 1024 optimizer updates, epoch-4-final
selection, exact training-config/base-checkpoint/checkpoint hashes, fewer than
100,000 adapter parameters, unchanged base model, and absent base gradients.
Focused tests cover rejection of epoch-1, smoke/non-formal, and config-hash
mismatches.

CUDA smoke completed adapter training and five-arm TEST inference. It verified
condition exactness, truth blindness, base-model immutability, and independent
`adapter_scale=0` Flow equivalence. The first smoke exposed a relative asset
path provenance bug; asset records were changed to resolved absolute paths and
the successful rerun was written under new `*_fix1` directories rather than
overwriting the diagnostic run.

Exactly one formal frozen training run then completed 4 epochs and 1024
updates. Validation losses were `0.142025`, `0.132090`, `0.133736`, and
`0.129919`; epoch 4 is the only formal checkpoint. Adapter parameter count is
54,003. The checkpoint SHA-256 is
`cdb740bfeccc08f7010048c69deb8e44a13917fe53e3b2f4b46910bf536f20bb`.
The base-model tensor hash before and after training is identically
`e623079c4e3fa95a64195e2e3470ca0f8ab93f6b743c634bce8d42fb1b30f244`;
base gradients were absent and base fine-tuning was false.

## Formal inference and case-first result

Formal truth-blind inference completed exactly 12 cases x 3 source seeds x 5
frozen arms = 180 outputs. All condition violations are zero, the base model
is unchanged, and the run manifest records `truth_loaded_by_runner: false`.

Case-first evaluation produced `LEARNED_EVIDENCE_ADAPTER_VALIDATED`:

- learned target-IoU improvement: 12/12 cases;
- correct adapter above zero adapter: 12/12 cases;
- correct adapter above wrong-case adapter: 12/12 cases;
- hard-seismic RMSE improvement: 12/12 cases;
- cross-case median correct-minus-Flow target-IoU delta: `+0.236709`;
- cross-case median hard-seismic RMSE delta: `-0.008943`;
- cross-case median truth-present mean-IoU delta: `+0.039260`;
- all engineering, evidence reuse, learned coupling, specificity, hard
  physics, volume, structure, and global-geology gates passed.

The correct adapter's cross-case median target IoU is `0.410565`, versus
`0.169669` for Flow-only, `0.174973` for Stage18 continuous target,
`0.100748` for adapter-zero, and `0.098252` for wrong-case adapter. Its median
hard-seismic RMSE is `0.050981`, versus `0.063596` for Flow-only.

This is evidence for learned evidence-to-velocity coupling and case-specific
reuse under the frozen Stage19 experiment. It remains a binary,
high-contrast, noiseless inverse-crime upper bound and does not establish field
generalization. Stage19R stops here for research discussion.

## Exact execution and validation

```bash
../../.venv/bin/python scripts/stage19/audit_generator_reuse.py
../../.venv/bin/python scripts/stage19/build_cohort_v2.py
../../.venv/bin/python scripts/stage19/build_observations.py --device cuda
../../.venv/bin/python scripts/stage19/run_evidence.py --device cuda
../../.venv/bin/python scripts/stage19/audit_evidence.py
../../.venv/bin/python scripts/stage19/train_adapter.py --smoke \
  --output-dir experiments/stage19_learned_evidence_adapter/smoke/training_v2_fix1
../../.venv/bin/python scripts/stage19/run_inference.py --smoke \
  --training-manifest experiments/stage19_learned_evidence_adapter/smoke/training_v2_fix1/training_manifest.json \
  --adapter-checkpoint experiments/stage19_learned_evidence_adapter/smoke/training_v2_fix1/adapter_checkpoint.pt \
  --output-dir experiments/stage19_learned_evidence_adapter/smoke/inference_v2_fix1
../../.venv/bin/python scripts/stage19/train_adapter.py
../../.venv/bin/python scripts/stage19/run_inference.py
../../.venv/bin/python scripts/stage19/evaluate.py --device cuda
```

Final focused validation:

```bash
../../.venv/bin/python -m py_compile scripts/stage19/*.py
../../.venv/bin/python -m pytest -q \
  tests/test_stage19_learned_evidence_adapter.py \
  tests/test_stage18_evidence_semantics.py \
  tests/test_stage17_frozen_design.py \
  tests/test_stage15_binary_trace_boundary.py \
  tests/test_stage15_binary_seismic_consensus.py \
  tests/test_phase2_property_sampling.py
git diff --check
```

Result: `77 passed, 13 warnings`; compilation and whitespace checks pass.
