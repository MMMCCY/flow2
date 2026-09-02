# Stage19 development report — cohort gate stop

## Repository audit

- Remote: `xmj@172.27.231.254:22`
- Repository: `/home/xmj/mcy/flowtrain_stochastic_interpolation-main/project/geodata-3d-conditional`
- Starting branch/commit: `main @ 3134684cb9715a8cc58e3758d8770fde075cdd5d`
- Initial status: only the user's untracked
  `project/FLOW2_STAGE19_LEARNED_EVIDENCE_ADAPTER_CODEX_SPEC.md`
- Checkpoint SHA-256:
  `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`
- Runtime: PyTorch 2.8.0+cu128; NVIDIA RTX 4090 D; CUDA available

The required project instructions, baseline/protocol/research/handoff files,
Stage17/18 reports and formal outputs, Phase6 adapter spec/report/source and
the old Full StructuralGeo benchmark were inspected before editing.

## Implementation

Stage19 now has four frozen configurations and separate programs for:

1. immutable 64/8/12 StructuralGeo cohort generation;
2. binary acoustic/seismic observation generation;
3. unchanged Stage17A full-trace evidence inversion;
4. VAL-only AP-skill and wrong-case reuse audit;
5. four-epoch, 1024-update adapter-only training;
6. truth-blind five-arm test inference;
7. hard-geology, topology and hard-seismic case-first evaluation.

The adapter remains the existing `ResidualVelocityAdapter`, configured with
one evidence channel, width 12 and dilations 1/2/4/1.  The code freezes every
base parameter, uses EMA, keeps the output correction zero at conditions,
caps its norm at 0.25, records noise hashes and fixes epoch 4 as the only
formal checkpoint.  Inference statically excludes truth assets.

Frozen config hashes:

- cohort: `f0acfd6be07b870b58cf3a75451516d4f6b762c2de957375b653ae066608368c`
- evidence: `5c272f1a66b0f9ad3cb426f95dfe4c929f358b0fbc1e33199e2623028c8bc458`
- training: `ac3d5bf73b23a97168e0b200b0484e520b5b7f27c0f442702abf186996a49220`
- inference: `42092177caa132bf63741626df115f237978ad77d7096093f500fb0da91c5801`

## Validation

Executed before formal assets:

```bash
../../.venv/bin/python -m py_compile scripts/stage19/*.py
git diff --check

../../.venv/bin/python -m pytest -q \
  tests/test_stage19_learned_evidence_adapter.py \
  tests/test_stage18_evidence_semantics.py \
  tests/test_stage17_frozen_design.py \
  tests/test_stage15_binary_trace_boundary.py \
  tests/test_stage15_binary_seismic_consensus.py \
  tests/test_phase2_property_sampling.py
```

Result: `65 passed in 2.46s`.

## Formal Step 1 and mandatory stop

Command:

```bash
../../.venv/bin/python scripts/stage19/build_cohort.py
```

The frozen TRAIN range examined exactly 256 candidates.  Only 33 were
eligible, at seeds:

```text
220260007 220260015 220260025 220260042 220260044 220260051
220260068 220260073 220260082 220260098 220260102 220260106
220260120 220260137 220260141 220260142 220260146 220260149
220260157 220260164 220260169 220260171 220260187 220260196
220260198 220260201 220260213 220260218 220260225 220260228
220260231 220260236 220260245
```

The required count is 64.  The runner therefore stopped exactly as specified:

```text
RuntimeError: STOP: insufficient eligible train cases
```

Failed cohort manifest SHA-256:
`ae47b6666b1b4c4ea0402548a3e8154a67aeba3de97c38d82c00d4137b607d38`.

## Result boundary

This is a pre-training cohort availability failure, not an adapter scientific
result.  No observation/evidence assets, VAL evidence gate, CUDA smoke,
training trace, checkpoint, formal inference output or Stage19 success/failure
classification was produced.  The specification forbids expanding the seed
budget or changing eligibility after this result, so execution stopped.
