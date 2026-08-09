# Stage 8 protocol — flow-constrained structured posterior

Frozen before authoritative Stage-8 results at repository commit
`3394145674e04e3698952064f486cfa31eb876f9` with a dirty tree containing the
pre-existing untracked project `AGENTS.md` and this Stage-8 implementation.

## Scope and immutable scientific choices

Stage 8 implements continuous structured hard-geophysics search and its
integration with cached, condition-exact samples from the frozen EMA flow
checkpoint. It performs no training, adapter fitting, LoRA, U-Net change,
gravity/magnetic fusion, controller sweep, SMC, or RJMCMC.

Every candidate follows:

    structured parameters -> hard labels -> exact condition projection
      -> hard acoustic codebook -> hard convolutional seismic
      -> observed hard-seismic RMSE -> truth-blind beam selection

The selector cannot receive a truth tensor. Truth is opened only by a separate
retrospective evaluator after the selected state and response are fixed.

## Stage 8A

Cases are the existing analytic five-body regression and StructuralGeo-native
seeds `20260807`, `20260808`, and `20260809`. The latter include an unknown
concealed count in the registered range `0..3`. Search births are drawn from
the same broad domain bounds for every case and are not centered at native
truth anchors:

- center x/y: `4..59`; center z: `6..50`;
- size x/y: `6..16`; size z: `6..14`;
- orientation: `0..180` degrees;
- shape: ellipsoid or `dike_hemisphere`;
- material: raw label 9 only;
- maximum concealed body count: 3.

The analytic case retains its immutable fixed three-body regression base. Each
native case starts from background plus its explicitly observed air/surface and
well voxels only; no unobserved fixed-body geometry and no hidden-body mask is
used to construct the inference start.

The deterministic proposal kernel supports birth, death, translate, resize,
rotate, and shape changes. The fixed screening budget is beam 8, 10
generations, and 12 proposals per parent: 961 hard-forward calls per arm.
Correct, zero, shuffled-xy, and independent wrong-case observations use the
same proposal seed and budget. All final states are cross-evaluated against
the common correct field.

The analytic regression may additionally run the immutable Stage-7 finite
library path only as a regression fixture. It is not evidence for the new
continuous gate.

## Stage 8B

The primary realistic case is `cond_generation_0`. Its split provenance is
unknown and it is same-distribution synthetic validation data; Stage 8B is an
integration/mechanism benchmark, not held-out generalization.

Arms:

- `FLOW_ONLY`: cached alpha-zero EMA/fixed-Euler samples, unchanged;
- `STRUCTURED_ONLY`: deterministic condition-only background plus structured
  search under the same domain and conditions;
- `FLOW_PLUS_STRUCTURED`: each paired flow sample plus structured edits.

The screening cohort is seed 42, four cached samples. Expansion to seeds 142
and 242 is allowed only from the inference-visible screening verdict in
`stage8_gate.json`, before retrospective truth metrics are inspected.

For the combined method, all four observation controls use identical search
budgets. The wrong-case observation is the hard response of a predeclared
frozen-flow member under the same acquisition geometry. The edit domain is all
unconditioned subsurface at least four cells below the observed local surface;
it is computed from `subsurface_mask.pt` and the borehole condition only. It is
not a truth-derived ROI. Bounds are the broad Stage-8A bounds above, with
maximum body count 4 and label-9 additions only in v1.

## Observation and likelihood

Stage 8 reuses the immutable Stage-4c `distinct_upper_bound_v1_fix2` synthetic
observation, complete acoustic codebook, full 64-by-64 trace grid, 320 samples
at 8 ms, and 25 Hz peak Ricker wavelet. Selection uses unnormalized hard
amplitude RMSE. Noise is zero. Cross-evaluation always uses the same correct
observation.

## Reproducibility and stop rules

Each run must store the repository commit/dirty status, source/config hashes,
checkpoint/base/condition/observation hashes, proposal seed and complete
selected state, trace, forward-call count, inference-visible audit, and a
separate retrospective section. Existing output directories are immutable.

If Stage 8A fails its frozen gate, Stage 8B must not run. If Stage 8B improves
hard seismic without concealed hard geometry, or does so through nonspecific
wrong-lithology substitution, Stage 8 stops and reports one minimal causal
follow-up rather than tuning broadly.

## Commands

CPU development gate:

```bash
PYTHONPATH=project/geodata-3d-conditional .venv/bin/python -m pytest -q \
  project/geodata-3d-conditional/tests/test_stage8_structured_posterior.py \
  project/geodata-3d-conditional/tests/test_stage7_structured_hard.py \
  project/geodata-3d-conditional/tests/test_phase4_seismic.py
```

Intended CUDA runs after the CPU gate:

```bash
PYTHONPATH=project/geodata-3d-conditional .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage8/run_stage8.py \
  --stage 8a --config project/geodata-3d-conditional/experiments/stage8_structured_posterior/configs/stage8_v1.json \
  --output-dir project/geodata-3d-conditional/experiments/stage8_structured_posterior/runs/stage8a_v1 --device cuda

PYTHONPATH=project/geodata-3d-conditional .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage8/run_stage8.py \
  --stage 8b --config project/geodata-3d-conditional/experiments/stage8_structured_posterior/configs/stage8_v1.json \
  --stage8a-summary project/geodata-3d-conditional/experiments/stage8_structured_posterior/runs/stage8a_v1/stage8a_summary.json \
  --output-dir project/geodata-3d-conditional/experiments/stage8_structured_posterior/runs/stage8b_screen_seed42_v1 --device cuda
```
