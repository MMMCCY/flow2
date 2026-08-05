# Experiment protocol

## Two baselines with different purposes

The project now uses two explicitly named baselines.

1. **Historical reference**: EMA model, adaptive Dopri5, `rtol=1e-6`,
   `t0=0.0001`, `tf=0.9999`. This preserves continuity with the original
   inference implementation.
2. **Paired no-guidance control**: EMA model, fixed Euler midpoint updates,
   identical initial noise and all other settings, with active guidance
   strength exactly zero. This is the only baseline used to attribute changes
   to guidance.

The two solvers are not expected to produce identical samples. Their decoded
and continuous differences must be measured with `audit_baseline_solvers.py`,
not hidden by calling both outputs “the baseline.”

## Strict pairing requirements

A guided run is paired only when all of the following match:

- checkpoint hash and EMA/raw policy;
- truth, boreholes, property configs, and supplied observation hashes;
- sampler, runtime, and geophysics source hashes;
- device, seed, sample count, initial-noise policy;
- integrator, step count, temperature, schedule, clipping, physics mode, and
  physics weights.

For relative guidance, baseline `alpha` must be zero and guided `alpha` must be
positive. For absolute guidance, baseline `mu` must be zero and guided `mu`
must be positive. A legacy config without protocol version and asset hashes is
not accepted as strictly paired.

## Stage-0 server commands

Run from the repository root and adjust only the root path if needed:

```bash
PROJECT=project/geodata-3d-conditional
CKPT="$PROJECT/demo_model/conditional-weights.ckpt"
CASE="$PROJECT/samples/jupyter-demo/cond_generation_0"
DEMO="$PROJECT/dike-demo-manual/cond_generation_0_label9"
OUT="$PROJECT/stage0-baseline-audit"

PYTHONPATH=src python "$PROJECT/audit_baseline_solvers.py" \
  --ckpt-path "$CKPT" \
  --samples-dir "$CASE" \
  --model-weights ema \
  --n-samples 1 \
  --n-steps 32 \
  --seed 42 \
  --device cuda \
  --output-dir "$OUT/solver-gap"
```

Generate a new strict paired control:

```bash
PYTHONPATH=src python "$PROJECT/guided_geophysical_sampling.py" \
  --ckpt-path "$CKPT" \
  --model-weights ema \
  --samples-dir "$CASE" \
  --density-config "$DEMO/density_config.json" \
  --n-samples 4 \
  --n-steps 32 \
  --guidance-mode relative \
  --alpha 0 \
  --mu 0.01 \
  --tau 0.1 \
  --guidance-start 0.5 \
  --guidance-schedule late_quadratic \
  --kernel-size 9 \
  --grad-clip-norm 1.0 \
  --seed 42 \
  --device cuda \
  --output-dir "$OUT/paired-alpha0"
```

Generate the guided member with all non-guidance arguments unchanged:

```bash
PYTHONPATH=src python "$PROJECT/guided_geophysical_sampling.py" \
  --ckpt-path "$CKPT" \
  --model-weights ema \
  --samples-dir "$CASE" \
  --density-config "$DEMO/density_config.json" \
  --n-samples 4 \
  --n-steps 32 \
  --guidance-mode relative \
  --alpha 0.05 \
  --mu 0.01 \
  --tau 0.1 \
  --guidance-start 0.5 \
  --guidance-schedule late_quadratic \
  --kernel-size 9 \
  --grad-clip-norm 1.0 \
  --seed 42 \
  --device cuda \
  --baseline-dir "$OUT/paired-alpha0" \
  --output-dir "$OUT/paired-alpha005"
```

Each run writes:

- `config.json` with strict protocol fields and asset hashes;
- `model_load_report.json` with EMA coverage;
- `input_validation.json` with label, borehole, and target observability facts;
- generated samples and guidance trace.

The guided command stops before sampling if its baseline is not strictly
paired.

## Gate before 3-D probability guidance

Proceed to the probability-volume upper-bound experiment only if:

- EMA coverage reports no missing trainable parameters;
- input validation reports zero borehole/truth mismatches;
- strict pairing passes;
- the solver-gap audit is recorded;
- the alpha-zero rerun is reproducible from the same config and assets.

## Phase-2 strict-pair extension

Phase 2 keeps every invariant above and adds the following equality fields:

- Phase-2 protocol and property-loss versions;
- complete property-config hash and parsed property-table tensor hash;
- truth-derived target-property tensor hash;
- confidence/missing-data mask hash;
- property channel names, units and normalized channel weights;
- three-dimensional scale sigmas and normalized scale weights;
- property temperature, controller, cap and schedule fields;
- property runner and implementation source hashes.

The first Phase-2 alpha-zero baseline must be newly generated. It cannot reuse
any Phase-1 probability baseline because its audited target assets, source
hashes, loss diagnostics and strict-pair fields differ. Phase-1 probability
loss and all 2-D forward-field losses must remain inactive.

Before the first Phase-2 GPU run, the runner must prove that alpha zero takes an
explicit no-gradient branch, reproduces a reference projected fixed-Euler
trajectory, refuses a non-empty output directory and re-projects conditions
after every step. See `docs/PHASE2_SPEC.md`.

## Phase-3/Phase-4 observation extension

Phase 3 and later acquisition-domain physics add an explicit observation
operator `H`. Strict pairing includes the observation config, resolved
observation, noiseless response, confidence, noise and operator-source hashes.
The observed target is generated once and is never resampled inside the
sampler. Deterministic response `H` is applied to the prediction; observation
noise is not applied to the prediction.

Before `H`, predicted soft properties at air, surface and borehole conditions
are replaced by the exact hard properties. This prevents finite-temperature
soft decoding at known voxels from leaking an erroneous contribution through a
blur or global forward operator. The state itself is still projected before
sampling and after every fixed-Euler step.

Phase 3 spatial degradation and Phase 4 acquisition physics are separate
attribution experiments even when they share this interface. If a 3-D
inversion product and a surface field derive from the same survey, they cannot
be presented as independent evidence or naively counted twice in a joint loss.
See `docs/PHASE3_SPEC.md` and `docs/PHASE4_SPEC.md`.
