# Stage 8A v2 — deterministic geophysics-informed birth-center re-gate

## Frozen scope

Stage8A v1 and Stage8A-R1 are immutable inputs. Stage8A v2 changes exactly one
algorithmic choice: the three uniformly sampled birth-center coordinates are
replaced by deterministic, truth-blind sensitivity-ranked coordinates. The
original random draws are still consumed and discarded so the frozen random
streams for body size, orientation, shape, material, and later moves remain
unchanged.

The following remain exactly equal to `stage8_v1.json`: cases, controls,
seeds, center and body bounds, shape set (`ellipsoid`, `dike_hemisphere`),
material set, maximum count, beam size 8, 10 generations, 12 proposals per
parent, proposal/move schedule, hard-condition projection, hard petrophysical
mapping, hard seismic forward, hard-RMSE selection, and the original
`stage8_gate.json` thresholds. Each arm therefore remains exactly 961 hard
forward evaluations. No cuboid or other shape is added.

No Stage8B, training, fine-tuning, LoRA, budget increase, broadening, sweep,
or second algorithmic change is authorized. A failed v2 gate stops Stage 8.

## Truth-blind multifield score

For the current hard state, let the fixed petrophysical mapping give the two
physical property fields

    P = (Z, s),

where `Z` is acoustic impedance and `s` is slowness. Let `F(P)` be the frozen
convolutional seismic forward, `y` the arm's observed seismic, and

    L(P) = mean((F(P) - y)^2).

One differentiable forward/backward computes both property gradients
`g_p(x) = dL/dP_p(x)`. For an integer candidate center `c`, `M_c(x)` is a
canonical label-9 ellipsoid insertion with size `(11,11,10)`, the untuned
midpoint of the frozen v1 size bounds. It is intersected with the inference-
visible edit mask. The ranked predicted loss decrease is

    score(c) = - sum over p in {Z,s} and voxels x of
                 g_p(x) M_c(x) [P_p(label 9) - P_p(current,x)].

MSE and nonzero RMSE directional rankings differ only by a positive scalar,
so this ranks the same first-order descent direction while avoiding division
by zero. Both impedance and slowness contributions are included. Higher score
ranks first; exact ties use ascending `(center_x, center_y, center_z)`.

Valid candidate centers are integer voxel centers inside the frozen v1 center
bounds and the current case's condition-excluding edit mask. The ranking is
recomputed for each distinct current beam state in each generation. Duplicate
copies of the same state within one generation share one ranking and consume
successive ranked centers, preventing duplicate hard proposals.

This score is proposal guidance only. Every proposed body still follows

    hard geology -> exact condition projection -> hard two-field mapping
      -> hard seismic forward -> observed hard RMSE -> truth-blind selection.

The differentiable guidance forward/backward counts and runtime are recorded
separately and never enter the 961-call hard-search budget or final selection.

## Audit artifacts

Each arm saves:

- `sensitivity_rankings.json`, with current state/label, prediction,
  observation, residual, sensitivity-map, edit-mask, and both property-gradient
  hashes; ranked centers/scores; tie-breaking; selected births; operation
  counts/runtime; and an explicit truth-blindness record;
- `sensitivity_maps/*.pt`, containing the complete score map and valid-center
  mask, making the full ranking reconstructible;
- `birth_proposals.csv`, containing every selected birth center, predicted
  first-order score, hard RMSE, delta versus parent/empty, and condition audit;
- the original selection, proposal trace, hard outputs, retrospective metrics,
  paired table, summary, and report.

Loss-only selection files are frozen before retrospective truth metrics are
computed. The final decision first applies the unchanged original Stage8A
gates. The summary additionally reports `delta_rmse < 0` birth counts for every
correct arm.

## Stop rule and authoritative command

The output directory is new and immutable:

```bash
PYTHONPATH=project/geodata-3d-conditional .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage8/run_stage8.py \
  --stage 8a \
  --config project/geodata-3d-conditional/experiments/stage8_structured_posterior/configs/stage8a_v2.json \
  --output-dir project/geodata-3d-conditional/experiments/stage8_structured_posterior/runs/stage8a_v2 \
  --device cuda
```

If the original gate fails, stop without Stage8B or another modification.
