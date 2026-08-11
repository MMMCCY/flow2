# Full StructuralGeo benchmark provenance audit

Date: 2026-08-11  
Repository: `main @ 72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f`

## Scope and stop condition

This is a read-only provenance/design audit for a future geology-only benchmark
builder. It does not repair or continue Stage 11. No geology, seismic data, Flow
sample, Stage11-A output, training output, or new benchmark registry was generated.
The only new artifact is this audit document.

The worktree was already dirty before this document was added. The pre-audit
`git status --short` included modified handoff/prompt documents and untracked
Stage 9/10/11, guidance, paper, script, and test assets. None of those existing
changes was modified by this audit.

## Executive finding

The Flow training data and `cond_generation_0` use the same full stochastic
StructuralGeo path:

```text
GeoData3DStreamingDataset
  -> MarkovGeostoryGenerator
  -> default_markov_matrix.csv
  -> categorical event template
  -> full GeoWord/GeoProcess history
  -> GeoModel.compute_model(normalize=True)
  -> fill NaN with raw label -1
```

`guidance/native_geology_audit.py` does not sample that distribution. It is a
controlled five-body fixture built with `ParametricGeoEngine`: five fixed-location
hemispheres, flat label-0 basement, three body-aligned wells, no stochastic
stratigraphic history, no fault, no regional fold, no erosion, and an audit-only
merge of temporary labels 9--13 into label 9. It is useful for mechanism isolation
but must not be used as the builder for a Flow-distribution benchmark.

A new builder can produce unedited, full-history draws from the training recipe
with prospectively frozen deterministic seeds, but two caveats are mandatory:

1. the present legacy generator is not end-to-end seedable; setting only
   `np.random.seed` or only a Torch seed is insufficient;
2. the original streaming training run has no saved geology-seed/sample manifest,
   so exact seed-disjointness from training cannot be proven retrospectively.

The defensible claim is therefore **prospectively frozen, independently generated,
same-recipe test cases**. It is not a cryptographically proven training-disjoint
split. If cases are required to contain a hidden raw-label-9 target and specified
structural event types, the benchmark is additionally a predeclared conditional
subset of the training generator, not an unconditional sample from it.

## 1. Authoritative training generation entrypoint

The training entrypoint is
`project/geodata-3d-conditional/model_train_sh_inference_cond.py`.
`get_data_loader()` constructs `GeoData3DStreamingDataset` with:

- resolution `(64, 64, 64)`;
- bounds `((-1920, 1920), (-1920, 1920), (-1920, 1920))`;
- epoch size `20_000`;
- no `generator_config`, so the packaged default Markov matrix is used.

The dataset implementation is
`StructuralGeo-main/src/geogen/dataset/dataset.py`. Its default generator is
`MarkovGeostoryGenerator`, and each `__getitem__` call performs:

```python
model = self.model_generator.generate_model()
model.fill_nans()
data = model.get_data_grid()
```

The integer-valued float tensor is returned as shape `[1, 64, 64, 64]`. During
training, `make_combined_mask(batch)` supplies stochastic surface+borehole hard
conditioning. Raw geology labels are `-1..13`; the Flow embedding indices are
the shifted values `0..14`.

The full history generator is
`StructuralGeo-main/src/geogen/generation/model_generators.py`:

- Markov start state: `BaseStrata`;
- end state: `End`;
- maximum sequence length: 20;
- transition recipe:
  `StructuralGeo-main/src/geogen/generation/markov_matrix/default_markov_matrix.csv`;
- model construction: `GeoModel(..., height_tracking=True)` followed by
  `compute_model(normalize=True)`;
- remaining NaN/air is converted to `-1` by the dataset.

The default matrix can select sedimentation, erosion, dike, sill, pluton,
ore-deposit, fold, and fault events. Event subtypes and their probabilities are
defined in `categorical_events.py`; their geometries and parameter distributions
are defined in `geowords.py`. `Mountains` is exported as an event class but is not
a state in the current default matrix. A full-recipe draw therefore has access to
the complete event family but is not guaranteed to contain every event type.

## 2. `cond_generation_0/true_model.pt` provenance

The code lineage is unambiguous:

1. `project/geodata-3d-conditional/inference_demo.ipynb`, execution cell 4,
   imports `create_cond_data` from `model_inference_experiments.py`;
2. the notebook sets `save_dir` to `samples/jupyter-demo`, title
   `cond_generation`, and `num_samples=4`;
3. `create_cond_data()` constructs the same `GeoData3DStreamingDataset` with
   `(64,64,64)` and `(-1920,1920)^3`;
4. `synthetic_model = dataset[0].unsqueeze(0)` becomes
   `cond_generation_0/true_model.pt` through `save_model_and_boreholes()`;
5. `make_combined_mask(synthetic_model)` creates its random surface/borehole
   condition, and all values outside that mask are set to sentinel `-1` in
   `boreholes.pt`.

`dataset_size=100_000` in this helper affects only `__len__`; it does not select a
stored sample or a dataset index. `dataset[0]` triggers a fresh stochastic model.

Current artifact anchors are:

| Asset | SHA-256 |
|---|---|
| `samples/jupyter-demo/cond_generation_0/true_model.pt` | `a14f5a740a3ea6af8c2eec8507d79fe3a94f1b704626f0478a9e526f33b71cb3` |
| `samples/jupyter-demo/cond_generation_0/boreholes.pt` | `1e2592ce0e820569d0b3fa13eaa011c3bc70c312a19e5eab878c428facdbdb41` |

The truth is float32 `[1,1,64,64,64]` with observed raw labels
`-1,1,2,3,4,5,6,9,13`; label 9 has 8,968 voxels. The saved condition contains
nine full vertical wells at these `(x,y)` coordinates:

```text
(8,46), (9,5), (10,24), (27,17), (35,26),
(39,59), (44,60), (48,6), (57,32)
```

Only 13 of the 8,968 label-9 voxels are observed by the saved condition.

The boundary of what can be proven is equally important:

- `*.pt` is ignored by Git, so Git history does not contain these tensors;
- neither the notebook nor `create_cond_data()` records a geology seed;
- the notebook's `inference.seed=None` is not passed to StructuralGeo;
- the Torch RNG state used for the jittered boreholes is not recorded;
- no generated Markov sequence, selected event subtype, event parameter log,
  source hash manifest, NumPy RNG state, or StructuralGeo environment manifest
  accompanies the case.

Consequently the saved tensor's code path and content hash are known, but its
exact random realization cannot be regenerated from recorded provenance. Phase 1
and Phase 2 consume this saved case; they do not independently create a different
truth. Existing project documents correctly classify it as same-distribution
synthetic validation data and not held-out evidence.

## 3. Full training/Phase-1 generation versus the native audit fixture

| Property | Training and `cond_generation_0` | `native_geology_audit.py` |
|---|---|---|
| Generator | `GeoData3DStreamingDataset` -> `MarkovGeostoryGenerator` | `ParametricGeoEngine` with explicit `GeoModelSpec` |
| Domain | physical bounds `(-1920,1920)^3`, `64^3` | index-like bounds `(0,63)^3`, `64^3` |
| History | stochastic Markov history, up to 20 states | exactly five explicit intrusion events plus basement |
| Background | stochastic `BaseStrata`: basement and/or full sedimentary foundation | flat label-0 basement with fixed top at `z=55` |
| Stratigraphy | labels 1--5; uniform/Markov/tilted base strata and later fine/coarse/single sediment events | absent |
| Erosion/unconformity | flat, tilted, cut-fill, or wave variants can occur | absent |
| Fault | normal, reverse, strike-slip, horst/graben, random, or sequence can occur | absent |
| Fold | simple, shaped, or Fourier regional folds can occur; some intrusion/fault words also contain local fold transforms | no separate regional fold event |
| Dike/intrusion | dikes, warped dikes, dike groups, sills/sill systems, laccoliths/lopoliths, and ore blobs can occur | five `IntrusionSpec(kind="hemisphere")` bodies only |
| Locations | sampled by generator and transformed/backtracked through the history | fixed anchors `(10,10,17)`, `(26,10,37)`, `(42,10,25)`, `(12,42,21)`, `(42,42,39)` |
| Geometry variability | event-specific stochastic sizes, orientations, waves, deformation and overprinting | only diameter, height, minor-axis scale and rotation vary around fixed anchors |
| Surface | generated from normalized geological history; air is filled as `-1` | fixed planar air boundary at `z=56` |
| Wells | training: 8--31 jittered-grid wells; `cond_generation_0`: one random nine-well realization | exactly the first three intrusion anchors, deliberately body-aligned |
| Raw labels | `-1` air, `0` bedrock, `1..5` sediment, `6..8` dike, `9..11` intrusion, `12..13` blob | temporary event IDs `9..13`, then all five bodies are rewritten to 9 |
| Normalization | enabled; height tracking enabled | disabled; height tracking disabled |
| Intended use | learned prior and same-recipe synthetic validation | controlled causal/mechanism audit |

The training label constants in `geowords.py` are:

```text
bedrock 0
sediment 1..5
dike 6..8
intrusion 9..11
blob/ore 12..13
air/unfilled -1
```

Therefore raw label 9 is an intrusion-class value, but it is not one universal
shape or event identity. It can arise from a dike group, sill/sill system,
laccolith, or lopolith recipe. Conversely, temporary use of 12 and 13 as
hemisphere event IDs in the native fixture deliberately does not preserve the
training label semantics; the later merge is valid only for that audit.

## 4. Deterministic same-recipe generation: feasibility and limitation

### Feasibility

Yes: a future builder can use the same default Markov matrix, event templates,
GeoWords, bounds, resolution, normalization, label mapping, and final NaN fill,
while assigning new deterministic root seeds. This preserves the generator
recipe and leaves each resulting truth unedited.

### Why the current seed interface is insufficient

The current code mixes three RNG mechanisms:

1. `pydtmc.MarkovChain.simulate(..., seed=None)` creates its own RNG;
2. each `GeoWord(seed=None)` creates `np.random.default_rng(None)`, seeded from
   fresh entropy;
3. several GeoWords, probability helpers, Fourier helpers, and height
   normalization still call legacy global `np.random.*` functions.

The training `DataLoader` also uses 16 workers. No worker-to-geology seed contract
is recorded. Thus `torch.manual_seed(seed)` or `np.random.seed(seed)` alone cannot
make a model repeatable.

Before building cases, implement one explicit root `SeedSequence` that spawns
named child streams for:

- Markov sequence selection;
- event-template subtype selection;
- each instantiated GeoWord and nested GeoWord;
- legacy random-variable/Fourier helpers;
- height normalization;
- condition-layout generation, if a generated rather than fixed layout is used.

The preferred implementation is proper RNG injection, not a runtime monkeypatch
of `np.random.default_rng`. Record NumPy, PyDTMC, SciPy, Torch, Python and source
versions. The current inspected environment is NumPy 1.26.4, PyDTMC 8.0.0,
SciPy 1.15.3 and Torch 2.8.0+cu128.

Required deterministic regression checks before any benchmark generation:

1. two fresh processes with the same root seed produce identical Markov/event
   histories, categorical tensor bytes and hashes;
2. a different root seed changes the history or categorical tensor;
3. single-process and fresh-process results agree;
4. raw output remains finite, integer-valued and within `-1..13`;
5. bounds, resolution, normalization, default matrix hash and all generator
   source hashes match the frozen recipe;
6. generation does not invoke the Flow checkpoint, sampling, or geophysics.

### What “held out” may mean here

Because the historical training samples and RNG states were not saved, strict
non-overlap with the training stream cannot be audited. Continuous event
parameters make an exact accidental duplicate unlikely, but likelihood is not
provenance.

Use this wording:

> Independently generated, prospectively registered same-recipe test cohort;
> no case was used to choose inference parameters or to train/update the frozen
> checkpoint. Historical sample-level training overlap cannot be certified
> because the original streaming run did not retain a seed/sample manifest.

Do not use “proven training-disjoint held-out set” without new training provenance.

## 5. Automatic hidden-label-9 target without modifying truth

The canonical target should retain the training label semantics:

```python
raw_label9 = truth == 9
hidden_label9 = raw_label9 & ~condition_mask
```

No voxel in `truth` is changed. In particular, do not implant an intrusion, move
an intrusion, force an anchor, delete an intersected part, or merge labels 10--13
into label 9.

Use 6-connected components of the final raw-label-9 raster for geometry reporting.
The recommended deterministic policy is:

- primary evaluation target: all `hidden_label9` voxels, matching the Phase-1
  class-level target semantics;
- primary-body diagnostic: the final raw-label-9 connected component with the
  largest number of hidden voxels; ties are resolved by the component ID produced
  by lexicographic raster scan;
- retain and report every other label-9 component; the primary-body designation
  does not alter truth or suppress other target-class voxels.

This selects an unobserved target automatically after the fixed condition is
formed. Truth/component information remains retrospective and must not enter
Flow input, seismic inversion, bridge construction, ranking, tuning or stopping.

Final connected components are spatial bodies, not guaranteed one-to-one event
identities: separate same-label intrusions can merge, and later events can split
or overprint a body. If event identity is required, save the generated packed and
unpacked history and add deposition provenance during generation. Do not obtain
event identity by repurposing valid raw rock labels.

For a target-specific benchmark, predeclare a seed-ordered eligibility rule before
running any downstream method. The minimum logical rule is:

```text
raw label-9 voxel count > 0
hidden label-9 voxel count > 0
```

For a `full_complexity_targeted_v1` cohort, a stricter history-only rule may also
require:

- a sedimentary/base-strata history;
- at least one fold or fault event;
- at least one label-9-producing intrusion event;
- finite, nonempty final label-9 support outside the condition mask.

Select the first `K` eligible seeds in a predeclared seed order. Preserve every
rejected seed and reason in the registry. Do not select on target centroid,
similarity to another truth, future seismic response, bridge performance, or Flow
performance. This cohort samples
`p_training(geology | declared full-complexity and hidden-label9 eligibility)`,
not the unconditional training distribution; the report must say so.

## 6. Unified truth-independent borehole layout

Reuse the nine `(x,y)` positions recovered from the canonical Phase-1 case for
every new truth:

```python
CANONICAL_NINE_WELL_XY = (
    (8, 46), (9, 5), (10, 24),
    (27, 17), (35, 26), (39, 59),
    (44, 60), (48, 6), (57, 32),
)
```

This layout is fixed before seeing any new truth, uses nine wells within the
training range of 8--31, and is close to the original jittered-grid conditioning
style. It also makes cross-case comparisons use identical acquisition geometry.

Construct the condition as:

```text
fixed vertical-well mask from CANONICAL_NINE_WELL_XY
OR
make_surface_mask(truth)
```

The surface values necessarily depend on the generated topography because the
surface is an observed condition; the borehole coordinates do not. Save an
explicit boolean `condition_mask.pt` in addition to `boreholes.pt`, because the
sentinel `-1` represents both true air and unobserved voxels and cannot by itself
recover the mask unambiguously.

No truth-independent sparse layout can mathematically guarantee that every random
label-9 realization leaves target support unobserved. The guarantee must be made
without moving wells: generate with the frozen layout, compute
`hidden_label9`, and apply the predeclared `hidden_label9_voxels > 0` eligibility
gate. If a seed fails, record it; do not relocate a well or edit the geology.

Recommended saved condition assets are:

- `surface_mask.pt`;
- `borehole_mask.pt`;
- `condition_mask.pt`;
- `condition_values.pt` or compatibility-named `boreholes.pt`;
- `well_xy.json`;
- hashes for every asset.

## 7. Recommended benchmark builder

Create, only after separate authorization, a geology-only builder such as:

```text
project/geodata-3d-conditional/
  guidance/full_structuralgeo_benchmark.py
  scripts/benchmarks/build_full_structuralgeo_benchmark.py
  experiments/full_structuralgeo_benchmark/
    seed_registry.json
    cases/<case_id>/
      truth/true_model.pt
      truth/label9_mask.pt
      truth/hidden_label9_mask.pt
      truth/history.json
      condition/boreholes.pt
      condition/condition_mask.pt
      condition/surface_mask.pt
      condition/borehole_mask.pt
      condition/well_xy.json
      manifest.json
```

The builder's frozen algorithm should be:

1. validate a prospectively written seed registry and frozen recipe/source hashes;
2. instantiate the seed-plumbed `MarkovGeostoryGenerator` with the training
   bounds, resolution and default matrix;
3. retain packed/unpacked event history and generated parameter provenance;
4. compute the model with training normalization/height-tracking semantics;
5. call `fill_nans()` and save the raw unmodified categorical truth;
6. form the fixed nine-well mask and the existing surface mask;
7. derive raw/hidden label-9 masks and 6-connected geometry metadata;
8. apply only the predeclared history/target-availability eligibility rule;
9. write all accepted and rejected seed records, source/version metadata and
   content hashes atomically;
10. stop. Do not create seismic, load the checkpoint, run Flow, or launch Stage11-A.

The benchmark manifest should contain at least:

- root seed and named child-seed derivation/version;
- seed-order position, eligibility result and rejection reason;
- Markov state sequence, selected event subtypes and packed/unpacked history;
- generator matrix, source and environment hashes;
- bounds, resolution, normalization and height-tracking settings;
- raw label counts and label-semantics version;
- well coordinates and condition-mask hashes;
- label-9 total/observed/hidden voxel counts;
- 6- and 26-connected component summaries, centroids and bounding boxes;
- truth-only versus inference-visible asset roles;
- a statement that no seismic/Flow/downstream metric was consulted.

Freeze all accepted cases together before any downstream run. The builder should
refuse overwrite by default.

## 8. Existing code/configuration to reuse

Reuse without changing scientific semantics:

- `model_train_sh_inference_cond.py:get_config` for canonical bounds,
  resolution and category count;
- `StructuralGeo-main/src/geogen/dataset/dataset.py` for the authoritative
  training data path and final `fill_nans`/tensor convention;
- `MarkovGeostoryGenerator`, `MarkovMatrixParser`, and
  `default_markov_matrix.csv` for the history distribution;
- `categorical_events.py` and `geowords.py` for event subtype distributions,
  process geometry and raw label semantics;
- `GeoModel.compute_model(normalize=True)`, height tracking, `fill_nans()`,
  `get_data_grid()`, and `get_history_string(unpacked=True)`;
- `boreholes.py:make_surface_mask` for the existing surface-conditioning
  convention;
- the tensor-saving convention in
  `model_inference_experiments.py:save_model_and_boreholes` where compatibility
  is needed;
- `native_geology_audit.py:connected_target_statistics` for deterministic
  6-connected reporting, after generalizing paths/types if necessary;
- existing project SHA-256/manifest/truth-firewall utilities for asset roles and
  immutable provenance.

Add rather than silently alter:

- explicit end-to-end RNG injection for the legacy Markov generator;
- a fixed-coordinate `make_fixed_boreholes_mask(shape, well_xy)` helper;
- a geology-only case manifest and accepted/rejected seed registry;
- explicit condition masks and truth-only target masks.

Do not reuse as the full benchmark truth builder:

- `build_structuralgeo_native_case()`;
- its fixed anchors or drilled/hidden body roles;
- its fixed `z=56` planar surface;
- its labels-9--13 temporary event-ID/merge mechanism;
- its three body-aligned wells;
- any Stage11 seismic, bridge, candidate or Flow result to choose cases.

## Final answers

1. **Training entrypoint:** `model_train_sh_inference_cond.py:get_data_loader`
   -> `GeoData3DStreamingDataset` -> default `MarkovGeostoryGenerator`.
2. **`cond_generation_0` source:** notebook cell -> `create_cond_data()` ->
   `dataset[0]` -> `save_model_and_boreholes()`. The content hashes are known;
   its geology seed, event history and exact source/environment snapshot are not.
3. **Generator comparison:** the native audit is a controlled five-hemisphere
   fixture, not a draw from the full training/Phase-1 generator distribution.
4. **New full truths:** feasible with the same recipe after end-to-end seed
   plumbing. They can be prospectively independent/same-recipe, but historical
   training-disjointness cannot be proven from existing records.
5. **Hidden label 9:** use the unmodified raw-label-9 support outside the fixed
   condition mask; use deterministic connected-component reporting, not relabeling
   or manual geometry edits.
6. **Boreholes:** freeze the canonical nine-well `(x,y)` layout for all cases,
   combine it with `make_surface_mask`, save the explicit mask, and enforce hidden
   support through a recorded eligibility check rather than moving wells.
7. **Next implementation:** a standalone geology-only
   `build_full_structuralgeo_benchmark.py` with seed registry, histories, masks,
   source/version hashes and a no-overwrite rule. It should stop before seismic,
   Flow and Stage11-A.

