# Full StructuralGeo benchmark build report

Date: 2026-08-11
Machine decision: `FULL_STRUCTURALGEO_BENCHMARK_READY`

## Scope and frozen stop rule

This build is geology-only. It did not generate seismic, load a Flow checkpoint, run inference, run property inversion, build a probability bridge, or execute Stage11/Stage12B.

The benchmark uses the training generator path `MarkovGeostoryGenerator` with the packaged default Markov matrix, `(-1920,1920)^3` bounds, `64^3` resolution, height tracking, height normalization, and final NaN-to-`-1` fill. Truth voxels were not edited, implanted, moved, deleted, relabeled, or merged.

## Repository snapshot

- Branch: `main`
- HEAD: `72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f`
- Dirty at build start: `True`
- Pre-build `git status --short`:

```text
 M StructuralGeo-main/src/geogen/dataset/dataset.py
 M StructuralGeo-main/src/geogen/generation/categorical_events.py
 M StructuralGeo-main/src/geogen/generation/geowords.py
 M StructuralGeo-main/src/geogen/generation/model_generators.py
 M StructuralGeo-main/src/geogen/model/geomodel.py
 M StructuralGeo-main/src/geogen/model/metaballs.py
 M StructuralGeo-main/src/geogen/probability/random_varibles.py
 M StructuralGeo-main/src/geogen/probability/sedimentbuilders.py
 M StructuralGeo-main/src/geogen/probability/wavegenerators.py
 M project/geodata-3d-conditional/docs/DEVELOPMENT_HANDOFF.md
 M project/geodata-3d-conditional/docs/NEXT_CONVERSATION_PROMPT.md
?? StructuralGeo-main/src/geogen/generation/rng_contract.py
?? project/geodata-3d-conditional/docs/FULL_STRUCTURALGEO_BENCHMARK_AUDIT.md
?? project/geodata-3d-conditional/docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md
?? project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/
?? project/geodata-3d-conditional/experiments/stage10_geophysical_probability_bridge/
?? project/geodata-3d-conditional/experiments/stage11_diverse_geometry_bridge/
?? project/geodata-3d-conditional/experiments/stage9_flow_prior_posterior/
?? project/geodata-3d-conditional/guidance/full_structuralgeo_benchmark.py
?? project/geodata-3d-conditional/guidance/geophysical_probability_bridge.py
?? project/geodata-3d-conditional/guidance/prior_ensemble.py
?? project/geodata-3d-conditional/guidance/stage11_diverse_geometry.py
?? project/geodata-3d-conditional/paper/
?? project/geodata-3d-conditional/scripts/benchmarks/
?? project/geodata-3d-conditional/scripts/paper_figures/
?? project/geodata-3d-conditional/scripts/stage10/
?? project/geodata-3d-conditional/scripts/stage11/
?? project/geodata-3d-conditional/scripts/stage9/
?? project/geodata-3d-conditional/tests/test_paper_figures.py
?? project/geodata-3d-conditional/tests/test_stage10_probability_bridge.py
?? project/geodata-3d-conditional/tests/test_stage9_enrichment.py
?? project/geodata-3d-conditional/tests/test_stage9_prior_ensemble.py
?? project/geodata-3d-conditional/tests/test_stage9_truth_firewall.py
```

## Deterministic RNG gate

RNG contract: `structuralgeo_named_seedsequence_v1`. Gate passed: `True`.

The contract starts from one root `numpy.random.SeedSequence` and derives order-independent named streams for the Markov sequence, categorical event subtype, each event/GeoWord and nested GeoWord, legacy probability helpers, Fourier helpers, sediment helpers, metaballs, and height normalization. The fixed condition layout requires no random stream.

- `same_seed_two_fresh_processes`: `True`
- `single_process_matches_fresh_process`: `True`
- `different_seed_changes_history_or_tensor`: `True`
- `source_and_matrix_hashes_identical_across_processes`: `True`
- `config_matches_training_recipe`: `True`
- `shape`: `True`
- `finite`: `True`
- `integer_valued`: `True`
- `labels_within_frozen_range`: `True`
- `bounds`: `True`
- `resolution`: `True`
- `height_tracking`: `True`
- `normalize`: `True`
- `fill_nans`: `True`

The full probe output, including two fresh-process replicates, a single-process replicate, a different-seed probe, categorical tensor SHA-256 values, histories, shape/range checks, matrix hash, and frozen source hashes, is in `audit/determinism_audit.json`.

## Actual source files changed

- `StructuralGeo-main/src/geogen/dataset/dataset.py`
- `StructuralGeo-main/src/geogen/generation/categorical_events.py`
- `StructuralGeo-main/src/geogen/generation/geowords.py`
- `StructuralGeo-main/src/geogen/generation/model_generators.py`
- `StructuralGeo-main/src/geogen/generation/rng_contract.py`
- `StructuralGeo-main/src/geogen/model/geomodel.py`
- `StructuralGeo-main/src/geogen/model/metaballs.py`
- `StructuralGeo-main/src/geogen/probability/random_varibles.py`
- `StructuralGeo-main/src/geogen/probability/sedimentbuilders.py`
- `StructuralGeo-main/src/geogen/probability/wavegenerators.py`
- `project/geodata-3d-conditional/guidance/full_structuralgeo_benchmark.py`
- `project/geodata-3d-conditional/scripts/benchmarks/build_full_structuralgeo_benchmark.py`
- `project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/configs/full_complexity_targeted_v1.json`

## Prospectively frozen cohort rule

- Eligibility: `full_complexity_targeted_v1`
- Target accepted cases: `5`
- Maximum seed budget: `128`
- Seeds evaluated in the arithmetic order starting at `120260001` with step `1`.
- Required before acceptance: BaseStrata history; at least one Fold or Fault event; at least one raw-label9-producing Dike/Sills/Pluton event; final raw label9 > 0; hidden raw label9 under the fixed condition > 0.
- Forbidden selection variables: target centroid, visual attractiveness, similarity to cond_generation_0, seismic response, future bridge performance, and Flow performance.

## Accepted cases

| Case | Root seed | label9 total | observed | hidden | hidden fraction | Markov history |
|---|---:|---:|---:|---:|---:|---|
| fullgeo_case01 | 120260003 | 120 | 2 | 118 | 0.983333 | BaseStrata → Pluton → Dike → Sills → Dike → Sediment → Dike → Sediment → Fold → Fold → OreDeposit → Fold → Dike → OreDeposit → Fold → Fault → Sediment → Fold → Fold → Fold → Sediment |
| fullgeo_case02 | 120260004 | 16083 | 37 | 16046 | 0.997699 | BaseStrata → Sills → Erosion → Fold → Fold → Erosion → Fault → Dike → Sediment → Erosion → Fold → Fold → Fault → Sediment → OreDeposit → Fault → Sediment → Fold → Fault → Fold → Dike |
| fullgeo_case03 | 120260008 | 1736 | 3 | 1733 | 0.998272 | BaseStrata → Dike → OreDeposit → Sediment → Erosion → Dike → OreDeposit → Sediment → Erosion → Sediment → Dike → OreDeposit → Sediment → Erosion → Fold → Fold → Fold → Dike → OreDeposit → Sediment → Fold |
| fullgeo_case04 | 120260011 | 13293 | 364 | 12929 | 0.972617 | BaseStrata → Erosion → Sediment → Erosion → Fold → Fault → OreDeposit → Fault → Fold → Fold → Erosion → Fault → Fault → Fault → Erosion → Fault → Fold → Fault → Dike → Dike → OreDeposit |
| fullgeo_case05 | 120260012 | 4111 | 9 | 4102 | 0.997811 | BaseStrata → OreDeposit → OreDeposit → Fault → Dike → Erosion → Dike → Fold → Fold → Sediment → Erosion → Fault → Sediment → Fault → Erosion → Fault → Sediment → Fold → Fault → Sediment → Erosion |

Each case manifest records event subtypes, packed/unpacked histories, raw label counts, fixed well coordinates, condition hashes/statistics, 6/26-connected label9 geometry, and the retrospective primary-body diagnostic. `seed_registry.json` and `rejected_seed_registry.json` preserve the complete prospective search trace.

## Rejected seeds

Rejected count: `7`.

| Root seed | Rejection reason(s) | Markov sequence |
|---:|---|---|
| 120260001 | missing_label9_producing_intrusion_event | BaseStrata → Fold → Fault → Erosion → Sediment → Erosion → End |
| 120260002 | missing_label9_producing_intrusion_event | BaseStrata → Fold → Sediment → Fault → End |
| 120260005 | missing_label9_producing_intrusion_event | BaseStrata → Sediment → Dike → Erosion → Fault → Fold → Fault → Fold → Fold → Sediment → Erosion → Sediment → Fault → Dike → End |
| 120260006 | missing_label9_producing_intrusion_event | BaseStrata → Pluton → Fault → Erosion → Fold → Sediment → Fold → Dike → Fault → Fold → Fold → End |
| 120260007 | missing_label9_producing_intrusion_event | BaseStrata → Fault → Fault → Fault → Fold → Fault → OreDeposit → OreDeposit → Sediment → Fold → Fault → Fault → Dike → OreDeposit → Fold → Sediment → Dike → Sediment → Erosion → Sediment → End |
| 120260009 | missing_label9_producing_intrusion_event | BaseStrata → Fault → Erosion → Sediment → Fold → Fold → Fault → OreDeposit → Fold → Dike → OreDeposit → Fault → Fault → Fault → Sediment → OreDeposit → Erosion → Erosion → Dike → Sediment → Fault |
| 120260010 | missing_label9_producing_intrusion_event | BaseStrata → Fold → Fold → Sediment → End |

## Fixed condition and hidden-target semantics

All cases use the same nine full-depth wells: `(8,46), (9,5), (10,24), (27,17), (35,26), (39,59), (44,60), (48,6), (57,32)`. The condition mask is exactly the fixed borehole mask OR the existing `make_surface_mask(truth)`. Explicit surface, borehole, and union masks are saved; no inference relies on sentinel `-1` alone.

The target is exactly `raw_label9 = truth == 9`; hidden target is exactly `raw_label9 & ~condition_mask`. Labels 10–13 remain unchanged and are not merged into label9.

## Provenance hashes

Default matrix SHA-256: `cb068ecb1d65487828899179fa02e0aff279703504686c569451fc871dd3784b`

| Source | SHA-256 |
|---|---|
| `StructuralGeo-main/src/geogen/dataset/dataset.py` | `abd2aca08258bc1c4f34cca0ee585878280ab204cad4aef52ce637700847db3a` |
| `StructuralGeo-main/src/geogen/generation/categorical_events.py` | `63f8431e79c6b1fc4937a480c83e18bd69dd04cc18a2b81c427a4149e3233a15` |
| `StructuralGeo-main/src/geogen/generation/geowords.py` | `7acb56bf56a07a1bbe4ef8a2a2c7245b3942cbcf84d55931b2025bf2b016ca03` |
| `StructuralGeo-main/src/geogen/generation/markov_matrix/default_markov_matrix.csv` | `cb068ecb1d65487828899179fa02e0aff279703504686c569451fc871dd3784b` |
| `StructuralGeo-main/src/geogen/generation/model_generators.py` | `2ee1b5fe4bcd89e70d80c3b78af02a7696af7fdedacca3ed391145e3df67e397` |
| `StructuralGeo-main/src/geogen/generation/rng_contract.py` | `96e284b99df9ef8444b00e15e2ecfc92ce85c3c3bfdb9328474690937782a022` |
| `StructuralGeo-main/src/geogen/model/geomodel.py` | `a7d575bb2864cff02bc6925f61ca7bc1a4d2c7b9f75006e61b3b30a4dc33a094` |
| `StructuralGeo-main/src/geogen/model/metaballs.py` | `c0e5d2c2ccd5cdc58d4ea35639221079718a7629d6960711b75b99e53e845be6` |
| `StructuralGeo-main/src/geogen/probability/random_varibles.py` | `7188a98bba406c60ea402e28d94f67aad41503d463bd25663717398226bd384e` |
| `StructuralGeo-main/src/geogen/probability/sedimentbuilders.py` | `01066bd55e01f5d2354406419a70dd1056456df0da5257f7a07763840db67243` |
| `StructuralGeo-main/src/geogen/probability/wavegenerators.py` | `2ff37e2df0e5b76e4d1d39a93c1af661fdda787d707da0ddde5a096f115d1e56` |
| `project/geodata-3d-conditional/boreholes.py` | `b0656303987fef04241a70a1df1a6ca4c2897d6c1e4de6470f492487fc038630` |
| `project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/configs/full_complexity_targeted_v1.json` | `39bb41e8b7f0d6b1597c8ba81d5762603616a9484c1e3b855c8c43fd570f50b0` |
| `project/geodata-3d-conditional/guidance/full_structuralgeo_benchmark.py` | `3d292e6b48dbc2ed6911315d9e736afd770e7e164785208584dc268cd1d18097` |
| `project/geodata-3d-conditional/model_train_sh_inference_cond.py` | `36d0dc613d8860677f137a2f1d681ebcc50829a6d90f5547c6eac3fd12a977c2` |
| `project/geodata-3d-conditional/scripts/benchmarks/build_full_structuralgeo_benchmark.py` | `96f05f1ab3e87fbb7ffcd73790cf1fa9ecbd0741c145577221d4324ea8bfda24` |

## QC figures

QC figures were generated only after acceptance. They use identical lithology colors and camera settings and were not used to accept, reject, or replace any seed.

- `qc/fullgeo_case01_qc.png`
- `qc/fullgeo_case01_qc.pdf`
- `qc/fullgeo_case02_qc.png`
- `qc/fullgeo_case02_qc.pdf`
- `qc/fullgeo_case03_qc.png`
- `qc/fullgeo_case03_qc.pdf`
- `qc/fullgeo_case04_qc.png`
- `qc/fullgeo_case04_qc.pdf`
- `qc/fullgeo_case05_qc.png`
- `qc/fullgeo_case05_qc.pdf`

## Training-overlap statement

independently generated, prospectively registered same-recipe test cohort; no case was used to choose inference parameters or update the frozen checkpoint. Historical sample-level overlap with the original streaming training run cannot be certified because no seed/sample manifest was retained.

## Machine decision

`FULL_STRUCTURALGEO_BENCHMARK_READY`

Stage 12A stops here and awaits manual approval. No downstream experiment was started.
