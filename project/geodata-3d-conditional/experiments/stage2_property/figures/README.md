# Phase-2 fixed-camera visual results

All directories were generated from completed strict alpha-zero/guided pairs
with `scripts/stage2/visualize_property_guidance.py`. Each directory contains:

- `truth_baseline_guided_3d.png`;
- `paired_changes_3d.png`;
- `ensemble_probability_isosurfaces_3d.png`;
- `vtk/{truth,baseline,guided}.vti`;
- `manifest.json` with run hashes, selected sample and hard metrics.

## Phase-2a successful samples

The following fixed sample IDs are shown without selecting only the best
realization. All belong to the 12/12 passing distinct-codebook confirmation.

| Seed | Sample | Figure directory | Label-9 IoU / P / R | Improved truth-present classes |
|---:|---:|---|---|---:|
| 42 | 0 | `phase2a_distinct_seed42_n4/` | 0.4816 / 0.9005 / 0.5087 | 6/8 |
| 42 | 3 | `phase2a_distinct_seed42_sample3/` | 0.5090 / 0.8943 / 0.5416 | 6/8 |
| 142 | 0 | `phase2a_distinct_seed142_sample0/` | 0.4568 / 0.9419 / 0.4700 | 6/8 |
| 142 | 3 | `phase2a_distinct_seed142_sample3/` | 0.4626 / 0.9196 / 0.4820 | 5/8 |
| 242 | 0 | `phase2a_distinct_seed242_sample0/` | 0.5235 / 0.9020 / 0.5551 | 5/8 |
| 242 | 3 | `phase2a_distinct_seed242_sample3/` | 0.4736 / 0.9005 / 0.4998 | 6/8 |

## Phase-2b sensitivity examples

- `phase2b_c100_seed42_n4_sample2/`: the sole failing sample of the 3/4 c100
  fallback, retained to show the major-body robustness miss;
- `phase2b_c010_seed42_n4_sample0/`: a low-contrast confirmed-failure example.

The empirical ensemble panels show hard-label occurrence frequency across the
four saved realizations. They are not soft decoder probability volumes.
