# Phase 1b protocol-v4 aggregate report

This report is generated from immutable saved run artifacts. The target is a
truth-derived label-9 probability oracle, not measured geophysics.

- Strict pairs: 12
- Seeds: 42, 142, 242
- Strict gate outcome: `not_full_pass`
- Stage decision: `mechanism_validated_with_topology_and_endpoint_caveats`

| Metric | Baseline mean | Guided mean | Mean delta |
|---|---:|---:|---:|
| `global_voxel_accuracy` | 0.597210 | 0.643152 | +0.045942 |
| `global_mean_iou` | 0.180441 | 0.217816 | +0.037376 |
| `target_iou` | 0.031445 | 0.809861 | +0.778416 |
| `target_precision` | 0.078787 | 0.827396 | +0.748609 |
| `target_recall` | 0.051963 | 0.974725 | +0.922762 |
| `target_absolute_volume_error_fraction` | 0.334746 | 0.185130 | -0.149615 |
| `target_centroid_distance` | 16.951820 | 3.379813 | -13.572007 |
| `target_connected_components` | 52.166667 | 66.833333 | +14.666667 |
| `largest_component_fraction` | 0.723943 | 0.385557 | -0.338386 |
| `selected_roi_iou` | 0.035114 | 0.939234 | +0.904121 |
| `selected_roi_precision` | 0.105030 | 0.962678 | +0.857648 |
| `selected_roi_recall` | 0.051963 | 0.974725 | +0.922762 |
| `selected_absolute_volume_error_fraction` | 0.510119 | 0.013883 | -0.496237 |
| `outside_roi_voxel_accuracy` | 0.639290 | 0.639312 | +0.000022 |

## Pre-registered gate audit

| Gate | Status | Value |
|---|---|---|
| `absolute_target_iou_ge_0p15` | PASS | 0.8098611425842716 |
| `absolute_precision_ge_0p25` | PASS | 0.8273961957231815 |
| `absolute_recall_ge_0p25` | PASS | 0.9747249479631281 |
| `mean_iou_delta_ge_0p08` | PASS | 0.77841633823898 |
| `centroid_reduction_ge_20_percent` | PASS | 0.8006223968342312 |
| `mean_hard_change_between_1_and_10_percent` | PASS | 0.051810900370279946 |
| `component_ratio_le_1p25` | FAIL | 1.2811501597444088 |
| `roi_component_ratio_le_1p25_diagnostic` | FAIL | 1.266540642722117 |
| `no_raw_largest_component_fraction_loss` | FAIL | 0.38555663693183084 |
| `inside_roi_change_share_ge_90_percent` | PASS | 0.9875876625169496 |
| `zero_condition_violations` | PASS | 0 |
| `outside_roi_accuracy_preserved` | PASS |  |
| `ensemble_diversity_preserved` | PASS |  |
| `visible_truth_aligned_structure` | PENDING | requires_protocol_v4_fixed_camera_render_and_visual_review |

## Truth-relative topology context

- Truth component sizes: [4079, 2192, 2043, 627, 22, 4, 1].
- Mean guided ROI top-four sizes: [4074.4166666666665, 2202.5, 2129.0833333333335, 600.1666666666666].
- Every guided sample has exactly four ROI components with at least 20 voxels.
- The raw component-count failure is retained; size-stratified evidence does not
  retroactively change the pre-registered threshold.
