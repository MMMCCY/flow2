# Ensemble run: stage19_test_case004

Actual independent draws: 3; planned: 200; failed: 0.

Case selection: historical_adapter_median_iou_best; uses historical true target IoU. This selected-case illustration is not an unbiased cross-case comparison.

Baseline is computed once per source and shared across the two identical-baseline comparisons. No pseudo-pairing between different selected Top100 sets.

Frozen Flow, 32-step fixed Euler, raw labels -1..13, target9; XYZ 100/100/50 m, Z upward. Stage19R has a trained adapter. Binary high-contrast noiseless inverse-crime upper bound.

Observation IDs were written before evaluation truth was opened. Full-target oracle ranking is explicitly retrospective. No truth-controlled tuning or seed selection.

Undefined empty-union IoU and empty-prediction precision are null. Threshold masks are frequency summaries, not full geological realizations and carry no inherited seismic RMSE.

No new five-body experiment or per-object recovery claim. A mask-only condition overlay is labelled as condition voxels.



## STAGE15_TRACE_V4

Paired median IoU delta: -0.013908; paired median hard-seismic RMSE delta: +0.004628.

all3: median candidate IoU 0.373882, hard seismic RMSE 0.062727; threshold IoUs [0.4886213478197034, 0.4886213478197034, 0.6774263588333213, 0.6774263588333213]

obs_top3: median candidate IoU 0.373882, hard seismic RMSE 0.062727; threshold IoUs [0.4886213478197034, 0.4886213478197034, 0.6774263588333213, 0.6774263588333213]

oracle_top3: median candidate IoU 0.373882, hard seismic RMSE 0.062727; threshold IoUs [0.4886213478197034, 0.4886213478197034, 0.6774263588333213, 0.6774263588333213]

random3: median candidate IoU 0.373882, hard seismic RMSE 0.062727; threshold IoUs [0.4886213478197034, 0.4886213478197034, 0.6774263588333213, 0.6774263588333213]

Unique decoded models: 3. Identical independent outputs retain multiplicity.

## STAGE19R_ADAPTER

Paired median IoU delta: +0.267342; paired median hard-seismic RMSE delta: -0.017485.

all3: median candidate IoU 0.671035, hard seismic RMSE 0.040153; threshold IoUs [0.8123587693377369, 0.8123587693377369, 0.7784228350266086, 0.7784228350266086]

obs_top3: median candidate IoU 0.671035, hard seismic RMSE 0.040153; threshold IoUs [0.8123587693377369, 0.8123587693377369, 0.7784228350266086, 0.7784228350266086]

oracle_top3: median candidate IoU 0.671035, hard seismic RMSE 0.040153; threshold IoUs [0.8123587693377369, 0.8123587693377369, 0.7784228350266086, 0.7784228350266086]

random3: median candidate IoU 0.671035, hard seismic RMSE 0.040153; threshold IoUs [0.8123587693377369, 0.8123587693377369, 0.7784228350266086, 0.7784228350266086]

Unique decoded models: 3. Identical independent outputs retain multiplicity.
