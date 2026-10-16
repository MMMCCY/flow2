# Ensemble run: stage19_test_case004

Actual independent draws: 200; planned: 200; failed: 0.

Case selection: historical_adapter_median_iou_best; uses historical true target IoU. This selected-case illustration is not an unbiased cross-case comparison.

Baseline is computed once per source and shared across the two identical-baseline comparisons. No pseudo-pairing between different selected Top100 sets.

Frozen Flow, 32-step fixed Euler, raw labels -1..13, target9; XYZ 100/100/50 m, Z upward. Stage19R has a trained adapter. Binary high-contrast noiseless inverse-crime upper bound.

Observation IDs were written before evaluation truth was opened. Full-target oracle ranking is explicitly retrospective. No truth-controlled tuning or seed selection.

Undefined empty-union IoU and empty-prediction precision are null. Threshold masks are frequency summaries, not full geological realizations and carry no inherited seismic RMSE.

No new five-body experiment or per-object recovery claim. A mask-only condition overlay is labelled as condition voxels.



## STAGE15_TRACE_V4

Paired median IoU delta: +0.020074; paired median hard-seismic RMSE delta: +0.004201.

all200: median candidate IoU 0.348795, hard seismic RMSE 0.065094; threshold IoUs [0.5020539470428111, 0.6441281138790036, 0.8198107678137821, 0.5284208452369225]

random100: median candidate IoU 0.349123, hard seismic RMSE 0.065168; threshold IoUs [0.5055569399451781, 0.649403922574029, 0.8209018120522545, 0.5718925302627694]

obs_top100: median candidate IoU 0.371867, hard seismic RMSE 0.063446; threshold IoUs [0.517630249527989, 0.6567705970052505, 0.8284938020037358, 0.5613862360933346]

oracle_top100: median candidate IoU 0.373809, hard seismic RMSE 0.063686; threshold IoUs [0.5257048092868989, 0.6635905491197068, 0.8385242385242385, 0.5653800708940527]

Unique decoded models: 200. Identical independent outputs retain multiplicity.

## STAGE19R_ADAPTER

Paired median IoU delta: +0.294359; paired median hard-seismic RMSE delta: -0.019122.

all200: median candidate IoU 0.626976, hard seismic RMSE 0.041399; threshold IoUs [0.8706099815157117, 0.8869247766356038, 0.7913563308506468, 0.2789826498422713]

random100: median candidate IoU 0.626064, hard seismic RMSE 0.041457; threshold IoUs [0.8742043847241867, 0.8894873521208041, 0.7928802588996764, 0.34608181370133073]

obs_top100: median candidate IoU 0.654765, hard seismic RMSE 0.039906; threshold IoUs [0.8884982873625383, 0.8832433474739684, 0.7880898655940352, 0.3577484227129338]

oracle_top100: median candidate IoU 0.654906, hard seismic RMSE 0.040173; threshold IoUs [0.8912789127891279, 0.8862321635171616, 0.7939185875429132, 0.3638604100946372]

Unique decoded models: 200. Identical independent outputs retain multiplicity.
