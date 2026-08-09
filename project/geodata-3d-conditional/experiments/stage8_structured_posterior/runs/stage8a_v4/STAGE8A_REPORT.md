# Stage 8A report

Decision: **FAIL_STAGE8A_STOP_BEFORE_STAGE8B**

## Frozen gates

- `all_required_cases_present`: `True`
- `correct_specificity`: `False`
- `analytic_hidden_iou`: `False`
- `analytic_hidden_recall`: `False`
- `native_mean_iou`: `False`
- `native_mean_recall`: `False`
- `geometry_improves_from_empty`: `False`
- `conditions_exact`: `True`
- `truth_blind_selection`: `True`
- `identical_budgets`: `True`
- `unknown_count`: `False`

## Correct-arm birth diagnostics

- `analytic_five_body`: births=560, delta_rmse<0 vs parent=0, delta_rmse<0 vs empty=0
  - v4 improving scale-0.25 seeds=0; continuation attempts=0; transition successes={'0.25_to_0.50': 0, '0.50_to_0.75': 0, '0.75_to_1.00': 0}; final lineage candidates surviving global beam=40
- `native_seed20260807`: births=430, delta_rmse<0 vs parent=0, delta_rmse<0 vs empty=1
  - v4 improving scale-0.25 seeds=0; continuation attempts=0; transition successes={'0.25_to_0.50': 0, '0.50_to_0.75': 0, '0.75_to_1.00': 0}; final lineage candidates surviving global beam=21
- `native_seed20260808`: births=310, delta_rmse<0 vs parent=0, delta_rmse<0 vs empty=0
  - v4 improving scale-0.25 seeds=0; continuation attempts=0; transition successes={'0.25_to_0.50': 0, '0.50_to_0.75': 0, '0.75_to_1.00': 0}; final lineage candidates surviving global beam=8
- `native_seed20260809`: births=570, delta_rmse<0 vs parent=0, delta_rmse<0 vs empty=0
  - v4 improving scale-0.25 seeds=0; continuation attempts=0; transition successes={'0.25_to_0.50': 0, '0.50_to_0.75': 0, '0.75_to_1.00': 0}; final lineage candidates surviving global beam=32

Selection used hard observed seismic RMSE only. Truth metrics were computed only after selection files were frozen.

No training, fine-tuning, LoRA, Stage-9 SMC/RJMCMC, gravity, or magnetics were run.
