# Stage 8A report

Decision: **FAIL_STAGE8A_STOP_BEFORE_STAGE8B**

## Frozen gates

- `all_required_cases_present`: `True`
- `correct_specificity`: `True`
- `analytic_hidden_iou`: `False`
- `analytic_hidden_recall`: `False`
- `native_mean_iou`: `False`
- `native_mean_recall`: `False`
- `geometry_improves_from_empty`: `False`
- `conditions_exact`: `True`
- `truth_blind_selection`: `True`
- `identical_budgets`: `True`
- `unknown_count`: `True`

## Correct-arm birth diagnostics

- `analytic_five_body`: births=380, delta_rmse<0 vs parent=1, delta_rmse<0 vs empty=5
  - v3 smallest-scale improving births=1; growth probes=0; improving growth probes=0; matched v2 full-size failures rescued at v3 smallest scale=0
- `native_seed20260807`: births=490, delta_rmse<0 vs parent=1, delta_rmse<0 vs empty=0
  - v3 smallest-scale improving births=1; growth probes=0; improving growth probes=0; matched v2 full-size failures rescued at v3 smallest scale=0
- `native_seed20260808`: births=410, delta_rmse<0 vs parent=0, delta_rmse<0 vs empty=0
  - v3 smallest-scale improving births=0; growth probes=0; improving growth probes=0; matched v2 full-size failures rescued at v3 smallest scale=0
- `native_seed20260809`: births=560, delta_rmse<0 vs parent=1, delta_rmse<0 vs empty=0
  - v3 smallest-scale improving births=1; growth probes=0; improving growth probes=0; matched v2 full-size failures rescued at v3 smallest scale=0

Selection used hard observed seismic RMSE only. Truth metrics were computed only after selection files were frozen.

No training, fine-tuning, LoRA, Stage-9 SMC/RJMCMC, gravity, or magnetics were run.
