# Stage 8A-v3-R3 — trust-region beam / lineage survival audit

## Primary classification

`GLOBAL_BEAM_PRUNES_LOCALLY_IMPROVING_SEEDS`

The frozen `(hard RMSE, state_id)` beam ordering was reconstructed from
the existing 961-row trace of every correct arm. No proposal, geology
materialization, hard petrophysical mapping, seismic forward, or truth
metric was evaluated.

## Counts over all correct-arm scale-0.25 births

- Total smallest-scale births: 1840.
- `N(delta_parent < 0)`: 3.
- `N(delta_empty < 0)`: 5.
- `N(delta_parent < 0 and entered_next_beam)`: 0.
- `N(delta_empty < 0 and not entered_next_beam)`: 5.

Locally improving child-minus-cutoff margins were all positive: min=8.94628465176e-06, median=9.83476638794e-06, max=5.4975040257e-05. A positive margin is worse than the
eighth-place cutoff and therefore means correct deterministic pruning.

## Three locally improving correct-arm births

| Case | Gen | Δ parent | Δ empty | Rank / 96 | Cutoff margin | Entered beam | Exact v2 geometry match |
|---|---:|---:|---:|---:|---:|---|---|
| analytic_five_body | 9 | -3.40305268764e-06 | -6.59199431539e-05 | 10 | 9.83476638794e-06 | False | 0 |
| native_seed20260807 | 4 | -3.31550836563e-07 | 5.4975040257e-05 | 50 | 5.4975040257e-05 | False | 0 |
| native_seed20260809 | 6 | -1.3055279851e-05 | 8.94628465176e-06 | 30 | 8.94628465176e-06 | False | 0 |

All three satisfied the frozen local continuation rule but ranked 10th,
50th, and 30th, respectively, so none entered the eight-member beam.
The analytic child also beat the empty reference; the two native children
improved their immediate parents but remained worse than empty. Complete
ancestor-by-ancestor records are in `locally_improving_lineages.csv`.

## Selection and implementation audit

For generations 1–9, every persisted next-generation parent list exactly
equals the eight candidates reconstructed by ascending hard RMSE with
state-id tie-breaking. No candidate with reconstructed rank 1–8 was
discarded, and none with rank above 8 entered. All 1,840 birth parents
were valid frozen beam states. The selection-defect count is zero.

Continuation was not executed for the three improving children for one
deterministic reason: each child was pruned by global beam competition
before it could become a later proposal parent. The other 1,837 births
failed the strict local hard-RMSE improvement rule.

## Frozen v2-v3 alignment

None of the three locally improving branches has an exact match in its
existing v2-v3 full-target geometry table. Therefore R3 preserves the
frozen finding that no matched geometry demonstrates `v2 full-size fails /
v3 small-scale succeeds`; unmatched proposals are not reinterpreted as
such evidence.

## Exactly one future recommendation

Use lineage-preserving local hard-loss continuation before global beam competition.

This recommendation is not implemented in R3. It must reuse the fixed 961
hard-forward slots and may not increase beam width, relax within-lineage
hard-RMSE monotonicity, add shapes, change the v2 ranker, or use truth.
Stage8A-v4, Stage8B, training, and all new forwards remain unexecuted.
