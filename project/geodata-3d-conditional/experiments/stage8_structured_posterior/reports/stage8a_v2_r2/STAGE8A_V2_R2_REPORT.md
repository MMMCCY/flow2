# Stage 8A-v2-R2 — Sensitivity-to-Hard-Proposal Alignment Audit

**Primary classification: `FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY`**

This is a read-only postmortem. Stage8A-v1, R1, and v2 remained unchanged. No proposal was generated or selected, no hard proposal seismic forward was called, Stage8A-v2 was not rerun, and Stage8B/training/v3 were not run.

## A. ORACLE_POSTMORTEM_ONLY retrospective center rank

The complete 140,985-center analytic initial ranking was reconstructed from the frozen score map and written before Stage7 successful geometry was opened.

| candidate | nearest grid center | distance | rank | top-rank percentile | score percentile | score | top-1 distance | nearest top-96 distance |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| candidate_04 | (9,41,21) | 0.8660254 | 5 | 0.003546477% | 99.99716% | 6.169795e-05 | 36.53423 | 0.8660254 |
| candidate_06 | (41,41,39) | 0.8660254 | 1 | 0.0007092953% | 100% | 6.173096e-05 | 0.8660254 | 0.8660254 |

## B. Recorded score versus frozen hard loss

| scope | births | Spearman(score, hard Δ) | Kendall tau | higher score -> lower hard RMSE? | top-10 overlap | top-10% overlap |
|---|---:|---:|---:|:---:|---:|---:|
| all_correct | 2240 | 0.01958888 | 0.01421926 | False | 0/10 | 35/224 |
| all_controls | 6530 | -0.634799 | -0.4555653 | True | 0/10 | 111/653 |
| control_zero | 2050 | -0.7872437 | -0.6296944 | True | 2/10 | 108/205 |
| control_shuffled_xy | 2240 | -0.546461 | -0.3757443 | True | 2/10 | 81/224 |
| control_wrong_case_observation | 2240 | 0.02278757 | 0.01586571 | False | 1/10 | 46/224 |

## C. Canonical score -> actual-mask derivative -> hard delta

| scope | births | rho(canonical, actual decrease) | rho(actual decrease, hard Δ) | canonical/actual sign agreement | actual/hard sign agreement | actual predicts improve | hard improves |
|---|---:|---:|---:|---:|---:|---:|---:|
| all_correct | 2240 | 0.1541902 | -0.08248442 | 0.9401786 | 0.05982143 | 2106 | 0 |
| all_controls | 6530 | 0.7957309 | -0.5789724 | 0.9537519 | 0.2679939 | 4784 | 4 |

Raw gradients were absent, so they were indispensably reconstructed. Observation tensors were also absent and required eight observation-reconstruction forwards. These are separately counted and never used to evaluate a new proposal.

## D. ORACLE_POSTMORTEM_ONLY known finite-step sign audit

| candidate | first-order ΔMSE | first-order rank / 12 | hard singleton ΔRMSE | hard rank / 12 | first-order improves? | hard improves? | sign agreement |
|---|---:|---:|---:|---:|:---:|:---:|:---:|
| candidate_04 | -0.0001042769 | 1 | -0.002602592 | 2 | True | True | True |
| candidate_06 | -0.0001042769 | 2 | -0.002602604 | 1 | True | True | True |

## Classification and stop boundary

Localization is not the failure: the two successful analytic centers rank 1 and 5 of 140,985. After replacing the canonical mask by every proposal's actual mask, first order still predicts improvement for 2,106/2,240 correct-arm births, while the frozen finite hard evaluation improves for 0/2,240; sign agreement is only 5.98%. The known Stage7 singleton masks themselves have matching improving first-order and finite-step signs, confirming that the implementation can represent the descent signal. The dominant break is therefore extrapolation from an infinitesimal property direction to the one-shot finite hard insertion used by v2.

The frozen v2 protocol deviation remains separate: the analytic zero arm kept 961 hard forwards but realized 370 births rather than v1's 560 because the unchanged state-dependent kernel followed a different beam path. R2 does not reinterpret, repair, or rerun it.

## Exactly one next algorithmic recommendation

Replace one-shot full-size births with a deterministic truth-blind hard-loss trust-region continuation that starts from a nested small allowed-shape insertion and grows it only through already-budgeted hard-RMSE-improving proposals, without increasing the hard-forward budget.

This recommendation was not implemented. Stage8B and Stage8A-v3 remain unrun.
