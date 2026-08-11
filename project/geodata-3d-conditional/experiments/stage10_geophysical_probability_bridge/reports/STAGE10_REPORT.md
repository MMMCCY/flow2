# Stage 10 final report: truth-blind geophysical probability bridge

## Decision

**STOP_BRIDGE_NO_GEOPHYSICAL_INFORMATION.** The strict Stage10-A gate passed in only 1/3 registered cases; at least 2/3 were required. Stage10-B, Stage10-C and Stage10-D were not executed, and no Flow forward pass was run by Stage 10.

The negative classification is specifically caused by the wrong-case control: the correctly located bridge beats the constant and XY-shuffled controls and has lower Brier score in all three cases, but a cyclic wrong-case bridge has higher AUPRC in `native_seed20260901` and `native_seed20260903`. The frozen rule therefore does not permit a claim of case-specific geophysical discrimination.

## Repository and frozen Flow

- Git branch: `main`
- Git HEAD: `72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f`
- The complete dirty state is recorded verbatim in `STAGE10_MACHINE_DECISION.json` and the pre-Stage10 state in `audit/repository_state.json`.
- Checkpoint: `project/geodata-3d-conditional/demo_model/conditional-weights.ckpt`
- Checkpoint SHA-256: `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`
- Weight policy: EMA trainable parameters with raw frozen embedding; 32-step midpoint Euler, canonical condition projection and hard decode remained frozen.

## Registered cases and seeds

Cases: `native_seed20260901`, `native_seed20260902`, `native_seed20260903`. The pilot/formal/spatial Flow seeds were frozen in `configs/flow_seed_bank.json` before evaluation, but none were consumed because Stage10-A failed.

## Property inversion and petrophysical model

Each case reuses the Phase-5a linearized post-stack log-impedance Tikhonov inversion. The low-frequency prior comprises fixed unranked Stage9A candidates 100--111; no seismic or truth ranking was used. Inputs are observed synthetic seismic, the registered forward/operator configuration, sparse hard conditions and the registered acoustic codebook. Only inverted log acoustic impedance enters the bridge; fixed prior slowness and susceptibility do not.

The bridge averages `P(k|q_s)` over all 12 posterior samples. Class means are registered codebook log impedances, every class uses the pre-registered half-median adjacent-codebook spacing, and the class prior is uniform. These choices were frozen before the retrospective evaluator opened truth.

## Truth firewall proof

The bridge builder and Flow-facing loader have no truth path/tensor argument. All bridge tensors and manifests were written and hashed first; constant, shuffled and wrong-case controls were then written and hashed. Only after both collections validated did `evaluate_bridge_information.py` load `retrospective/truth_labels.pt`. The inference builder used truth only indirectly to the permitted extent that synthetic observed seismic was originally generated from truth.

## Stage10-A bridge discrimination

| Case | Correct AP | Constant AP | Shuffled AP | Wrong-case AP | Correct Brier | Constant Brier | Pass |
|---|---:|---:|---:|---:|---:|---:|:---:|
| native_seed20260901 | 0.252243 | 0.007996 | 0.012416 | 0.285093 | 0.010095 | 0.011374 | False |
| native_seed20260902 | 0.281201 | 0.010866 | 0.013872 | 0.182565 | 0.010404 | 0.013861 | True |
| native_seed20260903 | 0.165361 | 0.010057 | 0.013683 | 0.218775 | 0.011455 | 0.013161 | False |

All metrics use the unconstrained subsurface (225,115 voxels per case). Constant-prior AUPRC equals prevalence by construction. The strict per-case pass requires correct AUPRC above constant, shuffled and wrong-case controls plus lower Brier than constant.

## Later stages

- Stage10-B paired Flow pilot: **not executed**.
- Stage10-C formal paired experiment: **not executed**.
- Stage10-D shuffled Flow control: **not executed**.
- Consequently there are no paired IoU deltas, no hard-output gain claim, no paired-delta figure and no new main-paper probability-bridge figure.

## Conference-paper artifacts

`figures/bridge_diagnostic.{pdf,svg,png}` is suitable only as a retrospective negative-result or supplementary diagnostic. It marks observed seismic and bridge quantities as inference-visible and truth as retrospective-only. Existing main-paper Figure 1 and Figure 4 are not changed.

## Limitations and prohibited claims

This stage uses noiseless inverse-crime seismic, a deliberately distinctive label-9 synthetic acoustic codebook and an uncalibrated 12-member spread. It does not show measured-geophysics performance. It does not show that frozen Flow can exploit the bridge, because Flow execution was forbidden by the gate. Do not tune class variance, smoothing, thresholds or guidance against truth to overturn this result, and do not automatically proceed to D-Flow, SMC, posterior ranking, more wells or retraining without research-plan reassessment.

## Authoritative files

- `audit/property_inversion_provenance.json`
- `audit/leakage_audit.json`
- `bridge/manifest.json` and each case `bridge/<case_id>/manifest.json`
- `controls/manifest.json`
- `diagnostics/bridge_information_metrics.csv`
- `diagnostics/bridge_controls.csv`
- `diagnostics/stage10a_decision.json`
- `reports/STAGE10_MACHINE_DECISION.json`
