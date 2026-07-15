# Dike Guidance Demo Experiment Development Report

Date: 2026-07-02

This report reviews the dike / intrusion guidance demo work performed so far.
All guidance work stayed at inference-time or post-processing level. Training
code, U-Net structure, embedding, and Flow Matching training loss were not
modified.

## Original Goal

The target paper claim was:

> Under sparse borehole conditioning, a trained conditional Flow Matching model
> can be used as a learned geological prior, and inference-time lightweight
> geophysical proxy guidance can reduce geophysical misfit and partially
> constrain dike-like target-label uncertainty without retraining.

The expected demo evidence was:

- Same truth model, same boreholes, same observed proxy fields for baseline and guided.
- Baseline ensemble has uncertainty in dike / intrusion extent.
- Guided ensemble has lower proxy misfit.
- Guided target-label probability volume is more concentrated and closer to truth.
- Guided hard decoded samples improve target IoU / recall / centroid distance without damaging geology or borehole consistency.

## Stage 1: Gravity-Only Guidance Baseline

### Code State

The original guidance path used:

- `soft_decode_to_probs(x, embedding_weight)`
- soft lithology probabilities to expected density
- `SimpleGravityForward`
- normalized lightweight gravity-proxy misfit
- gradient correction only inside `guided_geophysical_sampling.py`

This was sampling-only and preserved model training and architecture.

### Result Pattern

Gravity-only guidance generally reduced the same proxy metric used for
guidance, while preserving global geology metrics. However, target-label
improvements were small.

Representative results:

| Case | Gravity proxy misfit | Target IoU | Target recall | Interpretation |
|---|---:|---:|---:|---|
| `cond_generation_0_label9` | 0.883 -> 0.706 | 0.01271 -> 0.01278 | 0.01385 -> 0.01395 | Proxy improved, target still failed |
| `cond_generation_1_label10` | 0.324 -> 0.231 | 0.42636 -> 0.43020 | 0.68161 -> 0.68216 | Best gravity-only candidate, but improvement small |
| `paper_cond_gen_0_label7` | 0.215 -> 0.180 | 0.40705 -> 0.40848 | 0.54949 -> 0.55099 | Proxy improved, target improvement small |

For `cond_generation_1_label10`, residual RMS reduction was about `0.1168`.
For `paper_cond_gen_0_label7`, residual RMS reduction was about `0.1962`.

### Interpretation

The simple gravity method appeared to "work" mainly because:

- The training-free guidance loss and evaluation metric were the same smooth gravity proxy.
- The gravity proxy is spatially smooth, so small continuous-state changes can lower misfit without requiring many hard label changes.
- The decoded geological model was mostly preserved.

However, it did not strongly solve the target dike reconstruction problem. Even
the best target-IoU improvements were on the order of `0.001` to `0.004`, which
is too small for a strong paper figure claim.

## Stage 2: Manual Target Label and Demo Tooling

### Code Additions

New post-processing and visualization tools were added:

- `geology_io_utils.py`
- `select_dike_demo_case.py`
- `analyze_dike_observability.py`
- `evaluate_target_feature.py`
- `visualize_dike_ensemble.py`
- `compare_gravity_residuals.py`
- `select_dike_demo_samples.py`
- `make_dike_guidance_demo.py`
- `visualize_truth_model_labels.py`
- 3D visualization helpers for truth / baseline / guided / target-only panels

The target-label workflow was corrected from automatic guessing to manual QA:

1. Visualize `truth_model`.
2. Manually identify dike-like lithology id.
3. Build controlled `density_config`.
4. Generate observed lightweight gravity-proxy from truth and config.
5. Run paired baseline/guided sampling.
6. Evaluate global proxy and target-label metrics.
7. Create residual and ensemble figures.

### Result Pattern

This tooling was useful and exposed the core problem clearly:

- Some labels were unsuitable because the model rarely generated them or the target evidence was weak.
- Candidate screening identified `cond_generation_1_label10` and `paper_cond_gen_0_label7` as the better gravity-only candidates.
- Even for these better candidates, target improvements remained small.

### Interpretation

This stage improved experimental discipline. It did not create the expected
large visual difference between baseline and guided. It showed that the
gravity-only method can support a cautious "proxy misfit improves" claim, but
not yet a strong "dike geometry is reconstructed" claim.

## Stage 3: Magnetic-Proxy and Gravity-Gradient-Proxy Guidance

### Code Additions

To make geophysical guidance more local and target-sensitive, the following
were added:

- `MagneticTMIForward`
- `GravityGradientForward`
- `create_susceptibility_config.py`
- `generate_observed_geophysics.py`
- multi-physics guidance in `guided_geophysical_sampling.py`
- optional magnetic / gravity-gradient columns in `evaluate_geophysics.py`

Supported modes:

- `gravity`
- `magnetic`
- `gravity_gradient`
- `joint`

### Joint Alpha=1.0 Result

Case: `cond_generation_1_label10`

Baseline:

- `geo_misfit`: `0.3239`
- `magnetic_proxy_misfit`: `1.3506`
- `gravity_gradient_proxy_misfit`: `0.2114`
- `joint_proxy_misfit`: `1.5654`

Joint guided, alpha=1.0:

- `geo_misfit`: `0.2565` improved
- `magnetic_proxy_misfit`: `2.9140` worse
- `gravity_gradient_proxy_misfit`: `0.3375` worse
- `joint_proxy_misfit`: `3.1267` worse

The guidance was definitely active:

- final-step `effective_guidance_ratio`: about `0.959`

But hard decoded changes were tiny:

- mean changed voxel fraction: `0.00262`

Target-label metrics did not improve:

- target IoU: `0.42636 -> 0.42611`
- target recall: `0.68161 -> 0.68091`
- target F1: `0.59568 -> 0.59543`
- probability entropy: `0.06443 -> 0.06483`

### Single-Physics Diagnostics

Magnetic-only alpha=1.0:

- `magnetic_proxy_misfit`: `1.3506 -> 3.2895`
- target IoU: `0.42636 -> 0.42328`
- changed voxel fraction: `0.00248`

Gravity-gradient-only alpha=1.0:

- `gravity_gradient_proxy_misfit`: `0.2114 -> 0.3619`
- target IoU: `0.42636 -> 0.42146`
- changed voxel fraction: `0.00387`
- borehole consistency dropped more visibly: `0.99906 -> 0.96900`

### Interpretation

The magnetic and gravity-gradient additions did not reveal a stronger demo.
They revealed a failure mode:

- Guidance velocity is strong in continuous embedding space.
- The final hard categorical decode changes very few voxels.
- The decoded samples get worse under the very proxy they were supposed to improve.
- The issue is not only joint weighting, because single-physics magnetic and gravity-gradient also fail.

## Why Simple Gravity Looked Better

The earlier gravity-only method did not necessarily solve the dike problem; it
solved the easiest version of the proxy-matching problem.

Reasons it looked better:

1. The proxy was smooth.
   Gravity integrates broad mass effects. Small continuous changes can reduce
   the gravity-proxy residual without forcing many categorical boundaries to
   move.

2. The metric and guidance were aligned.
   The same `SimpleGravityForward` and density config were used for guidance
   and evaluation. This made misfit reduction easier to observe.

3. The target claim was weaker than desired.
   Gravity-only improved global proxy misfit more than it improved target-label
   reconstruction. Target IoU changes were small even in the best cases.

4. It preserved the geological prior.
   Because hard decoded labels barely moved, voxel accuracy and mean IoU did
   not degrade much. This made the method look stable, but also limited its
   ability to reconstruct missing target geometry.

## Why Magnetic / Gravity-Gradient Got Worse

Likely causes:

1. Soft guidance to hard decode mismatch.
   The loss acts on soft probabilities derived from continuous embedding
   states. Evaluation acts on hard decoded categories. Strong continuous changes
   do not necessarily cross class boundaries in the desired locations.

2. Relative guidance ignores loss scale.
   In relative mode, the gradient direction is normalized and then scaled by
   prior velocity norm. This can make a poor or noisy magnetic/gradient
   direction very strong.

3. High-frequency kernels are less forgiving.
   Magnetic and gravity-gradient proxies are more local and sharper than the
   original gravity proxy. Small geometry errors can increase the final decoded
   proxy residual.

4. Endpoint dynamics are not guaranteed to descend final decoded loss.
   The trace showed magnetic/gradient losses could be lower on average across
   the continuous trajectory but worse at the final decoded endpoint.

5. Target property contrast may be too artificial.
   The target susceptibility/density contrast was deliberately large to make
   the target observable, but that may create gradients that conflict with the
   learned categorical manifold.

6. Borehole / prior constraints dominate hard labels.
   The model prior and sparse conditioning may keep most categorical regions
   stable. The guidance changes the continuous state, but only about
   `0.25% - 0.39%` of voxels cross decode boundaries.

## Current Technical Diagnosis

We are likely in a wrong sub-path if the goal is a strong paper figure showing
clear dike reconstruction improvement.

The current evidence supports:

- Inference-time proxy guidance can be made active.
- Gravity-proxy misfit can be reduced.
- The geology prior is mostly preserved.

The current evidence does not support:

- Magnetic-proxy guidance improves magnetic-proxy misfit after hard decode.
- Gravity-gradient guidance improves gravity-gradient-proxy misfit after hard decode.
- Target-label uncertainty is meaningfully reduced.
- Dike-like geometry becomes visibly more truth-like.

## Recommended Pause Point

Do not keep increasing alpha or adding more proxy terms without first resolving
the soft-to-hard mismatch.

The next useful diagnostic, before any new guidance design, would be:

1. For each sampling step, compute proxy loss before and after applying the
   guidance velocity to confirm local descent in continuous space.
2. Save final soft target-label probability volume before hard decode.
3. Compare:
   - soft target probability improvement
   - hard target IoU improvement
   - decoded proxy misfit improvement
4. If soft improves but hard does not, the blocker is decode-boundary crossing.
5. If soft does not improve locally, the blocker is guidance direction / sign /
   scaling / forward operator.

## Strategic Recommendation

For the paper demo, the safest claim from current results is not magnetic or
gravity-gradient guidance. It is a limited gravity-proxy result:

> A lightweight gravity-proxy guidance can reduce proxy residual while largely
> preserving the learned geological prior, but in the current categorical
> Flow-Matching setup, this does not yet robustly translate into strong
> hard-decoded dike reconstruction improvement.

If the paper requires Figure-8/Figure-9-style target probability improvement,
the current implementation is not sufficient. The project should pause and
decide whether to:

- redesign inference guidance around categorical logits / decode boundaries;
- introduce a post-sampling reweighting or selection method instead of
  modifying the ODE trajectory;
- use gravity-proxy improvement as a modest result rather than a strong dike
  reconstruction claim;
- or change the experimental target to one where the pretrained prior already
  frequently generates the dike and guidance only selects among plausible
  alternatives.

