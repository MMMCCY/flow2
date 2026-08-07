# Phase 1–4 operator call graph

Evidence freeze: branch `main`, Git SHA
`11caa498b15e6b89891604e9537830b30df504fa`, clean worktree.

This document is derived from the code at the SHA above.  It is a diagnostic
map, not a replacement implementation.  Phase names refer to the historical
project stages; Stage6Q uses the same functions directly.

## Shared state, decoder and generator

- Continuous optimized/sampled state: `x [B,E,X,Y,Z]`.
- Soft decoder: `guided_geophysical_sampling.soft_decode_to_probs`, cosine
  normalization followed by temperature-scaled softmax.
- Hard decoder: `Geo3DStochInterp.decode`, cosine normalization followed by
  `argmax`; raw labels are returned by subtracting one from category indices.
- Frozen prior velocity: `Geo3DStochInterp.net(x, conditioning, t)`.
- Canonical checkpoint policy: `inference_runtime.load_model_with_weight_policy`;
  the raw frozen `embedding.weight` is retained and EMA is applied to every
  trainable U-Net parameter.
- Canonical reusable condition projection:
  `guidance.generator_posterior.project_conditions`.  Historical Phase 1/2/4
  samplers contain behaviorally equivalent private `_project_conditions`
  helpers and project before the first step and after every Euler update.

## Phase 1: target probability volume

```text
x
 -> soft_decode_to_probs(x, embedding.weight, tau)
 -> target-label probability channel
 -> probability_volume_loss
      target may itself be a weighted Gaussian multiscale volume built by
      build_probability_volume / gaussian_blur_3d
 -> autograd dL/dx
 -> clip_gradient_by_norm
 -> build_probability_guidance_velocity
 -> v_physics = -guidance_velocity
 -> x_next_pre_projection = x + dt * (v_base + v_physics)
 -> exact hard-condition projection
 -> model.decode only for hard trace/final evaluation
```

Optimized variable is the continuous embedding state.  The U-Net supplies the
frozen base velocity.  The loss is soft; hard decode is after the state update
and does not participate in autograd.  Best/final selection belongs to the
runner/evaluator, not the sampler.

## Phase 2: matched multiscale property volume

```text
x
 -> soft_decode_to_probs
 -> probabilities_to_expected_properties
 -> matched_multiscale_property_loss
      gaussian_blur_property_channels is applied to both prediction and target
      for every explicitly configured sigma
 -> autograd / clipping / relative controller
 -> v_physics = -guidance_velocity
 -> fixed Euler
 -> condition projection
 -> hard decode for audit
```

The Gaussian operators are part of the Phase 2 loss and are not part of the
canonical seismic forward.

## Phase 3: spatially degraded property observation

Phase 3 keeps the same continuous state, decoder, controller, Euler sign and
condition projection.  Its observation adapter explicitly applies the selected
spatial operator to both sides after exact known-property overwrite.  Identity
and configured Gaussian degradation are attribution arms.  This historical
blur must not be inserted into `guidance/seismic.py`.

## Phase 4A: gravity

```text
x
 -> soft_decode_to_probs
 -> probabilities_to_density
 -> overwrite_exact_condition_density
 -> RectangularPrismGravity.__call__
 -> gravity_field_loss
 -> autograd / clipping / relative controller
 -> v_physics = -guidance_velocity
 -> fixed Euler
 -> condition projection
 -> hard decode
 -> hard_labels_to_density -> same RectangularPrismGravity for hard audit
```

The optimized variable remains `x`; the U-Net is involved only through frozen
`v_base`.  The gravity operator and loss are single-source functions in
`guidance/gravity.py`.

## Phase 4C: canonical convolutional seismic

```text
x
 -> soft_decode_to_probs
 -> probabilities_to_subsurface_acoustic
      rock-conditional renormalization inside known subsurface
 -> overwrite_exact_condition_acoustic
 -> ConvolutionalSeismic.interface_response
      impedance reflectivity + slowness-derived TWT
 -> deposit_reflectivity (linear deposition; out-of-window prediction cropped)
 -> convolve_reflectivity_spikes (fixed Ricker wavelet)
 -> seismic_field_loss
 -> autograd / clipping / relative controller
 -> v_physics = -guidance_velocity
 -> x_next_pre_projection = x + dt * (v_base + v_physics)
 -> condition projection
 -> hard decode
 -> hard_labels_to_acoustic -> same ConvolutionalSeismic for hard audit
```

`guidance/seismic.py` contains no Gaussian blur.  The physical update sign is
negative loss gradient because the sampler constructs a positive
`guidance_velocity` aligned with `dL/dx` and evaluates
`guided_velocity = prior_velocity - guidance_velocity`.

## Stage6Q endpoint and best-iterate policies

- `optimize_embedding_endpoint` optimizes checkpoint embedding vectors in the
  Q2 search mask, with no U-Net, and selects only minimum hard physics RMSE.
- `optimize_endpoint_state` optimizes a full continuous endpoint, projects
  conditions after every Adam step, and selects only the configured hard loss.
- Q0/Q1/Q2/Q3 hard evaluation uses `AnalyticObservationSuite`, which dispatches
  to the same seismic/gravity operators above.

## Audit invariants exposed by this graph

1. Soft and hard responses require separate baselines and attainment
   denominators.
2. The actually applied physics velocity is `-guidance_velocity`.
3. State projection and exact-property overwrite are distinct operations.
4. TWT deposition/cropping occurs before wavelet convolution.
5. Gaussian blur is explicit only in probability/property/spatial arms, never
   implicit in canonical seismic.
