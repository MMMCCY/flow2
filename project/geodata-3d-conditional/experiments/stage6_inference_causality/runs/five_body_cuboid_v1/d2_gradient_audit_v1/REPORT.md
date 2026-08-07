# Phase 6Q D2 gradient and controller semantics report

Gradient verdict: **PASS**
Decoder/mapping verdict: **PASS**
Applied-controller verdict: **FAIL_RAW_GRADIENT_PRODUCTION_DTYPE**

| Chain | Best eps | Best relative error | -grad descends |
|---|---:|---:|---:|
| embedding_to_acoustic_property | 1.0e-02 | 1.8292e-08 | True |
| explicit_blurred_property | 1.0e-02 | 2.71447e-09 | True |
| property_to_reflectivity_and_twt_deposition | 1.0e-02 | 6.71693e-09 | True |
| wavelet_convolved_seismic_loss | 1.0e-02 | 4.13111e-09 | True |
| gravity_forward_and_loss | 1.0e-02 | 1.79122e-08 | True |

## Applied updates

| Update | Norm | Soft delta | Hard delta | Truth-direction fraction |
|---|---:|---:|---:|---:|
| raw_negative_gradient_small_step | 3.37147e-12 | 0 | 0 | 0 |
| normalized_negative_gradient_small_step | 0.0001 | 4.07454e-10 | 0 | -6.73245e-07 |
| actual_controller_velocity_unit_time | 276.934 | -3.32255e-05 | 0.000526596 | 0.273604 |
| actual_euler_applied_physics_update | 8.65417 | -1.13354e-05 | 0 | 0.0566232 |
| actual_euler_update_after_condition_projection | 8.65417 | -1.13354e-05 | 0 | 0.0566232 |
