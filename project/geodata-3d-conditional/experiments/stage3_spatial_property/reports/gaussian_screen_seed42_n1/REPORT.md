# Phase-3 Gaussian spatial-resolution seed-42 n=1 screen

## Decision

**SCREEN CLOSED: no nonzero Gaussian blur passed the complete n=1 gate**

This is a truth-derived 3-D spatial-degradation screen, not measured or acquisition-domain geophysics.

## Frozen levels

| Level | Gate | Accuracy delta | Fixed mIoU delta | Label-9 IoU / P / R | Major mean recall |
|---|---:|---:|---:|---|---:|
| identity_anchor_v1 | True | 0.0413 | 0.0676 | 0.4881 / 0.9032 / 0.5151 | 0.4984 |
| gaussian_sigma1_v1 | False | 0.0312 | 0.0458 | 0.3357 / 0.6758 / 0.4001 | 0.3981 |
| gaussian_sigma2_v1 | False | 0.0217 | 0.0272 | 0.2064 / 0.4935 / 0.2619 | 0.2584 |
| gaussian_sigma4_v1 | False | 0.0128 | 0.0117 | 0.1026 / 0.2997 / 0.1349 | 0.1305 |

## Frozen promotion result

- Status: `no_nonzero_blur_passed`.
- Selected level: `identity_anchor_v1`.
- Seed-42 n=4 bracket: `['identity_anchor_v1', 'gaussian_sigma1_v1']`.
- Label-9 IoU is non-increasing with blur: `True`.
- Major-body mean recall is non-increasing with blur: `True`.
- Lower observation loss alone did not promote any nonzero blur.
