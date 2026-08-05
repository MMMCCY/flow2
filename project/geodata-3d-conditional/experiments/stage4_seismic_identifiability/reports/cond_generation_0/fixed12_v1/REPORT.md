# Phase-4d seismic identifiability and posterior-selection audit

## Decision

**FAIL: frozen prior pool lacks geological support**

No sample was generated or modified. Ranking used hard seismic loss only; truth was revealed only for audit.

## Frozen pool

- Candidates: `12` from seeds 42/142/242.
- Geological support gate: `False`.
- Seismic ranking gate: `False`.
- Promotion: `False`.

## Seismic-selected top candidate

- Candidate: `seed42_sample1`.
- Hard seismic RMSE: `0.041072`.
- Global accuracy / truth-present mIoU: `0.6122` / `0.2759`.
- Label-9 IoU / precision / recall: `0.0245` / `0.0876` / `0.0329`.
- Major-component minimum / mean recall: `0.0000` / `0.0329`.

## Rank relationship

- loss vs global_voxel_accuracy: rho `-0.2937062937062937`, one-sided p `0.16968303169683033`.
- loss vs truth_present_mean_iou: rho `-0.04195804195804196`, one-sided p `0.458054194580542`.
- loss vs target_iou: rho `0.5524475524475524`, one-sided p `0.9676032396760323`.
- loss vs target_recall: rho `0.5874125874125874`, one-sided p `0.9758024197580242`.
- loss vs major_component_mean_recall: rho `0.5804195804195804`, one-sided p `0.9748025197480252`.

## Label-9 whole-class substitution oracle

- Unconditioned truth voxels changed: `8955`.
- Least-visible replacement label: `12`.
- Minimum substitution RMSE: `0.017692`.

Whole-class substitution sensitivity does not establish local lithology uniqueness.

A lower continuous field loss alone is not geological recovery.
