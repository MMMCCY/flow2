# Stage15-G illustrative binary result

This is an explicitly **post-hoc selected validation example**, not a held-out generalization claim and not a Phase1-equivalent success claim. The fixed selection rule chose seed `15180114` because it has the largest absolute `AUPRC - prevalence` among the five label9-positive Stage15-G validation cases.

- Binary target: label9 versus background
- Label9 voxels: 17601
- Natural prevalence: 0.067142
- Stage15-F global-histogram AUPRC: 0.096528
- Stage15-G linear-mapper AUPRC: 0.139013
- AUPRC minus prevalence: 0.071871
- Truth/background mean P9: 0.065056 / 0.030643

The figure shows that the frozen binary seismic inversion plus the simple linear mapper can highlight part of the true label9 region. It supports feasibility of the inversion-to-label9 bridge in one favorable case; it does not establish robust performance across cases or justify Flow guidance yet.
