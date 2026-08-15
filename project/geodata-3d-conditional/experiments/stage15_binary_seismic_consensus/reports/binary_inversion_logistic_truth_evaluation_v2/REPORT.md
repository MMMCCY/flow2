# Stage15-G — Binary case-relative inversion logistic mapper

The mapper is a single 4-to-1 linear layer. Its target is strictly binary: raw label9 versus every other subsurface voxel. It reuses frozen Stage15-F inversion scores and does not rerun seismic, inversion, or Flow.

## Validation

- Pooled prevalence / AUPRC: 0.0080414098 / 0.093510412
- Truth/background mean P9: 0.087461792 / 0.011856851
- Positive-case median AUPRC: 0.096166638
- Positive cases with AUPRC above own prevalence: 4/5
- Positive cases with truth P9 above background: 4/5

## Retrospective cond_generation_0

- AUPRC: 0.091581668
- Truth mean/median P9: 0.082984142 / 0.019993041
- Background mean/median P9: 0.038600627 / 0.018136095
- Probability range: 0.0021958235 to 0.36426514
- >=0.8 positive precision/recall/IoU: None / 0.0 / 0.0
- Truth positive/unknown/negative: 0 / 1618 / 7350

Historical AUPRC: B2 0.057284505; C 0.045394953; F global histogram 0.089872368.
