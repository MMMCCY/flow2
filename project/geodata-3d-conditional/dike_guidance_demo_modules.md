# Dike Guidance Demo Modules

These modules are post-processing and paper-figure tools for saved categorical
geology realizations. They do not modify training, U-Net architecture,
embedding, Flow Matching loss, or the core guidance formula in
`guided_geophysical_sampling.py`.

All geophysical wording and outputs use `lightweight gravity-proxy` terminology.
The tools are designed for headless execution and read saved `.pt` results plus
`metrics.csv` where needed.

## P0

- `geology_io_utils.py`: shared loading, tensor-shape normalization, sample file
  indexing, target-label masks, target-label probability volumes, CSV/JSON I/O,
  connected components, and conservative `paired_by_seed` inference.
- `select_dike_demo_case.py`: selects a dike-like target label/component from a
  truth model using geometry, sparse conditioning, and lightweight
  gravity-proxy observability evidence.
- `analyze_dike_observability.py`: replaces/deletes the target label in the
  truth model, measures the change in lightweight gravity-proxy response,
  saves proxy fields and target-mask slice QA figures, and records
  `recommended_for_demo`.
- `evaluate_target_feature.py`: computes target-label-specific reconstruction
  metrics, including IoU, precision, recall, volume error, centroid distance,
  connected components, probability threshold metrics, and ensemble probability
  overlap. It also saves `target_probability.pt` and probability slice figures.

## P1

- `visualize_dike_ensemble.py`: creates baseline-vs-guided target-label
  ensemble probability QA figures and 3D Figure 8/Figure 9 style realization
  and probability-threshold renderings. PyVista/Plotly availability is recorded,
  while Matplotlib voxel rendering is used as the stable PNG fallback.
- `compare_gravity_residuals.py`: creates side-by-side baseline and guided
  lightweight gravity-proxy residual figures plus individual observed,
  predicted, residual, and residual-difference PNGs with a shared residual
  color scale.
- `select_dike_demo_samples.py`: combines global metrics and target-label
  metrics to select named display roles without implying sample-wise
  improvement unless seed/config evidence supports it.
- `plot_guidance_sweep.py`: plots `guidance_sweep_summary.csv` into a combined
  guidance response curve when an alpha/mu sweep summary is available.

## P2

- `make_dike_guidance_demo.py`: orchestrates case selection, observability,
  target metrics, residual comparison, ensemble figures, sample selection, and
  `demo_report.md` generation from saved outputs. It uses selected samples for
  residual/visualization outputs and reports observability warnings.
