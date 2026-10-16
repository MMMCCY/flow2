# Stage23 reference-style revision

From repository root on labrs-codex:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_stage23_reference_figures.py --out project/geodata-3d-conditional/experiments/stage23_full_pseudo_wells/reports/conference_paper_v2
```

Choose a fresh --out directory for redraw; existing output is protected. Only two scientific plates; each has normal/clean PNG/PDF. Clean versions retain only panel letters. Captions and metrics are external. PDF3D is raster, text/legend vector. Previous conference_paper_v1 is retained.

Files:

- fig01_full_geology_comparison: [PNG](fig01_full_geology_comparison.png) / [PDF](fig01_full_geology_comparison.pdf)
- fig02_target_body_comparison: [PNG](fig02_target_body_comparison.png) / [PDF](fig02_target_body_comparison.pdf)
- fig01_full_geology_comparison_clean: [PNG](fig01_full_geology_comparison_clean.png) / [PDF](fig01_full_geology_comparison_clean.pdf)
- fig02_target_body_comparison_clean: [PNG](fig02_target_body_comparison_clean.png) / [PDF](fig02_target_body_comparison_clean.pdf)

[Captions](CAPTIONS.md) · [Render config](RENDER_CONFIG.json) · [Input identities](INPUT_ASSETS.json)
