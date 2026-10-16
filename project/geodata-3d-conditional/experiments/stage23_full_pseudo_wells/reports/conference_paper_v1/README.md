# Stage23 visualization

From the repository root, on labrs-codex with the repository .venv:

```bash
PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python project/geodata-3d-conditional/scripts/reports/build_stage23_conference_figures.py --out project/geodata-3d-conditional/experiments/stage23_full_pseudo_wells/reports/conference_paper_v1
```

The output path must not exist; use another --out for a fresh redraw. Inputs are read-only. No checkpoint is loaded. Original renders: panels/. Editable vector text is preserved in PDF; 3D is raster. Scientific captions: CAPTIONS.md. Geometry/render/identity: RENDER_CONFIG.json, INPUT_ASSETS.json.

Final QC: all three PNGs opened; one common camera per row, full fixed cube, no viewport clipping, all target voxels retained. PDF fonts/text checked with pdffonts/pdftotext. RENDER_CHECKS.json records checks. BUILD_SCRIPT_SNAPSHOT.py is an archival copy; invoke the canonical scripts/reports path above to redraw.
