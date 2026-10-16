#!/usr/bin/env python3
"""Use the validated ensemble workflow with publication layout and shaded exact faces.

Accepts the same postprocess / plot-only CLI as ensemble_workflow.py. Statistics,
IDs and masks are untouched; only the renderer is replaced. Sampling is excluded.
"""
import sys
from pathlib import Path
PROJECT=Path(__file__).resolve().parents[1]
REPO=PROJECT.parents[1]
for path in (REPO,PROJECT,Path(__file__).resolve().parent): sys.path.insert(0,str(path))
import ensemble_rendering
import ensemble_workflow
if '--action' not in sys.argv or sys.argv[sys.argv.index('--action')+1] not in ('postprocess','plot-only'):
    raise SystemExit('publication entry requires --action postprocess or plot-only')
ensemble_workflow.occurrence.render_panels=ensemble_rendering.render_panels
ensemble_workflow.main()
args=sys.argv
report=Path(args[args.index('--plot-dir')+1]) if '--plot-dir' in args else Path(args[args.index('--report-dir')+1]) if '--report-dir' in args else Path(args[args.index('--out')+1])/'postprocess_v1'
ensemble_workflow.write_json(report/'renderer_manifest.json',dict(renderer=ensemble_workflow.runtime.asset_record(Path(ensemble_rendering.__file__)),entry=ensemble_workflow.runtime.asset_record(Path(__file__)),statistics_transformed=False,exact_exposed_faces=True,spacing_preserved=True,camera='front +Y; iso from -X,-Y,+Z',paper_layout=True,presentation_canvas=[1920,1080]))
