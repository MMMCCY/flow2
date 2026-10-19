"""Execute exact approved function ASTs with frozen low-level rendering dependencies."""
import ast
import hashlib
import itertools
import json
from pathlib import Path
import types
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pyvista as pv

ROOT=Path(__file__).resolve().parent
manifest=json.loads((ROOT/'MANIFEST.json').read_text())
for name,digest in manifest['source_sha256'].items():
    assert hashlib.sha256((ROOT/'source_snapshot'/name).read_bytes()).hexdigest()==digest
style=types.ModuleType('approved_v1_frozen_style')
style.__file__=str(ROOT/'source_snapshot/scripts/paper_figures/style.py')
exec(compile(Path(style.__file__).read_text(),style.__file__,'exec'),style.__dict__)
namespace=dict(np=np,plt=plt,pv=pv,Patch=Patch,itertools=itertools,WINDOW=(1600,1440),
    DIRECTION=np.array([1.72,-1.35,1.38]),TITLES=['Ground truth','Frozen Flow','Ours'])
for name in ('LABEL_COLORS','LABEL9_COLOR','OBSERVATION_COLOR','binary_surface','cutaway_mask',
             '_new_plotter','_add_surface','_add_bounding_box'):
    namespace[name]=getattr(style,name)
entry=ROOT/'source_snapshot'/manifest['entry']
tree=ast.parse(entry.read_text())
for node in tree.body:
    if isinstance(node,ast.FunctionDef) and node.name in manifest['frozen_runtime_functions']:
        exec(compile(ast.Module(body=[node],type_ignores=[]),str(entry),'exec'),namespace)
render=namespace['render']
compose=namespace['compose']
camera=namespace['camera']
cutaway_mask=namespace['cutaway_mask']
CONFIG=json.loads((ROOT/'approved_samples/RENDER_CONFIG.json').read_text())
