"""Small meaningful guards, geometry and distance tests; no network invocation."""
import sys,subprocess
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.stage25.evaluate import surface_distance,centroid_distance
from scripts.stage25.generate import build_geometry
from scripts.stage25.common import W9,SPACING

def test_surface_metric_all_components_and_empty():
    a=np.zeros((8,8,8),bool);a[2,2,2]=True
    assert surface_distance(a,a,[2]*3)==0
    b=np.roll(a,1,axis=0)
    assert surface_distance(a,b,[2]*3)==2
    b[6,6,6]=True
    assert surface_distance(a,b,[2]*3)>2
    assert surface_distance(a,np.zeros_like(a),[2]*3) is None
    assert centroid_distance(a,np.roll(a,1,axis=1),[2]*3)==2

def test_native_geometry_one_center_well_and_cap():
    labels,params=build_geometry();v=labels[0,0].numpy();t=v==9
    from scipy import ndimage
    assert ndimage.label(t,np.ones((3,3,3)))[1]==1
    assert sum(t[x,y].any() for x,y in W9)==1
    assert t[35,26,29:36].all() and t[35,26,:29].all()
    assert not t[:,:,52:].any() and not t[[0,-1]].any()
    assert set(np.unique(v))=={-1,0,1,3,5,9}
    assert params['spacing_m']==[SPACING]*3

def test_truth_and_history_firewall_in_subprocess():
    code='''
from scripts.stage25.common import guard,ROOT,PROJECT
guard('flow')
for p in (ROOT/'truth/labels.pt',ROOT/'GEOMETRY.json',ROOT/'evaluation_registry.json',ROOT/'preprocessing/properties.pt',PROJECT/'scripts/stage22/forbidden.py'):
    try:p.open('wb' if p.name=='forbidden.py' else 'rb')
    except PermissionError:continue
    raise AssertionError(p)
'''
    r=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
    assert r.returncode==0,r.stderr
