import sys
from pathlib import Path
import numpy as np
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[3]))
from ensemble_rendering import render_panels


def test_full_empty_boundary_faces_leave_masks_unchanged_and_use_16_9(tmp_path):
    mask=np.zeros((2,3,4),bool); mask[0,0,0]=True; before=mask.copy()
    panels=[('Empty threshold region',[(np.zeros_like(mask),'#e7aa32')]),('Boundary cell',[(mask,'#e7aa32')]),('Full mask',[(np.ones_like(mask),'#e7aa32')])]
    render_panels(tmp_path/'fig',panels,mask.shape,(100,100,50),'SOFTWARE TEST ONLY',presentation=True,columns=3)
    np.testing.assert_array_equal(mask,before)
    with Image.open(tmp_path/'fig.png') as image: assert image.size==(1920,1080)
    svg=(tmp_path/'fig.svg').read_text()
    assert 'SOFTWARE TEST ONLY' in svg and 'Empty threshold region' in svg
    assert 'Red: independent observed target' in svg and 'cell 100/100/50 m' in svg
