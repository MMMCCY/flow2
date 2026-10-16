"""Only volumes/case/output vary; exact approved_v1 rendering and composition functions."""
import argparse
import importlib.util
import os
import shutil
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

def standard():
    manifest=read_json(STANDARD/'MANIFEST.json')
    assert file_hash(STANDARD/'renderer.py')==manifest['renderer_sha256']
    for rel,h in manifest['source_sha256'].items():assert file_hash(STANDARD/'source_snapshot'/rel)==h
    for name,h in manifest['approved_files'].items():assert file_hash(STANDARD/'approved_samples'/name)==h
    spec=importlib.util.spec_from_file_location('geology_approved_v1',STANDARD/'renderer.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module

def plot(case,regression=False):
    guard('plot')
    approved=standard()
    if case=='fullgeo_case03':
        from scripts.reports.build_stage23_conference_figures import load_inputs
        arrays,_,_,_,assets,_=load_inputs()
        volumes=[arrays[k] for k in ('truth','A_9','FLOW_STAGE23')]
        root=ROOT/'standard_regression'
    else:
        r=ROOT/'cases'/case;e=read_json(r/'EXECUTION.json');assert e['case']==case and e['status']=='completed'
        by={s['arm']:s for s in e['samples']}
        assets={'truth':read_json(r/'EVALUATION_INPUTS.json')['truth'],
            'A':by['A_9']['prediction'],'Ours':by['FLOW_RESTORED_PSEUDO']['prediction']}
        volumes=[verify_tensor(a).numpy()[0,0] for a in assets.values()]
        root=ROOT/'reports'/case
    if root.exists():raise FileExistsError(root)
    root.mkdir(parents=True);(root/'panels').mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    wells=read_json(OLD/'layout.json')['original_wells']
    cams=approved.CONFIG['cameras'];keep=np.ones((64,64,64),dtype=bool)
    cut=approved.cutaway_mask((64,64,64),.52)
    images=[];checks=[]
    for i in range(2):
        for j,volume in enumerate(volumes):
            image,check=approved.render(volume,cut if i else keep,cams['geology'])
            Image.fromarray(image).save(root/'panels'/f'geology_r{i+1}_c{j+1}.png',dpi=(600,600))
            images.append(image);checks.append(dict(kind='geology',row=i,col=j,**check))
    for clean in (False,True):approved.compose(images,root,'full_geology',clean)
    target=[]
    for j,volume in enumerate(volumes):
        image,check=approved.render(volume,keep,cams['target'],wells)
        Image.fromarray(image).save(root/'panels'/f'target_c{j+1}.png',dpi=(600,600))
        target.append(image);checks.append(dict(kind='target',col=j,**check))
    for clean in (False,True):approved.compose(target,root,'target',clean)
    write_json(root/'RENDER_CONFIG.json',dict(case=case,seed=42,standard='geology_approved_v1',
        standard_manifest_sha256=file_hash(STANDARD/'MANIFEST.json'),cameras=cams,panel_checks=checks,
        inputs=assets,scientific_compute=0,visual_inspection='pending'))
    if case=='fullgeo_case03':
        comparisons={}
        for f in root.glob('*.png'):
            expected=STANDARD/'approved_samples'/f.name
            assert np.array_equal(np.asarray(Image.open(f)),np.asarray(Image.open(expected))),f'Visible approved style regression: {f.name}'
            comparisons[f.name]='pixel_exact'
        write_json(root/'STYLE_REGRESSION.json',dict(case=case,comparisons=comparisons,passed=True,
            renderer_sha256=file_hash(STANDARD/'renderer.py'),source_manifest_sha256=file_hash(STANDARD/'MANIFEST.json')))
        dest=ROOT/'reports'/case;dest.mkdir(parents=True)
        for f in (STANDARD/'approved_samples').iterdir():
            if f.suffix in ('.png','.pdf'):shutil.copyfile(f,dest/f.name)
        shutil.copytree(root/'panels',dest/'panels')
        write_json(dest/'REUSE.json',dict(source=str((STANDARD/'approved_samples').relative_to(REPO)),
            main_figures='original approved bytes reused',panels='necessary pixel-exact case03 standard regression; no Flow'))
    print(f'{case}: exact approved_v1 plates and 9 independent panels',flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=(*CASES,'fullgeo_case03'),required=True)
    plot(p.parse_args().case)
