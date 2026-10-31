"""Use existing approved_v1 renderer; fixed Y26; all fragments and false positives."""
import argparse
import os
import sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(PROJECT))
import numpy as np
from .common import *
from scripts.stage24.plot import standard
from scripts.stage25.plot import save_panel,sections

def main(root):
    root=root.resolve();install_firewall(root);torch.set_num_threads(4)
    ev=read_json(root/'EVALUATION.json');er=read_json(root/'evaluation_registry.json')
    out=root/'figures';out.mkdir(exist_ok=True)
    approved=standard();cams=approved.CONFIG['cameras']
    assets={**ev['prediction_assets'],**{k+'_Truth':a for k,a in er['truths'].items()}}
    arrays={k:verify_tensor(a).numpy()[0,0] for k,a in assets.items()}
    wells=read_json(root/'layout.json')['original_wells'];keep=np.ones((64,64,64),dtype=bool)
    cut=approved.cutaway_mask((64,64,64),.52);checks=[]
    def plate(name,keys,geology=False):
        folder=out/name
        if (folder/'DONE.json').exists():
            return
        folder.mkdir(exist_ok=True);(folder/'panels').mkdir(exist_ok=True)
        approved.namespace['TITLES']=keys
        images=[]
        if geology:
            for row,mask in enumerate((keep,cut)):
                for key in keys:
                    im,check=approved.render(arrays[key],mask,cams['geology'])
                    save_panel(im,folder/'panels'/f'geology_{key}_{row}')
                    images.append(im);checks.append(dict(arm=key,kind='geology',**check))
            approved.compose(images,folder,'full_geology',False)
        images=[]
        for key in keys:
            im,check=approved.render(arrays[key],keep,cams['target'],wells)
            save_panel(im,folder/'panels'/f'target_{key}');images.append(im)
            checks.append(dict(arm=key,kind='target',**check))
        approved.compose(images,folder,'target',False)
        sections([arrays[k] for k in keys],keys,approved,folder/'section_Y26')
        write_json(folder/'DONE.json',dict(keys=keys,all_fragments_retained=True))
    plate('G1_main',['G1_Truth','Fy_G1','Fyr_G1'],True)
    plate('G1_reference',['G1_Truth','A','F0'])
    plate('G1_structure',['G1_Truth','F0','Fr'])
    if 'Fy_G2' in arrays:
        plate('G2_main',['G2_Truth','Fy_G2','Fyr_G2'],True)
        plate('G2_reference',['G2_Truth','F0','Fr'])
    write_json(out/'RENDER_CONFIG.json',dict(standard='geology_approved_v1',
        manifest_sha256=file_hash(STANDARD/'MANIFEST.json'),cameras=cams,checks=checks,
        section_Y=26,all_components_retained=True,hard_labels_only=True,
        target_cutaway=False,visual_inspection='pending',network_evaluations=0))
    (out/'CAPTIONS.md').write_text('''# Figure captions

G1/G2 main: truth, physics-only Fy, physics+TV Fyr. Reference plates retain the
old A, new-path F0 and structure-only Fr controls. Every hard raw9 component and
false positive is retained; cyan lines show the original nine boreholes.
Whole-geology main plates use full and the approved fixed cutaway views.
Target plates show the complete volume. All cameras, color tables, lighting and
composition are the existing approved_v1 functions; only inputs and titles vary.
Sections are fixed Y26, Z upward, identical discrete colors, nearest sampling.
No truth-based choice of section, component deletion, surface smoothing or air
cleanup is performed. Metrics and conclusions remain outside the figures.
Spacing is3840/63m. 3D PDF panels are raster images with vector titles.
This is a noiseless synthetic tracewise convolution/inverse-crime mechanism
experiment with one source, not measured seismic, held-out generalization,
posterior calibration, or unique geological reconstruction.
''')
    print('FIGURES_COMPLETE',out,flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    main(p.parse_args().out)
