#!/usr/bin/env python3
"""Minimal Stage15-style Stage23 plates; read-only saved assets, no inference."""
import argparse
import itertools
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES']=''
os.environ['PYTHONDONTWRITEBYTECODE']='1'
sys.dont_write_bytecode=True
PROJECT=Path(__file__).resolve().parents[2]
REPO=PROJECT.parents[1]
sys.path.insert(0,str(PROJECT))
from scripts.reports.build_stage23_conference_figures import load_inputs, protect, R21, R23
from scripts.stage21.common import read_json, write_json, file_hash, git_snapshot
from scripts.paper_figures.style import (
    LABEL_COLORS, LABEL9_COLOR, OBSERVATION_COLOR, binary_surface, cutaway_mask,
    _new_plotter, _add_surface, _add_bounding_box,
)
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import pyvista as pv
import vtk

OUT=R23.parent/'reports/conference_paper_v2'
WINDOW=(1600,1440)
TITLES=['Ground truth','Frozen Flow','Ours']
DIRECTION=np.array([1.72,-1.35,1.38])


def camera(focal_z,scale):
    focal=np.array([32.,32.,focal_z])
    forward=-DIRECTION/np.linalg.norm(DIRECTION)
    right=np.cross(forward,[0,0,1.]);right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    return dict(position=(focal+DIRECTION*64).tolist(),focal_point=focal.tolist(),
        direction=DIRECTION.tolist(),view_up=[0,0,1],parallel_projection=True,
        parallel_scale=scale,zoom=1.,right=right.tolist(),up=up.tolist())


def project(points,cam):
    rel=np.asarray(points)-np.asarray(cam['focal_point'])
    return np.column_stack((.5+rel@cam['right']/(2*cam['parallel_scale']*WINDOW[0]/WINDOW[1]),
                            .5+rel@cam['up']/(2*cam['parallel_scale'])))


def render(volume,keep,cam,wells=None):
    plotter=_new_plotter(WINDOW)
    counts={}
    target=wells is not None
    for label in ([9] if target else sorted(int(i) for i in np.unique(volume) if i!=-1)):
        selected=(volume==label)&keep
        counts[str(label)]=int(selected.sum())
        mesh=binary_surface(selected)
        if mesh is not None:
            assert np.allclose(mesh.points,np.round(mesh.points))
            projected=project(mesh.points,cam)
            assert (projected>.005).all() and (projected<.995).all(), 'Nonair voxel face would be clipped'
            _add_surface(plotter,mesh,LABEL_COLORS[label],1.)
    if target:
        _add_bounding_box(plotter,volume.shape)
        for x,y in wells:
            line=pv.Line((x+.5,y+.5,0.),(x+.5,y+.5,64.))
            _add_surface(plotter,line.tube(radius=.12,n_sides=12),OBSERVATION_COLOR,1.)
        box_corners=np.array(list(itertools.product([0.,64.],repeat=3)))
        assert (project(box_corners,cam)>.005).all() and (project(box_corners,cam)<.995).all()
        assert counts['9']==int((volume==9).sum())
    plotter.camera_position=[cam['position'],cam['focal_point'],cam['view_up']]
    plotter.camera.parallel_projection=True
    plotter.camera.parallel_scale=cam['parallel_scale']
    plotter.camera.clipping_range=(1.,600.)
    image=np.asarray(plotter.screenshot(return_img=True))[:,:,:3]
    plotter.close()
    assert np.all(image[:2]>250) and np.all(image[-2:]>250)
    assert np.all(image[:,:2]>250) and np.all(image[:,-2:]>250)
    return image,dict(label_voxels_represented=counts,full_target_voxels=int((volume==9).sum()),
        all_faces_inside_viewport=True,target_only=target)


def compose(images,out,kind,clean=False):
    geology=kind=='full_geology'
    rows=2 if geology else 1
    height=(109 if clean else 104) if geology else (55 if clean else 62)
    fig,axes=plt.subplots(rows,3,figsize=(180/25.4,height/25.4),squeeze=False)
    fig.subplots_adjust(left=.004,right=(.875 if geology and not clean else .996),
        top=(.995 if clean else .925),bottom=.006,wspace=.008,hspace=.008)
    for i in range(rows):
        for j in range(3):
            ax=axes[i,j]
            ax.imshow(images[i*3+j],interpolation='none')
            ax.set_axis_off()
            ax.text(.015,.97,f'({chr(97+i*3+j)})',transform=ax.transAxes,fontsize=9,va='top')
            if i==0 and not clean:
                ax.set_title(TITLES[j],fontsize=10,pad=3)
    if geology and not clean:
        fig.legend(handles=[Patch(facecolor=LABEL_COLORS[i],label=str(i)) for i in range(14)],
            loc='center left',bbox_to_anchor=(.89,.49),frameon=False,ncol=1,
            fontsize=8,handlelength=.9,handletextpad=.5,labelspacing=.65,borderaxespad=0)
    stem=('fig01_full_geology_comparison' if geology else 'fig02_target_body_comparison')+('_clean' if clean else '')
    fig.savefig(out/f'{stem}.png',dpi=600,facecolor='white')
    fig.savefig(out/f'{stem}.pdf',dpi=600,facecolor='white',metadata={'Title':stem})
    plt.close(fig)


def captions(out,metrics,quality,outside):
    a=metrics[('A_9','E_original_free')];n=metrics[('FLOW_STAGE23','E_original_free')]
    ah=metrics[('A_9','E_holdout')];nh=metrics[('FLOW_STAGE23','E_holdout')]
    direct=metrics[('RESTORED_CLASSIFICATION','E_original_free')]
    q={r['arm']:r for r in quality}
    off={r['arm']:r for r in outside}
    text=f'''# Stage23 revised captions / 修订图注

Both plates show **fullgeo_case03, seed42, synthetic property surrogate with known Gaussian kernel; frozen network**. Column order: ground truth, original nine-well frozen Flow A, Stage23 frozen Flow supplied with restored-property full multi-lithology pseudo-wells. “Ours” denotes preprocessing/conditioning, not retraining. The original boreholes and surface constraints are retained; Stage23 uses the old total512 layout, including503 additional well columns. No new inference, training, restoration solve or condition generation was performed. Each PDF contains raster3D with editable vector titles/panel letters/legend, not vector3D geometry.

## Figure A / 图A

Upper row: full nonair geology. Lower row: identical display-only front-corner removal using the existing cutaway_mask with cut_fraction0.52 (X≥33.28 and Y<33.28 in cell-index coordinates). Actual cell centers/array ordering are not transformed: XYZ tensor axes, VTK ImageData dimensions=shape+1, unit cell spacing, Fortran-order ravel. All panels use the same orthographic camera, focal point, scale and opaque exact voxel faces; no contour interpolation, smoothing, repair or per-model image cropping. Air−1 is invisible. The shared discrete legend gives raw labels0…13 from the verified project LABEL_COLORS; label9 remains orange. No invented lithological names. Wells and coordinate decorations are omitted. The camera is uniformly centered lower than the target plate to fill the panels with geology rather than empty air; every nonair mesh point in all three FULL models is verified inside the viewport. Scene data remain the complete64³ volume.

上排完整多岩性实体，下排同一前角缺口剖切。统一使用Stage15低层体素外露面、色表和光照，不更改标签几何；所有非空气岩性不透明。相机只按图统一调整，没有按每个模型的目标bbox放大或隐藏误报。色标数字为raw岩性标签；原井/地表条件在模型中保留但不叠加密集井柱。干净版仅保留(a)…(f)，没有列名或色标。

## Figure B / 图B

The FULL raw9 body in each model is shown as opaque #D95F02 voxel surfaces. All predicted components, including remote FP fragments, remain; there is no target cutaway, truth ghost, outline, ROI restriction, small-component filtering, hole filling or error recolouring. One common oblique orthographic camera frames the complete0…64 cube, including original nine wells rendered identically in thin #168A89 tubes (radius0.12 voxel) and a faint gray cube. Lines are original condition-position references, NOT512 pseudo-wells, and do not change colour according to truth hits. No surface or non-target opacity fog is added. Clean version has only(a)…(c).

三列分别只显示各自完整目标体；预测列不叠真值。保持单行三格，与Stage15三面板风格一致；青色细线是同一组原9井位置参照，浅灰盒为完整64³空间参照。所有碎片、漏失和过大部分如实展示。目标体的阶梯面与明暗来自真实体素几何和同一光照。

## Saved evidence, kept OUTSIDE figures / 外置数值与限制

Saved fixed E_holdout all-class mIoU: A={float(ah['fixed_class_miou']):.4f}, Stage23={float(nh['fixed_class_miou']):.4f}. On E_original_free, A target P/R/IoU={float(a['target_precision']):.6f}/{float(a['target_recall']):.6f}/{float(a['target_iou']):.6f}; Stage23={float(n['target_precision']):.6f}/{float(n['target_recall']):.6f}/{float(n['target_iou']):.6f}. Volume ratios are{float(a['target_pred_truth_volume_ratio']):.5f} and{float(n['target_pred_truth_volume_ratio']):.5f}. Displayed geometry is FULL volume, while these target metrics are in the fixed original-unknown domain.

On identical added_mask, old smooth versus restored pseudo-well accuracy is{float(q['C_512']['accuracy'])*100:.2f}%→{float(q['STAGE23_WELLS']['accuracy'])*100:.2f}%; target FN{q['C_512']['target_fn']}→{q['STAGE23_WELLS']['target_fn']}, FP{q['C_512']['target_fp']}→{q['STAGE23_WELLS']['target_fp']}. Hidden object1 common subset outside the Stage23 condition mask is{off['FLOW_STAGE23']['voxels']} voxels: A={off['A_9']['tp']}, COPY={off['COPY_ONLY_STAGE23']['tp']}, C={off['C_512']['tp']}, Stage23={off['FLOW_STAGE23']['tp']} correct target voxels.144/510 is coverage inside that truth-object subset, not object IoU or global recall.

Direct full restored-field classification remains much better: original-free target IoU{float(direct['target_iou']):.6f}, mIoU{float(direct['fixed_class_miou']):.6f}; it uses full-field information whereas Flow receives sampled wells. The result still misses/splits target bodies and has FP, so it is not accurate complete reconstruction. Known-kernel/no-extra-noise/artificial-surrogate/reused-case evidence does not establish real geophysical inversion, cross-case robustness or the paper's total objective. Holdout locations were not added as conditions but contributed to the global property surrogate; they are not independent observations/geological TEST.

视觉参照为实际Stage15两张PNG及本地Synthetic Geology论文图7/8（DOI10.1029/2025JH000986）；不复制旧任务ghost、粉色命中井、多实现阵列或论文原图。v1保留不覆盖。最终PNG需人工打开对照参考核验，配置和输入身份见同目录JSON。
'''
    (out/'CAPTIONS.md').write_text(text,encoding='utf-8')


def main(out):
    protect()
    if out.exists():
        raise FileExistsError(f'Preserve existing output: {out}')
    torch.set_num_threads(4)
    arrays,metrics,quality,outside,assets,identities=load_inputs()
    wells=read_json(R21/'layout.json')['original_wells']
    assert len(wells)==9
    volumes=[arrays[k] for k in ('truth','A_9','FLOW_STAGE23')]
    cams={'geology':camera(26.5,51.8),'target':camera(29.44,54.5)}
    out.mkdir(parents=True)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    keep=np.ones((64,64,64),dtype=bool)
    cut=cutaway_mask((64,64,64),.52)
    images=[];checks=[]
    for row in range(2):
        for col,volume in enumerate(volumes):
            image,check=render(volume,cut if row else keep,cams['geology'])
            images.append(image);checks.append(dict(kind='geology',row=row,col=col,**check))
            print(f'Full geology {row+1}/{col+1}',flush=True)
    for clean in (False,True):
        compose(images,out,'full_geology',clean)
    targets=[]
    for col,volume in enumerate(volumes):
        image,check=render(volume,keep,cams['target'],wells)
        targets.append(image);checks.append(dict(kind='target',col=col,**check))
        print(f'Complete target {col+1}',flush=True)
    for clean in (False,True):
        compose(targets,out,'target',clean)
    captions(out,metrics,quality,outside)
    write_json(out/'INPUT_ASSETS.json',dict(tensors=assets,metadata_file_sha256=identities,
        original_wells=wells,layout_sha256=file_hash(R21/'layout.json'),
        specification_sha256=file_hash(REPO/'STAGE23_VISUALIZATION_REVISION_PROMPT_2026-10-17.md')))
    write_json(out/'RENDER_CONFIG.json',dict(git=git_snapshot(),script_sha256=file_hash(__file__),
        shared_style_sha256=file_hash(PROJECT/'scripts/paper_figures/style.py'),cameras=cams,
        axis_order='XYZ; Fortran cell ravel; cell coordinates0..64',spacing=[1,1,1],
        panel_pixels=WINDOW,independent_crop=False,geology_opacity=1.,target_opacity=1.,
        light=dict(ambient=.34,diffuse=.66,specular=0,smooth_shading=False,show_edges=False),
        cut_fraction=.52,target_cutaway=False,truth_ghost=False,label_colors=LABEL_COLORS,
        original_well_color=OBSERVATION_COLOR,original_well_radius=.12,original_well_count=9,
        target_box='existing pale gray wireframe; complete0..64',dpi=600,
        panels=checks,pyvista=pv.__version__,vtk=vtk.vtkVersion.GetVTKVersion(),
        visual_inspection='pending',new_model_forwards=0,training=0,deconvolution=0,condition_generation=0))
    (out/'README.md').write_text('# Stage23 reference-style revision\n\nFrom repository root on labrs-codex:\n\n```bash\nPYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python '+
        str(Path(__file__).relative_to(REPO))+' --out '+str(out.relative_to(REPO))+
        '\n```\n\nChoose a fresh --out directory for redraw; existing output is protected. Only two scientific plates; each has normal/clean PNG/PDF. Clean versions retain only panel letters. Captions and metrics are external. PDF3D is raster, text/legend vector. Previous conference_paper_v1 is retained.\n')
    assert all(file_hash(REPO/p)==h for p,h in identities.items())
    print('Finished two plates and clean variants; scientific compute=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=OUT)
    main(parser.parse_args().out.resolve())
