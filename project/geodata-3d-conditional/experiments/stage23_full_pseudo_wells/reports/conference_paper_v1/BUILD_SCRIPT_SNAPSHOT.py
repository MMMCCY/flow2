#!/usr/bin/env python3
"""Read-only Stage23 figures from sealed tensors: no model, solve or new conditions."""
from __future__ import annotations
import argparse
import csv
import itertools
import os
from pathlib import Path
import sys

os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
sys.dont_write_bytecode = True
PROJECT = Path(__file__).resolve().parents[2]
REPO = PROJECT.parents[1]
sys.path.insert(0, str(PROJECT))
from scripts.stage21.common import read_json, write_json, write_csv, file_hash, verify_tensor, git_snapshot
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
import pyvista as pv
import vtk
from scripts.paper_figures.style import LABEL_COLORS, LABEL9_COLOR, binary_surface, _add_surface

R23 = PROJECT/'experiments/stage23_full_pseudo_wells/run_20261017_v2'
R21 = PROJECT/'experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1'
R22 = PROJECT/'experiments/stage22_blur_aware_bridge/run_20261016_v1'
DEFAULT_OUT = R23.parent/'reports/conference_paper_v1'
WINDOW = (1500, 1360)
DPI = 600
TITLES = ['Ground truth', 'Frozen Flow\n(9 wells)', 'Frozen Flow + pseudo-wells\nblur-aware preprocessing']


def protect():
    protected = [R21, R22, R23, R23.parent/'run_20261017_v1',
        PROJECT/'experiments/stage15_binary_seismic_consensus',
        PROJECT/'scripts/stage21v2', PROJECT/'scripts/stage22', PROJECT/'scripts/stage23']
    def audit(event,args):
        if event!='open' or not isinstance(args[0],(str,bytes,os.PathLike)):
            return
        p=Path(os.fsdecode(args[0])).resolve()
        writing=((isinstance(args[1],str) and any(c in args[1] for c in 'wax+')) or
                 (isinstance(args[2],int) and args[2] & (os.O_WRONLY|os.O_RDWR|os.O_CREAT|os.O_TRUNC|os.O_APPEND)))
        if p.suffix=='.ckpt':
            raise PermissionError('Visualization must never open a checkpoint')
        if writing and any(p==r or r in p.parents for r in protected):
            raise PermissionError(f'Sealed history is read-only: {p}')
    sys.addaudithook(audit)


def csv_rows(path):
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def load_inputs():
    execution=read_json(R23/'EXECUTION.json')
    assert execution['status']=='completed' and execution['new_Flow_outputs']==1 and execution['velocity_evaluations']==32
    samples={s['arm']:s for s in execution['samples']}
    ereg=read_json(R23/'EVALUATION_INPUTS.json')
    assets={'truth':ereg['truth']}
    identities={}
    def identity(path):
        identities[str(path.relative_to(REPO))]=file_hash(path)
    for arm in ('A_9','B_512','C_512','FLOW_STAGE22','RESTORED_CLASSIFICATION','FLOW_STAGE23','COPY_ONLY_STAGE23'):
        sample=samples[arm]
        assets[arm]=sample['prediction']
        if sample.get('receipt'):
            path=REPO/sample['receipt']['path']
            assert file_hash(path)==sample['receipt']['file_sha256']
            receipt=read_json(path)
            assert receipt['status']=='completed' and receipt.get('seed',receipt.get('source_seed'))==42
            if 'case_id' in receipt:
                assert receipt['case_id']=='fullgeo_case03'
            assert receipt['prediction']==sample['prediction']
            complete=path.parent/'completion_receipt.json'
            assert read_json(complete)['receipt_sha256']==file_hash(path)
            identity(path); identity(complete)
    # Bind truth to the registered case, rather than guessing from a file basename.
    oldcase=read_json(R21/'evaluation_registry.json')['cases']['fullgeo_case03']
    assert assets['truth']==oldcase['truth']
    assert read_json(R23/'PROTOCOL.json')['case']=='fullgeo_case03'
    assert read_json(R23/'inference_registry.json')['noise'].keys()=={'42'}
    cond=read_json(R23/'inference_registry.json')['conditions']
    assets.update({f'condition_{k}':v for k,v in cond.items()})
    arrays={}
    for key,asset in assets.items():
        t=verify_tensor(asset)
        assert t.shape==(1,1,64,64,64)
        if 'mask' not in key:
            assert torch.isfinite(t).all() and torch.equal(t,t.round()) and t.min()>=-1 and t.max()<=13
        arrays[key]=t.numpy()[0,0]
    m=arrays['condition_mask']; v=arrays['condition_values']
    assert not ((arrays['FLOW_STAGE23']!=v)&m).any()
    assert np.array_equal(m,verify_tensor(read_json(R21/'inference_registry.json')['conditions']['fullgeo_case03']['B_512']['mask']).numpy()[0,0])
    rows=csv_rows(R23/'comparison.csv')
    metrics={(row['arm'],row['domain']):row for row in rows}
    # CSV-to-sealed-EXECUTION consistency, not a repeat of scientific evaluation.
    for (arm,domain),row in metrics.items():
        saved=samples[arm]['metrics'][domain]
        for key in ('target_iou','target_precision','target_recall','fixed_class_miou','target_pred_truth_volume_ratio'):
            assert abs(float(row[key])-saved[key])<1e-12
    quality=csv_rows(R23/'condition_quality.csv')
    outside=csv_rows(R23/'object1_common_outside.csv')
    for row in quality:
        saved=next(q for q in execution['conversion'] if q['arm']==row['arm'])
        assert float(row['accuracy'])==saved['accuracy']
        assert int(row['target_fn'])==saved['target_fn'] and int(row['target_fp'])==saved['target_fp']
    for row in outside:
        saved=next(q for q in execution['object1_common_subset'] if q['arm']==row['arm'])
        assert int(row['tp'])==saved['tp'] and int(row['voxels'])==saved['voxels']
    for name in ('EXECUTION.json','EVALUATION_INPUTS.json','inference_registry.json','PROTOCOL.json',
                 'comparison.csv','condition_quality.csv','object1_common_outside.csv','object_metrics.csv'):
        identity(R23/name)
    identity(R21/'evaluation_registry.json')
    return arrays,metrics,quality,outside,assets,identities


def camera(direction):
    """One camera scale from the FULL fixed cube, never an individual model bbox."""
    direction=np.asarray(direction,dtype=float)
    forward=-direction/np.linalg.norm(direction)
    right=np.cross(forward,[0,0,1.]); right/=np.linalg.norm(right)
    up=np.cross(right,forward)
    corners=np.array(list(itertools.product([0.,64.],repeat=3)))
    centered=corners-32
    extent_x=max(abs(centered@right)); extent_y=max(abs(centered@up))
    scale=max(extent_y,extent_x/(WINDOW[0]/WINDOW[1]))*1.15
    return dict(position=(np.ones(3)*32+direction*64).tolist(),focal_point=[32.,32.,32.],
        view_up=[0.,0.,1.],parallel_projection=True,parallel_scale=float(scale),
        screen_right=right.tolist(),screen_up=up.tolist(),scene_bounds=[0,64,0,64,0,64])


def project_point(point,cam):
    relative=np.asarray(point)-np.asarray(cam['focal_point'])
    sx=2*cam['parallel_scale']*WINDOW[0]/WINDOW[1]
    sy=2*cam['parallel_scale']
    return [.5+float(relative@np.asarray(cam['screen_right']))/sx,
            .5+float(relative@np.asarray(cam['screen_up']))/sy]


def render(volume,keep,cam,target_only,path):
    pv.OFF_SCREEN=True
    plotter=pv.Plotter(off_screen=True,window_size=WINDOW,border=False)
    plotter.set_background('white')
    plotter.enable_anti_aliasing('ssaa')
    counts={}
    labels=[9] if target_only else sorted(int(i) for i in np.unique(volume) if i!=-1)
    for label in labels:
        selected=(volume==label)&keep
        counts[str(label)]=int(selected.sum())
        mesh=binary_surface(selected)
        if mesh is not None:
            # Exact exposed voxel faces, no contour interpolation, smoothing or repair.
            assert np.allclose(np.mod(mesh.points,1),0)
            _add_surface(plotter,mesh,LABEL_COLORS[label],1.)
    box=pv.Box(bounds=(0,64,0,64,0,64))
    plotter.add_mesh(box,style='wireframe',color='#A9AFB4',line_width=.8,lighting=False)
    plotter.camera_position=[cam['position'],cam['focal_point'],cam['view_up']]
    plotter.camera.parallel_projection=True
    plotter.camera.parallel_scale=cam['parallel_scale']
    plotter.camera.clipping_range=(1.,600.)
    picture=plotter.screenshot(return_img=True)
    plotter.close()
    Image.fromarray(picture[:,:,:3]).save(path,dpi=(600,600))
    return picture[:,:,:3],dict(path=path.name,label_voxels_represented=counts,
        original_target_voxels=int((volume==9).sum()),target_only=target_only,
        no_component_filter=True,no_ghost=True,full_frame_no_trim=True)


def image_panel(ax,picture,cam,letter,side=False):
    ax.imshow(picture,extent=(0,1,0,1),interpolation='none',aspect=WINDOW[1]/WINDOW[0])
    ax.set(xlim=(0,1),ylim=(0,1)); ax.set_axis_off()
    ax.text(.01,.99,f'({letter})',transform=ax.transAxes,va='top',fontsize=9,fontweight='bold')
    # Vector axis labels / triad. No text is baked into the VTK render.
    origin=np.array([.08,.10 if side else .13])
    labels=[('X',np.array([1,0,0.])),('Z',np.array([0,0,1.]))] if side else [
        ('X',np.array([1,0,0.])),('Y',np.array([0,1,0.])),('Z',np.array([0,0,1.]))]
    for label,axis in labels:
        d=np.array([axis@cam['screen_right'],axis@cam['screen_up']])*(.045 if side else .095)
        ax.annotate('',xy=origin+d,xytext=origin,xycoords='axes fraction',
            arrowprops=dict(arrowstyle='-',lw=.65,color='#3F454B'))
        ax.text(*(origin+d*1.28),label,transform=ax.transAxes,ha='center',va='center',fontsize=8)


def save(fig,out,stem):
    fig.savefig(out/f'{stem}.png',dpi=DPI,facecolor='white')
    fig.savefig(out/f'{stem}.pdf',dpi=DPI,facecolor='white',metadata={'Title':stem,'Creator':'Stage23 CPU visualization'})
    # Check publication dimensions and minimum vector text size in the actual layout.
    from matplotlib.text import Text
    sizes=[t.get_fontsize() for t in fig.findobj(Text) if t.get_text().strip()]
    assert min(sizes)>=8
    plt.close(fig)


def full_figure(images,cam,metrics,out):
    fig,axes=plt.subplots(2,3,figsize=(180/25.4,138/25.4))
    fig.subplots_adjust(left=.015,right=.985,top=.885,bottom=.19,wspace=.035,hspace=.10)
    for j,title in enumerate(TITLES):
        axes[0,j].set_title(title,fontsize=9,pad=11)
        for i in range(2):
            image_panel(axes[i,j],images[('geology',i,j)],cam,chr(97+3*i+j))
    for j,arm in ((1,'A_9'),(2,'FLOW_STAGE23')):
        value=float(metrics[(arm,'E_holdout')]['fixed_class_miou'])
        axes[1,j].text(.5,-.055,f'Holdout mIoU = {value:.4f}',transform=axes[1,j].transAxes,
            ha='center',fontsize=9)
    handles=[Patch(facecolor=LABEL_COLORS[i],label=f'raw {i}') for i in range(14)]
    fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.045),ncol=7,fontsize=8,
        frameon=False,handlelength=.85,columnspacing=1.1,handletextpad=.35)
    fig.text(.5,.140,'Top: complete volume   |   Bottom: identical quadrant cut at X/Y = 32',ha='center',fontsize=8)
    fig.text(.5,.014,'X / Y / Z: voxel indices, 0–64; unit spacing; air hidden; original conditions retained',ha='center',fontsize=8)
    save(fig,out,'fig01_full_geology_comparison')


def target_figure(images,cams,metrics,outside,truth,out):
    fig,axes=plt.subplots(2,3,figsize=(180/25.4,138/25.4))
    fig.subplots_adjust(left=.015,right=.985,top=.885,bottom=.17,wspace=.035,hspace=.20)
    for j,title in enumerate(TITLES):
        axes[0,j].set_title(title,fontsize=9,pad=11)
        for i in range(2):
            image_panel(axes[i,j],images[('target',i,j)],cams[i],chr(97+3*i+j),side=i==1)
        if j:
            arm='A_9' if j==1 else 'FLOW_STAGE23'
            row=metrics[(arm,'E_original_free')]
            text=f"IoU {float(row['target_iou']):.4f} | P {float(row['target_precision']):.4f} / R {float(row['target_recall']):.4f}"
            axes[1,j].text(.5,-.07,text,transform=axes[1,j].transAxes,ha='center',fontsize=8)
    from scipy.ndimage import label
    ids,count=label(truth==9,structure=np.ones((3,3,3)))
    assert count==2 and int((ids==1).sum())==572
    for object_id,text,xytext in ((1,'Hidden object 1\n(572 voxels)',(.14,.95)),(2,'Object 2',(.64,.20))):
        center=np.argwhere(ids==object_id).mean(0)+.5
        axes[0,0].annotate(text,xy=project_point(center,cams[0]),xytext=xytext,
            xycoords='axes fraction',textcoords='axes fraction',fontsize=8,va='top',
            arrowprops=dict(arrowstyle='-',lw=.65,color='#444444'),
            bbox=dict(facecolor='white',edgecolor='none',pad=1,alpha=.85))
    row=next(r for r in outside if r['arm']=='FLOW_STAGE23')
    base=next(r for r in outside if r['arm']=='A_9')
    assert int(row['voxels'])==int(base['voxels'])==510
    fig.text(.5,.100,f"Hidden object: {row['tp']}/{row['voxels']} previously unconstrained voxels recovered",ha='center',fontsize=9,fontweight='bold')
    fig.text(.5,.065,f"Baseline: {base['tp']}/{base['voxels']} | common object-1 subset outside Stage23 conditions; coverage, not IoU",ha='center',fontsize=8)
    fig.text(.5,.022,'Full 64³ scene; all target components retained | top: isometric; bottom: orthographic −Y view',ha='center',fontsize=8)
    save(fig,out,'fig02_target_body_comparison')


def evidence_figure(metrics,quality,out):
    fig,axes=plt.subplots(1,2,figsize=(180/25.4,89/25.4))
    fig.subplots_adjust(left=.09,right=.985,top=.85,bottom=.31,wspace=.38)
    q={r['arm']:r for r in quality}
    vals=[float(q[a]['accuracy']) for a in ('C_512','STAGE23_WELLS')]
    axes[0].bar([0,1],vals,color=['#8F9CA8','#D95F02'],width=.58)
    axes[0].set(ylim=(0,1.14),xticks=[0,1],xticklabels=['Smoothed\npseudo-wells','Restored\npseudo-wells'],ylabel='All-label accuracy')
    for i,arm in enumerate(('C_512','STAGE23_WELLS')):
        r=q[arm]
        axes[0].text(i,vals[i]+.035,f"{vals[i]*100:.2f}%",ha='center',fontsize=9,fontweight='bold')
        axes[0].text(i,.045,f"Target FN {r['target_fn']}\nTarget FP {r['target_fp']}",ha='center',va='bottom',fontsize=8,color='white')
    axes[0].set_title('(a) Conversion at the same added mask',loc='left',fontsize=9,pad=12)
    arms=['A_9','C_512','FLOW_STAGE23','B_512']; x=np.arange(4)
    for offset,key,title,color in ((-.18,'target_iou','Target IoU','#D95F02'),(.18,'fixed_class_miou','All-class mIoU','#5A91C8')):
        vals=[float(metrics[(a,'E_holdout')][key]) for a in arms]
        axes[1].bar(x+offset,vals,width=.34,label=title,color=color)
        for xx,val in zip(x+offset,vals):
            axes[1].text(xx,val+.012,f'{val:.4f}',ha='center',va='bottom',rotation=90,fontsize=8)
    axes[1].set(xticks=x,xticklabels=['A\n9 wells','C\nsmooth','Stage23\nrestored','B\ntrue wells'],ylim=(0,.76),ylabel='Score on fixed holdout')
    axes[1].set_title('(b) Spatial holdout evidence',loc='left',fontsize=9,pad=12)
    axes[1].legend(loc='upper left',bbox_to_anchor=(-.05,1.015),ncol=2,fontsize=8,frameon=False,handlelength=.8,columnspacing=.7)
    for ax in axes:
        ax.spines[['top','right']].set_visible(False)
        ax.tick_params(labelsize=8)
    direct=metrics[('RESTORED_CLASSIFICATION','E_original_free')]
    fig.text(.5,.155,f"Full-field restored direct classification: target IoU {float(direct['target_iou']):.4f}; mIoU {float(direct['fixed_class_miou']):.4f}",ha='center',fontsize=8)
    fig.text(.5,.105,'Direct classification uses the full restored field; Flow receives sampled pseudo-wells.',ha='center',fontsize=8)
    fig.text(.5,.045,'One case / one seed; no error bars. Holdout locations contributed to the global property surrogate.',ha='center',fontsize=8)
    save(fig,out,'figS01_bridge_evidence')


def captions(metrics,out):
    text='''# Stage23 figure captions / 图注

All main figures: **fullgeo_case03, seed42, synthetic property surrogate with known Gaussian kernel; frozen network**. No extra noise, no training, no new deconvolution, no new conditioning or model inference. These are retrospective single-case illustrations, not ensemble uncertainty or independent geological TEST. Original nine boreholes and surface constraints are retained in all models. The Stage23 model additionally uses full multi-lithology pseudo-wells from the old total512 layout (503 extra columns); wells are omitted visually. Shapes and values are unchanged.

## Figure 1 / 图1：Full geology comparison / 整体多岩性模型

Ground truth, frozen nine-well Flow A, and frozen Flow conditioned by restored-property pseudo-wells. Upper row: complete nonair64³ geology. Lower row: the SAME display-only quadrant removal X≥32 and Y<32. All panels use one camera, parallel scale, fixed scene bounds, unit voxel spacing and Stage15 discrete raw-label colours. Air (raw−1) is hidden, and raw labels0…13 have no invented geological names. Labels use XYZ array order, as verified against the VTK ImageData Fortran-order cell mapping. Fixed E_holdout all-class mIoU comes from the saved CSV; ground truth has no performance score. No independent panel trimming or rescaling.

真值、原9井冻结Flow及恢复属性完整伪井条件下的冻结Flow。上排完整非空气体，下排统一剖切（仅显示移除X≥32且Y<32的象限）。同相机、焦点、尺度、色表、光照和64³边框；无几何平滑修补。原条件保留但不绘井柱。原始张量轴为XYZ，体素边界0…64、中心0.5…63.5，不虚构米或海拔。

## Figure 2 / 图2：Complete target bodies / 完整目标岩体

All raw9 voxels in the FULL64³ scene, including every FP fragment, with opaque orange voxel surfaces and no truth ghost or target cutaway. Upper row uses a common isometric camera, lower row the same orthographic view along−Y. Truth-only object labels are retrospective annotations; there are no ROI restrictions on predictions. Original-unknown-domain (E_original_free) IoU, precision and recall are saved metrics while the scene displays the complete volume. For A versus Stage23, predicted/true target volume ratios are2.63993 versus1.90075. The common hidden-object subset is truth object1 intersect E_original_free outside the Stage23 NEW mask:510 voxels; A/C/COPY recover0, Stage23 recovers144 (28.24%). This percentage is coverage inside that object subset, NOT object IoU or global recall. Whole object1 any-prediction coverage is206/572, with one-to-one matched IoU0.122137. Missing parts, splitting and distant false positives remain visible.

三列保留全部目标预测，不按真值或连通分量筛选、不填洞、不叠加真值ghost。上下排为统一等轴和沿−Y正交视角。真值对象1标注仅用于回顾性解释。整场景展示与指标域分开：P/R/IoU在固定原未知域计算，144/510是同一井外对象子集内覆盖率，不能标成IoU。本轮仍有漏失、分裂和误报，不能称准确完整重建。

## Figure S1 / 补图：Bridge evidence / 桥接证据

Left: all-label pseudo-well accuracy on the identical18555-voxel added_mask, with target FN/FP; no truth-based removal of incorrect segments. Right: saved fixed E_holdout target IoU and all-class mIoU for A, smooth pseudo-well C, Stage23, and accurate-well B. No error bars for one seed. Direct restored classification is explicitly retained: E_original_free target IoU0.9677 and mIoU0.6627, much better than Flow, but it uses the FULL spatial field rather than sampled pseudo-well information. B uses accurate true well labels and is not an information-matched deployment competitor. Holdout has no added condition, but its property information participated in the global surrogate; it is not independent observation holdout.

左图保留伪井全部错误：准确率60.99%→92.44%，目标FN211→7、FP0→1。右图保留准确井B对照与直接分类的优势，避免只展示有利结果。转换和井外恢复改善不等于真实地球物理反演、跨案例稳健或论文总目标完成。

## Rendering and editable output / 渲染与编辑

PNG600dpi. PDF embeds raster3D renders, with vector titles, axes labels/triads, metrics, annotations and discrete legend (embedded TrueType fonts); it is NOT wholly vector3D geometry. Original1500×1360 panel PNGs are in panels/. Style reference: actual Stage15 figures. Structural reference: local Synthetic Geology PDF, DOI10.1029/2025JH000986, pages12–13 / Figures7–8; no source artwork or multiple-realization claim copied. Reproduction command is in README.md; RENDER_CONFIG and INPUT_ASSETS bind geometry, rendering and exact saved evidence.
'''
    (out/'CAPTIONS.md').write_text(text,encoding='utf-8')


def main(out):
    protect()
    torch.set_num_threads(4)
    if out.exists():
        raise FileExistsError(f'Refuse unknown/existing output: {out}; choose a new --out')
    arrays,metrics,quality,outside,assets,identities=load_inputs()
    out.mkdir(parents=True); (out/'panels').mkdir()
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.labelsize':9,
        'xtick.labelsize':8,'ytick.labelsize':8,'legend.fontsize':8,'pdf.fonttype':42,
        'ps.fonttype':42,'figure.facecolor':'white','axes.linewidth':.65})
    cams=[camera([1.72,-1.35,1.38]),camera([0,2.6,0])]
    keep=np.ones((64,64,64),dtype=bool)
    cut=keep.copy(); cut[32:,:32,:]=False
    volumes=[arrays[k] for k in ('truth','A_9','FLOW_STAGE23')]
    images={}; panel_checks=[]
    for kind in ('geology','target'):
        for row in range(2):
            for col,volume in enumerate(volumes):
                cam=cams[row] if kind=='target' else cams[0]
                mask=cut if kind=='geology' and row==1 else keep
                path=out/'panels'/f'{kind}_r{row+1}_c{col+1}.png'
                image,check=render(volume,mask,cam,kind=='target',path)
                if kind=='target':
                    assert check['label_voxels_represented']['9']==int((volume==9).sum())
                images[(kind,row,col)]=image
                panel_checks.append(check)
                print(f'Rendered {path.name}',flush=True)
    full_figure(images,cams[0],metrics,out)
    target_figure(images,cams,metrics,outside,arrays['truth'],out)
    evidence_figure(metrics,quality,out)
    captions(metrics,out)
    write_csv(out/'figure_metrics.csv',[dict(arm=arm,domain=domain,**{k:v for k,v in row.items() if k not in ('arm','domain')}) for (arm,domain),row in metrics.items()])
    write_csv(out/'figure_condition_metrics.csv',quality)
    write_csv(out/'figure_object1_common_subset.csv',outside)
    config=dict(case='fullgeo_case03',seed=42,git=git_snapshot(),script_sha256=file_hash(__file__),
        reference_commit='90ca753f0413f48aa57e989ba41f327ae98573b7',
        array_order='XYZ; cell_data=array.ravel(order=F); dimensions=array.shape+1',
        voxel_spacing=[1,1,1],scene_bounds=[0,64,0,64,0,64],units='voxel index',
        voxel_centers='0.5..63.5; cell boundaries0..64',cameras=cams,
        full_geology_opacity=1.,target_opacity=1.,target_background_opacity=0.,
        removed_quadrant='display only, geology lower row: X>=32 AND Y<32',target_cutaway=False,
        label_colors=LABEL_COLORS,air_hidden=-1,truth_ghost=False,wells_rendered=False,
        surface='exact binary voxel cell faces; no contour interpolation/smoothing/repair',
        panel_size_pixels=WINDOW,figure_size_mm=[180,138],evidence_size_mm=[180,89],dpi=DPI,
        independent_image_trim=False,minimum_font_pt=8,PDF='raster3D with vector text/legend',
        pyvista=pv.__version__,vtk=vtk.vtkVersion.GetVTKVersion(),matplotlib=matplotlib.__version__,
        new_model_forwards=0,training=0,deconvolution=0,condition_generation=0)
    write_json(out/'RENDER_CONFIG.json',config)
    write_json(out/'INPUT_ASSETS.json',dict(tensors=assets,metadata_file_sha256=identities,
        reference_style_sha256=file_hash(PROJECT/'scripts/paper_figures/style.py'),
        specification_sha256=file_hash(REPO/'STAGE23_VISUALIZATION_PROMPT_2026-10-17.md')))
    write_json(out/'RENDER_CHECKS.json',dict(status='rendered_pending_visual_inspection',panels=panel_checks,
        all_target_voxels_represented=True,hashes_verified=True,metrics_match_saved_execution=True,
        minimum_vector_font_pt=8,new_model_forwards=0))
    command=f'PYTHONDONTWRITEBYTECODE=1 CUDA_VISIBLE_DEVICES="" .venv/bin/python {Path(__file__).relative_to(REPO)} --out {out.relative_to(REPO)}'
    (out/'README.md').write_text('# Stage23 visualization\n\n'+
        'From the repository root, on labrs-codex with the repository .venv:\n\n```bash\n'+command+
        '\n```\n\nThe output path must not exist; use another --out for a fresh redraw. Inputs are read-only. '+
        'No checkpoint is loaded. Original renders: panels/. Editable vector text is preserved in PDF; 3D is raster. '+
        'Scientific captions: CAPTIONS.md. Geometry/render/identity: RENDER_CONFIG.json, INPUT_ASSETS.json.\n')
    assert all(file_hash(REPO/p)==h for p,h in identities.items())
    print(f'Complete: {out}; all model/solve/training/condition-generation counts=0',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out',type=Path,default=DEFAULT_OUT)
    main(parser.parse_args().out.resolve())
