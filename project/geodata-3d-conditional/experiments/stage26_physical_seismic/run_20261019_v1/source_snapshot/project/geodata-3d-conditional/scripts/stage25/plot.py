"""Approved_v1 exact rendering/composition, plus fixed Y26 categorical sections."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage25.common import *
from scripts.stage24.plot import standard
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch
from PIL import Image

def save_panel(im,path):
    Image.fromarray(im).save(path.with_suffix('.png'),dpi=(600,600))
    fig=plt.figure(figsize=(4,3.6));ax=fig.add_axes([0,0,1,1]);ax.imshow(im);ax.axis('off');fig.savefig(path.with_suffix('.pdf'),dpi=600);plt.close(fig)

def sections(volumes,names,approved,path):
    colors=approved.namespace['LABEL_COLORS'];cmap=ListedColormap([colors[i] for i in range(-1,14)])
    fig,axes=plt.subplots(1,len(volumes),figsize=(3.1*len(volumes)+.6,3.3),sharex=True,sharey=True)
    for ax,v,title in zip(np.atleast_1d(axes),volumes,names):
        ax.imshow(v[:,26,:].T,origin='lower',interpolation='nearest',cmap=cmap,norm=BoundaryNorm(np.arange(-1.5,14,1),15),extent=(-.5,63.5,-.5,63.5))
        ax.set_title(title,fontsize=10);ax.set_xlabel('X index');ax.set_aspect('equal');ax.set_xticks([0,16,32,48,63]);ax.set_yticks([0,16,32,48,63])
    np.atleast_1d(axes)[0].set_ylabel('Z index (upward)')
    used=sorted(set(int(i) for v in volumes for i in np.unique(v))-{ -1 })
    fig.legend(handles=[Patch(facecolor=colors[i],label=str(i)) for i in used],loc='center right',frameon=False,fontsize=8)
    fig.subplots_adjust(left=.07,right=.92,bottom=.16,top=.9,wspace=.08)
    for ext in ('png','pdf'):fig.savefig(path.with_suffix('.'+ext),dpi=600,facecolor='white')
    plt.close(fig)

def main(out=None):
    guard('plot');e=read_json(ROOT/'EVALUATION.json');assert e['status']=='completed'
    out=Path(out).resolve() if out else ROOT/'figures'
    if out.exists():raise FileExistsError(out)
    out.mkdir();(out/'panels').mkdir()
    approved=standard();cams=approved.CONFIG['cameras'];layout=read_json(ROOT/'layout.json');wells=layout['original_wells']
    truth=read_json(ROOT/'evaluation_registry.json')['truth']
    assets={'Truth':truth,**{a:e['samples'][a] for a in 'ABC'},'DIRECT':e['offline']['DIRECT']}
    arrays={a:verify_tensor(asset).numpy()[0,0] for a,asset in assets.items()}
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    keep=np.ones((64,64,64),dtype=bool);cut=approved.cutaway_mask((64,64,64),.52)
    checks=[];ims=[]
    approved.namespace['TITLES']=['Ground truth','Frozen Flow A','Pseudo-well Flow C']
    for row in range(2):
        for j,key in enumerate(('Truth','A','C')):
            im,check=approved.render(arrays[key],cut if row else keep,cams['geology'])
            save_panel(im,out/'panels'/f'geology_{key}_r{row+1}');ims.append(im);checks.append(dict(arm=key,kind='geology',row=row,**check))
    for clean in (False,True):approved.compose(ims,out,'full_geology',clean)
    ims=[]
    for key in ('Truth','A','C'):
        im,check=approved.render(arrays[key],keep,cams['target'],wells)
        save_panel(im,out/'panels'/f'target_{key}');ims.append(im);checks.append(dict(arm=key,kind='target',**check))
    for clean in (False,True):approved.compose(ims,out,'target',clean)
    for key in ('B','DIRECT'):
        for kind,mask,cam,ww in [('full',keep,cams['geology'],None),('cutaway',cut,cams['geology'],None),('target',keep,cams['target'],wells)]:
            im,check=approved.render(arrays[key],mask,cam,ww);save_panel(im,out/'panels'/f'{kind}_{key}');checks.append(dict(arm=key,kind=kind,**check))
    sections([arrays[a] for a in ('Truth','A','C')],['Ground truth','Frozen Flow A','Pseudo-well Flow C'],approved,out/'fig03_section_Y26')
    sections([arrays[a] for a in ('Truth','B','DIRECT')],['Ground truth','Accurate-well Flow B','Direct classification'],approved,out/'fig04_controls_section_Y26')
    write_json(out/'RENDER_CONFIG.json',dict(standard='geology_approved_v1',standard_sha256=file_hash(STANDARD/'MANIFEST.json'),cameras=cams,section=dict(axis='Y',index=26,origin='pre-frozen generation center'),inputs=assets,checks=checks,label_filtering=False,morphology_repair=False,target_cutaway=False,rendered_all_components=True,model_forward_calls=0,visual_inspection='pending'))
    (out/'CAPTIONS.md').write_text('''# 图注

主图列顺序：真值、原9井冻结Flow A、完整属性伪井冻结Flow C。整体岩性图上排完整、下排认可版固定显示缺口；目标图显示完整raw9，保留所有分量、误报及碎片，青色细线为原9井。所有3D面板使用approved_v1封存函数、色表、光照、相机与布局；沿用Stage24实际视口验证修复。列标题仅区分本次臂。

fig03是预先固定Y=26中心剖面，XYZ原轴、Z向上，同色表、无插值；展示拱顶、肩部及向下供给柱。fig04和panels中的B/DIRECT为独立解释面板；它们不扩大主图布局。

网格采样位置为每轴[-1920,1920]含端点64点，实际相邻间距3840/63米。认可3D渲染把每个离散标签画为单位体素面，边界位于0…64，其半格图形偏移不影响数组或距离指标。PDF中的3D部分为高清栅格，标题/图例为矢量。

数字见图外metrics.csv、shape_metrics.csv、condition_quality.csv、outside_added_wells.csv及中文REPORT.md。直接分类使用全场，Flow使用512井采样，信息量不同。单案例、单seed、已知Gaussian核、无额外噪声、人工双属性，不能表述为实测物理验证或泛化结论。holdout未硬写入，但其观测参与全场恢复，不是独立观测测试。
''',encoding='utf-8')
    print('FIGURES_COMPLETE',out,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);main(p.parse_args().out)
