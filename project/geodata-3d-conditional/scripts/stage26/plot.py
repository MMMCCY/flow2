"""Approved_v1 Truth/A/D plates, D0 control and fixed-Y26 time-domain seismic."""
import os,sys
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26.common import *
from scripts.stage24.plot import standard
from scripts.stage25.plot import save_panel,sections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def main(out=None):
    guard('plot');torch.set_num_threads(4)
    e=read_json(ROOT/'EVALUATION.json');assert e['status']=='completed'
    out=Path(out).resolve() if out else ROOT/'figures'
    if out.exists():raise FileExistsError('Choose a new redraw directory')
    out.mkdir(parents=True);(out/'panels').mkdir()
    truth=read_json(OLD/'evaluation_registry.json')['truth']
    assets={'Truth':truth,**{a:e['prediction_assets'][a] for a in ('A','D','D0')}}
    arrays={k:verify_tensor(v).numpy()[0,0] for k,v in assets.items()}
    approved=standard();approved.namespace['TITLES']=['Ground truth','Frozen Flow A','Seismic-well Flow D']
    cams=approved.CONFIG['cameras'];wells=read_json(ROOT/'layout.json')['original_wells']
    keep=np.ones((64,64,64),dtype=bool);cut=approved.cutaway_mask((64,64,64),.52)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    images=[];checks=[]
    for row in range(2):
        for key in ('Truth','A','D'):
            im,check=approved.render(arrays[key],cut if row else keep,cams['geology']);save_panel(im,out/'panels'/f'geology_{key}_r{row+1}');images.append(im);checks.append(dict(arm=key,kind='geology',row=row,**check))
    for clean in (False,True):approved.compose(images,out,'full_geology',clean)
    images=[]
    for key in ('Truth','A','D'):
        im,check=approved.render(arrays[key],keep,cams['target'],wells);save_panel(im,out/'panels'/f'target_{key}');images.append(im);checks.append(dict(arm=key,kind='target',**check))
    for clean in (False,True):approved.compose(images,out,'target',clean)
    im,check=approved.render(arrays['D0'],keep,cams['target'],wells);save_panel(im,out/'panels/target_D0');checks.append(dict(arm='D0',kind='target',**check))
    sections([arrays[k] for k in ('Truth','A','D')],['Ground truth','Frozen Flow A','Seismic-well Flow D'],approved,out/'fig03_section_Y26')
    sections([arrays['D0']],['Prior-only Flow D0'],approved,out/'fig04_D0_section_Y26')
    freeze=read_json(ROOT/'INPUT_FREEZE.json');inv=read_json(ROOT/'INVERSION_STATE.json')
    seismic=[verify_tensor(a).numpy()[0,0,:,26,:].T for a in (freeze['observation'],inv['assets']['prior_seismic'],inv['assets']['updated_seismic'])]
    limit=max(float(np.abs(v).max()) for v in seismic);dt=8.;nt=seismic[0].shape[0]
    fig,axes=plt.subplots(1,3,figsize=(9.9,4.4),sharex=True,sharey=True)
    for ax,field,title in zip(axes,seismic,['Observed','A prior','Updated impedance']):
        im=ax.imshow(field,origin='upper',interpolation='nearest',aspect='auto',extent=(-.5,63.5,(nt-.5)*dt,-.5*dt),cmap=approved.style.SEISMIC_CMAP,vmin=-limit,vmax=limit)
        ax.set_title(title,fontsize=10);ax.set_xlabel('X index');ax.set_xticks([0,16,32,48,63]);ax.set_yticks([0,800,1600,2400,3192])
    axes[0].set_ylabel('Two-way time (ms)')
    fig.subplots_adjust(left=.08,right=.90,bottom=.14,top=.9,wspace=.12)
    cax=fig.add_axes([.92,.16,.015,.7]);fig.colorbar(im,cax=cax,label='Amplitude')
    for ext in ('png','pdf'):fig.savefig(out/f'fig05_seismic_Y26.{ext}',dpi=600,facecolor='white')
    plt.close(fig)
    write_json(out/'RENDER_CONFIG.json',dict(standard='geology_approved_v1',standard_sha256=file_hash(STANDARD/'MANIFEST.json'),cameras=cams,section=dict(axis='Y',index=26,selection='pre-fixed Stage25 center'),inputs=assets,checks=checks,all_components_retained=True,target_cutaway=False,visual_inspection='pending',render_source_sha256=file_hash(Path(__file__)),seismic=dict(section_Y=26,time_axis='TWT milliseconds; not depth',sample_interval_ms=8.,num_samples=nt,shared_color_limits=[-limit,limit],independent_scaling=False),plot_model_calls=0))
    (out/'CAPTIONS.md').write_text('''# 外置图注

主图为同一真值、旧原9井冻结Flow A、本次地震阻抗伪井冻结Flow D。整体岩性上排完整、下排认可版固定缺口；目标图保留完整64³全部raw9分量、误报和碎片，不截目标、不修补、不筛分量。相同approved_v1封存renderer、色表、光照、相机、缩放与布局，仅改资产及短标题。原9井用认可青色细线显示。

fig03是预先固定Y26深度域中心剖面，XYZ不翻转、Z向上、同一离散色表、无插值。fig04及panels/target_D0是无地震更新的A先验伪井Flow D0独立机制对照；不混入主图列。所有单面板均有PNG/PDF。

fig05是同一Y26的时间域地震：观测、A先验模拟、连续阻抗更新模拟。统一振幅色标，时间轴为双程时ms、0…3192ms，与深度剖面分开；无逐图归一化或选最好位置。更新模拟仍使用A先验慢度，未估计新速度。图内没有指标，数值见主目录CSV。

网格真实spacing=3840/63米。边界距离是全部边界体素双向距离的合并均值，不能将均值小于一格解读成每处边界达到亚网格精度。holdout未硬写入但物理观测覆盖，不是独立观测泛化测试。合成逐柱法向入射、已知波子、无额外噪声、人工强目标声学对比及同算子inverse-crime，不含三维波场/绕射/偏移成像/实测验证。PDF3D部分为高清栅格，文本图例为矢量。
''',encoding='utf-8')
    print('FIGURES_COMPLETE',out,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);main(p.parse_args().out)
