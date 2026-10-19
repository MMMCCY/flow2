"""Only two fixed-Y26 explanatory figures; saved arrays, no inversion or Flow."""
import sys,os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']='';sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage26_diagnostics.common import *
from scripts.paper_figures.style import LABEL_COLORS,RESIDUAL_CMAP
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap,BoundaryNorm
from matplotlib.patches import Patch

def geometry_axes(ax,truth):
    ax.contour(np.arange(64),np.arange(64),(truth[:,26,:]==9).T,levels=[.5],colors='#252525',linewidths=.65)
    ax.set_xlim(-.5,63.5);ax.set_ylim(-.5,63.5);ax.set_aspect('equal');ax.set_xticks([0,16,32,48,63]);ax.set_yticks([0,16,32,48,63]);ax.set_xlabel('X index')

def main(out=None):
    guard('evaluate');torch.set_num_threads(4)
    e=read_json(ROOT/'EVALUATION.json');f=read_json(ROOT/'INPUT_FREEZE.json');assert e['status']=='completed'
    out=Path(out).resolve() if out else ROOT/'figures'
    if out.exists():raise FileExistsError('Choose a new figure directory')
    out.mkdir(parents=True)
    truth=verify_tensor(read_json(STAGE25/'evaluation_registry.json')['truth']).numpy()[0,0]
    cmap=ListedColormap([LABEL_COLORS[i] for i in range(-1,14)]);norm=BoundaryNorm(np.arange(-1.5,14,1),15)
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'pdf.fonttype':42,'ps.fonttype':42})
    fig,axes=plt.subplots(3,2,figsize=(7.4,9.2),sharex=True,sharey=True)
    names={'prior_exact':'Saved prior','updated':'Saved update','oracle_updated':'ORACLE update'}
    for row,field in enumerate(FIELDS):
        for col,candidate in enumerate(CANDIDATE_NAMES):
            key=f'{field}_{candidate}';v=verify_tensor(e['classification_assets'][key]).numpy()[0,0];ax=axes[row,col]
            ax.imshow(v[:,26,:].T,origin='lower',interpolation='nearest',cmap=cmap,norm=norm,extent=(-.5,63.5,-.5,63.5))
            geometry_axes(ax,truth);ax.set_title(names[field]+' / '+candidate,fontsize=10)
            if col==0:ax.set_ylabel('Z index (upward)')
    fig.legend(handles=[Patch(facecolor=LABEL_COLORS[i],label=str(i)) for i in range(14)],loc='center right',bbox_to_anchor=(1,.51),frameon=False,fontsize=8)
    fig.subplots_adjust(left=.08,right=.91,top=.96,bottom=.06,hspace=.26,wspace=.15)
    for ext in ('png','pdf'):fig.savefig(out/f'fig01_classification_Y26.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    table,_=setup();sub=verify_tensor(f['subsurface']).numpy()[0,0]
    true_log=table[0,torch.from_numpy(truth).long()+1].log().numpy()
    a=verify_tensor(f['old_assets']['prior_exact']).numpy()[0]
    u=verify_tensor(f['old_assets']['updated']).numpy()[0]
    oracle=verify_tensor(read_json(ROOT/'ORACLE_STATE.json')['assets']['updated']).numpy()[0]
    prior_error=np.abs(np.log(a[0])-true_log)
    delta_u=np.abs(np.log(u[0])-true_log)-prior_error;delta_o=np.abs(np.log(oracle[0])-true_log)-prior_error
    twt_a=verify_tensor(e['twt_assets']['A_twt_xyz_ms']).numpy()[0,0];twt_t=verify_tensor(e['twt_assets']['truth_twt_xyz_ms']).numpy()[0,0]
    values=[(twt_a-twt_t)[:,26,:].T,delta_u[:,26,:].T,delta_o[:,26,:].T]
    support=sub[:,26,:].T
    lim_t=max(float(np.abs(values[0][support]).max()),1e-12);lim_e=max(float(np.abs(v[support]).max()) for v in values[1:])
    fig,axes=plt.subplots(1,3,figsize=(11.2,4.5),sharex=True,sharey=True)
    titles=['A TWT - true TWT','Saved update: error change','ORACLE update: error change']
    for i,(ax,v,title) in enumerate(zip(axes,values,titles)):
        lim=lim_t if i==0 else lim_e
        image=ax.imshow(np.ma.masked_where(~support,v),origin='lower',interpolation='nearest',extent=(-.5,63.5,-.5,63.5),cmap=RESIDUAL_CMAP,vmin=-lim,vmax=lim)
        geometry_axes(ax,truth);ax.set_title(title,fontsize=10)
        cb=fig.colorbar(image,ax=ax,orientation='horizontal',fraction=.05,pad=.22);cb.set_label('ms' if i==0 else 'Delta absolute log-impedance error')
    axes[0].set_ylabel('Z index (upward)');fig.subplots_adjust(left=.06,right=.99,top=.90,bottom=.25,wspace=.12)
    for ext in ('png','pdf'):fig.savefig(out/f'fig02_twt_property_errors_Y26.{ext}',dpi=450,facecolor='white')
    plt.close(fig)
    write_json(out/'RENDER_CONFIG.json',dict(section_Y=26,axes='original XYZ, Z upward; TWT top-down output flipped back before plotting',panels_fig01='3 fields x 2 fixed candidate sets',fig01_palette='existing LABEL_COLORS, all raw-1..13',fig02_limits=dict(twt_ms=[-lim_t,lim_t],shared_log_error_change=[-lim_e,lim_e]),truth_target_outline='evaluation reference only, never input to classifier or initial impedance',geology_edits=False,new_3D_figures=False,total_figures=2,visual_inspection='pending',plot_source_sha256=file_hash(Path(__file__))))
    (out/'CAPTIONS.md').write_text('''# 图外说明

图1：固定Y26六个离线分类剖面。行分别为保存prior_exact、固定A慢度的保存updated、额外一次正确慢度ORACLE updated；列分别为完整raw0…13与从原9井/地表提取的OBSERVED候选集。相同log阻抗距离/first argmin、原条件优先、空气定义及离散色表。黑线为真值raw9轮廓，仅作事后可视参照，未输入分类器。没有Flow预测或标签修补。ORACLE行不能当作可部署方法。

图2：同一Y26。左为同深度网格点A与真值的中心TWT之差（ms），并非已匹配地震反射事件；中/右为保存更新或ORACLE更新相对同一旧prior的绝对log阻抗误差变化，蓝色负值表示误差减少、红色正值表示变差，两幅共用相同色标。已观测地表以上不显示。TWT函数返回top-down数组，先flip回原XYZ再绘图，Z向上；右图的真值阻抗只用于计算误差，不进入oracle求解。

ORACLE仅替换固定慢度，影响先验地震到时和时间修正向深度映射，携带真实地下结构信息；原A阻抗起点逐体素不变。所有其他输入和单次求解配置不变。数值见图外CSV及REPORT.md。分类候选OBSERVED在现实中可能漏掉未见类别，不能据单案例推广。没有改变正式Stage26结果。
''',encoding='utf-8')
    print('TWO_EXPLANATORY_FIGURES_COMPLETE',out,flush=True)
if __name__=='__main__':
    import argparse
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path);main(p.parse_args().out)
