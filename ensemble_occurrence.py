"""Hard-label empirical occurrence; truth-blind physics ranking and exact voxel surfaces.

Restored implementation for the supplied README/tests (the referenced file was absent).
Only numpy and Pillow are required; Flow sampling and hard forward live in the project.
"""
from __future__ import annotations
import argparse
import csv
import hashlib
import html
import json
import math
from pathlib import Path
import numpy as np


def write_json(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False)+'\n')


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(8*1024*1024), b''): h.update(block)
    return h.hexdigest()


def load_array(path, key='labels'):
    value = np.load(path, mmap_mode='r', allow_pickle=False)
    if isinstance(value, np.lib.npyio.NpzFile):
        with value: return value[key]
    return value


def normalize_labels(labels):
    a = np.asanyarray(labels)
    if a.ndim == 5 and a.shape[1] == 1: a = a[:, 0]
    if a.ndim != 4 or not np.issubdtype(a.dtype, np.integer) or any(d <= 0 for d in a.shape):
        raise ValueError('labels must be nonempty integer [N,X,Y,Z] or [N,1,X,Y,Z]')
    return a


def read_ids(path, n):
    ids = json.loads(Path(path).read_text())
    if not isinstance(ids, list) or len(ids) != n or any(not isinstance(i, str) or not i for i in ids) or len(set(ids)) != n:
        raise ValueError('IDs must be N unique nonempty strings')
    return ids


def minimum_counts(n, strict=True):
    if n <= 0: raise ValueError('N must be positive')
    return [(n*p)//100+1 if strict else (n*p+99)//100 for p in (40,60,80)] + [n]


def counts_for(labels, indices, target_label):
    count = np.zeros(labels.shape[1:], np.uint32)
    for i in indices: count += labels[int(i)] == target_label
    return count


def metrics(prediction, truth, mask=None):
    p, t = np.asarray(prediction, bool), np.asarray(truth, bool)
    if p.shape != t.shape: raise ValueError('prediction/truth shape mismatch')
    if mask is not None:
        if mask.shape != p.shape: raise ValueError('evaluation mask shape mismatch')
        p, t = p & mask, t & mask
    tp, fp, fn = int((p&t).sum()), int((p&~t).sum()), int((~p&t).sum())
    return dict(iou=tp/(tp+fp+fn) if tp+fp+fn else None,
                precision=tp/(tp+fp) if tp+fp else None,
                recall=tp/(tp+fn) if tp+fn else None,
                tp=tp, fp=fp, fn=fn, predicted_voxels=int(p.sum()), truth_voxels=int(t.sum()))


def select_indices(ids, mode, keep, scores=None, higher=False, seed=20261009):
    n = len(ids)
    if len(set(ids)) != n or not 0 < keep <= n: raise ValueError('invalid keep/IDs')
    if mode == 'all': return np.arange(n)
    if mode == 'random': return np.sort(np.random.default_rng(seed).choice(n, keep, replace=False))
    if mode not in ('physics', 'oracle-iou'): raise ValueError(mode)
    scores = np.asarray(scores, dtype=float)
    if scores.shape != (n,) or not np.isfinite(scores).all(): raise ValueError('scores must be N finite values')
    return np.argsort(-scores if higher else scores, kind='stable')[:keep]


def read_scores(path, ids, column):
    with Path(path).open(newline='') as f: rows = list(csv.DictReader(f))
    mapping = {r['sample_id']: float(r[column]) for r in rows}
    if len(mapping) != len(rows) or set(mapping) != set(ids): raise ValueError('score CSV must align exactly with IDs')
    result = np.array([mapping[i] for i in ids])
    if not np.isfinite(result).all(): raise ValueError('nonfinite observation score')
    return result


def exposed_faces(mask, spacing=(1,1,1)):
    """Return every exposed cell face, including full masks and boundary cells."""
    m = np.asarray(mask, bool)
    if m.ndim != 3: raise ValueError('surface requires XYZ')
    padded = np.pad(m, 1)
    faces = []
    for axis in range(3):
        tangents = [a for a in range(3) if a != axis]
        for side in (-1,1):
            slices = [slice(1,-1)]*3
            slices[axis] = slice(2,None) if side == 1 else slice(None,-2)
            cells = np.argwhere(m & ~padded[tuple(slices)])
            points = np.repeat(cells[:,None,:].astype(float),4,axis=1)
            points[:,:,axis] += side == 1
            for j, (u,v) in enumerate(((0,0),(1,0),(1,1),(0,1))):
                points[:,j,tangents[0]] += u; points[:,j,tangents[1]] += v
            faces.append(points*np.array(spacing))
    return np.concatenate(faces, axis=0)


def render_panels(out, panels, shape, spacing=(1,1,1), title='', front=False, z_down=False, columns=4, presentation=False):
    """Vector SVG plus PNG with the same orthographic, physical-aspect projection."""
    from PIL import Image, ImageDraw
    w, h = (360,270) if presentation else (300,300)
    cols = min(columns, len(panels)); rows = math.ceil(len(panels)/cols)
    width, height = cols*w, rows*h+62
    canvas = Image.new('RGB',(width,height),'white'); draw = ImageDraw.Draw(canvas)
    draw.text((12,10), title, fill='#222222')
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>',f'<text x="12" y="22" font-size="14">{html.escape(title)}</text>']
    # Front looks along -Y, Z increases upward by default. Isometric has equal orthogonal axis foreshortening.
    basis = np.array([[1,0,0],[0,0,-1 if not z_down else 1],[0,1,0]],float) if front else np.array([[1,-1,0],[1/math.sqrt(3),1/math.sqrt(3),-2/math.sqrt(3) if not z_down else 2/math.sqrt(3)],[1,1,1]],float)
    corners = np.array([[x,y,z] for x in (0,shape[0]) for y in (0,shape[1]) for z in (0,shape[2])])*spacing
    projected = corners@basis.T
    lo, hi = projected[:,:2].min(0), projected[:,:2].max(0)
    scale = min((w-42)/max(hi[0]-lo[0],1),(h-65)/max(hi[1]-lo[1],1))
    for index, (name, layers) in enumerate(panels):
        ox,oy = (index%cols)*w+20,(index//cols)*h+60
        draw.text((ox,oy-22),name,fill='#222222')
        svg.append(f'<text x="{ox}" y="{oy-10}" font-size="12">{html.escape(name)}</text>')
        def project(points): return (points@basis.T)[...,:2]*scale-lo*scale+np.array([ox,oy])
        xy = project(corners)
        for a in range(8):
            for b in range(a+1,8):
                if np.count_nonzero(corners[a]!=corners[b]) == 1:
                    coords = [tuple(xy[a]),tuple(xy[b])]; draw.line(coords,fill='#bbbbbb',width=1)
                    svg.append(f'<path d="M {xy[a,0]:.1f},{xy[a,1]:.1f} L {xy[b,0]:.1f},{xy[b,1]:.1f}" stroke="#bbbbbb" fill="none"/>')
        for mask,color in layers:
            faces = exposed_faces(mask,spacing)
            if not len(faces): continue
            depth = (faces.mean(1)@basis.T)[:,2]
            for face in faces[np.argsort(-depth,kind='stable')]:
                polygon = project(face); coords = [tuple(p) for p in polygon]
                draw.polygon(coords,fill=color,outline=None)
                svg.append('<polygon points="'+' '.join(f'{p[0]:.1f},{p[1]:.1f}' for p in polygon)+f'" fill="{color}"/>')
        if not any(np.any(mask) for mask,_ in layers):
            draw.text((ox+40,oy+60),'Empty region',fill='#666666')
            svg.append(f'<text x="{ox+40}" y="{oy+70}" fill="#666666">Empty region</text>')
        draw.text((ox,oy+h-57),'X / Y / Z; physical aspect',fill='#777777')
        svg.append(f'<text x="{ox}" y="{oy+h-47}" font-size="10" fill="#777777">X / Y / Z; physical aspect</text>')
    svg.append('</svg>'); out=Path(out); out.parent.mkdir(parents=True,exist_ok=True)
    canvas.save(out.with_suffix('.png')); out.with_suffix('.svg').write_text('\n'.join(svg))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    for flag in ('labels','ids','provenance','out'): p.add_argument('--'+flag,type=Path,required=True)
    for flag in ('truth','score-csv','condition-mask','condition-labels','evaluation-mask'): p.add_argument('--'+flag,type=Path)
    p.add_argument('--mode',choices=['all','random','physics','oracle-iou'],default='all')
    p.add_argument('--target-label',type=int,default=9); p.add_argument('--keep',type=int,default=100)
    p.add_argument('--expected-count',type=int,default=200); p.add_argument('--score-column',default='hard_physics_rmse')
    p.add_argument('--higher-is-better',action='store_true'); p.add_argument('--at-least',action='store_true')
    p.add_argument('--spacing',type=float,nargs=3,default=(1,1,1)); p.add_argument('--z-down',action='store_true'); p.add_argument('--skip-plots',action='store_true')
    args=p.parse_args(argv)
    if args.out.exists(): raise FileExistsError(args.out)
    labels=normalize_labels(load_array(args.labels)); n=len(labels); ids=read_ids(args.ids,n)
    if n != args.expected_count: raise ValueError(f'actual N={n}, expected {args.expected_count}')
    if not np.isfinite(args.spacing).all() or min(args.spacing)<=0: raise ValueError('positive finite spacing required')
    provenance=json.loads(args.provenance.read_text())
    if not provenance.get('case_id') or not provenance.get('method_id'): raise ValueError('case_id/method_id required')
    condition=None
    if bool(args.condition_mask) != bool(args.condition_labels): raise ValueError('condition mask and labels required together')
    if args.condition_mask:
        cm=load_array(args.condition_mask,'mask').astype(bool); cv=load_array(args.condition_labels)
        if cm.shape != labels.shape[1:] or cv.shape != cm.shape: raise ValueError('conditions shape mismatch')
        for i in range(n):
            if np.any(labels[i][cm]!=cv[cm]): raise ValueError(f'condition violation: {ids[i]}')
        condition=cm & (cv==args.target_label)
    scores=read_scores(args.score_csv,ids,args.score_column) if args.mode=='physics' else None
    # Physics selection is persisted BEFORE truth is opened.
    chosen=select_indices(ids,args.mode,args.keep,scores,args.higher_is_better) if args.mode!='oracle-iou' else None
    args.out.mkdir(parents=True)
    if chosen is not None: write_json(args.out/'inference_selected_ids.json',[ids[i] for i in chosen])
    truth=load_array(args.truth)==args.target_label if args.truth else None
    if truth is not None and truth.shape!=labels.shape[1:]: raise ValueError('truth shape mismatch')
    em=load_array(args.evaluation_mask,'mask').astype(bool) if args.evaluation_mask else None
    if args.mode=='oracle-iou':
        if truth is None: raise ValueError('oracle requires truth')
        scores=np.array([metrics(a==args.target_label,truth)['iou'] for a in labels],dtype=float)
        chosen=select_indices(ids,args.mode,args.keep,scores,True)
    groups={'all':np.arange(n),'selected':chosen,'random_control':select_indices(ids,'random',args.keep)}
    audit=dict(actual_count=n,provenance=provenance,mode=args.mode,strict=not args.at_least,
               undefined_metric_convention='null; empty union or denominator is undefined',
               frequency_interpretation='empirical occurrence; uncalibrated',spacing=list(args.spacing),groups={},
               input_hashes={str(path):file_hash(path) for path in (args.labels,args.ids,args.provenance,args.truth,args.score_csv,args.condition_mask,args.condition_labels,args.evaluation_mask) if path})
    for name,indices in groups.items():
        folder=args.out/name; folder.mkdir(); count=counts_for(labels,indices,args.target_label); cuts=minimum_counts(len(indices),not args.at_least)
        masks=np.array([count>=c for c in cuts]); write_json(folder/'selected_ids.json',[ids[i] for i in indices])
        np.savez_compressed(folder/'occurrence.npz',count=count,frequency=count/len(indices),threshold_masks=masks,minimum_counts=cuts)
        audit['groups'][name]=dict(n=len(indices),minimum_counts=cuts,threshold_metrics=[metrics(m,truth,em) if truth is not None else dict(predicted_voxels=int(m.sum())) for m in masks])
        if not args.skip_plots:
            overlay=[] if condition is None else [(condition,'#c83232')]
            left=[] if truth is None else [('Truth (retrospective)',[(truth,'#e7aa32')]+overlay)]
            member_indices=indices[np.unique(np.linspace(0,len(indices)-1,min(12,len(indices)),dtype=int))]
            write_json(folder/'figure8_ids.json',[ids[i] for i in member_indices])
            members=left+[(ids[i],[(labels[i]==args.target_label,'#e7aa32')]+overlay) for i in member_indices]
            thresholds=left+[(f'C >= {c} / N={len(indices)}',[(m,'#e7aa32')]+overlay) for m,c in zip(masks,cuts)]
            for front in (False,True):
                suffix='front' if front else 'iso'
                for fig,panels in [('figure8',members),('figure9',thresholds)]:
                    render_panels(folder/f'{fig}_{suffix}',panels,labels.shape[1:],args.spacing,f'{provenance["method_id"]} / {name} / N={len(indices)}; empirical occurrence',front,args.z_down)
    rows=[]
    for i in range(n):
        row=dict(sample_id=ids[i],selected=int(i in chosen),score=None if scores is None else float(scores[i]))
        if truth is not None: row.update(metrics(labels[i]==args.target_label,truth,em))
        rows.append(row)
    with (args.out/'candidate_audit.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    write_json(args.out/'audit.json',audit)
    return audit


if __name__=='__main__': main()
