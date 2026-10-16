#!/usr/bin/env python3
"""Independent audit of actual ensemble identities, source pairing, ranking and counts."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image


def read(path): return json.loads(Path(path).read_text())


def verify(run,report):
    protocol=read(run/'protocol.json'); state=read(run/'sampling_state.json'); context=read(report/'plot_context.json')
    sources=protocol['source_ids']; ids=context['ids']; n=len(ids)
    assert len(sources)==200 and len({x['sample_id'] for x in sources})==200
    assert len({x['source_seed'] for x in sources})==200
    assert ids==[x['sample_id'] for x in sources], 'requires complete frozen 200 protocol'
    assert state['status']=='completed' and state['complete_source_count']==200 and not state['failed_ids']
    assert state['base_model_unchanged'] and state['independent_baseline_equivalence']
    arms=['FLOW_ONLY']+protocol['methods']; assert len(state['records'])==len(arms)*200
    records={(x['sample_id'],x['arm']):x for x in state['records']}
    assert len(records)==len(state['records'])
    assert len({r['decoded_path'] for r in state['records']})==len(state['records'])
    assert len({records[(sid,'FLOW_ONLY')]['initial_noise_sha256'] for sid in ids})==200
    for sid in ids:
        rows=[records[(sid,arm)] for arm in arms]
        assert len({x['initial_noise_sha256'] for x in rows})==1
        assert all(x['condition_violations']==0 for x in rows)
        for row in rows:
            for rel,expected in [(row['decoded_path'],row['decoded_file_sha256']),(str(Path(row['decoded_path']).parent/'hard_predicted_seismic.pt'),row['hard_prediction_file_sha256'])]:
                assert hashlib.sha256((run/rel).read_bytes()).hexdigest()==expected
    truth=np.load(report/'truth_target.npy'); cm=np.load(report/'condition_mask.npy'); cv=np.load(report/'condition_labels.npy')
    audits=[]
    for arm in arms:
        a=np.load(report/f'{arm}_labels.npy',mmap_mode='r')
        assert a.shape==(200,64,64,64) and np.issubdtype(a.dtype,np.integer)
        assert a.min()>=-1 and a.max()<=13
        assert np.all(a[:,cm]==cv[cm])
        scores=np.array([records[(sid,arm)]['hard_seismic_rmse'] for sid in ids]); assert np.isfinite(scores).all()
        observed=read(report/f'{arm}_observation_selected_ids.json')
        assert observed['obs_top100']==[ids[i] for i in np.lexsort((np.arange(200),scores))[:100]]
        assert read(report/'inference_selection_complete.json')['truth_loaded'] is False
        binary=a==9
        union=np.logical_or(binary,truth).sum((1,2,3)); intersection=np.logical_and(binary,truth).sum((1,2,3))
        iou=intersection/union
        oracle=[ids[i] for i in np.lexsort((np.arange(200),-iou))[:100]]
        for name,idx in context['groups'][arm].items():
            folder=report/arm/name; chosen=read(folder/'selected_ids.json'); idx=np.asarray(idx)
            assert len(chosen)==len(set(chosen))==len(idx)
            assert chosen==[ids[i] for i in idx]
            assert len(chosen)==(200 if name=='all200' else 100)
            if name=='oracle_top100': assert chosen==oracle
            if name=='random100': assert idx.tolist()==sorted(np.random.default_rng(20261009).choice(200,100,replace=False).tolist())
            if name=='obs_top100': assert chosen==observed['obs_top100']
            data=np.load(folder/'occurrence.npz'); count=data['count']; masks=data['threshold_masks']; cuts=data['minimum_counts']
            manual=np.sum(binary[idx],axis=0,dtype=np.uint32)
            np.testing.assert_array_equal(count,manual)
            np.testing.assert_allclose(data['frequency'],count/len(idx),rtol=0,atol=0)
            assert cuts.tolist()==([81,121,161,200] if len(idx)==200 else [41,61,81,100])
            np.testing.assert_array_equal(masks,count[None]>=cuts[:,None,None,None])
            assert all(np.all(~b|a) for a,b in zip(masks,masks[1:]))
            assert np.all(masks[-1]==np.all(binary[idx],axis=0))
            members=read(folder/'figure8_ids.json')
            assert len(members)==12 and len(set(members))==12 and set(members)<=set(chosen)
            audits.append(dict(arm=arm,group=name,count=len(idx),thresholds=cuts.tolist(),full_all_hit_voxels=int(masks[-1].sum()),free_all_hit_voxels=int((masks[-1]&~cm).sum())))
    pngs=list(report.rglob('*.png'))
    for path in pngs:
        with Image.open(path) as img: img.verify()
    result=dict(status='PASS',planned_count=200,actual_count=200,models=len(records),source_pairing=True,conditions_exact=True,all_counts_exact=True,strict_thresholds=True,all_subsets_unique=True,obs_ranking_exact=True,oracle_ranking_exact=True,group_checks=audits,validated_png_files=len(pngs),note='Automated artifact audit; manual PNG inspection reported separately.')
    (report/'VALIDATION_REPORT.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({key:value for key,value in result.items() if key!='group_checks'},indent=2))

if __name__=='__main__':
    parser=argparse.ArgumentParser(); parser.add_argument('--run',type=Path,required=True); parser.add_argument('--report',type=Path,required=True)
    args=parser.parse_args(); verify(args.run,args.report)
