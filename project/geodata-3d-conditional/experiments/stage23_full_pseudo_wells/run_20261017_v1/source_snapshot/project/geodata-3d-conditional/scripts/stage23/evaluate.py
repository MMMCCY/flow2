"""Retrospective evaluation only after the sole Stage23 Flow has been sealed."""
import argparse
import os
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.stage23.common import *
from scripts.stage21.evaluate import voxel_metrics, connected, match_objects, ratio
import numpy as np
import torch


def class_metrics(pred, truth, domain, classes):
    rows = []
    for label in classes:
        p, t = (pred == label) & domain, (truth == label) & domain
        tp, pn, tn = int((p&t).sum()), int(p.sum()), int(t.sum())
        rows.append(dict(raw_label=label, tp=tp, fp=pn-tp, fn=tn-tp,
            predicted_volume=pn, truth_volume=tn, precision=ratio(tp,pn),
            recall=ratio(tp,tn), iou=ratio(tp,pn+tn-tp)))
    return rows


def conversion(pred, truth, added):
    rows = class_metrics(pred, truth, added, list(range(-1,14)))
    target = next(r for r in rows if r['raw_label']==9)
    count = int(added.sum())
    errors = int(((pred!=truth)&added).sum())
    target_errors = int(((pred!=truth)&added&(truth==9)).sum())
    summary = dict(voxels=count, errors=errors, accuracy=ratio(count-errors,count),
        truth_target_errors=target_errors, truth_non_target_errors=errors-target_errors,
        truth_air_voxels=int(((truth==-1)&added).sum()),
        target_tp=target['tp'], target_fp=target['fp'], target_fn=target['fn'],
        target_precision=target['precision'], target_recall=target['recall'], target_iou=target['iou'])
    # All raw classes, including truth air; row=true, column=predicted.
    codes = (truth[added].astype(np.int64)+1)*15 + pred[added].astype(np.int64)+1
    matrix = np.bincount(codes, minlength=225).reshape(15,15)
    assert int(matrix.sum()) == count and count-int(np.trace(matrix)) == errors
    assert target['tp']+target['fp'] == int(((pred==9)&added).sum())
    assert target['tp']+target['fn'] == int(((truth==9)&added).sum())
    return summary, rows, matrix


def sealed_prediction(folder):
    r = read_json(folder/'receipt.json')
    assert r['status']=='completed'
    assert read_json(folder/'completion_receipt.json')['receipt_sha256'] == file_hash(folder/'receipt.json')
    return r, verify_tensor(r['prediction'])


def evaluate(root):
    guard('evaluate')
    if (root/'EXECUTION.json').exists():
        raise FileExistsError('Completed evaluation is immutable')
    torch.set_num_threads(4)
    torch.set_grad_enabled(False)
    freeze = read_json(root/'METHOD_FREEZE.json')
    seal = read_json(root/'CONDITION_RECEIPT.json')
    r = read_json(root/'inference_registry.json')
    assert freeze['source_sha256']==source_hashes()
    assert seal['freeze_file_sha256']==file_hash(root/'METHOD_FREEZE.json')
    assert seal['protocol_sha256']==json_hash(read_json(root/'PROTOCOL.json'))
    assert seal['inference_registry_sha256']==file_hash(root/'inference_registry.json')
    assert seal['condition_record_sha256']==json_hash(r['conditions'])
    newreceipt, new = sealed_prediction(root/'predictions/seed42')
    assert newreceipt['velocity_evaluations']==32
    assert newreceipt['condition_record_sha256']==seal['condition_record_sha256']
    assert newreceipt['initial_noise_tensor_sha256']==r['noise']['42']['asset']['tensor_sha256']
    assert newreceipt['base_hash_before']==newreceipt['base_hash_after']==r['noise']['42']['expected_model_hash']
    conditions = check_condition(r['conditions'])
    assert not ((new!=conditions['values'])&conditions['mask']).any()
    added_t = verify_tensor(seal['added_mask'])
    assert torch.equal(added_t, conditions['mask']&~conditions['original_mask'])
    # First truth access happens here, AFTER both conditions and one new prediction are sealed.
    started = utc()
    e = read_json(OLD/'evaluation_registry.json')['cases'][CASE]
    truth_t = verify_tensor(e['truth'])
    truth = truth_t.numpy()[0,0]
    added = added_t.numpy()[0,0]
    domains = {name:(verify_tensor(a['tensor']).numpy()[0,0],a['classes']) for name,a in e['domains'].items()}
    assert int(((truth==9)&domains['E_original_free'][0]).sum())==1733
    assert int(((truth==9)&domains['E_holdout'][0]).sum())==103
    ti, nt = connected(truth==9)
    assert nt==2 and int((ti==1).sum())==572
    assert not ((ti==1)&conditions['original_mask'].numpy()[0,0]).any()
    common = (ti==1)&domains['E_original_free'][0]&~conditions['mask'].numpy()[0,0]
    samples, allmetrics, objects, predicted, classes_rows, common_rows = [],[],[],[],[],[]
    predictions = {}
    def measure(arm, pred, asset, receipt_identity=None):
        assert pred.shape==truth_t.shape and pred.min()>=-1 and pred.max()<=13
        v = pred.numpy()[0,0]
        pi,_ = connected(v==9)
        sample = dict(arm=arm, prediction=asset, receipt=receipt_identity, metrics={}, objects={})
        predictions[arm] = v
        common_tp = int(((v==9)&common).sum())
        common_rows.append(dict(arm=arm, subset='truthID1 & E_original_free & ~Stage23_new_mask',
            voxels=int(common.sum()), tp=common_tp, coverage=ratio(common_tp,int(common.sum()))))
        for name,(domain,cs) in domains.items():
            m = voxel_metrics(v,truth,domain,cs)
            assert m['target_tp']+m['target_fp']==m['target_pred_volume']
            assert m['target_tp']+m['target_fn']==m['target_truth_volume']
            oo,pp,stats,_ = match_objects(ti,pi,domain,RULES)
            m.update(stats)
            m['target_pred_truth_volume_ratio'] = ratio(m['target_pred_volume'],m['target_truth_volume'])
            m['full_target_pred_volume'] = int((v==9).sum())
            m['full_target_truth_volume'] = int((truth==9).sum())
            sample['metrics'][name], sample['objects'][name] = m, oo
            allmetrics.append(dict(arm=arm, domain=name, **m))
            classes_rows.extend(dict(arm=arm, domain=name, **row) for row in class_metrics(v,truth,domain,cs))
            objects.extend(dict(arm=arm,domain=name,full_truth_voxels=int((ti==o['object_id']).sum()),**o) for o in oo)
            predicted.extend(dict(arm=arm,domain=name,**p) for p in pp)
        samples.append(sample)
    def receipt_id(folder):
        return dict(path=str((folder/'receipt.json').relative_to(REPO)),file_sha256=file_hash(folder/'receipt.json'))
    for arm in ('A_9','B_512','C_512'):
        folder = OLD/'predictions'/CASE/'seed42'/arm
        old,pred = sealed_prediction(folder)
        assert old['initial_noise_tensor_sha256']==r['noise']['42']['asset']['tensor_sha256']
        assert old['base_hash_after']==r['noise']['42']['expected_model_hash']
        measure(arm,pred,old['prediction'],receipt_id(folder))
    folder = RESTORED/'predictions/seed42'
    old,pred = sealed_prediction(folder)
    assert old['initial_noise_tensor_sha256']==r['noise']['42']['asset']['tensor_sha256']
    assert old['base_hash_after']==r['noise']['42']['expected_model_hash']
    measure('FLOW_STAGE22',pred,old['prediction'],receipt_id(folder))
    old_exec = read_json(RESTORED/'EXECUTION.json')
    direct = next(s for s in old_exec['samples'] if s['arm']=='RESTORED_CLASSIFICATION')['prediction']
    measure('RESTORED_CLASSIFICATION',verify_tensor(direct),direct)
    base = torch.from_numpy(predictions['A_9']).reshape_as(truth_t)
    copy = torch.where(conditions['mask'],conditions['values'].to(base.dtype),base)
    held = torch.from_numpy(domains['E_holdout'][0]).reshape_as(truth_t)
    assert torch.equal(copy[held],base[held])
    measure('COPY_ONLY_STAGE23',copy,save_tensor(root/'offline/copy_only.pt',copy))
    measure('FLOW_STAGE23',new,newreceipt['prediction'],receipt_id(root/'predictions/seed42'))
    oldreg = read_json(OLD/'inference_registry.json')
    oldc = check_condition(oldreg['conditions'][CASE]['C_512'])
    assert torch.equal(oldc['mask'],conditions['mask'])
    conversion_summaries, conversion_classes, confusion = [],[],{}
    for arm,labels in (('C_512',oldc['values']),('STAGE23_WELLS',conditions['values'])):
        summary, cls, matrix = conversion(labels.numpy()[0,0],truth,added)
        conversion_summaries.append(dict(arm=arm,domain='added_mask',**summary))
        conversion_classes.extend(dict(arm=arm,domain='added_mask',**c) for c in cls)
        confusion[arm] = matrix.tolist()
        write_csv(root/f'confusion_{arm}.csv', [dict(truth_raw_label=i-1,
            **{f'pred_{j-1}':int(matrix[i,j]) for j in range(15)}) for i in range(15)])
    by = {s['arm']:s for s in samples}
    n, c = by['FLOW_STAGE23']['metrics']['E_holdout'], by['C_512']['metrics']['E_holdout']
    off = {a['arm']:a for a in common_rows}
    checks = dict(holdout_recall_improves=n['target_recall']>c['target_recall'],
        holdout_iou_improves=n['target_iou']>c['target_iou'],
        holdout_miou_maintained=n['fixed_class_miou']>=c['fixed_class_miou'],
        object1_outside_tp_above_A=off['FLOW_STAGE23']['tp']>off['A_9']['tp'],
        object1_outside_tp_above_COPY=off['FLOW_STAGE23']['tp']>off['COPY_ONLY_STAGE23']['tp'])
    result = dict(status='completed', completed_at_utc=utc(), evaluation_started_at_utc=started,
        condition_sealed_at_utc=seal['sealed_at_utc'], samples=samples,
        conversion=conversion_summaries, object1_common_subset=common_rows,
        modeling_checks=checks, modeling_all_criteria=all(checks.values()),
        new_Flow_outputs=1, velocity_evaluations=32, no_retries=True,
        copy_holdout_equal_A=True, source_sha256=source_hashes(), stop_reason='single_prediction_budget_complete')
    for name,rows in (('sample_metrics',allmetrics),('class_metrics',classes_rows),('object_metrics',objects),
        ('predicted_object_metrics',predicted),('object1_common_outside',common_rows),
        ('condition_quality',conversion_summaries),('condition_class_metrics',conversion_classes)):
        write_csv(root/f'{name}.csv',rows)
    write_json(root/'EVALUATION_INPUTS.json',dict(truth=e['truth'],truth_role='retrospective only',
        domains=e['domains'],evaluation_registry_sha256=file_hash(OLD/'evaluation_registry.json'),
        common_object1_subset=save_tensor(root/'offline/object1_common_outside.pt',torch.from_numpy(common)),
        started_at_utc=started, prediction_completion_sha256=file_hash(root/'predictions/seed42/completion_receipt.json')))
    write_json(root/'EXECUTION.json',result)
    from scripts.stage23.report import report
    report(root,result,predictions,truth)
    assert source_hashes()==freeze['source_sha256']
    print('Retrospective evaluation complete; stop at one Flow output / 32 calls',flush=True)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root',type=Path,default=ROOT)
    evaluate(parser.parse_args().root.resolve())
