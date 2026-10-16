"""CPU, truth-blind reuse of restored labels; seal complete pseudo-wells before Flow."""
import argparse
import os
import shutil
import sys
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
sys.dont_write_bytecode = True
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.stage23.common import *
from guidance.property_volume import property_table_from_config
import torch


def prepare(out):
    guard('prepare')
    if out.exists():
        raise FileExistsError(out)
    torch.set_num_threads(4)
    assets = read_json(RESTORED/'preprocessing_assets.json')
    seal = read_json(RESTORED/'PREPROCESSING_RECEIPT.json')
    r22 = read_json(RESTORED/'inference_registry.json')
    freeze22 = read_json(RESTORED/'METHOD_FREEZE.json')
    assert seal['status'] == 'completed' and seal['converged']
    assert seal['assets'] == assets
    assert seal['freeze_file_sha256'] == file_hash(RESTORED/'METHOD_FREEZE.json')
    assert seal['protocol_sha256'] == json_hash(read_json(RESTORED/'PROTOCOL.json'))
    assert seal['inference_registry_sha256'] == file_hash(RESTORED/'inference_registry.json')
    assert seal['condition_record_sha256'] == json_hash(r22['conditions'])
    # No solver import or invocation. One consistency recomputation of the old classifier.
    labels = verify_tensor(assets['classification'])
    properties = verify_tensor(assets['properties'])
    config = read_json(OLD/'codebook.json')
    classifier = read_json(OLD/'classifier.json')
    table, weights, _ = property_table_from_config(config, 15)
    assert file_hash(OLD/'codebook.json') == freeze22['codebook_sha256']
    assert file_hash(RESTORED/'codebook.json') == file_hash(OLD/'codebook.json')
    assert weights.tolist() == classifier['weights']
    assert (table[:,1:].max(1).values-table[:,1:].min(1).values).tolist() == classifier['scales']
    assert properties.dtype == torch.float32 and properties.shape == (1,2,64,64,64)
    assert labels.dtype == torch.int64 and labels.shape == (1,1,64,64,64)
    assert labels.min() >= 0 and labels.max() <= 13
    assert torch.equal(classify(properties, table, weights), labels)
    oldreg = read_json(OLD/'inference_registry.json')
    arms = oldreg['conditions'][CASE]
    original = check_condition(arms['A_9'])
    layout = read_json(OLD/'layout.json')
    values, mask = dense_conditions(original['values'], original['mask'], labels, layout['levels']['512'])
    added = mask & ~original['mask']
    for arm in ('B_512', 'C_512'):
        # Only compare masks, never oracle labels during preparation.
        assert torch.equal(mask, verify_tensor(arms[arm]['mask']))
    assert not (added & well_mask(layout['holdout'])).any()
    assert torch.equal(values[added], labels[added].to(values.dtype))
    assert torch.equal(values[original['mask']], original['values'][original['mask']])
    assert (values[~mask] == -1).all()
    assert mask[well_mask(layout['levels']['512'])].all()
    paired = r22['noise']['42']
    assert file_hash(REPO/paired['A_receipt_path']) == paired['A_receipt_sha256']
    noise = verify_tensor(paired['asset'])
    a = read_json(REPO/paired['A_receipt_path'])
    assert tensor_hash(noise) == a['initial_noise_tensor_sha256']
    assert paired['expected_model_hash'] == a['base_hash_after']
    # Identity checks for all historically frozen sources, without modifying their glob sets.
    historical_sources = {**oldreg['source_sha256'], **freeze22['source_sha256']}
    assert all(file_hash(REPO/p) == h for p,h in historical_sources.items())
    out.mkdir(parents=True)
    p = protocol()
    sources = source_hashes()
    write_json(out/'PROTOCOL.json', p)
    for relative in sources:
        destination = out/'source_snapshot'/relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO/relative, destination)
    shutil.copyfile(REPO/'STAGE23_FULL_PSEUDO_WELLS_PROMPT_2026-10-17.md', out/'source_snapshot/TASK_SPEC.md')
    inputs = {str(path.relative_to(REPO)): file_hash(path) for path in (
        OLD/'PROTOCOL.json', OLD/'layout.json', OLD/'classifier.json', OLD/'codebook.json',
        OLD/'inference_registry.json', RESTORED/'PROTOCOL.json', RESTORED/'METHOD_FREEZE.json',
        RESTORED/'PREPROCESSING_RECEIPT.json', RESTORED/'preprocessing_assets.json', RESTORED/'inference_registry.json')}
    write_json(out/'METHOD_FREEZE.json', dict(protocol_sha256=json_hash(p), source_sha256=sources,
        historical_source_sha256=historical_sources, input_file_sha256=inputs,
        reused_assets={k:assets[k] for k in ('classification','properties')},
        original_conditions=arms['A_9'], frozen_at_utc=utc(), git=git_snapshot()))
    records = {name:save_tensor(out/'conditions'/f'{name}.pt', tensor) for name,tensor in (
        ('values',values), ('mask',mask), ('original_values',original['values']), ('original_mask',original['mask']))}
    check_condition(records)
    added_asset = save_tensor(out/'conditions/added_mask.pt', added)
    registry = dict(schema='stage23_conditions_only_v1', protocol_sha256=json_hash(p),
        source_sha256=sources, checkpoint=CHECKPOINT, checkpoint_sha256=CHECKPOINT_SHA,
        conditions=records, noise={'42':paired})
    write_json(out/'inference_registry.json', registry)
    write_json(out/'CONDITION_RECEIPT.json', dict(status='completed', sealed_at_utc=utc(),
        freeze_file_sha256=file_hash(out/'METHOD_FREEZE.json'), protocol_sha256=json_hash(p),
        inference_registry_sha256=file_hash(out/'inference_registry.json'),
        condition_record_sha256=json_hash(records), added_mask=added_asset,
        added_voxels=int(added.sum()), added_columns=int(added.any(-1).sum()), total_well_layout=512,
        added_label_counts={str(i):int((values[added]==i).sum()) for i in range(14)},
        classification_recompute_equal=True, masks_equal_B_C=True, holdout_additions=0,
        truth_access=False, checkpoint_loaded=False, deconvolution_runs=0, scientific_gate=False))
    assert sources == source_hashes()
    print(f'Sealed {int(added.sum())} added voxels in {int(added.any(-1).sum())} columns; no scientific gate', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--out', type=Path, default=ROOT)
    prepare(parser.parse_args().out.resolve())
