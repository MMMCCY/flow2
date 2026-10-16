import sys
from pathlib import Path

import numpy as np
import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.stage21v2 import common as c
from guidance.property_volume import property_table_from_config, hard_labels_to_properties


def test_nested_layout_and_holdout():
    plan = c.layout()
    assert plan == c.layout()
    held = set(map(tuple, plan["holdout"]))
    assert len(held) == 256
    assert len({(x//4,y//4) for x,y in held}) == 256
    extras = set(map(tuple, plan["extra_order"]))
    assert len(extras) == 3831
    assert not held & extras and not held & set(c.W9)
    assert not extras & set(c.W9)
    prior = set(c.W9)
    for n in c.LEVELS:
        points = set(map(tuple, plan["levels"][str(n)]))
        assert len(points) == n and prior < points and not points & held
        prior = points
    assert held | extras | set(c.W9) == {(x,y) for x in range(64) for y in range(64)}


def test_conditions_original_priority_and_copy_holdout():
    plan = c.layout()
    shape = (1,1,64,64,64)
    truth = torch.full(shape, 9.)
    pseudo = torch.full(shape, 2)
    om = c.well_mask(c.W9)
    om[...,63] = True
    ov = torch.full(shape,-1.)
    ov[om] = 4
    a = torch.ones(shape, dtype=torch.long)
    held = c.well_mask(plan["holdout"]) & ~om
    for n in c.LEVELS:
        bv,bm = c.dense_conditions(ov,om,truth,plan["levels"][str(n)])
        cv,cm = c.dense_conditions(ov,om,pseudo,plan["levels"][str(n)])
        assert torch.equal(bm,cm) and not (cm & held).any()
        assert (cv[om] == 4).all() and (bv[om] == 4).all()
        assert (cv[cm & ~om] == 2).all() and (bv[bm & ~om] == 9).all()
        assert (cv[~cm] == -1).all() and (bv[~bm] == -1).all()
        d = c.copy_only(a,cv,cm)
        assert torch.equal(d[held],a[held])


def test_classifier_roundtrip_all_classes_and_ties():
    table, weights, _ = property_table_from_config(c.read_json(c.CODEBOOK),15)
    truth = torch.arange(14).reshape(1,1,14,1,1)
    properties = hard_labels_to_properties(truth,table)
    assert torch.equal(c.classify(properties,table,weights),truth)
    # Exact collision chooses the smaller raw label, regardless of current truth frequencies.
    collision = table.clone()
    collision[:,2] = collision[:,1]
    result = c.classify(hard_labels_to_properties(torch.ones(1,1,1,1,1),collision),collision,weights)
    assert result.item() == 0
    # Distances must scale using NONAIR range (air is deliberately an extreme value).
    manual = table.clone()
    manual[:,0] = -1000
    assert torch.equal(c.classify(properties,manual,weights),truth)


def test_fixed_domains_and_classes():
    plan = c.layout()
    truth = torch.zeros(1,1,64,64,64)
    truth[...,0] = -1
    truth[...,10:20] = 9
    om = c.well_mask(c.W9)
    domains = c.evaluation_domains(truth,om,plan["holdout"])
    hashes = {k:c.tensor_hash(v) for k,v in domains.items()}
    counts = {k:torch.unique(truth[v]).tolist() for k,v in domains.items()}
    for n in c.LEVELS:
        c.dense_conditions(truth,om,truth,plan["levels"][str(n)])
        again = c.evaluation_domains(truth,om,plan["holdout"])
        assert hashes == {k:c.tensor_hash(v) for k,v in again.items()}
        assert counts == {k:torch.unique(truth[v]).tolist() for k,v in again.items()}
    assert counts == {"E_holdout":[0.,9.], "E_original_free":[0.,9.]}


def test_inference_rejects_evaluation_fields():
    r = dict(schema="stage21v2_inference_v1",protocol_sha256="h",source_sha256={},
             checkpoint=c.CHECKPOINT,checkpoint_sha256=c.CHECKPOINT_SHA,conditions={})
    c.validate_inference(r)
    for name in ("truth","pseudo","roi","evaluation"):
        with pytest.raises(ValueError):
            c.validate_inference({**r,name:"hidden.pt"})


def test_classifier_uses_normalized_config_weights():
    # Independent scalar arithmetic detects channel-weight/range mistakes.
    table, weights, _ = property_table_from_config(c.read_json(c.CODEBOOK),15)
    prop = torch.tensor([.31,.043]).reshape(1,2,1,1,1)
    scores=[]
    for label in range(14):
        score=0.
        for ch in range(2):
            vals=[float(table[ch,k+1]) for k in range(14)]
            score += float(weights[ch]) * ((float(prop[0,ch,0,0,0])-vals[label])/(max(vals)-min(vals)))**2
        scores.append(score)
    assert c.classify(prop,table,weights).item() == int(np.argmin(scores))
