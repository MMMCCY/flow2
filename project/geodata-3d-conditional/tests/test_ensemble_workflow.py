"""Scientific integration contracts: chunk offsets, geometry, ranking and counts."""
import gzip
import importlib.util
from pathlib import Path
import numpy as np
import pytest
import torch

REPO=Path(__file__).resolve().parents[3]
spec=importlib.util.spec_from_file_location('ensemble_workflow',REPO/'project/geodata-3d-conditional/scripts/ensemble_workflow.py')
workflow=importlib.util.module_from_spec(spec); spec.loader.exec_module(workflow)
occ=workflow.occurrence


def test_chunk_offset_is_mandatory_and_raw_label9_is_preserved(tmp_path):
    chunk=torch.zeros((2,1,3,4,5),dtype=torch.long); chunk[1,0,1,2,3]=9
    path=tmp_path/'chunk.pt.gz'
    with gzip.open(path,'wb') as f: torch.save(chunk,f)
    with pytest.raises(ValueError,match='offset'): workflow.load_hard(path)
    a=workflow.load_hard(path,1)
    assert a.shape==(3,4,5) and a[1,2,3]==9 and a.sum()==9
    with pytest.raises(ValueError): workflow.load_hard(path,2)


def test_exposed_cell_surfaces_full_boundary_and_spacing():
    faces=occ.exposed_faces(np.ones((2,3,4),bool),(100,100,50))
    assert faces.shape==(2*(2*3+2*4+3*4),4,3)
    np.testing.assert_array_equal(faces.min((0,1)),[0,0,0])
    np.testing.assert_array_equal(faces.max((0,1)),[200,300,200])
    assert len(occ.exposed_faces(np.zeros((2,3,4),bool)))==0


def test_200_subsets_exact_counts_and_stable_ties():
    ids=[str(i) for i in range(200)]
    selected=occ.select_indices(ids,'physics',100,np.ones(200))
    np.testing.assert_array_equal(selected,np.arange(100))
    random=occ.select_indices(ids,'random',100)
    assert len(np.unique(random))==100
    np.testing.assert_array_equal(random,occ.select_indices(ids,'random',100))
    labels=np.zeros((200,2,2,2),dtype=np.int8); labels[:81,0,0,0]=9; labels[:121,1,0,0]=9; labels[:,1,1,1]=9
    counts=occ.counts_for(labels,np.arange(200),9)
    assert counts[0,0,0]==81 and counts[1,0,0]==121 and counts[1,1,1]==200
    masks=[counts>=cut for cut in occ.minimum_counts(200)]
    assert all(np.all(~b|a) for a,b in zip(masks,masks[1:]))
    top_count=occ.counts_for(labels,selected,9)
    assert top_count[0,0,0]==81 and top_count[1,0,0]==100


def test_truth_free_score_api_and_conditions():
    import inspect
    assert 'truth' not in inspect.signature(occ.select_indices).parameters
    empty=np.zeros((2,2,2),bool)
    result=workflow.summarize_mask(empty,empty,{'free':np.ones_like(empty)},[100,100,50])
    assert result['free']['iou'] is None and result['free']['connected_components_6']==0


def test_reject_fractional_models(tmp_path):
    path=tmp_path/'wrong.pt'; torch.save(torch.ones((3,4,5))*0.5,path)
    with pytest.raises(ValueError): workflow.load_hard(path)
