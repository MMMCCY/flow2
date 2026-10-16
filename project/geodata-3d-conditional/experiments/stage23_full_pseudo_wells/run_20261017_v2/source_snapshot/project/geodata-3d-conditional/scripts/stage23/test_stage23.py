"""Focused regression for full-column semantics, original priority and all-label accounting."""
import sys
import unittest
import subprocess
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import numpy as np
import torch
from scripts.stage23.common import dense_conditions, classify
from scripts.stage23.evaluate import conversion, class_metrics


class Stage23Tests(unittest.TestCase):
    def test_firewall_allows_dependencies_and_blocks_experiment_geometry(self):
        code = '''
from scripts.stage23.common import guard, PROJECT
guard('flow')
import sympy.geometry
from model_train_sh_inference_cond import Geo3DStochInterp
for path in (PROJECT/'experiments/example/geometry/hidden.pt',
             PROJECT/'experiments/example/truth/true_model.pt',
             PROJECT/'experiments/example/preprocessing/restored_classification.pt'):
    try:
        path.open('rb')
    except PermissionError:
        continue
    raise AssertionError('Firewall did not block forbidden asset')
'''
        completed=subprocess.run([sys.executable,'-c',code],cwd=Path(__file__).resolve().parents[2],capture_output=True,text=True)
        self.assertEqual(completed.returncode,0,completed.stderr)

    def test_full_multilithology_columns_and_original_priority(self):
        labels=torch.arange(14).reshape(1,1,1,1,14).expand(1,1,3,3,14).clone()
        values=torch.full_like(labels,-1)
        mask=torch.zeros_like(labels,dtype=torch.bool)
        mask[0,0,1,1,3]=True  # Original hard air overrides predicted lithology3.
        mask[0,0,0,0,5]=True
        values[0,0,0,0,5]=7
        v,m=dense_conditions(values,mask,labels,[(1,1)])
        self.assertTrue(m[0,0,1,1].all())
        self.assertEqual(v[0,0,1,1].tolist(),[0,1,2,-1,4,5,6,7,8,9,10,11,12,13])
        self.assertEqual(v[0,0,0,0,5].item(),7)
        self.assertTrue((v[~m]==-1).all())
        self.assertFalse(m[0,0,2,2].any())

    def test_classifier_fixed_nonair_and_first_tie(self):
        table=torch.stack((torch.arange(15),torch.arange(15))).float()
        table[:,2]=table[:,1]
        props=table[:,1].reshape(1,2,1,1,1)
        self.assertEqual(classify(props,table,torch.tensor([.5,.5])).item(),0)

    def test_confusion_retains_air_and_target_arithmetic(self):
        truth=np.array([-1,0,9,9,2,3])
        pred=np.array([0,0,9,2,9,3])
        s,rows,matrix=conversion(pred,truth,np.ones(6,dtype=bool))
        self.assertEqual((s['target_tp'],s['target_fp'],s['target_fn']),(1,1,1))
        self.assertEqual((s['errors'],s['truth_target_errors'],s['truth_non_target_errors']),(3,1,2))
        self.assertEqual(matrix[0,1],1)
        self.assertEqual(s['accuracy'],.5)
        self.assertEqual(s['target_iou'],1/3)
        air=next(r for r in rows if r['raw_label']==-1)
        self.assertIsNone(air['precision'])
        self.assertEqual(air['recall'],0)


if __name__=='__main__':
    unittest.main()
