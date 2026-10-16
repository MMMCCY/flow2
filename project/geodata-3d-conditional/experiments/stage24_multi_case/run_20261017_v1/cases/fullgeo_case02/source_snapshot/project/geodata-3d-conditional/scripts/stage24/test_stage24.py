"""Necessary case binding, full-column, classifier and firewall checks; no solves/forwards."""
import sys
from pathlib import Path
import unittest
import subprocess
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *
from scripts.stage24.run import run
import torch

class Stage24Tests(unittest.TestCase):
    def test_case_binding_and_actual_noise(self):
        for case in CASES:
            n=paired_noise(case)
            self.assertEqual(read_json(REPO/n['A_receipt_path'])['case_id'],case)
            self.assertEqual(verify_tensor(n['asset']).dtype,torch.float32)
        with self.assertRaises(ValueError):run('fullgeo_case03')

    def test_full_multilithology_original_priority(self):
        labels=torch.arange(14).reshape(1,1,1,1,14).expand(1,1,2,2,14).clone()
        values=torch.full_like(labels,-1);mask=torch.zeros_like(labels,dtype=torch.bool)
        mask[0,0,1,1,3]=True
        v,m=dense_conditions(values,mask,labels,[(1,1)])
        self.assertTrue(m[0,0,1,1].all());self.assertEqual(v[0,0,1,1,3].item(),-1)
        self.assertEqual(v[0,0,1,1,9].item(),9);self.assertTrue((v[~m]==-1).all())

    def test_fixed_classifier_tie(self):
        table=torch.stack((torch.arange(15),torch.arange(15))).float();table[:,2]=table[:,1]
        self.assertEqual(classify(table[:,1].reshape(1,2,1,1,1),table,torch.tensor([.5,.5])).item(),0)

    def test_firewall_dependency_import_and_truth_block(self):
        code='''
from scripts.stage24.common import guard,PROJECT
guard('prepare')
import sympy.geometry
from model_train_sh_inference_cond import Geo3DStochInterp
for p in (PROJECT/'experiments/example/truth/true_model.pt',PROJECT/'experiments/example/geometry/ids.pt'):
    try:p.open('rb')
    except PermissionError:continue
    raise AssertionError('Truth firewall failed')
'''
        r=subprocess.run([sys.executable,'-c',code],cwd=PROJECT,capture_output=True,text=True)
        self.assertEqual(r.returncode,0,r.stderr)

if __name__=='__main__':unittest.main()
