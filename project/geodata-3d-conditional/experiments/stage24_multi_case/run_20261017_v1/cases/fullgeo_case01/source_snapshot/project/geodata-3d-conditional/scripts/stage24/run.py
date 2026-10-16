"""Thin adapter to the repaired Stage23 runner; one canonical prediction per new case."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.common import *
from scripts.stage23 import run as engine

def run(case):
    if case not in CASES:raise ValueError(case)
    root=ROOT/'cases'/case
    seal=read_json(root/'CONDITION_RECEIPT.json')
    assert seal['case']==case and seal['status']=='completed' and seal['converged']
    assert read_json(root/'PROTOCOL.json')['case']==case
    # Only source identity and firewall are adapted. Model/noise/sampler are unchanged.
    engine.guard=guard
    engine.source_hashes=sources
    engine.run((root/'inference_registry.json').resolve(),42,(root/'predictions/seed42').resolve())

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=CASES,required=True)
    run(p.parse_args().case)
