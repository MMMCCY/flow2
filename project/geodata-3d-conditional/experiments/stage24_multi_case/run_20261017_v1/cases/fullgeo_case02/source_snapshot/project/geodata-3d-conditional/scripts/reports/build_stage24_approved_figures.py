"""Independent reports entry for the frozen approved_v1 Stage24 plates."""
import argparse
import sys
from pathlib import Path
sys.dont_write_bytecode=True
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from scripts.stage24.plot import plot
from scripts.stage24.common import CASES
if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--case',choices=(*CASES,'fullgeo_case03'),required=True)
    plot(p.parse_args().case)
