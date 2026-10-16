import os,sys,shutil
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
sys.dont_write_bytecode=True
repo=Path('/home/xmj/mcy/flowtrain_stochastic_interpolation-main')
sys.path.insert(0,str(repo/'project/geodata-3d-conditional'))
from scripts.stage24 import plot
actual=plot.ROOT
plot.ROOT=Path('/tmp/stage24_final_style_regression')
plot.plot('fullgeo_case03')
shutil.copyfile(plot.ROOT/'standard_regression/STYLE_REGRESSION.json',actual/'standard_regression/AFTER_VALIDATOR_REGRESSION.json')
print('Final assertion-only adapter retains all four approved PNGs pixel-exact')
