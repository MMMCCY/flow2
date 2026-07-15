import sys
sys.path.insert(0, "/home/mcy/Geoflow/flowtrain_stochastic_interpolation-main/src")
sys.path.insert(0, "/home/mcy/Geoflow/StructuralGeo-main/src")

import torch
from utils import plot_2d_slices

x = torch.load("/home/mcy/Geoflow/repro_samples/structuralgeo_64/geology_64_0.pt")
plot_2d_slices(x.squeeze(0), save_path="/home/mcy/Geoflow/repro_samples/structuralgeo_64/geology_64_0_slices.png")
print("saved slice plot")