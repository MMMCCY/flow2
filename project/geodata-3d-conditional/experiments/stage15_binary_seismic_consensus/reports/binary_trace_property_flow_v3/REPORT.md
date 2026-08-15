# Stage15-H — Full-trace binary boundary inversion and property-guided Flow

- Inversion voxel AUPRC: 0.512386623
- Boundary AUPRC: 0.580512428
- XY footprint AUPRC: 1.000000000
- Fixed 0.5 core precision / recall / IoU: 0.949953 / 0.112177 / 0.111518
- Flow pairs improving IoU/precision/recall/global mIoU: 3/3 / 3/3 / 3/3 / 3/3
- Figure seed 242 IoU: 0.067156 -> 0.294259
- Figure seed precision: 0.183057 -> 0.413749
- Figure seed recall: 0.095897 -> 0.504683
- Figure seed centroid distance: 9.063101 -> 4.273142

The binary trace inversion uses complete 320-sample traces and never splits the vertical forward model. The normalized binary property endpoint is label9=1 and every other class=0; the continuous inversion score supplies confidence without a threshold.
