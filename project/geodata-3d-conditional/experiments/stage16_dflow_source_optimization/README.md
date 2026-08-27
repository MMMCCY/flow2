# Stage16: D-Flow source optimization

This stage is an inference-only mechanism test. It compares a projected
32-step fixed-Euler `FLOW_ONLY` sample with D-Flow optimization initialized
from the exact same Gaussian source tensor. The CFM checkpoint, EMA weights,
embedding and network parameters remain frozen.

The three frozen configs reuse, respectively:

- the authoritative Phase1 probability target, target core, ROI and calibrated
  probability loss at endpoint `tau=0.1`;
- the authoritative Phase2A two-channel property table, full target property
  volume, confidence, scales and endpoint `tau=0.1`;
- the Stage15 frozen observation, binary acoustic mapping and convolutional
  seismic operator with raw seismic MSE. The seismic runner asset API contains
  no truth geology.

The optimizer setting is pre-registered once in each config: Adam, learning
rate 0.01, 20 iterations, no source-shell regularization. It is not selected
by a sweep and must not be changed after inspecting outcomes.

One-seed smoke commands are documented in `reports/DEVELOPMENT_REPORT.md`.
Generated runs belong in the objective subdirectories and must always use a
new empty output directory.
