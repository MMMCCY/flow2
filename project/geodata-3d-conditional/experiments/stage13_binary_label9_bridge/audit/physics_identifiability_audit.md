# Stage 13 binary label-9 physics-identifiability audit

## Decision

`PHYSICS_BINARY_BRIDGE_REQUIRES_LEARNED_MAPPER`

The current registered seismic physics does not identify a voxelwise,
target-specific label-9 likelihood from the acoustic codebook alone. Stage13-A
therefore stops before bridge construction and retrospective truth evaluation.
No heuristic anomaly map is substituted for a probability model.

## Registered forward and the missing information

The current operator is a normal-incidence, post-stack convolutional upper
bound. For each independent lateral trace it:

1. maps hard lithology to acoustic impedance and slowness;
2. computes adjacent vertical reflection coefficients
   `r=(Z_below-Z_above)/(Z_below+Z_above)`;
3. places each coefficient at a cumulative, slowness-dependent two-way time;
4. convolves the spikes with a fixed zero-phase 25 Hz Ricker wavelet.

This representation is sensitive to interfaces, not to the class identity of a
homogeneous voxel interior. A label-9 body can create top and bottom reflections,
but their amplitude and polarity depend jointly on which nuisance/background
lithologies touch it, their ordering, body thickness, tuning with the wavelet,
and the velocity-dependent time-depth map. Interfaces between non-label9 rocks
can produce the same local signatures. Absolute impedance and the low-frequency
component are not observed by this band-limited reflectivity experiment.

The registered acoustic codebook makes label 9 acoustically distinctive
(`density=3000 kg/m^3`, `Vp=6500 m/s`), but it only specifies class-conditional
point values. It does not specify the distributions of surrounding lithology,
geometry, thickness, orientation, or time-depth uncertainty required for either
`p(feature|label9)` or the nuisance mixture `p(feature|non-label9)`.

## Candidate feature audit

| Candidate | Physics audit |
|---|---|
| Local signed/absolute amplitude | A reflection is an interface property. Sign reverses with vertical ordering, and large amplitude is not unique to label 9. |
| Reflection coefficient / impedance contrast | Requires deconvolution and time-depth localization; even an exact contrast identifies a pair of impedances, not which adjacent voxel is label 9. |
| Envelope / local energy | Removes polarity but becomes still less class-specific; any strong non-label9 contrast can create high energy. |
| Vertical/local seismic context | Top/base pairs can encode a body pattern, but their distribution depends on thickness, surrounding rocks, interference, geometry, and velocity. Estimating this is pattern recognition, not a codebook-only likelihood. |
| Multiscale response | Changes representation and resolution but does not restore the missing absolute-impedance/DC component or interior support. |
| Inverted impedance anomaly | The Phase5a inversion supplies its missing low frequency and time-depth mapping from geological prior members. Converting that scalar volume to classes is the Stage12B route explicitly excluded here. |

Consequently, the rule `high absolute seismic amplitude = label9` has no
physical justification and is not implemented.

## Why independent calibration would be a learned mapper

A calibration simulator could draw full geological contexts, forward model
them, and estimate a target-specific local likelihood or train an encoder. That
would learn the missing joint distribution of waveform context and target
occupancy. It is scientifically plausible, but it is a learned pattern mapper
whose simulation distribution, features, split, calibration, and validation
must be prospectively authorized. It is not a fixed consequence of the current
forward operator and codebook, so it is outside this Stage13-A physics branch.

## Exact repository paths reviewed

Frozen benchmark and decisions:

- `project/geodata-3d-conditional/docs/FULL_STRUCTURALGEO_BENCHMARK_AUDIT.md`
- `project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/FULL_STRUCTURALGEO_BENCHMARK_BUILD_REPORT.md`
- `project/geodata-3d-conditional/experiments/full_structuralgeo_benchmark/FULL_STRUCTURALGEO_BENCHMARK_DECISION.json`
- `project/geodata-3d-conditional/experiments/stage10_geophysical_probability_bridge/reports/STAGE10_MACHINE_DECISION.json`
- `project/geodata-3d-conditional/experiments/stage10_geophysical_probability_bridge/diagnostic_addendum/STAGE10R_DIAGNOSTIC_REPORT.md`
- `project/geodata-3d-conditional/experiments/stage12b_fullgeo_probability_bridge/reports/STAGE12B_REPORT.md`
- `project/geodata-3d-conditional/experiments/stage12b_fullgeo_probability_bridge/reports/STAGE12B_A_MACHINE_DECISION.json`
- `project/geodata-3d-conditional/experiments/stage12b_fullgeo_probability_bridge/evaluation/stage12b_a/`

Phase1 probability construction, guidance, and evaluation:

- `project/geodata-3d-conditional/guidance/probability_volume.py`
- `project/geodata-3d-conditional/guidance/probability_sampling.py`
- `project/geodata-3d-conditional/guidance/probability_evaluation.py`
- `project/geodata-3d-conditional/scripts/stage1/run_probability_guidance.py`
- `project/geodata-3d-conditional/docs/PHASE1_REPORT.md`

Seismic forward, inversion, and acoustic/petrophysical bridge:

- `project/geodata-3d-conditional/guidance/seismic.py`
- `project/geodata-3d-conditional/experiments/stage4_seismic/configs/acoustic_distinct_label9_upper_bound_v1.json`
- `project/geodata-3d-conditional/guidance/seismic_inversion.py`
- `project/geodata-3d-conditional/scripts/stage5/build_acoustic_inversion_posterior.py`
- `project/geodata-3d-conditional/docs/PHASE5A_SPEC.md`
- `project/geodata-3d-conditional/docs/PHASE5A_REPORT.md`
- `project/geodata-3d-conditional/guidance/geophysical_probability_bridge.py`
- `project/geodata-3d-conditional/scripts/stage12b/build_synthetic_observations.py`
- `project/geodata-3d-conditional/scripts/stage12b/build_probability_bridges.py`

## Frozen boundary preserved

The Stage10 FAIL and Stage10R interpretation are unchanged. Stage12A remains
`FULL_STRUCTURALGEO_BENCHMARK_READY`; its five cases and truth are unchanged.
Stage12B remains `STOP_FULLGEO_BRIDGE_NOT_CASE_SPECIFIC`. No Stage13 Flow,
inversion, bridge, calibration simulation, probability-volume construction, or
truth evaluation was run.
