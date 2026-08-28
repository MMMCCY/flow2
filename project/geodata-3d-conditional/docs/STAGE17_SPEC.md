# Stage17 frozen specification: evidence generalization and coupling attribution

Status: **frozen before Stage17 asset generation and scientific runs**  
Frozen date: 2026-08-27

## Scope

Stage17 separates two questions:

1. whether the frozen Stage15-H full-trace binary inversion produces
   case-specific label-9 spatial evidence across independent StructuralGeo
   cases, without running Flow;
2. whether, given identical evidence, authoritative trajectory guidance or
   D-Flow source optimization converts it into correct hard geology.

The checkpoint, raw frozen embedding, 411-entry EMA policy, 3-D U-Net,
32-step midpoint fixed-Euler solver, hard-condition semantics and decoder are
immutable. No training, parameter sweep, truth-selected seed/case, PGDM, DPS,
PnP, SGLD, topology loss or multiclass joint inversion is authorized.

## Frozen case cohort

Stage17A uses all five already-frozen `full_complexity_targeted_v1` Full
StructuralGeo benchmark cases, in registry order. No case is selected or
rejected using seismic, evidence, Flow or retrospective performance. The
cohort is independently generated and prospectively registered, but historical
sample-level exclusion from the original streaming checkpoint training cannot
be certified.

## Mandatory two-stage observation freeze

Synthetic observation construction and evidence inversion are separate
freezes:

1. `observation_generation_v1.json` is frozen first. It contains the ordered
   case list, exact truth/condition hashes, binary acoustic config, seismic
   config and generation semantics.
2. The observation builder may then load truth only to generate the immutable
   synthetic observations. It writes output tensor hashes and a completed
   manifest into a new empty directory.
3. Only after those observations exist and are hashed may
   `evidence_multicase_v1.json` be created and frozen. The inversion config
   contains exact observation/input hashes and never exposes truth to the
   inversion runner.

An inversion run whose config predates or does not hash the completed
observation assets is invalid.

The full-complexity generator leaves 23 isolated air-coded cells below the
highest non-air cell in case04. The seismic operator requires one contiguous
column below each local surface. Observation config v2 therefore freezes the
truth-independent operator-support rule
`columnwise_fill_below_highest_nonair_v1`: in every XY column, seismic support
is every cell at or below the highest non-air cell. Any enclosed air-coded
cell is represented by the binary background endpoint and remains excluded
from truth scoring/Flow hard-label semantics. Failed observation v1 is
preserved and is not scientific evidence.

## Stage17A: prevalence-corrected case specificity

The Stage15-H definition is unchanged: complete 320-sample traces, the frozen
binary acoustic endpoints, two linearized refinement passes, prior relative
weight `0.001`, vertical smoothness relative weight `0.01`, no lateral
filtering, and continuous `binary_impedance_score` without thresholding.

Per-case metrics include raw voxel AUPRC, prevalence, `AUPRC-prevalence`, score
mean/median for target and background, boundary AUPRC, XY-footprint AUPRC and
the fixed-0.5 core diagnostic.

The primary cross-case comparison domain is one fixed mask: the intersection
of every registered case's unconditioned subsurface mask. This prevents
case-dependent air surfaces or borehole masks from becoming the specificity
signal. Per-truth unconditioned-subsurface matrices are retained as secondary
diagnostics.

For evidence `q_i` and truth `m_j`, first compute raw average precision

`AP_ij = AP(q_i, m_j)`

and the truth-column prevalence `p_j` on the same primary mask. The primary
specificity matrix is prevalence-corrected AP skill:

`K_ij = (AP_ij - p_j) / (1 - p_j)`.

Raw `AP_ij` is mandatory auxiliary output, not the main specificity statistic.
The primary diagonal advantage is

`Delta_i = K_ii - median_{j != i}(K_ij)`.

The frozen decision language is:

- `EVIDENCE_CASE_SPECIFIC`: at least 4/5 `Delta_i > 0` and group median
  `Delta_i > 0`;
- `EVIDENCE_INFORMATIVE_BUT_NOT_CASE_SPECIFIC`: evidence exceeds prevalence
  within cases but the AP-skill diagonal advantage is unstable;
- `EVIDENCE_NOT_SUPPORTED`: most cases do not exceed prevalence or AP-skill
  specificity does not hold.

No high absolute AUPRC threshold is added. If the first decision is not
reached, Stage17C is stopped. Stage17B oracle/property calibration remains
independent and may continue.

## Engineering smoke versus calibration confirmation

Engineering smoke and Stage17B optimizer calibration confirmation are
different run classes and different evidence:

- `engineering_smoke` selects only the first preregistered source seed while
  retaining every optimizer arm and all scientific parameters. It verifies
  CUDA execution, memory, gradients, solver behavior, output completeness,
  hashes and exact conditions.
- `optimizer_calibration_confirmation` uses every preregistered source seed and
  is the only Stage17B run class allowed to support optimizer progression or
  configuration selection.

Smoke results must not tune or change optimizer, learning rate, iteration
count, source regularization, solver, loss, decode or gate thresholds. A
scientific-semantic code change after smoke requires a new config version, a
new output directory and a new smoke; old output remains immutable and is
marked superseded, never overwritten.

## Stage17B arms and optimizer configurations

For each Phase1 authoritative oracle-probability and Phase2A authoritative
ideal-property task, every source seed generates from the identical initial
noise:

- `FLOW_ONLY`;
- `REFERENCE_TRAJECTORY_GUIDANCE`, calling the original authoritative sampler
  with its original parameters;
- `DFLOW_LEGACY_ADAM`: Adam, lr `0.01`, 20 source steps;
- `DFLOW_PAPER_ALIGNED_LBFGS`: LBFGS, lr `1.0`, 5 outer steps, max 5 inner
  iterations and strong-Wolfe line search.

Both D-Flow arms use standard Gaussian initialization, the projected 32-step
fixed-Euler map, gradient checkpointing and zero source regularization. The
LBFGS setting is a paper-aligned project choice based on the paper's
standard-Gaussian QM9 implementation; it is not claimed to be a universal
paper hyperparameter for categorical 64-cubed geology.

Each arm reports outer steps, closure evaluations, endpoint solves, runtime and
peak CUDA memory. Objective reduction, nonzero source updates, immutable model
hashes and exact hard conditions establish only mechanism activity.

## Stage17B task-separated progression gate

Progression is evaluated separately for oracle and property. Pooled
aggregation cannot compensate for failure of either task.

For each task and optimizer, a seed has material hard control when target IoU
improves by at least `0.01` and at least one of the following holds: target
recall improves by `0.02`, centroid distance falls by `0.5` voxel, or absolute
volume-error fraction falls by `0.02`. At least 2/3 fixed seeds must satisfy
that rule in **each** task.

Reference recovery fractions are evaluated only where the reference gain is
non-micro: `0.01` for IoU/recall/volume-error fraction and `0.5` voxel for
centroid-distance reduction. For each optimizer, median target-IoU recovery
fraction must be at least `0.25` separately in oracle and property.

`DFLOW_HARD_CONTROL_MATERIAL` requires all of the above independently in both
tasks, objective/source mechanism activity, immutable model state and zero
condition violations. If both optimizers qualify, select the optimizer with
the larger minimum of its two task-specific median IoU recovery fractions;
ties use fewer endpoint solves, then lower runtime. If neither qualifies,
record `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL` and do not run the D-Flow
geophysical arm in Stage17C.

## Stage17C case-level inference

Stage17C is authorized only after `EVIDENCE_CASE_SPECIFIC`. Its trajectory arm
uses the exact Stage17A evidence tensor. Its D-Flow arm additionally requires
`DFLOW_HARD_CONTROL_MATERIAL` and uses the optimizer selected by the frozen
Stage17B rule. No evidence reconstruction or case-specific normalization is
allowed.

Each case uses source seeds `42`, `142`, and `242`. These are stochastic
replicates nested within the geology case. The five independent geology cases
are the primary statistical units: first compute fixed-seed paired deltas,
then a case summary, and finally group medians and positive-case fractions over
the five case summaries. The pooled 15-pair distribution is mandatory but
auxiliary; it cannot replace or override the case-level conclusion.

All coupling arms share checkpoint, condition tensors, evidence tensor hash,
initial-noise tensor hash, solver, time grid, steps, decode and evaluator.
Truth is unavailable to the runner and enters only after completed manifests
and hashes are validated by the retrospective evaluator.

## Stop rules

- Stage17A not case-specific: do not run Stage17C.
- Stage17B weak in either oracle or property: do not run Stage17C D-Flow.
- Stage17C completion: report evidence, coupling, reachability and topology
  interpretation, then stop.
- Never automatically enter later methods or training.
