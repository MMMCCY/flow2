# Development handoff

## Stage25 — native laccolith shape experiment (2026-10-19)

The user requested read-only review before development over SSH LabRS; review
confirmed Stage21–24 reuse and the native endpoint grid spacing3840/63 m.
One native Laccolith (raw9 cap+downward stem), seed20261019, was frozen with
simple raw0/1/3/5 background, flat observed surfaceZ52, center(35,26), cap
baseZ28/height8/diameter22 cells and stem diameter4.4 cells. Exactly one W9
well intersects the target; truth has2217 voxels and one connected component.

Authoritative result:
`experiments/stage25_laccolith_shape/run_20261019_v3/FINAL_REPORT.md`.
A/B/C used the same fresh valid15-channel CPU seed42 source, same411EMA model,
32 midpoint-Euler steps and hard projection:3 completed outputs/96 velocity
calls, zero condition violations. Exactly one Stage22 bounded-TV solve
converged at609 iterations and was reused without rerunning. Full-target
IoU A/B/C=0.117973/0.792091/0.786812; ASSD=517.608946/22.754509/23.177768 m;
volume ratios=1.368065/0.982860/0.992332. C added-well quality is99.9541%,
12 non-target errors, targetTP/FP/FN247/0/0. Common target subset outside new
wells has1933 voxels: A/COPY446TP, B1659TP, C1661TP. Full restored direct
classification targetIoU=1.0, so compression into wells/frozen generation
still loses shape information. This is one noiseless known-kernel surrogate
case, not evidence of field validity or generalization.

Approved_v1 full geology/whole-target plates, fixedY26 sections, B/direct
panels, PNG/PDF, captions, CSV and receipts are saved.16 main focused tests
and23 existing seismic tests passed, plus a firewall read/write probe.
Two initial main-engineering failures consumed0 velocity calls: JSON byte
identity after formatting, and incorrect20-channel source inferred from the
class default instead of actual checkpoint15. v1/v2 retain those logs and
frozen inputs; v3 reused inputs/restoration and corrected only invalid noise
channels, without seed selection or geometry/solver reruns.

Optional seismic started only after MANDATORY_COMPLETE. One existing
single-member inversion returned, but its new firewall incorrectly blocked
its own first property-output write; updated fields were lost on process
exit. Stop rules were respected: no second inversion, noD. Synthetic seismic
observation/config/failed receipt/log/executed snapshots remain. The firewall
write exception was fixed and probed without rerunning numerical inversion.
Actual totals remain3Flow/96 calls,1Gaussian solve,1physical inversion attempt,
0 saved updated physical fields. No physical scientific conclusion is claimed.
See `physics/REPORT.md` and `OPTIONAL_PHYSICS.json` for exact limitations.

New code: `scripts/stage25/` and `scripts/stage25_physics/`; no old experiment,
model, Geo core or approved rendering standard was altered. Do not rerun these
protected directories or automatically add geometry/seed/parameter scans.
A same-observation/two-cap-shape experiment or optional-physics retry requires
separate research authorization/budget. No GitHub push was made.

## Stage21 step 1 — accurate extra full wells (2026-10-14)

The user explicitly authorized only step 1, including all 45 formal outputs,
after a read-only SSH/source/asset review of the supplied Stage21 prompt.
The complete route remains: accurate wells -> geophysical full-sequence
inference/error/uncertainty -> native Flow virtual-well validation. Steps 2/3,
new physics, erroneous-well experiments, adapters and new training were not run.

Development and execution used SSH `labrs-codex` on the RTX4090D with the
repository `.venv`. HEAD is `b780b9ae3650b426400d3258b8e33f10a349bf6c`;
existing user modifications and Stage1–20 outputs are preserved. All five
historical full StructuralGeo cases, manifest/tensor file and canonical content
hashes, original nine wells and original surface were verified. The checkpoint
matches SHA-256 `561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`.

New implementation: `scripts/stage21/{common,prepare,run,evaluate,report,audit}.py`
and focused tests. The truth-independent PCG64 seed21001 4x4 layout fixes
W9 < W16 < W25 across all cases. Offline preparation samples only declared
oracle wells and preserves the registered surface; outside-mask values are -1.
Separate inference/evaluation registries and a runtime open-file firewall keep
truth, geometry and evaluation metadata out of inference. The existing EMA
loader and pure projected fixed-Euler sampler are reused without modifying
training, architecture, embedding, sampler or older stages. All 411 trainable
entries use EMA before freezing; the embedding retains its checkpoint value.

Validation: 29 focused/legacy sampler/frozen-trajectory tests passed. Real
CUDA smoke is case03/seed142/N9,N16,N25, plus an independent 32-step native N9
continuous-state equivalence regression. Smoke is separate from formal data.
Formal inference completed five cases x seeds42/142/242 x three levels =45
outputs/1440 velocity evaluations. FP32/no TF32, deterministic algorithms,
32 midpoint-time Euler steps and CPU-generated pre-projection noise are fixed.
Every step is finite and condition-exact; all model state entries are unchanged
(aggregate before/after `10d2181fe3eeeb406afc3da5e78ded64e9f92abfb2b3c352ee57e60e06da9b68`).
Completed-run `--resume` independently verifies all45 assets without additional
model calls or changing the sealed ledger.

Authoritative root:
`experiments/stage21_virtual_borehole_conditioning/run_20261014_v2/`.
Formal predictions are in `formal/`; current tables, reports, all5 fixed-seed142
two-camera target figures and unoccluded well maps are in `reports_v3/`.
`reports_v1` retains a report-manifest relative/absolute-path packaging failure;
`reports_v2` passed the independent audit but retains a well-map legend overlap.
Both use identical predictions/metrics. `run_20261014_v1` retains the earlier
pre-sampling guard false positive on the SymPy geometry source (zero model
calls); a regression corrected experiment-geometry versus library-source
handling. Nothing was overwritten to conceal these engineering events.

All three levels share `E=(truth!=-1)&~M25`. The case-first cross-case median
label9 IoU is N9/N16/N25 `0.028397/0.049379/0.042132`; fixed truth-class mIoU
is `0.187515/0.218250/0.249923`. The distinct median of case-level median
same-seed paired IoU differences for25-9 is `+0.012432`, positive in5/5 cases.
The nonmonotonic dose-curve median and positive paired25-16 summary are different
statistics, both explicitly preserved rather than conflated. Absolute target
recovery remains weak and volume/recall do not uniformly improve.

Seven original M9-unobserved major26-connected target components occur in
case02/03/05; only one (case05 component3) is newly intersected, by8 voxels.
The other six remain unhit at25 wells. Original hidden-major matched IoU remains
very small; the25-9 cross-case median paired difference is0 (one positive,
one negative, one zero case). Under the predeclared *diagnostic*, not universal
success thresholds, none of the7 objects x3 realizations is detected in any
level. This supports bounded off-well categorical benefit, not reliable hidden
body recovery or successful geophysical virtual wells. Connected components are
object proxies; this reused cohort is not untouched TEST or field validation,
and pretraining overlap cannot be excluded.

Full/source/condition/noise/prediction hashes, all sample/case/object tables,
one-to-one full-ID/common-domain overlap matrices, denominators, original/new
exposure groups, paired differences and N/A handling are retained. See the
experiment README for exact commands, `reports_v3/REPORT.md` for findings and
`DELIVERY_AUDIT.json` for independent verification. The next research decision
is whether current layout information and low absolute hidden recovery justify
a separately preregistered information/layout diagnosis or a sequence bridge.
Do not automatically launch a new layout, physics method, error sweep or training.

## Selected-case 200 ensemble sampling, screening and figures (2026-10-09)

The user authorized Stage15 trace v4 and Stage19R on the same case, selected
retrospectively by the existing adapter three-seed median target IoU.
`stage19_test_case004` ranks first (historical median 0.6710347944). This is a
selected-case illustration, not an unbiased cross-case or generalization test.
Stage15's unchanged v4 positive-endpoint/confidence mapping was transferred to
the Stage19R TEST conditions, observation and continuous evidence. Stage19R
uses its immutable formal epoch4 trained adapter; no training or sweep occurred.

SSH GPU inference completed 200 independent CPU-generated source draws with
seeds `42 + 100*i`, i=0..199. Each source has one shared FLOW_ONLY baseline,
one STAGE15_TRACE_V4 output and one STAGE19R_ADAPTER output: 600 complete hard
geologies, zero failures and zero condition violations. The first three
baseline/adapter outputs exactly replay historical formal results; an
independent property-alpha-zero baseline matches adapter-scale-zero. Base
parameter hashes remain unchanged. Same-source paired median target-IoU
increments are +0.020074 (Stage15) and +0.294359 (Stage19R); hard-seismic RMSE
increments are +0.004201 and -0.019122 respectively. Thus Stage15's modest
geological gain does not imply improved physics on this case.

Each arm has all200/random100/obs_top100/oracle_top100 with independently
recomputed counts, strict >40/60/80% and all-hit masks, full/free/condition
metrics, connectivity, all-candidate distributions and prefix50/100/200
stability. Observation IDs are persisted before evaluation truth is opened;
oracle uses full true label9 IoU and is explicitly retrospective. Ranking,
source pairing, unique IDs, all counts and all 203 PNGs pass artifact audit.
Publication PNG/SVG, fixed 12-member panels, two views, paper and 1920x1080
layouts, independent captions, reports and hashes are delivered under
`experiments/ensemble_200_selected_case/run_v2/publication_v1/`.

The complete report is `DEVELOPMENT_REPORT.md`, the visual index is
`FIGURE_INDEX.md`, and numerical checks are in `VALIDATION_REPORT.json`.
`MANUAL_VISUAL_CHECK.md` records opened PNGs and the projection/thickness
limitation. The raw sampler saves each hard model, hard seismic prediction,
score, noise/model hash and per-step trace. The supplied reference
`ensemble_occurrence.py` was absent; it was reconstructed against the README
and six supplied passing tests. New integration/Stage19 regressions pass 36
pytest tests, and the publication-renderer test passes once (43 checks total,
13 existing Matplotlib warnings). Historical experiments and existing user
report/visualization changes are preserved; no push or publication occurred.

Entry points: `scripts/ensemble_workflow.py` (audit/sample/resume/postprocess/
plot-only), `scripts/render_ensemble_publication.py` (publication postprocess/
plot-only), `scripts/verify_ensemble_run.py`, and `scripts/finalize_ensemble_report.py`.
Commands and asset paths are in `experiments/ensemble_200_selected_case/README.md`.
`run_v1` retains the initial pre-sampling log-format engineering failure;
`run_v2` is the completed run. N3 smoke and artificial surface tests are
separately labelled and are not part of the 200-realization experiment.
Stop after this requested study; no five-body experiment, new method, training,
threshold tuning or additional case is implied.


## Stage19R cohort repair — learned evidence adapter validated (2026-09-04)

Stage19R was authorized from `main @
f97c8ca1713421fe31ff250a62671e0cbe06fb41`. The immutable Stage19 v1 cohort
stop below remains authoritative for v1 and was not edited or overwritten.
Stage19R changes only the frozen candidate search budgets (TRAIN 256 -> 1024,
VAL/TEST 128 -> 256) plus generator/asset provenance, TEST truth-firewall, and
formal-checkpoint guards. Eligibility, accepted counts, seed starts, generator
recipe, observation/inversion science, adapter, training, inference, metrics,
and gates are unchanged.

All 12 generator-critical source/default-matrix hashes match the authoritative
Full StructuralGeo benchmark. Root seed `120260003` exactly replays the
historical event sequence and truth/condition hashes, yielding
`GENERATOR_REUSE_VALIDATED`. Cohort v2 then completed 64/8/12 accepted cases
after examining 431/60/63 TRAIN/VAL/TEST candidates. Candidate-level traces
and non-exclusive rejection counts are retained. This repairs the cohort
availability engineering prerequisite; it is not by itself a scientific
result.

All 84 observations pass forward closure, and the unchanged Stage17A evidence
pipeline completed after exact source-asset hash validation. The VAL-only gate
passes with 8/8 positive AP-skill cases, median AP skill `0.549253`, and 8/8
positive specificity cases. CUDA smoke passes condition exactness, independent
scale-zero Flow equivalence, base immutability, truth blindness, and the formal
checkpoint guard. Exactly one frozen formal adapter training completed 4
epochs/1024 updates. The 54,003-parameter epoch-4 checkpoint SHA-256 is
`cdb740bfeccc08f7010048c69deb8e44a13917fe53e3b2f4b46910bf536f20bb`;
base tensor hashes are identical before/after and base gradients are absent.

Truth-blind formal inference completed 12 cases x 3 seeds x 5 arms = 180
outputs with zero condition violations. The stripped TEST registry has no
truth pointer. Retrospective case-first evaluation concludes
`LEARNED_EVIDENCE_ADAPTER_VALIDATED`: learned target IoU, correct-over-zero,
correct-over-wrong, and hard-seismic improvement each pass 12/12 cases. Median
correct-minus-Flow target-IoU delta is `+0.236709`, hard-seismic RMSE delta is
`-0.008943`, and truth-present mean-IoU delta is `+0.039260`; every frozen
engineering, reuse, coupling, specificity, physics, volume, structure, and
global-geology gate passes.

This result supports learned case-specific evidence-to-velocity coupling only
within the binary high-contrast noiseless inverse-crime upper bound. It does
not prove field generalization, and historical sample-level overlap with
streaming CFM pretraining cannot be excluded because no old seed/sample
manifest exists. Authoritative Stage19R artifacts are under
`experiments/stage19_learned_evidence_adapter/{generator_reuse_audit,cohort_v2,observations_v2,evidence_v2,evidence_audit_v2,checkpoints/formal_v2,formal/inference_v2,reports/formal_v2}`;
the full report is
`experiments/stage19_learned_evidence_adapter/reports/STAGE19R_DEVELOPMENT_REPORT.md`.
Focused validation passes 77 tests. Stop after Stage19R and wait for research
discussion; do not tune or launch a new method automatically.

## Stage19 learned evidence adapter — stopped at cohort gate (2026-09-02)

Stage19 was authorized from clean `main` at
`3134684cb9715a8cc58e3758d8770fde075cdd5d` (apart from the user's untracked
Stage19 instruction file).  The implementation adds separate cohort,
observation, evidence, VAL-audit, frozen-adapter training, five-arm truth-blind
inference and retrospective case-first evaluation programs.  It directly
reuses the Stage17A inversion and the Phase6 residual adapter; it does not
modify the base Flow, checkpoint, EMA policy, Stage1--18 outputs or the old
five-case benchmark.  Focused CPU regressions pass `65 passed`.

The first formal step triggered the protocol's mandatory cohort availability
stop.  In the fixed TRAIN candidate range `220260001..220260256`, only 33 of
256 candidates met the unchanged Full StructuralGeo eligibility, versus the
required first 64.  The run stopped with
`RuntimeError: STOP: insufficient eligible train cases`.  The frozen cohort
config SHA-256 is
`f0acfd6be07b870b58cf3a75451516d4f6b762c2de957375b653ae066608368c`;
the failed cohort manifest SHA-256 is
`ae47b6666b1b4c4ea0402548a3e8154a67aeba3de97c38d82c00d4137b607d38`.
The 33 accepted partial TRAIN cases are preserved under
`experiments/stage19_learned_evidence_adapter/cohort/cases/` and must not be
promoted to a smaller post-hoc training cohort.

Per the frozen Stage19 rule, do not expand the 256-candidate budget, weaken
eligibility, replace rejected cases, generate observations, train the adapter,
or run test inference under this protocol.  No evidence reuse gate, CUDA
smoke, formal training, adapter checkpoint, 180-output inference matrix or
scientific Stage19 classification exists.  A future attempt requires an
explicitly authorized new cohort protocol; it must not overwrite this failed
run.

## Stage18 evidence target semantics (2026-08-29)

Stage18 is complete.  It first audited all 30 immutable Stage17C hard
geologies with the frozen Stage17 binary acoustic/seismic observation model,
without running Flow.  The decision is `SURROGATE_OVERSHOOT_SIGNAL`: all 5/5
independent cases improve case-median target IoU while worsening hard-seismic
RMSE under positive-only trajectory guidance, and all five remain volume
overpredicted.  Truth-to-observation CUDA replay closes with maximum RMSE
`4.448e-7`.

The sole Stage18B intervention changed
`target_properties: 1 -> binary_impedance_score` while retaining Stage17C's
`confidence = binary_impedance_score * free_subsurface`.  Thus spatial
confidence weighting, evidence, source noise, checkpoint/EMA, conditions,
32-step fixed Euler, controller, schedules, decoder and every guidance
parameter remained fixed.  No threshold or score rescaling was used.  Formal
CUDA produced exactly 15 new `CONTINUOUS_PROPERTY_TARGET` outputs; all 15
pairs and model/condition hashes pass.

The decision is `VOLUME_REPAIR_WITH_LOCALIZATION_RETAINED`: case-median
absolute volume error improves versus positive-only in 5/5 cases and target
IoU remains above Flow-only in 4/5.  New hard-seismic RMSE is below both
positive-only and Flow-only in 5/5.  Across case medians, Flow / positive / new
target IoU is `0.02999 / 0.24451 / 0.09284`, absolute volume-error fraction is
`0.59467 / 2.31987 / 0.45016`, and hard-seismic RMSE is
`0.03961 / 0.04758 / 0.03797`.

This supports positive-only target semantics as a principal causal contributor
to Stage17C overprediction, but not as the only limitation.  New-arm topology
is substantially fragmented: cross-case median connected components rise to
`328`, largest-component fraction falls to `0.3770`, and top-8 mass falls to
`0.8531`.  Case04 also loses the Flow-only target-IoU advantage.  Fixed
truth-present mIoU is slightly lower (`0.19664`) than both references.

Soft and hard label-9 masses use the identical free-subsurface mask.  The
case-median `M_hard-M_soft` is `-12.46`, with all five case medians negative,
so hard decode does not systematically amplify the volume; the remaining
behavior is already present in the final soft state.

Focused remote regressions pass `61 passed, 13 warnings`.  Full details,
commands, hashes and tables are in
`experiments/stage18_evidence_semantics/reports/DEVELOPMENT_REPORT.md`.
Stage18 is stopped.  Do not automatically run a threshold arm, test 0.6,
retune guidance, add hard correction, change the decoder or start a new method.

## Stage16 D-Flow source optimization (2026-08-24)

Stage16 adds an inference-only, source-point D-Flow engine: the EMA CFM is
frozen and the only optimized tensor is the Gaussian source state.  The
authoritative projected 32-step fixed-Euler map is differentiated end to end,
with non-reentrant per-step model-net checkpointing and exact condition
projection at initialization and after every step.  Oracle Phase1 probability,
Phase2A property and direct Stage15 binary-seismic objectives share the same
engine and strict FLOW_ONLY/DFLOW source pairing.  The seismic runner's asset
API contains no truth; truth is loaded only by a separate evaluator after a
completed run.

CPU tests pass 9/9 and relevant Phase/Stage regressions pass 74/74.  One-seed
CUDA smoke runs (seed 42, frozen Adam 0.01, 20 iterations, no shell penalty)
completed for all three objectives with nonzero source gradients, nonzero
updates, zero condition violations and byte-identical model state before/after.
Endpoint losses decreased: oracle 2.05651 -> 1.78493, property 1.54070 ->
0.83186, seismic soft raw MSE 0.0016833 -> 0.0011523; hard seismic MSE also
decreased 0.0017588 -> 0.0011911.  Retrospective hard-geology changes were
mixed or negative, especially seismic mIoU -0.07879 and accuracy -0.15602.
This is retained negative mechanism evidence; no tuning was performed.  Formal
multi-seed experiments have not been run.  See
`experiments/stage16_dflow_source_optimization/reports/DEVELOPMENT_REPORT.md`.

## Stage15 topology-support stress test (2026-08-14)

A new controlled A/B experiment freezes the existing checkpoint and compares a
volume-matched solid oblate label-9 body (2543 voxels, beta1=0) with a toroidal
label-9 body (2524 voxels, beta1=1).  Both use the same cond_generation_0
background after replacing its original label 9 by raw label 4, the same
surface/nine boreholes, the same concealed center, and the frozen Stage15
binary acoustic convolutional forward.  The registered target intersects no
borehole.  Source seeds 42/142/242 are fixed and unselected.

All 18 Flow outputs completed on the RTX 4090 D with zero condition violations:
Flow-only, Stage15-H trace-inversion property guidance, and truth-derived
oracle probability guidance for both cases.  On the ring case, median label-9
IoU is 0.00019 / 0.30917 / 0.99370; median recall is
0.00040 / 0.98177 / 1.00000.  Oracle guidance produces target-crop beta1=1 in
2/3 samples and median azimuthal ring coverage 1.0, demonstrating useful
expressive support for the code-defined ring under direct inference guidance.
The Stage15-H bridge also produces beta1=1 in 2/3 and ring coverage 1.0, but
fills part of the center (median hole preservation 0.7117), has low precision
0.3110, and worsens median hard seismic RMSE from 0.02831 to 0.03430.  Its
geometric gain therefore cannot be called an observation-consistent correction.

The v1 retrospective gate is preserved as a metric-design failure: it required
central-hole preservation to exceed Flow-only, but an empty Flow prediction has
the degenerate score 1.0.  Evaluation v2 replaces only that clause with
azimuthal ring coverage and reruns no Flow or inversion.  Its decision is
`PRIOR_SUPPORTS_RING_BUT_SEISMIC_BRIDGE_INSUFFICIENT`.  This is a frozen-prior
stress test, not proof that the topology was absent from every historical
streaming training sample.  Authoritative artifacts are under
`experiments/stage15_topology_support/{cases_v1_fix1,runs_v1_fix1,reports/topology_support_v2}`.

Updated: 2026-08-11 after the new end-to-end exploratory Stage 14 pilot tested
whether the previous bridge-only stop rule was overly conservative. The frozen
Flow did not benefit from the existing Stage12B post-seismic P(label9) volume:
the machine decision is `GANSIM_STYLE_GEO_GUIDANCE_NOT_SUPPORTED`. Historical
Stage10, Stage12 and Stage13 machine decisions remain unchanged.

Stage15-C was completed on 2026-08-13 as a newly authorized non-neural direct
mapper. It used 128 fixed new Full StructuralGeo seeds, the unchanged Stage15
binary acoustic/seismic forward, one 17-sample (128 ms) centered local-energy
attribute, fixed-background-velocity time-to-depth resampling, and 64 pooled
quantile bins with Laplace probabilities. The calibration distribution was
strongly zero-inflated: ties left 63/64 quantile bins empty, and the held-out
`cond_generation_0` map became the constant natural calibration prevalence
`0.0098096961`. Retrospective AUPRC was `0.045394953`, below frozen B2
occupancy AUPRC `0.057284505`; truth/background means and medians were equal,
the `P9>=0.8` diagnostic mask was empty, and all 8,968 truth-label9 voxels fell
under `P9<=0.2`. Stage15-C stops here: do not tune the window, bins, fixed
velocity, diagnostic thresholds, add attributes/models, or run Flow guidance
under this protocol.

Stage15-D was explicitly stopped before implementation because Stage15-C did
not persist its per-case trace/column tensors and deterministic replay was not
authorized. Stage15-E then completed a Flow-free 4x4x4 coarse binary seismic
identifiability test. Since 64/4=16, each of the 64 coarse parameters was
nearest-repeated over a 16x16x16 fine block; this arithmetic clarification is
frozen in the config. Eight fixed Adam inversions used only raw seismic MSE for
300 iterations. Final MSEs clustered at `0.000704765–0.000705092`, but truth
localization failed: per-run median Pearson/Spearman were `-0.05205/0.00290`,
median target-minus-background q was only `0.000986`, and the eight-run mean
had Pearson/Spearman `-0.11286/-0.10096`, target/background q
`0.517821/0.517536`, top-k overlap `19/33`, and centroid distance `0.93652`
coarse cells. The machine decision is `COARSE_LABEL9_LOCALIZATION_FAILS`.
Do not reinterpret seismic-loss reduction as localization success or change
coarse size, add regularization, sweep parameters, or enter Flow guidance
under this protocol.

## Read this first in a new conversation

For a paste-ready continuation prompt and a compact statement of the terminal
Stage 9A result, first open `docs/NEXT_CONVERSATION_PROMPT.md`. This handoff
remains the authoritative detailed history.

The active implementation is `flow2` on branch `main`, starting from commit
`72a8eed6ffc9c3bc07d7942709a68fbc6bc9896f` for Stage 9A. The working tree
contains the new Stage-9 code and immutable formal run outputs. Never
discard, reset, upload-overwrite or silently reorganize those files. Inspect
`git status` and overlapping diffs before every edit.

Required reading:

1. `docs/AGENTS.md`
2. `docs/PROJECT_BASELINE.md`
3. `docs/EXPERIMENT_PROTOCOL.md`
4. this file
5. `docs/STAGE1_SPEC.md`
6. `docs/PHASE1_REPORT.md`
7. `docs/PHASE2_SPEC.md`
8. `docs/PHASE2A_REPORT.md`
9. `docs/PHASE2B_SPEC.md`
10. `docs/PHASE2B_FOLLOWUP_SPEC.md`
11. `docs/PHASE2B_REPORT.md`
12. `docs/RESEARCH_GOAL.md`
13. `experiments/stage1_probability/README.md`
14. `experiments/stage2_property/README.md`
15. `docs/PHASE3_SPEC.md`
16. `docs/PHASE4_SPEC.md`
17. `experiments/stage3_spatial_property/README.md`
18. `docs/PHASE3_REPORT.md`
19. `docs/PHASE4A_REPORT.md`
20. `docs/PHASE4C_SPEC.md`
21. `docs/PHASE4C_REPORT.md`
22. `docs/PHASE4D_SPEC.md`
23. `docs/PHASE4D_REPORT.md`
24. `docs/PHASE5A_SPEC.md`
25. `experiments/stage5_acoustic_inversion/README.md`
26. `docs/PHASE5A_REPORT.md`
27. `docs/PHASE5B_SPEC.md`
28. `docs/PHASE5B_REPORT.md`
29. `docs/PHASE5C_SPEC.md`
30. `experiments/stage5_generator_posterior/README.md`
31. `docs/PHASE5C_REPORT.md`
32. `docs/PHASE6_ADAPTER_SPEC.md`
33. `experiments/stage6_geo_adapter/README.md`
34. `docs/PHASE6A_REPORT.md`
35. `docs/NEXT_CONVERSATION_PROMPT.md`
36. `docs/PHASE6P_INFERENCE_LIMIT_SPEC.md`
37. `docs/PHASE6P_REPORT.md`
38. `experiments/stage6_inference_causality/runs/five_body_cuboid_v1/d7_observation_specificity_v1/D7_OBSERVATION_SPECIFICITY_REPORT.md`
39. `experiments/stage6_inference_causality/reports/stage7_v1_final_v2/STAGE7_REPORT.md`
40. `experiments/stage8_structured_posterior/runs/stage8a_v4/STAGE8A_V4_FINAL_REPORT.md`
41. `docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md`
42. `experiments/stage9_flow_prior_posterior/README.md`
43. `experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/STAGE9A_REPORT.md`

## Global goal and immutable boundary

The research goal is to improve decoded three-dimensional geology when surface
and sparse boreholes under-constrain deep intrusive structures, by introducing
spatially broad 3-D/property/geophysical evidence during inference. Success is
defined in hard labels, complete-model accuracy, geometry and ensemble
uncertainty. Continuous proxy loss alone is not success.

The final system is reciprocal: geophysical data should add global constraints
beyond sparse wells, while the frozen learned geological prior and exact
surface/borehole conditions should reduce the non-unique set of geophysical
solutions. The intended output is a condition-exact, observation-consistent,
geologically plausible ensemble, not a geophysics-only deterministic inverse.

Do not modify training, the 3-D U-Net, checkpoint or EMA convention. The normal
frozen `embedding.weight` is used with EMA values for all 411 trainable model
entries. Attribution experiments use identical inputs/noise/time grids and a
strict fixed-Euler alpha-zero pair. Conditions are projected before sampling
and after every integration step. Do not combine the current work with the
historical 2-D gravity path yet.

## Stage status

- Phase 0: complete.
- Phase 1: closed as mechanism validated with topology and endpoint caveats.
- Phase 2a: complete. The ideal full-resolution density-plus-susceptibility
  property upper bound passes all frozen gates in 12/12 strict pairs across
  seeds 42/142/242, with the documented class, geometry and CUDA-repeatability
  caveats.
- Phase 2b property-codebook ambiguity/contrast sensitivity: protocol,
  codebooks, stage-aware runner, single-sample launcher, report gate and CPU
  tests are complete. The distinct-codebook GPU anchor, `paired_c100` and the
  narrow `paired_c025` pass; `paired_c010` fails the target/component/topology
  gates, and the final exact-collision control also fails. The five-level
  screen is complete. The n=4 bracket is also complete: `paired_c025` is a 3/4
  transition and `paired_c010` is a 0/4 confirmed seed-42 failure. Neither is
  eligible for multi-seed promotion. The separately pre-registered
  `paired_c100` seed-42 n=4 fallback is also a 3/4 transition, so Phase 2b
  closes with no multi-seed-confirmed ambiguous-codebook operating point.
- Phase 3 spatial degradation: closed. Identity passes, but every nonzero
  Gaussian level fails; sigma 1 also fails its n=4 confirmation 0/4.
- Phase 4 acquisition-domain physics: the full-support gravity (4a),
  convolutional-seismic guidance (4c) and fixed-12 identifiability audit (4d)
  are implemented and closed negative. Each can improve a physical residual,
  but none improves the required hard label-9/major-body geology.
- Phase 5a no-training acoustic inversion bridge: complete. The continuous
  property gate passes, but direct nearest-codebook hard projections worsen;
  it authorized the now-completed single-pair Phase-5b flow bridge gate.
- Phase 5b inversion-posterior flow bridge: closed negative. The formal CUDA
  seed-42 n=1 pair is valid and lowers hard bridge loss, but fails global,
  class-majority, absolute label-9 and major-body gates. No n=4 or tuning is
  authorized.
- Phase 5c direct generator-posterior search: implemented and closed negative.
  Full-dimensional pCN over the Gaussian initial noise is technically active,
  condition-exact and inexpensive enough for a short chain, but the frozen
  8-proposal CUDA pilot lowers hard seismic loss while worsening label-9 IoU,
  recall and four-major-body recovery. No longer chain or pCN tuning is
  authorized on the repeatedly audited legacy case.
- Phase 6A frozen-flow residual adapter: engineering/oracle mechanism pass.
  The 54,327-parameter external adapter trains while the original model remains
  bitwise unchanged, preserves exact conditions and greatly improves the
  same-case truth-derived acoustic-oracle hard geology. This is not held-out or
  truth-blind geophysics; hard seismic loss worsens. Phase 6 is started but not
  complete. The next task is deterministic grouped data and a held-out oracle
  pilot.
- Phase 6P frozen inference-limit audit: complete. A predeclared trajectory
  ladder raises alpha/cap from 0.25 to 4.0; maximum hard-seismic RMSE
  attainment is only 9.88% at ratio 1.0, and stronger interference damages
  geology while physical fit regresses. A separate 200-step endpoint optimizer
  reaches only 1.69%; its soft response improves while the hard response
  diverges. All engineering, hash, condition and historical-regression gates
  pass. Same-case inference-strength tuning is closed.
- Phase 6Q D0–D6 and Stage 7: complete. D7 uses eight identical BASE states and
  ranks the similarity mechanism S1 residual (`0.9208`) > S4 hard transition
  (`0.0340`) > S2 VJP (`0.0242`) > S3 controller (`~0`). Structured hard
  inference then exactly recovers the analytic hidden pair from hard seismic
  alone and passes correct/zero/shuffled/wrong-case specificity. Across three
  deterministic StructuralGeo replicas the correct arm ranks strictly first
  on the shared correct observation in 3/3 cases; mean hard attainment is
  `42.70%`, hidden IoU is `0.914–0.987`, hidden recall is `0.916–0.996`, and
  condition/wrong-lithology violations are zero. No training was run.
- Stage 8A continuous structured hard-geophysics: complete negative gate. The
  immutable Stage-7 analytic library regression still reaches exact hard RMSE
  zero without truth candidate indices, but the pre-registered broad-domain
  continuous proposal search selects its empty start for the analytic correct
  arm and all three StructuralGeo-native correct arms. Every arm uses exactly
  961 hard-forward calls; hard conditions, truth-blind selection and matched
  control budgets pass. Analytic and native concealed IoU/recall remain zero,
  so correct/control specificity, geometry, unknown-count and recovery gates
  fail. Stage 8B is blocked and was not run. Stage 8A-R1 then analyzed every
  frozen correct-arm trace without replay: each contains the empty reference
  plus 960 proposals, and none of the 3,840 noninitial proposals has negative
  delta RMSE. The complete Stage-7 12-candidate loss library has two strictly
  improving paths, empty -> candidate_04/06 -> exact pair, so no simultaneous
  birth or energy-barrier crossing is required. In the explicitly oracle-only
  pre-existing local trace, all 62 perturbations of the exact pair remain
  better than empty, rejecting a too-narrow analytic hard-loss basin at those
  widths. The unique primary R1 category is `RANDOM_BIRTH_BASIN_MISS`; cuboid
  omission is a secondary analytic parameterization limitation, not the
  cross-case explanation because all three in-family DikeHemisphere cases fail
  identically. Do not increase budget or tune the proposal kernel post hoc.
  The sole minimal recommendation, only under a new protocol, is to replace
  uniform global birth-center selection with a deterministic truth-blind
  residual/sensitivity-ranked initializer inside the unchanged 961-call
  budget. R1 did not implement Stage 8A v2.
- Stage 8A-v2 deterministic birth-center re-gate: complete negative gate and
  stopped. The sole implemented change ranks integer birth centers by the
  negative first-order hard-state seismic-MSE directional derivative of a
  canonical label-9 ellipsoid, summing both impedance and slowness terms.
  Every correct arm still selects empty: each has 560 birth proposals and zero
  births with negative hard-RMSE delta versus either parent or empty. All four
  correct arms retain exactly 961 hard-forward evaluations, zero condition
  violations and truth-blind hard-RMSE selection. The unchanged original gate
  passes only case presence, conditions, truth firewall and equal budgets; it
  fails specificity, analytic/native recovery, geometry and unknown count.
  Across all 16 case/control arms, the run records 15,376 hard forwards plus
  725 separately accounted sensitivity forwards/backwards. One conformance
  caveat is explicit: the analytic zero-observation arm followed a different
  beam/body-count path and consequently realized 370 births plus 118 of each
  other move, rather than the v1 realized 560/80 counts, although the frozen
  state-dependent move kernel itself was unchanged. Do not repair or rerun in
  this task: the v2 failure stop rule forbids a second modification. Stage 8B
  remains blocked.
- Stage 8A-v2-R2 alignment audit: complete, with zero new hard proposal
  forwards and no v2 rerun. The complete analytic initial center ranking was
  frozen before opening oracle geometry: candidate_06 and candidate_04 map to
  ranks 1 and 5 of 140,985, so center localization is not the failure. Raw
  gradients/observations were absent from v2; R2 indispensably reconstructed
  eight observation fields and exactly replayed 725 differentiable
  forward/backward pairs. Every saved prediction, residual, impedance and
  slowness gradient, and sensitivity-map hash matched v2. On 2,240 correct-arm
  births, actual-mask first order predicts improvement for 2,106 but frozen
  hard RMSE improves for zero; sign agreement is only 5.98%. The Stage-7
  candidate_04/06 singleton masks rank first/second by the same first-order
  derivative and both have matching improving frozen finite-step signs. The
  unique primary category is `FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY`: v2's
  failure is the extrapolation from an infinitesimal property direction to its
  one-shot finite hard insertion, not sensitivity localization or a replay
  defect. Exactly one unimplemented recommendation is a deterministic
  truth-blind hard-loss trust-region continuation that starts with a nested
  small allowed-shape insertion and grows it only through already-budgeted
  hard-RMSE-improving proposals. The analytic-zero 370-versus-560 birth-count
  protocol deviation remains separately recorded and unrepaired.
- Stage 8A-v3 nested hard-loss birth continuation: complete negative gate and
  stopped. The only change used the preregistered common hard-categorical scale
  ladder `(0.25, 0.5, 0.75, 1.0)` at the unchanged v2-ranked center and frozen
  full geometry. All 16 arms retained exactly 961 hard forwards, exact
  conditions, truth-blind hard-RMSE selection, the unchanged ranker and the
  unchanged original gate. Correct-observation specificity passed in 4/4
  cases, but analytic IoU/recall reached only `0.040625`; native values were
  `0.002123`, `0`, and `0`, so analytic/native recovery and geometry gates
  failed. Across all arms, 10 of 8,270 scale-0.25 births improved their own
  parent; correct arms contributed 3. No improving child survived the frozen
  beam as a later proposal parent, so the run allocated zero continuation
  growth slots, evaluated no scale above 0.25, and has an empty improving-
  growth distribution. Frozen-v2 same-full-geometry alignment found zero
  matched full-size-failure/small-size-rescue examples. The machine decision is
  `FAIL_STAGE8A_STOP_BEFORE_STAGE8B`; do not rerun, run Stage8B, or implement a
  second algorithmic modification under this task.
- Stage 8A-v3-R3 beam/lineage survival audit: complete with zero new proposals,
  hard forwards, geology materializations, sensitivity passes or truth use.
  R3 reconstructs every 96-candidate generation by the frozen `(hard RMSE,
  state_id)` ordering. Across 1,840 correct-arm scale-0.25 births,
  `N(delta_parent<0)=3`, `N(delta_empty<0)=5`, locally improving and entering
  the next beam is zero, and empty-improving but pruned is five. The three local
  improvements rank 10/96, 50/96 and 30/96; their positive margins above the
  eighth-place cutoff are `9.835e-06`, `5.498e-05` and `8.946e-06`. All birth
  parents are valid beam states, every persisted next-generation parent list
  exactly matches the reconstructed top eight, and the selection-defect count
  is zero. None of the three branches exactly matches a frozen v2 full-target
  geometry, so R3 preserves the absence of matched full-fail/small-success
  evidence. The unique classification is
  `GLOBAL_BEAM_PRUNES_LOCALLY_IMPROVING_SEEDS`. Exactly one future, unimplemented
  recommendation is lineage-preserving local hard-loss continuation before
  global beam competition, reallocating the existing 961 slots without wider
  beams, relaxed lineage monotonicity, new shapes, ranker changes or truth.
- Stage 8A-v4 final standalone iteration: complete negative gate and terminal.
  All 16 arms use exactly 961 hard forwards with zero condition violations and
  truth-blind hard-RMSE selection. Across all arms there are 7,742 new-center,
  7,616 nonbirth and 2 reallocated growth evaluations. Both locally improving
  scale-0.25 seeds occur in zero controls; both scale-0.50 growth attempts fail,
  so no branch reaches scale 0.50 and no correct arm exercises continuation.
  Correct specificity is strict in only 2/4 cases. Analytic hidden IoU/recall
  are zero; native values are `0.00849`, `0.01`, and `0`, with mean
  `0.006164`. The original machine decision is
  `FAIL_STAGE8A_STOP_BEFORE_STAGE8B`. A post-run conformance caveat is frozen:
  v4 renamed the seed proposal-move metadata, and that field enters the
  downstream deterministic parent-state RNG hash, so paths diverged from v3
  even before continuation activated and the three R3 correct seeds were not
  reproduced. No repair/rerun is made because v4 is terminal and another full
  run is not authorized. Do not implement v5. Standalone Stage8A is closed;
  Stage8B remains unrun, and the next action is a newly authorized Flow-prior
  integration decision rather than another standalone structured-search edit.
- Stage 9A frozen Flow-prior support and geophysical enrichment: complete
  negative gate and terminal. Three new deterministic StructuralGeo cases each
  use 1024 independent Gaussian sources, frozen EMA, normal embedding,
  32-step midpoint fixed Euler, exact condition projection, hard decoding and
  one cached hard-seismic prediction. Correct, zero, shuffled-XY and
  independent wrong-case observations rank the same immutable predictions;
  all rankings and hashes freeze before retrospective truth is loaded. All
  3072 hard models are unique, but no candidate in any case passes the complete
  Phase-4d geological support gate. Correct-observation ranking does not
  jointly enrich label-9 IoU, label-9 recall and major-component mean recall;
  discrimination fails in 0/3 cases. Overall `SUPPORT=false`,
  `DISCRIMINATION=false`, and the machine action is
  `STOP_REASSESS_FROZEN_INFERENCE_ROUTE`. Do not implement Stage 9B/9C,
  posterior weighting, adaptive proposals, SMC, D-Flow, a new likelihood or
  training under this protocol.

## Stage-9A frozen Flow-prior support result

`docs/STAGE9A_FLOW_PRIOR_SUPPORT_SPEC.md` was frozen before CUDA evidence. The
candidate-generation API cannot receive truth; the ranking API sees only the
four observation controls; and the retrospective auditor validates immutable
pool/ranking hashes before it can load truth. Formal predictions are stored in
deterministic, lossless gzip-compressed float32 chunks with both compressed and
decompressed hashes. This reduces disk use without changing numerical values
or later readability; models remain hard int8 categorical volumes.

The three formal pools contain 1024 candidates each and use 3072 hard-seismic
forwards plus 98,304 Flow velocity forwards. Candidate-generation runtimes are
`3342.224`, `3508.668` and `3189.077` seconds, totaling `10039.969` seconds.
Every pool contains 1024/1024 unique hard models. The deployable correct-
observation tops are `candidate_000066`, `candidate_000862` and
`candidate_000567`, with hard-seismic RMSE `0.0318388`, `0.03101524` and
`0.03161827` respectively.

Retrospectively, the number of candidates passing the complete registered
support gate is `0/1024` in every case. Even the truth-oracle best label-9 IoU
is only `0.1170`, `0.1348` and `0.1289`; oracle selection is explicitly not
deployable. Correct-observation RMSE ranking enriches label-9 IoU in some
checks, but its Spearman/enrichment directions fail for label-9 recall and
major-component mean recall, so every case has `DISCRIMINATION_PASS=false`.
The aggregate gate is support `0/3` and discrimination `0/3`, both below the
required `2/3`.

The authoritative machine summary and human report are under
`experiments/stage9_flow_prior_posterior/reports/stage9a_prior_support_v1/`.
The immutable case evidence is under the sibling
`runs/stage9a_prior_support_v1/formal/` tree. Smoke runs are engineering only;
they are not scientific evidence. Stage 9A stops at
`STOP_REASSESS_FROZEN_INFERENCE_ROUTE`: lower seismic loss alone is not project
success, and these three synthetic inverse-crime cases do not establish field
generalization.

## Phase-0 result

- Checkpoint/input validation passed.
- EMA loading passed: 411 trainable entries from EMA; the frozen embedding kept
  its normal checkpoint value.
- Borehole/truth mismatches are zero.
- Adaptive Dopri5 and 32-step fixed Euler differ by about 4.8645% of decoded
  voxels and cannot be mixed in attribution.
- Historical 2-D gravity guidance at relative alpha 0.05 changed only 31.25
  hard voxels per sample on average. This proved an active path, not improved
  label-9 geometry.

## Phase-1 final decision

The authoritative evidence is:

- human-readable decision: `docs/PHASE1_REPORT.md`;
- generated report:
  `experiments/stage1_probability/reports/phase1b_v4_12pair/REPORT.md`;
- machine summary: the adjacent `summary.json`;
- per-pair and topology tables: `paired_samples.csv` and
  `topology_samples.csv`.

Rebuild those derived artifacts, without changing immutable runs, from the
repository root:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage1/summarize_phase1b.py \
  --overwrite
```

The aggregation revalidates each baseline/guided config, completion state,
EMA policy, initial-noise hash identity, saved pairing verdict, zero condition
violations and zero decoder mismatches. It then recomputes size-stratified
six-connected topology directly from the saved tensors.

### Protocol-v4 12-pair evidence

Seeds 42/142/242, four samples each, alpha/cap 0.25, 32 fixed-Euler steps,
`calibrated_soft_bce_hard_dice_v2`, `reference_norm_relative_v2` and
`windowed_sine`:

- target IoU `0.0314 -> 0.8099`, precision `0.0788 -> 0.8274`, recall
  `0.0520 -> 0.9747`;
- selected-ROI IoU `0.0351 -> 0.9392`, precision `0.1050 -> 0.9627`, absolute
  volume-error fraction `0.5101 -> 0.0139`;
- target-centroid distance `16.9518 -> 3.3798`;
- global voxel accuracy `0.5972 -> 0.6432`, mean IoU `0.1804 -> 0.2178`;
- all primary target metrics, global metrics and centroid distance improve in
  all 12 pairs;
- 5.1811% mean full-volume hard changes, 98.7588% inside ROI;
- outside-ROI accuracy change about `+0.000022` and all condition violations
  zero;
- all three guided ensembles retain four unique samples and outside-ROI
  disagreement.

### Honest caveats

The pre-registered strong gate is `not_full_pass`:

- full target component ratio is 1.2812 versus limit 1.25;
- ROI diagnostic component ratio is 1.2665;
- raw largest-component fraction falls `0.7239 -> 0.3856`;
- the Phase-2 handoff occurred before a protocol-v4 fixed-camera render was
  generated and visually reviewed;
- last-step hard churn remains 802-1,398 voxels rather than baseline-level
  stability.

Truth-relative size stratification shows why raw topology is misleading but
does not change the failed gate: truth component sizes are
4,079/2,192/2,043/627/22/4/1; all 12 guided samples have exactly four ROI
components of at least 20 voxels; mean guided top-four sizes are approximately
4,074/2,203/2,129/600; fragments of at most five voxels contain only 0.66% of
guided ROI target mass.

Phase 1 therefore proves inference-time 3-D control and soft-to-hard crossing.
It does not prove real geophysics, complete-model multi-lithology recovery, or
that boreholes/surface conditions can be removed.

## Phase-1 implementation paths

- `guidance/probability_volume.py`
- `guidance/probability_sampling.py`
- `guidance/probability_evaluation.py`
- `scripts/stage1/run_probability_guidance.py`
- `scripts/stage1/visualize_probability_guidance.py`
- `scripts/stage1/summarize_phase1b.py`
- `tests/test_stage1_probability_guidance.py`
- `tests/test_stage1_probability_visualization.py`
- `tests/test_stage1_phase_report.py`
- `experiments/stage1_probability/`

## Phase-2 design and current code

Phase 2a replaces target-label probability with complete expected properties:

```text
p_t(k,r) = softmax_k(cos(x_t(r), e_k) / tau_t)
q_t(j,r) = sum_k p_t(k,r) q(j,k)
L = sum_s lambda_s || W_s * [S_s(q_t) - S_s(q_true)] ||^2
```

The same 3-D Gaussian `S_s` is applied to predicted and truth-derived target
properties. Each property channel is divided by its weighted target standard
deviation before channel weighting. This implements the operator-matching rule
and prevents units with large numerical values from dominating.

Implemented in `guidance/property_volume.py`:

- strict `full_lithology_property_channels_v1` parser;
- mandatory explicit raw-label coverage `-1..13` with no silent default;
- hard-label to multi-channel property mapping;
- all-class soft probability to expected property mapping;
- depthwise matched multi-scale 3-D Gaussian operators;
- normalized multi-channel MSE and confidence/missing-region weights;
- end-to-end differentiable loss through the unchanged soft decoder;
- property/entropy diagnostics.

The first controlled configuration is
`experiments/stage2_property/configs/ideal_distinct_density_proxy_v1.json`.
It maps all classes to distinct scalar proxy values in controlled relative
units. It is deliberately an easy Phase-2a bridge and not calibrated density.
Because a single soft expected scalar remains non-unique across 15 category
probabilities, `ideal_density_susceptibility_label9_contrast_v1.json` adds a
second complete controlled channel with an explicit label-9 contrast. It tests
information identifiability, not measured magnetic inversion. Later Phase-2
ablation must introduce overlapping/ambiguous properties before Phase 3 adds
spatial degradation and noise.

CPU tests in `tests/test_phase2_property_volume.py` cover config completeness,
raw-label offset, hard/soft mapping equality, channel-isolated Gaussian blur,
matched-loss zero/positive behavior, unit invariance, finite soft-decoder
gradients and zero-confidence behavior.

## Implemented Phase-2 execution path

- `guidance/property_sampling.py`: isolated fixed-Euler sampler. It reuses the
  proven Phase-1 temperature, reference-norm controller and `windowed_sine`
  definitions without changing Phase-1 source or activating its probability
  loss. Alpha zero is an explicit no-gradient branch.
- `guidance/property_evaluation.py`: global and label-9 geometry, hard-property
  loss, all raw non-air class metrics, paired per-class deltas, and a stable
  truth-present-class mIoU alongside the historical dynamic-union mIoU.
- `scripts/stage2/run_property_guidance.py`: EMA loading, truth/borehole
  validation, explicit property parsing, target/confidence construction,
  immutable outputs, strict config comparison, sequential CPU noise hashes,
  sampling, hard evaluation and ensemble artifacts.
- `experiments/stage2_property/run_phase2a_ideal_property_smoke.sh`: isolated
  one-sample alpha-zero/alpha-0.10 GPU launcher.

The property confidence is the unconditioned non-air subsurface, so exact air,
surface and borehole conditions do not consume the full-resolution property
loss. They are still re-projected after every Euler step.

## First Phase-2 GPU result

The seed-42, one-sample, 32-step scalar-property pair under
`ideal_distinct_density_proxy_v1/phase2a_v1/seed42_n1_s32` is valid:

- strict pairing passed, EMA and identical initial noise were confirmed;
- all trace values were finite and all condition violations were zero;
- global accuracy changed `0.58737 -> 0.59684`, mean IoU
  `0.19291 -> 0.19527`, and hard-property loss
  `1.19629 -> 0.98169`;
- label-9 recall changed `0.04728 -> 0.04472`, predicted volume
  `6283 -> 3691` versus truth `8968`, and 2,623 label-9 voxels were converted
  away while only 31 converted into label 9.

This is a functioning sampler and controller but a failed scientific result
for the scalar observation. Do not raise alpha and do not run its n=4 batch.

## Second Phase-2 GPU result

The two-channel observability run under
`ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n1_s32`
also passed strict runtime validation. Global accuracy increased by `0.01129`,
truth-present fixed-set mIoU increased by `0.00606`, six of eight
truth-present classes improved IoU, and hard-property loss fell by 30.06%.
Label-9 IoU changed `0.02860 -> 0.05795`, precision
`0.06748 -> 0.22429`, recall `0.04728 -> 0.07248`, and true positives
`424 -> 650`.

This remains a partial result: label-9 predicted volume fell to 2,898 versus
truth 8,968, components increased `37 -> 109`, and the largest-component
fraction fell. The historical dynamic-union mIoU also fell by 0.01204 because
four erroneous label-8 voxels introduced an additional zero-IoU class into its
per-sample denominator. Both that metric and the new fixed truth-present mIoU
are retained. See `docs/PHASE2_PROGRESS.md`.

## Third Phase-2 GPU result

The same two-channel target at `alpha=cap=0.25` materially improves the hard
geology: label-9 IoU/precision/recall are 0.4816/0.9005/0.5087, centroid
distance falls to 3.42 voxels, global accuracy rises by 0.0410, and fixed-set
mIoU rises by 0.0667. Six of eight truth-present classes improve IoU.

Raw label-9 components increase to 202, but top-eight mass is 87.2% and tiny
mass is 5.37%. The four major truth components have
48.3%/53.6%/57.9%/37.2% recall. The result passes the single-sample directional
gate while retaining an explicit fragmentation caveat.

The Phase-2 evaluator now writes size-stratified topology fields and separate
truth-component recovery CSVs. Numeric n=4 gates are frozen in
`docs/PHASE2_PROGRESS.md`.

## Seed-42 n=4 confirmation

All eight pre-registered gates pass. Across four pairs, mean global accuracy
changes `0.59245 -> 0.63412`, fixed truth-present mIoU changes
`0.26969 -> 0.33847`, dynamic-union mIoU changes `0.17692 -> 0.19341`, and
hard-property loss changes `1.48448 -> 0.51833`. Mean guided label-9
IoU/precision/recall are 0.4905/0.9116/0.5152. Every pair improves six of eight
truth-present classes, all four major truth bodies pass the recovery gate,
topology/endpoint gates pass, conditions remain exact, and four guided samples
remain unique.

The two independent guided sample-0 invocations differ at 10 hard voxels
(0.0038%) despite identical inputs/noise; alpha zero is byte-identical. Strict
paired conclusions are stable, but cross-process bitwise CUDA determinism is
not claimed.

## Seed-142 n=4 confirmation

All eight gates pass again. Mean global accuracy changes
`0.60014 -> 0.64139`, fixed-set mIoU `0.28212 -> 0.34832`, hard-property loss
`1.51606 -> 0.52133`, and guided label-9 IoU/precision/recall are
0.4735/0.8940/0.5028. Topology, endpoint, exact conditions and diversity pass.
Sample 3 improves exactly five of eight truth-present class IoUs; label 2 and
the tiny label-6 prediction decline. Label 2 remains the consistent class-level
tradeoff.

## Seed-242 and Phase-2a decision

Seed 242 also passes all eight gates. The complete 12-pair decision is:

**PASS: Phase-2a ideal 3-D property upper bound validated with caveats.**

The authoritative human report is `docs/PHASE2A_REPORT.md`; generated evidence
is under `experiments/stage2_property/reports/phase2a_v1_12pair/`. Rebuild it:

```bash
PYTHONPATH=src .venv/bin/python \
  project/geodata-3d-conditional/scripts/stage2/summarize_phase2a.py \
  --overwrite
```

All 12 pairs improve global accuracy, fixed-set mIoU, hard-property loss and
label-9 IoU/precision/recall. Label 2 declines in all 12, label 13 remains
unrecovered, target bodies remain incomplete/fragmented, and cross-process
CUDA bitwise determinism is not claimed.

## Phase-2b code and frozen screen

The authoritative protocol is `docs/PHASE2B_SPEC.md`. The code paths are:

- manifest/configs:
  `experiments/stage2_property/configs/phase2b_codebook_ambiguity_v1/`;
- GPU launcher:
  `experiments/stage2_property/run_phase2b_codebook_screen.sh`;
- completed-level auditor:
  `scripts/stage2/summarize_phase2b_screen.py`;
- seed-42 n=4 bracket launcher:
  `experiments/stage2_property/run_phase2b_codebook_n4_bracket.sh`;
- seed-42 n=4 bracket auditor:
  `scripts/stage2/summarize_phase2b_n4_bracket.py`;
- post-bracket fallback protocol:
  `docs/PHASE2B_FOLLOWUP_SPEC.md`;
- `paired_c100` seed-42 n=4 fallback launcher/auditor:
  `experiments/stage2_property/run_phase2b_codebook_n4_fallback.sh` and
  `scripts/stage2/summarize_phase2b_n4_fallback.py`;
- strict Phase-2 fixed-camera renderer and VTK exporter:
  `scripts/stage2/visualize_property_guidance.py`;
- codebook and promotion regression:
  `tests/test_phase2b_codebook_ambiguity.py`.

The five levels are `distinct_c100_anchor`, `paired_c100`, `paired_c025`,
`paired_c010` and `paired_c004_overlap`. The final level makes truth-present
labels 6 and 9 exactly property-equivalent. All levels keep alpha/cap 0.25,
seed 42, n=1, 32 fixed-Euler steps and a separate alpha-zero strict baseline.

## Phase-3 implementation and current evidence

The frozen protocol is `docs/PHASE3_SPEC.md`. It starts from the distinct
two-channel Phase-2a codebook because this is the only 12/12 validated
operating point. The Phase-2b c100/c025 transition levels are not promoted.

Implemented paths:

- `guidance/spatial_property.py`: identity, Gaussian and average-pool response,
  exact known-property overwrite, depth/missing confidence, deterministic
  observation-only noise, soft loss and hard-observation loss;
- `guidance/property_sampling.py`: optional loss injection with the original
  Phase-2 property loss as its default, preserving the existing path;
- `scripts/stage3/run_spatial_property_guidance.py`: EMA/fixed-Euler runner,
  immutable observation tensors and hashes, strict pairs and full hard metrics;
- `scripts/stage3/audit_spatial_screen.py`: strict source/asset audit, bytewise
  Phase-2a alpha-zero regression and the frozen hard gate;
- `experiments/stage3_spatial_property/configs/`: frozen identity and Gaussian
  sigma 1/2/4 configs plus ordered manifest;
- `experiments/stage3_spatial_property/run_phase3_gaussian_screen.sh`: one
  level per invocation; no automatic all-level run;
- `scripts/stage3/summarize_gaussian_screen.py`: validates all four n=1
  reports/hashes and freezes the n=4 bracket decision;
- `experiments/stage3_spatial_property/run_phase3_gaussian_n4_bracket.sh`:
  restricted n=4 launcher for identity and sigma 1 only;
- `tests/test_phase3_spatial_property.py`: observation and sampler CPU gate.

The complete seed-42 n=1 screen is now available under
`experiments/stage3_spatial_property/reports/gaussian_screen_seed42_n1/`.
Identity passed the complete gate. Gaussian sigma 1, 2 and 4 all failed despite
lower observation loss. Label-9 IoU declined monotonically
`0.4881 -> 0.3357 -> 0.2064 -> 0.1026`, and major-body mean recall declined
`0.4984 -> 0.3981 -> 0.2584 -> 0.1305`.

The aggregate decision is **no nonzero Gaussian blur passed the n=1 gate**.
The identity side of the seed-42 n=4 bracket then passed 4/4 complete gates
and diversity. Mean accuracy and truth-present mIoU improve by 0.0420 and
0.0696; guided label-9 IoU/precision/recall average
0.4966/0.9144/0.5211. Conditions, hashes and Phase-2a alpha-zero regression
all pass. This confirms the undegraded anchor only.

The sigma-1 side then failed 0/4. All four pairs failed the label-9 target
thresholds because guided precision remained below 0.75; two also failed the
major-body recovery gate. Its mean guided label-9 IoU/precision/recall are
0.3482/0.6938/0.4120. Pairing, conditions, alpha-zero regression and diversity
remain valid. Phase 3 is therefore closed without a promoted nonzero-blur
working point. Do not run sigma 2/4 n=4 or seeds 142/242. Read
`docs/PHASE3_REPORT.md` for the authoritative conclusion.

Phase 4 is frozen separately in `docs/PHASE4_SPEC.md`. Phase-4a full-support
gravity development proceeds independently. The originally planned Phase-4b
joint guidance cannot assume a Phase-3 Gaussian working point because none was
promoted; it requires a new frozen design or an identity-only upper-bound arm.

## Phase-4a forward/operator implementation

Phase-4a CPU development is now active and isolated from the historical
`SimpleGravityForward` proxy:

- `guidance/gravity.py`: exact rectangular-prism downward `g_z`, SI geometry,
  mGal output, all-cell/all-station support through zero-padded FFT linear
  convolution, density mapping, field loss and immutable observations;
- `guidance/gravity_sampling.py`: gravity-loss injection into the proven
  projected fixed-Euler solver; alpha zero remains its explicit no-gradient
  branch;
- `scripts/stage4/build_gravity_observation.py`: immutable builder with source,
  config and tensor hashes;
- `experiments/stage4_gravity/configs/`: distinct synthetic upper bound,
  exact label-6/9 collision control and full-grid noiseless inverse-crime
  survey;
- `tests/test_phase4_gravity.py`: 18 operator, mapping, gradient, sampler,
  immutable-asset, pairing, controller and reranking gates.

The second implementation unit is also complete:

- `scripts/stage4/run_gravity_guidance.py`: immutable observation/source hash
  validation, EMA, CPU noise, strict fixed-Euler pairs and complete evaluation;
- `scripts/stage4/audit_gravity_screen.py`: alpha-zero regression and frozen
  geology-plus-gravity gate;
- `scripts/stage4/rerank_gravity_ensemble.py`: n=4 post-hoc baseline comparator;
- `experiments/stage4_gravity/run_phase4a_gravity_screen.sh`: first n=1 GPU
  launcher;
- `configs/gravity_controller_manifest_v1.json`: pre-registered alpha/cap 0.25
  and conditional alpha 0.10 diagnostic, included in strict pair hashes.

The Phase-4-focused CPU gate is 18/18 and the complete suite is 103 passed. The canonical
64-cubed observation is under
`experiments/stage4_gravity/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
It is finite with range -1.0706 to 1.5260 mGal and standard deviation 0.7621
mGal. A real-checkpoint CPU one-step alpha-zero runner smoke test completes with
EMA and zero condition violations.

The seed-42 n=1, 32-step, alpha/cap 0.25 GPU strict pair and audit are complete.
All pair, immutable-hash, alpha-zero regression and hard-condition checks pass,
but the complete geology-plus-gravity gate is `0/1`. Hard gravity RMSE decreases
by `0.06561 mGal` and global accuracy rises by `0.00907`; nevertheless label-9
IoU/precision/recall fall from `0.0286/0.0675/0.0473` to
`0.0159/0.0638/0.0207`. Prediction volume falls from `6283` to `2914`, major-
component minimum recall is `0`, and the `6562` hard changes contain `3392`
label-9-to-other versus only `23` other-to-label-9 transitions. Treat this as
field fitting under gravity non-uniqueness, not a partial Phase-4 success. It
must not advance to n=4. The manifest's condition for the single alpha=0.10
harm diagnostic was met, and that strict pair is also complete. It lowers hard
gravity RMSE to `0.87032 mGal` but degrades label-9 IoU/precision/recall to
`0.02111/0.06490/0.03033`; `2101` hard voxels leave label 9 and only `9` enter
it, with major-component minimum recall still zero. Phase 4a is closed without
n=4 or further alpha search. Read `docs/PHASE4A_REPORT.md`. The next independent
upper bound is convolutional seismic response; Phase-4b joint 3-D property plus
gravity remains unavailable because Phase 3 promoted no nonzero Gaussian level.

## Phase-4c convolutional-seismic implementation

The first two implementation units are complete under the frozen
`docs/PHASE4C_SPEC.md`:

- `guidance/seismic.py`: complete acoustic codebook, expected impedance and
  slowness, known-subsurface non-air renormalization, exact condition overwrite,
  local-surface TWT, linear deposition, zero-phase Ricker convolution, finite
  prediction-window cropping and normalized trace loss;
- `guidance/seismic_sampling.py`: seismic loss injection into the proven
  projected fixed-Euler solver;
- `scripts/stage4/build_seismic_observation.py`: immutable observation builder;
- `scripts/stage4/run_seismic_guidance.py`: EMA/CPU-noise strict-pair runner and
  complete hard geology/acoustic/seismic evaluator;
- `scripts/stage4/audit_seismic_screen.py`: pairing, tensor/source hashes,
  alpha-zero Phase-2a hard regression and complete geology-plus-seismic gate;
- `experiments/stage4_seismic/configs/`: complete distinct label-9 synthetic
  acoustic upper bound, full 320-sample noiseless observation and frozen
  alpha/cap controller order;
- `experiments/stage4_seismic/run_phase4c_seismic_screen.sh`: seed-42 n=1 GPU
  launcher;
- `tests/test_phase4_seismic.py`: 15 codebook, mapping, orientation, timing,
  deposition, boundary, gradient, condition, immutable-asset, controller,
  pairing and alpha-zero gates.

The canonical immutable observation is
`experiments/stage4_seismic/observations/cond_generation_0/distinct_upper_bound_v1_fix2/`.
It has shape `1 x 1 x 64 x 64 x 320`, amplitude range `-0.4716..0.4931`,
193459 valid interfaces and maximum truth TWT `1428.43 ms` within the fixed
`2552 ms` window. The original asset predates subsurface soft-air handling;
`fix1` predates explicit finite-record prediction cropping. Both remain
immutable historical assets and are rejected by current source-hash validation.

Real-checkpoint CPU one-step alpha-zero and alpha-0.25 strict-pair smokes pass.
They load all 411 EMA entries, share the exact initial-noise hash, produce a
finite nonzero seismic gradient, preserve strict pairing and end with zero hard
condition violations. These smokes are not scientific evidence.

The pre-registered seed-42 n=1, 32-step alpha/cap 0.25 GPU strict pair and
audit are complete. Pairing, immutable hashes, EMA, alpha-zero Phase-2a
regression and exact conditions pass, but the complete gate is `0/1`. Hard
seismic loss falls `17.86063 -> 15.24736` and RMSE falls
`0.042262 -> 0.039048`; global accuracy rises by only `0.002354`. Label-9
IoU/precision/recall change from `0.02860/0.06748/0.04728` to
`0.02593/0.07027/0.03947`, predicted volume falls `6283 -> 5038`, and the four
major truth-body recalls are `0.0306/0/0.1121/0`, all no better than baseline.
Of `3737` final paired hard changes, `1438` leave label 9 and only `193` enter.
The total paired change fraction is `1.4256%`; the distinct last-step churn is
`0.2678%`. Both pass their limits, so this is not the pre-registered excessive-
change case and alpha 0.10 is not authorized. Phase 4c is closed without n=4,
extra alpha search or gravity fusion. Read `docs/PHASE4C_REPORT.md`.

## Phase-4d seismic identifiability and posterior selection

Phase 4d is implemented and complete. It reads exactly the 12 existing
Phase-2a alpha-zero baselines from seeds 42/142/242, validates their EMA,
fixed-Euler, checkpoint/input/sample hashes and exact-condition invariants, and
requires seed-42 sample 0 to match the Phase-4c alpha-zero anchor. It generates
no samples. Candidate ranking uses only the immutable Phase-4c hard seismic
loss; complete truth geology is evaluated after ranking. A separate whole-
class truth substitution matrix measures operator sensitivity without entering
the selector.

Both frozen gates fail. No candidate meets the absolute label-9 and
major-component support thresholds; the oracle-best label-9 IoU/recall and
major-body mean recall are only `0.0672/0.0959/0.0963`. The seismic-selected
top candidate is `seed42_sample1` with label-9 IoU/precision/recall
`0.0245/0.0876/0.0329`. The selected top-three means are below ensemble means
for label-9 IoU (`0.02359 < 0.03144`), recall (`0.03702 < 0.05196`) and
major-body mean recall (`0.03905 < 0.05344`). Loss correlations with those
metrics have the wrong positive sign: `+0.552/+0.587/+0.580`.

Replacing all `8955` unconditioned truth label-9 voxels is visible to the
operator (least-visible replacement RMSE `0.017692`), but this target signal is
smaller than the full candidate RMSE of roughly `0.041..0.046` and is dominated
by other interface/timing errors. Phase 4d is closed without a larger pool,
posterior-selection promotion or ranking tuning. See `docs/PHASE4D_REPORT.md`.
The next main direction is a separately authorized geophysics-aware training or
fine-tuning protocol; do not modify training, U-Net or checkpoint without that
explicit authorization.

## Phase-5a no-training acoustic inversion bridge

The user authorized a bounded no-large-training route before any conditioning
adapter. `docs/PHASE5A_SPEC.md` freezes a truth-blind model-based log-impedance
inversion around the exact fixed 12-member Phase-4d prior pool. The builder
never loads truth geology or truth acoustics, and does not load the model or
checkpoint weights. It uses the Phase-4c full-cube noiseless observation, each
prior's fixed slowness/time-depth map, one pre-registered Tikhonov operating
point and exact surface/borehole overwrites. A separate audit opens truth only
after output/source hashes and the completed anti-leakage manifest pass.

All six Phase-5a bridge-eligibility checks pass. Exact nonlinear seismic RMSE,
unconstrained log-impedance RMSE and target-region log-impedance MAE improve in
12/12 members. Ensemble-mean log-impedance RMSE improves
`0.2514293 -> 0.2433104`; target-region MAE improves only
`0.6485778 -> 0.6430638`. Conditions have zero violations and posterior spread
is finite/nonzero.

This is not a geological success. Direct nearest-codebook projection reduces
12-member mean global accuracy `0.5972 -> 0.5163`, label-9 IoU
`0.03144 -> 0.01912` and recall `0.05196 -> 0.02496`. Read
`docs/PHASE5A_REPORT.md`. The only authorized next no-training step is a
single-seed/sample Phase-5b flow bridge with strict alpha-zero pairing and the
existing full hard-label/major-body gate. Stop if continuous loss improves
without hard geology; do not tune the inversion on truth, expand seeds, fuse
gravity, or claim calibrated uncertainty.

## Phase-5b completed negative evidence

`docs/PHASE5B_SPEC.md` freezes the only authorized next screen. The derived
bridge directory is
`experiments/stage5_acoustic_inversion/bridge_observations/cond_generation_0/fixed12_log_impedance_v1/`.
It contains the posterior mean log-impedance target, the complete log-impedance
category table and truth-blind spread confidence. Confidence is active over
193037 unconstrained subsurface voxels, has mean `0.51683`, and is exactly zero
at every hard condition. The builder only reads the Phase-5a pass bit as a stop
gate; no truth metric enters asset construction.

`scripts/stage2/run_property_guidance.py` now has an additive, validated
`phase5b_inversion_property_bridge_v1` external-target mode; historical
Phase-2a/2b behavior remains the default and their tests pass. The formal
launcher runs one alpha-zero/alpha-0.25 seed-42, n=1, 32-step EMA/fixed-Euler
pair and then `scripts/stage5/audit_inversion_property_bridge.py`. The audit
requires bytewise Phase-2a alpha-zero hard regression, lower hard bridge loss,
and the full existing global/per-class/label-9/major-body/topology gate.

The formal CUDA seed-42 n=1, 32-step pair and frozen audit are now complete.
Pairing, all hashes, EMA, exact conditions and bytewise Phase-2a alpha-zero
hard regression pass. Hard inversion-observation loss decreases by only
`0.000683`, while global accuracy decreases by `0.000213`. Guided label-9
IoU/precision/recall are `0.02890/0.06870/0.04750`, effectively unchanged and
far below the absolute gate. Only 4/8 truth-present classes improve. The four
major-body recalls are `0.04266/0.00046/0.12286/0`, giving minimum/mean
`0/0.04149` instead of the required `0.25/0.40`.

The path is active: 24 guidance steps are nonzero, soft loss reaches `0.5484`,
and 1212 hard voxels change. But 194 label-9 voxels leave the class and only
112 enter, with predicted label-9 volume decreasing `6283 -> 6201` versus
truth 8968. This is information/likelihood-to-lithology misalignment, not an
inactive soft-hard mechanism. Read `docs/PHASE5B_REPORT.md`.

Phase 5b is closed. Do not run n=4, more seeds, alternate alpha/confidence,
sharpening/blur or gravity fusion. The next distinct option is a lightweight
learned geophysics-conditioning adapter with the original U-Net/checkpoint
frozen. It needs a new split/anti-leakage/training protocol and explicit user
authorization before any training code is changed.

## Phase-5c completed direct-posterior evidence

The two newly supplied generator-prior seismic inversion papers motivated a
scientifically distinct no-training screen. `docs/PHASE5C_SPEC.md` freezes pCN
over the complete conditional flow's Gaussian initial noise. Each proposal is
generated by the canonical 32-step EMA fixed-Euler path, projected to exact
surface/borehole conditions after every step, hard decoded and scored by the
immutable Phase-4c hard convolutional-seismic likelihood. There is no
inversion-property bridge and no soft likelihood gradient. The proposal
preserves the Gaussian prior, so the Metropolis decision uses only the hard
seismic energy. Sampling and truth audit are separate programs.

The RTX 4090 D mechanism run is computationally feasible: one proposal takes
about 4 seconds after load and peak allocated memory is about 1.66 GB. The
primary chain accepts 7/8 proposals and retains eight unique condition-exact
hard models. The historical seed-42 initial-noise hash and decoded sample
exactly reproduce the Phase-4c alpha-zero baseline.

The minimum-seismic-loss retained state lowers hard loss
`17.860632 -> 17.653818`, raises global accuracy
`0.587366 -> 0.594143` and truth-present mIoU
`0.265249 -> 0.269580`, but label-9 IoU falls
`0.028596 -> 0.025298`, recall falls `0.047279 -> 0.040700`, and
four-major-body mean recall falls `0.041423 -> 0.034587`. This is another
physics-better/target-geology-worse result, now without a soft-hard gradient
bridge. Read `docs/PHASE5C_REPORT.md`.

The incomplete `performance_smoke_v1` directory records the first shape-audit
stop and must not be reused. The fixed successful smoke is
`performance_smoke_v1_fix1`; authoritative primary evidence is under
`experiments/stage5_generator_posterior/{runs,reports}/cond_generation_0/primary_pilot_v1/`.
Do not lengthen the chain, sweep beta/likelihood weight, select on truth or add
seeds for this exposed protocol. Proceed to the separately frozen Phase-6
adapter path.

## Phase-6A residual-adapter oracle mechanism

`docs/PHASE6_ADAPTER_SPEC.md` freezes an external residual-velocity adapter
while leaving the original EMA U-Net, embedding and checkpoint immutable. The
base velocity is evaluated under `no_grad`; AdamW contains only 54,327 adapter
parameters. The adapter receives state, detached base velocity, sparse
conditioning, condition mask, a two-channel 3-D acoustic feature and time. It
uses all-class endpoint CE/Dice in addition to normalized flow loss, and its
output is zero at hard conditions. `adapter_scale=0` takes an explicit branch
that exactly reproduces the paired fixed-Euler baseline.

The completed `oracle_tiny_overfit_v1` run is deliberately a legacy same-case
learnability screen. Its acoustic input is a normalized, full-resolution
truth-derived property volume; four cached flow states are repeated for 80
adapter-only steps. On the RTX 4090 D, the cached training section takes about
2.99 seconds and peaks at roughly 1.97 GB allocated CUDA memory. The original
model tensor hash is identical before/after and all base gradients remain
absent. First/last ten-step mean loss changes `0.205246 -> 0.084735`; cached
endpoint accuracy changes `0.941159 -> 0.971063`.

The strict seed-42 sample changes global accuracy `0.587366 -> 0.745053`,
truth-present mIoU `0.265249 -> 0.473283`, label-9 IoU
`0.028596 -> 0.511914`, precision `0.067484 -> 0.833687`, recall
`0.047279 -> 0.570138`, and four-major-body mean recall
`0.041423 -> 0.507007`. Conditions remain exact. Hard seismic loss worsens
`17.860632 -> 19.828848`, so the result proves categorical adapter capacity,
not the final reciprocal geophysics/geology objective. Read
`docs/PHASE6A_REPORT.md`.

Do not tune or promote this same-case checkpoint. Before formal Phase-6
training, materialize deterministic cases; split by complete geological
history with all sibling well/noise/property variants in one split; exclude
all current demos from formal test; freeze normalization, test manifest and
evaluator hashes. Then run a held-out oracle/degraded-feature pilot before a
truth-blind seismic encoder. Correct/zero/shuffled observation controls and a
small pre-registered physical auxiliary loss are mandatory, and both hard
geology and hard seismic must improve.

## Phase-6P frozen inference-limit result

Before formal training, Phase 6P directly tested the concern that the Phase-4C
guidance cap might simply be too weak. The truth-blind physical runner and the
independent truth auditor are specified in
`docs/PHASE6P_INFERENCE_LIMIT_SPEC.md`; the decision is in
`docs/PHASE6P_REPORT.md`.

The first arm exactly reproduces the historical alpha-zero baseline and the
historical alpha/cap-0.25 Phase-4C sample, then reports all predeclared
alpha=cap levels 0.25/0.5/1/2/4. Hard-seismic RMSE attainment is
7.60%/9.36%/9.88%/6.20%/4.43%. Ratio 1 is physically best, while ratio 4
changes 29.90% of hard voxels, lowers global mIoU `0.19291 -> 0.10201` and
lowers label-9 IoU `0.02860 -> 0.01334`. Increasing the cap beyond 1 therefore
amplifies off-direction updates rather than approaching the observation.

The second arm bypasses the trajectory and applies 200 Adam steps to the
continuous baseline endpoint with exact condition projection. Its best hard
state occurs at step 15 and reaches only 1.69% hard-seismic attainment. Soft
seismic RMSE nevertheless falls from `0.03177` at step 1 to `0.01682` at step
200 while final hard RMSE rises to `0.09732`. This is direct evidence of a
large soft-acoustic/hard-decode relaxation gap.

Both runs keep the base tensor hash unchanged, leave all base gradients absent,
have zero condition violations and distinguish file hashes, raw tensor hashes
and canonical int64 categorical-content hashes. The runner never selects on
truth geology; each frozen output is evaluated by a separate auditor.

The correct interpretation is low physical reachability under the tested
frozen generator/decoder/controller, not a mathematical impossibility result.
Geophysical non-uniqueness remains expected, but it cannot be the primary
explanation for the current notebook visualization because the hard seismic
response has not first reached the observation manifold. Further same-case
guidance-strength tuning is closed. Formal training still requires explicit
user confirmation and must use held-out grouped data, hard-aware categorical
alignment, a pre-registered physical term and correct/zero/shuffled controls.

## Stage-7 observation specificity and structured-hard result

The repository freeze passed at clean `main @ 85d5deb`. Historical D1–D4
runner hashes differed because those files were untracked when the formal runs
were made. Only D1–D4 were rerun under new provenance tags; D5 already matched.
Current source/config/checkpoint hashes match all five authoritative stages and
their observation hashes are present, so D7 records `provenance_verified=true`.

D7 regenerates the D4 BASE trajectory and exactly matches its endpoint hash.
At steps 8/12/16/20/24/28/32 and the common endpoint, all three observation
controls consume the identical state. Mean correct-to-control waveform
residual/raw-gradient/applied-velocity cosine is
`0.92084/0.94507/0.94507`; mean one-step target overlap is `0.97903`.
Clipping/scaling changes direction by less than `6e-12`, so the optional S3
controller experiment was not authorized. The 16-column local JVP basis keeps
effective rank 13 from probability through seismic, but condition numbers are
`1.16e3–4.43e3`; this is ill-conditioned without an additional L3 rank cliff.

Stage 7B implements a generator-free mixed discrete/continuous object space and
a StructuralGeo-native `IntrusionSpec(kind=hemisphere)` space. The population
beam can explore a finite hard-loss barrier, but accepted geology and final
selection use hard observed seismic RMSE only. Proposals are condition checked,
bounded below air, and limited to add/remove/translate/resize/rotate/change
shape/change lithology trust-region moves. Correct, zero, shuffled and
independent wrong-case observations receive the same search budget, and every
final model is cross-evaluated against every observation.

The analytic correct arm selects candidate bodies 4 and 6 without receiving
their truth indices, reaches 100% hard attainment and hidden IoU/recall 1.0.
The three native correct arms reach `45.00%`, `23.85%`, `59.23%` hard
attainment (mean `42.70%`) and rank strictly first when all arms are evaluated
against the same correct field. Hidden IoU/recall are
`0.969/0.996`, `0.914/0.916`, and `0.987/0.993`. All condition violations and
wrong-lithology volumes are zero. This clears the frozen aggregate gate while
retaining the weaker second native replica as an explicit limitation.

The authoritative outputs are:

- D7: `experiments/stage6_inference_causality/runs/five_body_cuboid_v1/d7_observation_specificity_v1/`;
- Stage 7B: `experiments/stage6_inference_causality/reports/stage7_v1_final_v2/`.

Do not promote the discarded `stage7_v1`, `fix1`–`fix5`, or partial
`stage7_v1_final` directories. They are preserved failure/debug evidence.
Training is still unnecessary for this bounded structured family; no adapter,
LoRA, U-Net fine-tuning or generator retraining was started.

## Stage-15F minimal inversion-score calibration

Stage15-F freezes the simplest population-level bridge requested after the
negative Stage15-E 4-cubed test. It uses 128 new StructuralGeo seeds
15180000--15180127, split by complete model into 96 train and 32 validation
cases. Every case receives the same deterministic all-zero initialization,
8-cubed sigmoid parameterization, 300-step Adam inversion and raw seismic MSE.
No Flow, regularization, class balancing, network, feature engineering or
parameter sweep is used. The 128 Stage15-C cases were not replayed because
their per-case geology/seismic tensors were not persisted.

The single calibration variable is the upsampled coarse inversion score. A
64-quantile, Laplace-smoothed empirical lookup maps that score to P(label9).
Only 18/96 training and 5/32 validation models naturally contain label9. The
pooled validation prevalence/AUPRC is `0.0080414/0.0677404`, and truth versus
background mean P9 is `0.0718413/0.0101997`. Across the five positive
validation models, however, only 3/5 have AUPRC above their own prevalence and
3/5 have positive truth/background separation; 58/64 bins are empty because
the deterministic inversion scores contain many ties. The signal is therefore
real in the pooled diagnostic but not uniformly model-stable.

The truth-blind cond_generation_0 probability volume reaches AUPRC `0.0898724`
versus Stage15-B2 `0.0572845` and Stage15-C `0.0453950`. Truth/background mean
P9 is `0.0846406/0.0388203`; raw coarse score Pearson/Spearman is
`0.167968/0.202518`, top-k is `80/129`, and centroid distance is `1.43386`
coarse cells. The calibrated range is only `0.0047967..0.116416`, so the
historical 0.8 diagnostic has zero positives and all truth voxels fall in the
<=0.2 partition. Do not interpret that historical threshold failure as absent
ranking information, and do not pass this map to Flow yet: cross-model
stability is not established. Authoritative artifacts are under
`experiments/stage15_binary_seismic_consensus/{inversion_probability,reports/inversion_score_probability_truth_evaluation_8x8x8_v2}`.

## Stage-15G binary case-relative linear mapper

Stage15-G reuses all frozen Stage15-F 8-cubed inversion scores. It reruns no
seismic forward or inversion and uses no Flow. Deterministic StructuralGeo
replay reconstructs only the subsurface support and the strictly binary target
`raw label9=1 / every other subsurface voxel=0`; tensor hashes must reproduce
the Stage15-F records. The mapper is one `nn.Linear(4,1)` over raw inversion
score, within-case percentile, vertical contrast and normalized depth. Its
soft coarse occupancy target weighted by subsurface voxel count is exactly
equivalent to natural-prevalence fine-voxel binary BCE. There is no class
balancing, hidden layer or parameter sweep.

The first `calibration_n128_8x8x8_v1` attempt completed replay/training but
stopped before held-out application because provenance hashing did not accept
the scalar bias tensor. The only fix stores bias as shape `[1]`; the complete
immutable run is `v2`. It reaches final train BCE `0.0503901`. On the same
frozen 32-model validation split, pooled AUPRC improves from Stage15-F
`0.0677404` to `0.0935104`; truth/background mean P9 is
`0.0874618/0.0118569`. Four of five label9-positive validation models have
AUPRC above their own prevalence and positive truth/background separation,
versus three of five under the global histogram.

On retrospectively exposed cond_generation_0, AUPRC is `0.0915817`, only a
small increase over Stage15-F `0.0898724` but above Stage15-B2 `0.0572845` and
Stage15-C `0.0453950`. Truth/background mean P9 is `0.0829841/0.0386006`; the
range is `0.00219582..0.364265`. The historical 0.8 diagnostic still has no
positive voxels; 1618/8968 truth voxels lie in 0.2--0.8 and 7350/8968 at or
below 0.2. This is improved binary ranking evidence, not yet a high-confidence
Flow-ready probability volume. The learned standardized coefficients are
raw-q `+0.198`, within-case percentile `-0.690`, vertical contrast `+0.127`,
depth `+0.749`; depth contribution means the result must not be attributed to
geophysics alone without a later matched control. Authoritative outputs are
under `experiments/stage15_binary_seismic_consensus/{binary_logistic,reports/binary_inversion_logistic_truth_evaluation_v2}`.

For near-term reporting, an explicitly post-hoc illustrative artifact was
built without new training, forward modeling or inversion. Among the five
label9-positive frozen validation cases, the fixed rule selects the largest
absolute `AUPRC - own prevalence`, yielding seed 15180114. It contains 17,601
label9 voxels; prevalence is `0.0671425`, Stage15-F AUPRC is `0.0965277`, and
Stage15-G AUPRC is `0.1390130`. Truth/background mean P9 separation is
`+0.0344134`. The three-projection figure shows coarse deep-region overlap but
also blocky spatial spread. It is stored under
`reports/binary_logistic_showcase_seed15180114_v1/` and must always be labeled
as a favorable post-hoc validation example, not held-out evidence, robust
generalization, or a Phase1-equivalent success.

## Stage-15 exploratory geophysical-evidence Flow demonstration

For a near-term presentation, Stage15 runs one explicitly post-hoc exploratory
Flow demonstration on the already exposed cond_generation_0 case. Historical
0.8/0.2 diagnostics are not changed. The frozen Stage15-G binary P9 volume is
divided by its subsurface maximum and clipped to `[0,1]`; this is disclosed as
`guidance evidence`, not calibrated probability. The truth-blind runner uses
the Phase1 protocol-v4 controller unchanged at alpha/cap 0.25 for eight fixed
strict pairs with seeds 42/142/242/342/442/542/642/742. It reads no truth and
does not select a sample. The independent retrospective evaluator reports all
eight pairs, then selects maximum label9-IoU improvement for visualization.

All 8/8 pairs improve label9 IoU and recall. Median changes are
IoU `+0.03336`, precision `+0.09134`, recall `+0.03841`; all hard-condition
violations are zero. Global mIoU decreases in all eight pairs, median
`-0.01584`, so this remains target-specific exploratory evidence rather than a
complete geology success. The post-hoc selected seed 142 changes label9
IoU `0.01896 -> 0.06711`, precision `0.07301 -> 0.17071`, recall
`0.02498 -> 0.09958`, and centroid distance `22.13 -> 18.05` fine voxels;
global mIoU changes `0.17453 -> 0.16335`. Full categorical models, all pair
metrics, the selected target comparison and fixed-camera 3-D presentation
figures are in
`experiments/stage15_binary_seismic_consensus/reports/flow_demo_posthoc_seed_screen_n8_v2/`.
Do not cite the selected seed alone as held-out or generalized evidence; cite
the 8/8 target result together with the global-geology tradeoff and post-hoc
selection disclosure.

A follow-up morphology correction removes the nearest-repeated fine target and
the voxelwise hard-Dice term.  The new loss averages the Flow soft label-9
probability within each frozen 8x8x8 cell and matches that coarse occupancy by
binary BCE, leaving within-cell morphology to the frozen generator.  This
eliminates the visibly cuboid target bodies, and again improves label-9 IoU and
recall in 8/8 fixed pairs.  Median deltas are IoU `+0.01996`, precision
`+0.06155`, recall `+0.02503`; median global mIoU delta is `-0.01311`.

It is not Phase1-equivalent.  The selected seed 142 changes IoU
`0.01896 -> 0.05389`, precision `0.07301 -> 0.13979`, recall
`0.02498 -> 0.08062`, and centroid distance `22.13 -> 12.45`, but its hard
label-9 connected-component count changes `45 -> 890` and largest-component
fraction `0.7298 -> 0.1883`.  Thus the block artifact is replaced by excessive
fragmentation: the geophysical map supplies useful coarse localization but not
the fine geometry available to the Phase1 truth-derived oracle.  The
Phase1-style fixed-camera six-panel figure and all metrics are under
`experiments/stage15_binary_seismic_consensus/reports/flow_demo_coarse_occupancy_seed_screen_n8_v3/`;
the truth-blind paired tensors are under sibling
`flow_demo/coarse_occupancy_seed_screen_n8_v2/`.  The earlier `v1` run stopped
before its first pair due to a missing trace-diagnostic compatibility field and
must not be interpreted as scientific output.

## Stage-15H full-trace binary boundary/property result

Stage15-H replaces coarse 3-D occupancy inversion with complete independent
320-sample trace inversion.  It never truncates the vertical forward model:
all 4096 XY traces are solved together using the frozen 25-Hz convolutional
operator, a background binary prior, two fixed linearized log-impedance
refinements and known background/label9 acoustic endpoints.  The inversion
runner is truth-blind and uses no Flow, training, threshold sweep, or lateral
filter.  The continuous score has held-out voxel AUPRC `0.5123866`, versus
Stage15-G `0.0915817`; truth/background means are `0.137269/0.002102`.
Vertical boundary AUPRC is `0.5805124` and XY footprint AUPRC is `1.0`.  The
fixed 0.5 display core has precision/recall/IoU
`0.949953/0.112177/0.111518`.

The successful Flow bridge uses the Phase2 matched multiscale property loss
with a strictly binary normalized codebook: label9 endpoint 1, all other
categories endpoint 0.  The inversion score is used as continuous positive
confidence, not as calibrated occupancy and without thresholding.  Seeds
42/142/242 all improve label9 IoU, precision, recall, global mIoU and global
accuracy with zero condition violations.  Guided IoUs are
`0.23149/0.27563/0.29426`; the figure-selected seed 242 changes IoU
`0.06716 -> 0.29426`, precision `0.18306 -> 0.41375`, recall
`0.09590 -> 0.50468`, centroid distance `9.06 -> 4.27`, and global mIoU
`0.17301 -> 0.19126`.  This is strong synthetic binary inverse-crime evidence,
not a field-generalization result or Phase2a truth-property equivalence.

Authoritative report, all three pairs, Phase1-style fixed-camera figure and
geophysical diagnostics are under
`experiments/stage15_binary_seismic_consensus/reports/binary_trace_property_flow_v2/`.
The first evaluation-output directory stopped during rendering because generic
Phase2 samples were 3-D rather than `[1,1,64,64,64]`; it contains no completed
report.  Property bridge attempts v1/v2 suppressed target because unresolved
background dominated confidence; v3 used unnormalized physical values and was
inert after non-finite gradient protection.  They are retained as negative
engineering evidence.  The successful normalized binary bridge is v4.

## Validation state at handoff

- Phase-1 report regeneration succeeded locally with `.venv/bin/python`.
- Phase-2/Phase-3 focused property and sampler tests: `24 passed`.
- Complete local lightweight suite after Stage-9A implementation and formal
  evidence: `231 passed`,
  with 13 existing Matplotlib/pyparsing deprecation warnings.
- The default system Python lacks PyTorch; use `.venv/bin/python` locally or the
  user's CUDA environment on the remote terminal.
- Static compilation and shell syntax pass; `git diff --check` is part of the
  final handoff check.
- Real 64-cubed observation construction succeeds for identity and Gaussian
  sigma 1/2/4; all outputs are finite and have stable distinct hashes.
- The first identity run tag `seed42_n1_s32_a025_c025` is incomplete and
  invalid: baseline sampling wrote `sample_0.pt`, then hard-observation
  evaluation rejected its unnormalized 3-D shape. The bug is fixed and covered
  by a regression test; use a new `..._fix1` run tag and never modify/reuse the
  partial directory.
- Phase 2a is complete: 12/12 strict pairs and 3/3 diversity gates pass with
  the documented caveats. The Phase-2b anchor, `paired_c100` and `paired_c025`
  pass; `paired_c010` and `paired_c004_overlap` fail. The n=1 screen is complete
  and the final n=4 bracket is c025 3/4 transition plus c010 0/4 failure. The
  c100 fallback is also a 3/4 transition. Phase 2b is closed. Phase-3 n=1 is
  complete: identity passes and every nonzero Gaussian blur fails. Identity
  also passes its seed-42 n=4 confirmation 4/4, while sigma 1 fails 0/4. Phase
  3 is closed without a spatially degraded primary working point. Phase-4a
  forward/operator, observation assets, GPU runner, strict audit and reranking
  comparator are implemented. Its seed-42 n=1 alpha/cap 0.25 and alpha=0.10
  strict pairs both fail the complete gate despite lower hard gravity residual;
  n=4 and further alpha tuning are blocked and Phase 4a is closed. The frozen
  convolutional-seismic upper bound is implemented and its seed-42 n=1
  alpha/cap 0.25 strict pair is complete. It lowers hard seismic residual but
  fails the geological gate `0/1`, with worse label-9 and major-body recovery.
  Phase 4c is closed; alpha 0.10, n=4 and gravity fusion are blocked by the
  frozen protocol. Phase-4b joint guidance also cannot assume a promoted
  Phase-3 Gaussian level. Phase 4d's fixed 12-sample gradient-free seismic
  selector also fails both support and ranking gates, showing that the current
  frozen route has both proposal-support and likelihood-alignment limitations.
  Phase 5a then converts the same seismic input to a fixed-12 uncertain 3-D
  impedance posterior without training. Its continuous bridge gate passes,
  but nearest-codebook hard geology worsens. The formal single-pair Phase-5b
  flow bridge is now complete and fails the frozen hard gate despite lower
  bridge loss. Phase 5c then directly samples the frozen generator posterior
  with hard seismic pCN. Its 8-proposal chain is active and physically better,
  but label-9 and major-body recovery worsen. The tested no-training routes are
  closed. Phase 6A now shows that a tiny learned adapter can strongly recover
  same-case oracle hard geology without changing the base model, but it does
  not yet generalize or improve physical fit. Phase 6P then shows that raising
  inference guidance to 4x or directly fitting the endpoint cannot close more
  than 9.88%/1.69% of the hard-seismic RMSE and progressively damages geology.
  Same-case inference-strength tuning is closed. Stage 7 subsequently restores
  correct-observation specificity and hidden recovery with a low-dimensional
  hard-physics search, including three native replicas. Formal Phase-6B
  training has not started and is not presently required for this bounded
  structured family. Standalone Stage 8A then closed terminally after four
  negative gates. Stage 9A finally tested the frozen Flow prior directly at
  N=1024 on three new cases: all 3072 hard models are unique, but no candidate
  passes the complete geological support gate and correct seismic ranking does
  not pass the complete discrimination gate. The current terminal action is
  `STOP_REASSESS_FROZEN_INFERENCE_ROUTE`; no later stage is authorized by this
  result.

## Stage-13 binary label-9 bridge identifiability stop

Stage 13 reviewed the frozen Stage10/10R, Stage12A and Stage12B decisions plus
the Phase1 probability path, convolutional seismic forward, Phase5a inversion,
acoustic codebook and scalar categorical bridge. Its pre-implementation result
is `PHYSICS_BINARY_BRIDGE_REQUIRES_LEARNED_MAPPER`, with final machine decision
`STOP_PHYSICS_BINARY_BRIDGE_NOT_IDENTIFIABLE`.

The existing normal-incidence post-stack operator observes band-limited
interface reflectivity at slowness-dependent travel times. It does not observe
absolute impedance or homogeneous voxel interiors. The codebook alone does not
specify the surrounding-lithology, geometry, thickness, orientation and
time-depth nuisance distributions required for a direct
`p(feature|label9)/p(feature|non-label9)` likelihood. Local amplitude, envelope,
energy and multiscale context are therefore not defensible label-9 probability
rules. Using the Phase5a scalar inversion would only reopen the prohibited
Stage12B bridge.

Stage13-A bridge construction and truth evaluation were not executed; no
probability/score/uncertainty volume or figure exists. Flow forward count is 0
and Stage13-B is not authorized. The explicit `NOT_EXECUTED` manifests,
identifiability audit, frozen protocol, report and machine decision are under
`experiments/stage13_binary_label9_bridge/`. A future learned
`seismic -> P(label9)` mapper requires a new prospectively frozen protocol and
manual approval; none was implemented.

## Stage-14 GANSim-style geophysical probability-guidance pilot

Stage 14 is a new end-to-end exploratory experiment, explicitly separated from
the historical Stage12B bridge-only decision. It tests whether that stop rule
was overly conservative by passing each existing Stage12B post-seismic
`P(label9)` volume directly through the already validated Phase1 probability-
guidance interface. It does not reopen or modify Stage12B.

The five frozen Stage12A full StructuralGeo cases each use the four source
seeds preregistered in the Stage12B protocol, for 20 strict pairs. The only
arms are `BASELINE` and `GEO_PROB_GUIDED`; each pair shares checkpoint, EMA
policy, source noise, 32-step midpoint fixed-Euler solver, conditions and every
non-alpha setting. Guidance reuses alpha/cap 0.25, cosine temperature,
windowed-sine scheduling, `reference_norm_relative_v2` and
`calibrated_soft_bce_hard_dice_v2`. No bridge, training, AUPRC gate, sweep,
probability preprocessing, truth-based tuning or best-sample selection was
performed. The Flow runner did not load truth; a separate retrospective
evaluator opened frozen truth only after all hard samples were saved.

The primary overall paired median hidden-label9 IoU change is `-0.031508`.
Only `1/5` cases has a positive case-median change: case medians are
`-0.003025`, `-0.093144`, `+0.001386`, `-0.166759` and `-0.031508`.
Across all 20 pairs, median label9 IoU/precision/recall changes are
`-0.030919/-0.001952/-0.155972`; hidden-label9 recall changes `-0.156332`, and
largest-hidden-component recall changes `-0.158806`. Truth-present mIoU changes
`-0.017098`, while global accuracy changes `+0.000198`. All per-step and final
condition violations are zero. No case crosses the preregistered `-0.1`
catastrophic median mIoU/accuracy degradation threshold.

The machine decision is therefore
`GANSIM_STYLE_GEO_GUIDANCE_NOT_SUPPORTED`: the positive-case and overall-
positive primary clauses fail, while the exact-condition and no-catastrophic-
degradation clauses pass. Stop after this decision; do not run more seeds,
guidance settings, probability transformations or training under this
protocol. Authoritative artifacts are under
`experiments/stage14_gansim_style_geo_guidance/`, including the frozen
protocol, truth-blind run manifests/traces, per-arm metrics, paired deltas,
human report and machine decision.

## Stage15 five-body prior/geophysics support experiment

The Stage7-style topology question was rerun through the current frozen Flow
and Stage15 binary-seismic guidance path after closing the case-definition
risks. One current-checkpoint-compatible 64-cubed background contains five
equal, disjoint 640-voxel bodies, all encoded as raw label9. Three bodies are
intersected by hard-conditioned wells; two bodies have exactly zero hard-
condition overlap. The current checkpoint, full hard conditions, Stage15
binary acoustic endpoints and convolutional seismic operator are unchanged.
The case manifest records a passed risk gate. Seeds 42, 142 and 242 were run
for `FLOW_ONLY`, `SEISMIC_GUIDED` and the diagnostic `ORACLE_GUIDED` arm; no
seed selection or parameter sweep was used.

Across the three seeds, Flow-only has median full/hidden IoU
`0.1146/0.0106` and aggregate hidden recall `0.1375`. Stage15 seismic guidance
raises these to `0.1608/0.0679` and hidden recall `0.9016`, but predicts a
median 18,675 label9 voxels for a 3,200-voxel target and merges all ten pairs
among the five truth bodies into the same connected system. Its hard-seismic
RMSE (`0.048009`) is also not lower than Flow-only (`0.044608`). Oracle spatial
guidance reaches full IoU `0.9459`, hidden recall `1.0`, and zero merged truth-
body pairs.

The experiment therefore separates spatial support from topology fidelity.
Current seismic guidance can reveal hidden label9 locations, but the indirect
bridge plus learned prior converts that evidence into an overconnected,
overpredicted morphology. The oracle arm proves that the frozen generator has
representational support for the separated five-body target, so this is not
evidence of an absolute inability to generate the topology. Authoritative
artifacts are under `experiments/stage15_five_body_flow/`.

## Stage17 evidence generalization and coupling attribution

Stage17 is complete under `docs/STAGE17_SPEC.md`. All five preregistered Full
StructuralGeo cases were used. Observation config v2 was frozen first, then
five observations were generated and hashed, then evidence config v1 was
frozen. Observation v1 is retained as a failed engineering run: case04 has 23
enclosed air-coded cells that violate the seismic operator's contiguous-column
support requirement. V2 freezes a deterministic support fill; it does not
change original-nonair retrospective scoring.

Stage17A concludes `EVIDENCE_CASE_SPECIFIC`: 5/5 prevalence-corrected AP-skill
diagonal advantages are positive and the group median is 0.558335. Raw AUPRC
is auxiliary. The runner was truth-blind and did not run Flow.

Stage17B concludes `DFLOW_ENGINE_ACTIVE_BUT_WEAK_HARD_CONTROL`. Formal
three-seed calibration is separate from smoke. LBFGS passes oracle but fails
property (median target-IoU recovery 0.215156 versus the frozen 0.25 gate);
Adam fails both. Model hashes remain unchanged, source updates/objective
reductions are active and condition violations are zero. Since oracle and
property cannot compensate one another, no optimizer is selected and no
Stage17C D-Flow geophysical arm is authorized. This is a negative result for
the frozen Adam/LBFGS protocols, not the entire D-Flow family.

Stage17C therefore runs only `FLOW_ONLY` versus `TRAJECTORY_EVIDENCE`, sharing
the exact Stage17A evidence and source tensor for every pair. Independent
geology case is the primary unit; three seeds are within-case replicates and
the pooled 15 pairs are auxiliary. All 5/5 cases have positive case-median
target-IoU deltas, with cross-case median +0.125051, yielding
`TRAJECTORY_COUPLING_POSITIVE`. Recall and localization improve, but absolute
volume-error worsens in every case, so this is not final-goal success; volume
calibration and topology remain unresolved. See
`experiments/stage17_evidence_coupling_attribution/reports/DEVELOPMENT_REPORT.md`.

Stop after Stage17. Do not automatically begin PGDM/PnP/SGLD, new topology
methods, multiclass inversion or training.


## Stage21v2 dense accurate/pseudo wells (2026-10-16)

The user explicitly authorized SSH review followed by the finite Stage21v2
experiment in `STAGE21V2_VSCODE_CODEX_PROMPT_2026-10-16.md`. This is separate
from the historical45-output Stage21 run. Development and GPU execution used
SSH labrs-codex and repository `.venv` on RTX4090D. HEAD remains
`b780b9ae3650b426400d3258b8e33f10a349bf6c`. Connected GitHub read-only comparison
confirmed main identical (server DNS could not resolve github.com). Existing
uncommitted handoff/Stage19R/ensemble/Stage21 work was preserved. No commit/push.

New source namespace `scripts/stage21v2/` contains common/prepare/run/workflow/
report/audit modules, with `tests/test_stage21v2_dense_virtual_boreholes.py`.
Reuses the unchanged Stage21 initial_noise/sample_native, EMA loader, pure
projected fixed Euler, property mapping/replicate Gaussian, and object matcher.
Checkpoint stays SHA256
`561e94bfda770ec41fc4cbed43436a7e2130eef5dfb7e5d666fcefc0724ff94c`.
The old source registry matched exactly for all reused A baselines; per-prediction
model hashes remain unchanged. Frozen complete source-map digest:
`2b3443bfb60ab976c385e69e6bc5931e7e190ddb97fcee2c7900ea9b52566ffb`; exact file hashes and snapshots are in the run registry/source_snapshot.

Authoritative delivery:
`experiments/stage21v2_dense_virtual_boreholes/run_20261016_v1/`.
See `report/REPORT.md`, PROTOCOL.json, REPO_DIFFERENCES.md in the experiment
root, execution_checklist.csv, TEST_RESULTS.json and DELIVERY_AUDIT.json.
PCG64 seed21601 gives256 stratified holdout columns and3831 nested extra columns.
Sigma1 and the original Phase2a distinct density/susceptibility-label9-contrast
codebook are fixed; all nonair labels participate in nearest-codebook mapping
with nonair codebook channel scales and normalized channel weights. Sigma0
CPU roundtrip is exact, with no codebook collisions. Original hard values/mask
always win, and outside-mask values remain-1. Inference registries contain
prepared conditions only; runtime guards reject truth/full pseudo/evaluation.

Actual progression: case03 seed42 N512 accurate B improved holdout mIoU, then
pseudo C passed the candidate rule. Higher N2048/3840 were therefore not run.
Only N512 was confirmed on seeds142/242, passing the fixed descriptive rule;
case01/02 then ran seed42 at that same density without searching. Total10 new
Flow outputs /320 velocity calls,5 audited original N9 pointers reused; new
inference including initialization 233.98s. D is offline copy-only,
P is full-property classification with original hard conditions restored.
No4096-well Flow, training, fine-tuning, adapter, real forward/inverse physics,
new noise experiments or parameter scans were performed.

Case03 three-seed holdout mIoU medians A/B512/C512/P:
0.148135/0.520606/0.284274/0.274543. C-A paired holdout differences are positive
for all3 seeds; median +0.140021. Original-free paired medians: mIoU +0.122987,
label9 IoU +0.003241. These are medians of paired differences. Seed242 still
has original-free target-IoU delta -0.00008167 and zero holdout target IoU.
Case01/02 seed42 holdout A/B/C/P mIoUs are respectively
0.190461/0.554534/0.367135/0.375864 and
0.177296/0.520625/0.333983/0.373385. Thus category gains replicate relative
to A, but C is not consistently better than P. Case02 holdout target IoU is
C0.396172 versus P0.668373. D equals A exactly on holdout; C-D original-free
paired-mIoU medians are +0.106091/+0.160910/+0.142567 for case03/01/02, so gains
are not explained solely by directly writing the additional conditions.

Object recovery and volume do not establish final-goal success. Sigma1 raw
classification drops every label9 voxel in the case03 original-free domain
(truth1733); C predicts388/165/160 voxels. The originally M9-hidden572-voxel
object has62 truth voxels at N512 well positions, but C matched IoU is0 for
all3 seeds. B matched IoUs0.0704-0.0817 also fail the prior diagnostic gate.
For case02's originally hidden926-voxel object, C matched IoU0.1805 is below
P0.7123. Case02 target absolute relative volume error increases A0.1995 ->
C0.3288. Connected components are object proxies; no reliable general hidden
body restoration, field validity, calibrated uncertainty or untouched TEST
claim follows from the descriptive confirmation rule.

Validation:34 focused/new/legacy tests passed; independent final audit passed.
Original and added hard-condition violations0, matched source/model hashes,
independent fixed-domain/class intersection/union arithmetic, COPY_ONLY holdout
identity, trace budgets and report hashes verified. Three fixed figure types
are delivered for each executed case. The supplemental offline classifier
analysis first encountered a ROOT variable-name collision; fixed to RUN_ROOT
and retained OFFLINE_POSTPROCESS_INITIAL_FAILURE.json, with no new model calls
or changes to frozen predictions. Supplement source and final hash are retained.

Stop here. Prioritize pseudo-well target loss and erroneous hard-conditioning
costs before a separately authorized real-physics bridge. Do not automatically
run higher densities, change sigma/codebook/layout/seeds, train, or extend the
case set. Reused cases/seeds are diagnostics, not independent geological TEST.


## Stage21v2 continuous-evidence offline diagnosis (2026-10-16)

The user authorized only the offline mechanism diagnosis described in
`STAGE21V2_OFFLINE_DIAGNOSTIC_CODEX_PROMPT_2026-10-16.md`. Current main HEAD
matches its review baseline `fbafbcb5893a1c262a247f53e0fee5dd7072b822`.
All analysis executed through SSH labrs-codex using repository `.venv`, CPU
loads and CUDA_VISIBLE_DEVICES="". No Flow checkpoint was opened; zero new
Flow, training, adapter or physical forward/inverse runs; no sigma/weight/
threshold/budget scan. The sealed old Stage21v2 run, old scripts/stage21v2
namespace and original tests were not modified.

New independent implementation: `scripts/stage21v2_diagnostics/`;
new delivery: `experiments/stage21v2_evidence_diagnostics/run_20261016_v1/`.
See REPORT.md, three figures in figures/, score_summary.csv,
margin_distributions.csv, true9_assignments.csv, object_diagnostics.csv,
CONFIG.json, INPUT_ASSETS.json, REPO_STATE.json, COMMAND.txt and completion/
source/artifact hashes. Original smoothed properties were available and used
without reconstruction. Registered asset file/content hashes, primary/auxiliary
target counts (case03 1733/103, case01 118/11, case02 16046/977), full argmin
versus pseudo_raw, and full26-connected object IDs/sizes all matched. No old
runner was imported. Input hashes remained unchanged after analysis.

Server sklearn is unavailable. The diagnostic implements the strictly equivalent
binary unweighted non-interpolated AP, accumulating complete equal-score groups
and summing recall increments times group precision. Five necessary small tests
passed: constant AP=prevalence, a tied-positive example, documented sklearn
example, all15 positive-bearing binary labelings of a four-point tied fixture,
and including all boundary ties in the fixed5% set. No dependency installation
or historical test suite was needed. Fixedtop5% is retrospective coverage,
not a chosen classifier threshold or proposed pseudo-well rule.

This extends the prior interpretation: the continuous proxy itself retains
strong readable localization in the two hard-classification failure cases.
Case03 E_original_free prevalence0.0115254; margin/susceptibility AP
0.810281/0.836303, AP_skill0.808069/0.834394. Both fixedtop5% sets select7519
voxels, precision0.230483 and recall1 over1733 true9 voxels. The original
well- and M9-zero-hit572-voxel object1 is covered572/572 by both scores, and
its holdout25/25 voxels also are covered. Yet object margin median -0.231197,
max -0.067478: allnegative, not a tie. Its full q9 median/p90/max are
0.306791/0.395538/0.475070, while susceptibility median/max0.032507/0.048630
remain below the pure target codebook0.1 but above typical background.
Continuous relative localization survives local mixture dilution; the pure
nearest-codebook hard interface selects no9 in either registered domain.

Case01 E_original_free prevalence0.000782680; margin/susceptibility AP
0.889537/0.875028, AP_skill0.889450/0.874931. Both fixedtop5% sets cover118/118,
but precision is only0.015652 because the fixed5% budget is broad for this
rare target. It has no zero-well-hit major object under fullsize>=20: the
three zero-hit components have1/6/3 voxels and all are fully covered in the
primary domain. Their holdout domain counts are0, so coverage is NA there;
its holdout11 positives do not validate those components. The110-voxel major
object4 has2 original well hits. Do not claim hidden-major restoration from
case01 or treat tiny objects as a strong generalization result.

Case02 global primary AP is margin0.967673/susceptibility0.967411, but its
original-zero-well32- and41-voxel components have0% top5% coverage for both
scores. The926-voxel object has69.44%/70.09%, and129-voxel object40.31%/38.76%.
Thus global ranking can coexist with missed hidden small components. Full
q9 is truth-derived explanation only, not inference input or a probability;
q9=0.5 is not a universal codebook decision boundary.

Research decision: a separately preregistered, strictly paired comparison of
hard pseudo-well labels versus conditions retaining continuous/uncertain
attributes is now informative, using identical evidence and original hard
conditions and retaining accurate-well/direct-classification comparisons.
This diagnosis does not implement that interface, select a new threshold,
prove soft conditioning works, prove field inversion validity, show every
hidden object's information survives, or justify retraining Flow. Stop here.


## Stage22 blur-aware TV restoration and sparse target hard bridge (2026-10-16)

User authorized the single case03 finite protocol in
`STAGE22_BLUR_AWARE_BRIDGE_PROMPT_2026-10-16.md`, with at most3 new Flow
outputs and no training, adapter, real physics, SDF or parameter scan. Initial
HEAD/main matches reference `9b46d1af2d08c32ce256ad18afab716b91c92ada`;
initial worktree contained only the untracked user prompt. All computations
used SSH labrs-codex and repository `.venv`. Old runs, Stage21v2 scripts and
original tests remain unchanged; no commit/push was performed.

New namespace `scripts/stage22/` contains common/solver/prepare/run/workflow/
report and focused test_stage22.py. Delivery:
`experiments/stage22_blur_aware_bridge/run_20261016_v1/` (REPORT.md,
PROTOCOL/METHOD_FREEZE/PREPROCESSING_RECEIPT/OFFLINE_GATE/EXECUTION,
solver_history, sample/object/predicted-object tables, two figures,
COMMANDS/TEST_RESULTS/FINAL_CHECKS, prediction receipts and artifact hashes).
Frozen source-map digest: `f7f55a2b1f820be4d7840c6fcc5e6e8cdc2ed645b9a49c04f7420dcbf28b6820`. Legacy source check verified all36
files in the old Stage21v2 registered source map; old sampler/model behavior
was not modified.

Preprocessing was blind to full truth, object IDs and q9. Its file-open guard
rejected truth/object/diagnostic tensors and checkpoints. It used saved two-
channel observations, fixed full codebook, scales and original conditions only.
Single CPU float64 projected accelerated-gradient solve used original sigma1
radius3 separable/replicate float32 kernel coefficients, exact autograd transpose,
isotropic forward-difference TVmean with exterior differences0 and sqrt epsilon
1e-8, fixed coefficient1e-4, full-codebook bounds including air and exact
unblurred original-condition property projection. Method/source freeze was at
12:53:28 UTC, final field/conditions sealed12:54:08 UTC, first full truth
read for evaluation12:56:12 UTC. No iterate selected by truth.

Converged after550 accepted iterations /39.52s: objective0.000166326000 ->
0.00000365432481, dataMSE1.13648481e-7, TVmean0.0354067633; final50 relative
changes<1e-6, hard and bound errors0. Final fixed classifier argmin9/d9<=.01/
margin>=.05 accepted79 target voxels in36 columns of the old512 layout outside
original conditions; these are partial target depths, not512 complete wells.
Offline gate:79TP,0FP, precision1;36 correct additions inside original-M9-
unhit572-voxel object1. Candidate conditions were sealed before evaluation and
never modified after truth was read. No non-target labels were added.

Exactly one new seed42 Flow /32 forward calls ran,23.77s including initialization.
Actual oldA CPU noise file/tensor was verified and loaded (canonical noise
SHA2566c4f6450e0e2eb2103d190b9747a1af6bc8e806c00a973a54511ea7bdd1ae81d).
EMA411/raw frozen embedding, FP32/noTF32/deterministic CUBLAS,32 midpoint-time
Euler and original sample_native are unchanged; model hash before/after equals
oldA. Original/additional decoded condition violations0.

The accurate sparse conditions elicited off-condition response, but NOT a
passing hidden-object restoration. On object1's common536-voxel subset outside
the36 new conditions, target TP rises A0 -> new392 (coverage0.731343); COPY_ONLY
stays0. Any prediction covers428/572 of object1 (0.748251). However, prediction
component2 has27758 original-free voxels,26581FP and significant overlaps with
both truth objects. One-to-one matching assigns it to object2, leaving object1
unmatched (matched IoU/coverage0); this is not absence of target localization.
Object2 matched IoU is only0.026589. Overall original-free newFlow:
TP1177/FP30711/FN556, precision0.036910, recall0.679169, IoU0.036278,
fixed-classmIoU0.106147, predicted target31888 versus truth1733.

Direct deconvolved full classification with original conditions restored is
much better: original-free TP1679/FP2/FN54, precision0.998810, recall0.968840,
IoU0.967723, mIoU0.662712; object1 matched IoU0.991289 and off-added TP533/536.
The newFlow is worse than direct preprocessing; do not report only gains over
A (A original-free IoU0.006382, precision0.008743, recall0.023081).

Seed42 passes original-free recall/precision comparison to A, but fails the
object1 matching gate and holdout mIoU guard: A0.148135 -> new0.131576, drop
0.016559 exceeds allowed0.01. Stop reason is
`seed42_research_gate_failed_no_confirmation`. Seeds142/242 were NOT run; do
not imply confirmation success/failure from unexecuted samples. OldA/B512/C512
are reference pointers only; B is not a matched-information-budget upper bound.
New and oldC masks/label contents differ, so these controls do not isolate the
causal effect of deconvolution versus condition count or positive-only evidence.

Validation:6 necessary new/basic sampler tests passed (1.86s), frozen method/
source identity, original conditions, target-only/noholdout additions, noise/
model identity, COPY_ONLY holdout equality, actual32-call budget and stop rule
were rechecked. Old assets/sources were preserved. No GPU preprocessing,
additional smoke, new seed/case, parameter change or retry occurred.

Decision: stop the current native sparse-target-hard-projection configuration;
retain the positive ideal-proxy restoration result and place any separately
registered next study at the condition-interface question, not more wrong/
unbalanced hard conditions or TV/sigma tuning. No new interface is implemented
here; success of soft conditioning, field validity or a need to retrain Flow
has not been demonstrated. This is a known-kernel/no-extra-noise/artificial-
strong-contrast reused-case result, not PGI replication or a novelty claim.


## Stage23 complete restored pseudo-wells (2026-10-17)

User authorized STAGE23_FULL_PSEUDO_WELLS_PROMPT_2026-10-17.md on main
8ac499a2b5fc0cacf205066f92b382625c314da4. New scripts/stage23 reuses saved
Stage22 classification, fixed Stage21v2 codebook/classifier and total512 layout.
No target-only acceptance filter: all non-original well depths receive raw0..13
predicted labels; original hard conditions take priority. CPU single classifier
consistency recomputation matched saved labels. Added18555 voxels in503 columns;
mask equals old B512/C512, holdout additions0. Conditions were sealed before
Flow/truth evaluation, no scientific precision gate. No restoration/training/
adapter/guidance/scan/trimming or other cases/seeds. Historical sources/results
and checkpoint were not modified. Development used SSH labrs-codex and .venv.

Delivery experiments/stage23_full_pseudo_wells/run_20261017_v2 contains Chinese
REPORT and reviewed conclusions, comparison/class/confusion/object CSVs, one
fixed Z15 fullXY six-panel categorical + binary-target figure, protocol/source/
condition/input seals, unique prediction and receipts, commands and tests.
Exactly1 new Flow output /32 actual midpoint-Euler velocity calls from paired
A_seed42 saved CPU noise; EMA411/raw embedding, FP32/noTF32/deterministic, all
parameters frozen, hard original/added condition violations0. No baseline reruns.

Conversion on same added_mask improves all-label errors7239->1402 (accuracy
0.609863->0.924441), non-target truth errors7028->1395; target TP0->204,
FN211->7 with1FP. Holdout C->new: recall0.029126->0.339806, IoU0.022059->
0.130112, fixed-class mIoU0.288156->0.503877. Object1 common subset is exactly
truthID1 & E_original_free & ~Stage23_new_mask, denominator510 for EVERY arm:
A/C/COPY0TP, B80, Stage22 366, direct507, Stage23 144 (coverage0.282353).
All preset modeling comparisons are supported for this sole sample, not repeats.

Original-free newFlow TP715/FP2579/FN1018, precision0.217061, recall0.412579,
IoU0.165816, mIoU0.454242; target3294 versus truth1733 (1.90075x). Stage22
had30711FP/18.40046x and1 cross-object merge; new has0 such merges but object1
is split across3 significant components. Object1 any coverage206/572, matched
112/572 and IoU0.122137; object2 any509/1161, matched455/1161, IoU0.158041.
Largest new component2173 original-free voxels/1718FP versus Stage22
27758/26581FP. Range control improves over Stage22 but sacrifices coverage
(relative to its366/510 shared off-condition TP) and remains inaccurate. Direct
restoration IoU0.967723/mIoU0.662712 remains far better. B true wells have
original-free IoU0.160974/mIoU0.492050 and holdout IoU0.140704, so do not
claim uniform superiority to accurate wells or to direct full-field conversion.

Engineering evidence is explicit: retained v1 guard blocked sympy.geometry
during dependency import,0 forward. v2 sampled all32 calls and saved tensor/
trace, then relative output path broke receipt serialization. CPU-only recovery
verified saved32 traces, exact times/conditions and prediction; no resampling.
failed_save_receipt and recovery source/log remain. After-model hash was recovered
from the before hash because equality/entry checks had already passed before
the save failure; receipt labels that evidence, not a fresh measurement. Timing
is approximate from trace mtime minus stored start time. Final code resolves CLI
paths; executed source snapshot and post-run three-file code diff are retained.
Five focused final CPU tests passed (14.989s); no GPU smoke or further forward.

Three judgments: conversion improved; prescribed single-sample off-condition
modeling benefit supported; relative range control improved but absolute FP,
volume bias, missed geometry and splitting remain substantial. Stop here after
one case/seed/output; user/MCYR decide any new protocol. Known-kernel/no-extra-
noise/artificial-proxy/reused-case evidence does not establish real geophysical
inversion, cross-model generalization, or publication novelty. No commit/push.


## Stage23 read-only scientific visualization (2026-10-17)

User authorized STAGE23_VISUALIZATION_PROMPT_2026-10-17.md. Current HEAD
90ca753f0413f48aa57e989ba41f327ae98573b7 matches its reference. Existing
unrelated geophysics-results deletions were preserved. New standalone script
scripts/reports/build_stage23_conference_figures.py reads sealed v2 tensors
through EXECUTION/registries/receipts, checking file/canonical tensor hashes,
case03/seed42/64cube and CSV-to-EXECUTION metrics. No sealed stage21v2/22/23
source or output was modified. All59 Stage23 sealed artifacts retain their hashes.

Delivery: experiments/stage23_full_pseudo_wells/reports/conference_paper_v1/.
fig01_full_geology_comparison and fig02_target_body_comparison are180x138mm,
2x3 GT/A/Stage23 grids; figS01_bridge_evidence is180x89mm. Each has600dpi PNG
and PDF with embedded TrueType/vector titles, labels, metrics and legend, plus
raster3D (not wholly vector geometry). Twelve1500x1360 original panel PNGs,
Chinese/English CAPTIONS, figure_metrics/condition/outside-object CSVs, rendering
config, input identities, redraw command, code snapshot and QC are retained.
Actual Stage15 reference PNGs and local Synthetic Geology PDF pp12-13/Figs7-8
were viewed. Reuse only pure exact voxel-face/palette helpers; no old ghost,
well layout, per-panel auto-bbox or white-margin trimming.

XYZ is array-axis order, VTK ImageData shape+1 and Fortran-order cell data; unit
spacing and fixed0..64 cube boundaries. All models share camera/focal/parallel
scale/lighting and raw0..13 colours; air hidden. Geology upper row is full,
lower row removes the same X>=32/Y<32 quadrant for display only. Target rows
retain ALL raw9 components without cutaway/ghost/ROI/filter/smoothing/repair;
shared isometric and orthographic along-Y views. Original wells/surface remain
in model conditioning but wells are omitted visually. Every fixed cube corner
projects inside camera, raw image borders are white and target represented
voxel counts equal full input counts, including FP fragments.

Figures retain positive conversion92.44% versus60.99%, holdout mIoU0.5039
versus A0.1481, and shared off-condition hidden-object144/510 versus0/510.
Explicitly label that as object-subset coverage, not IoU. Supplement preserves
true-well B and direct full-field IoU0.9677/mIoU0.6627; full-field direct and
sampled pseudo-well evidence are not equal information budgets. Missing target
segments, fragmentation and FP remain visible. One case/seed ideal surrogate
with known kernel is not field inversion, robustness or generalization proof.
All three PNGs opened and inspected; common vector axis annotations adjusted
into empty frame margin. PDF embedded fonts and extractable text checked.
New model forwards0, training0, deconvolution0, condition generation0. Only
CPU tensor reads/plots and off-screen graphics rendering were performed.
No commit/push, new scientific sample or follow-up experiment.


## Stage23 visualization revision: image-led reference plates (2026-10-17)

User replaced prior visualization layout through
STAGE23_VISUALIZATION_REVISION_PROMPT_2026-10-17.md. New independent script
scripts/reports/build_stage23_reference_figures.py; output
experiments/stage23_full_pseudo_wells/reports/conference_paper_v2/.
Preserve v1, old shared style, scientific source/output and unrelated changes.

Only two main plates: full opaque geology2x3 (whole volume above, common
cutaway_mask fraction0.52 below) and complete orange raw9 target1x3. Each
has600dpi PNG/PDF and a clean variant containing only panel letters. Normal
versions have only Ground truth/Frozen Flow/Ours, panel letters and one shared
discrete raw-label legend for geology. No metrics, conclusions, long titles,
annotations/arrows, target side-view row, evidence supplement, or diagram.
External bilingual CAPTIONS retains method, saved metrics and limitations.

Reuses Stage15 exact cell threshold/extract_surface, LABEL_COLORS, flat
lighting ambient0.34/diffuse0.66/specular0 and SSAA. Shared direction
[1.72,-1.35,1.38]. Per-plate cameras equal across ALL columns/rows:
geology focal[32,32,26.5], scale51.8; target focal[32,32,29.44], scale54.5;
parallel projection,1600x1440 raw render. Geology framing lowered uniformly
to reduce empty air, with every full-model nonair surface vertex verified
inside viewport. Target retains full0..64 box, all predicted components and
the same9 original wells as thin teal radius0.12 tubes. No ghost, ROI, target
cutaway, component filtering, smoothing, repairs, per-method auto-fit or
independent crop. Full target counts retained: GT1736/A4578/Stage23 3297.

Actual two Stage15 PNGs and local Synthetic Geology PDF pp12-13 were opened.
All four final variants opened; final plates were also inspected alongside
Stage15 three-panel and paper Fig7 references in a temporary QA contact
sheet. Final PDFs checked for embedded/extractable TrueType text and absence
of scientific metric text; clean variants lack column titles/legend.
v1 manifest and all59 sealed Stage23 artifacts unchanged; inputs verified
by file/canonical tensor hashes and registries. Config records camera,
colours, light, geometry validation and visual QC. No model/checkpoint load,
forward, training, deconvolution, new conditions or scientific changes.


## Stage24: approved visualization standard and two additional cases (2026-10-17)

User authorized STAGE24_MULTI_CASE_EXECUTION_PROMPT_2026-10-17.md. HEAD
90ca753f0413f48aa57e989ba41f327ae98573b7; previous handoff edits, plotting
outputs and unrelated geophysics-results deletions preserved. New scripts/stage24
and independent reports/build_stage24_approved_figures.py; no old stage21v2/22/23
source edits, no commit/push.

Actual accepted Stage23 conference_paper_v2 entry,4 local dependency sources,
original PNG/PDF/config/input identities/commands are frozen under
visualization_standards/geology_approved_v1 (MANIFEST, source_snapshot,
approved_samples, renderer.py, SEAL). Exact original function ASTs use frozen
style helpers; no new visual design. Four case03 PNG normal/clean variants
are pixel-exact in both initial and final regressions. Final case03 main
figures are original-byte copies;9 panels come from necessary style regression.

Delivery experiments/stage24_multi_case/run_20261017_v1 has two new cases,
three-case CSVs/PPT_COMPARISON, Chinese REPORT/REVIEWED_FINDINGS, external
bilingual CAPTIONS, PPT_ASSET_INDEX, commands, source snapshots/receipts and
FINAL_CHECKS. reports/<case03,case02,case01> has6 normal main plates,6 clean
variants, all600dpi PNG/PDF and27 independent panels; all main new plates
opened and checked next to approved case03 reference. Camera/scale, raw
colours, light, opaque voxel faces, complete target components, original9
well lines and full boxes unchanged; no ghost/ROI/filter/smoothing/repair.

Exactly2 CPU restoration solves using unchanged Stage22 solver and original
registered sigma1 properties: case01 converged617 iterations/40.4675s,
case02 601/43.7425s; same kernel/TV/scales/bounds/original lookup constraints.
No truth in preparation; fields and complete512 pseudo-wells sealed before
Flow/evaluation. Added18520/25476 voxels, masks equal old B512/C512, holdout
additions0, original hard priority and unknown placeholders preserved.
Two serial seed42 native Flow outputs only,32 calls each /64 total,23.29s/
22.89s, actual oldA saved CPU noise, EMA411/raw frozen embedding, FP32/no
TF32/autocast/deterministic, frozen model identity and hard conditions exact.
No scientific retries, extra seed/case/training/scans. Stop at budget.

case01 added accuracy C0.800918->0.968251, target FN5->0/FP0. Holdout
IoU0.014286->0.160000, recall0.090909->0.727273, mIoU0.367135->0.519594.
Primary TP53/FP468/FN65, IoU0.090444/P0.101727/R0.449153, volume ratio
4.415254. Only major object4 full110 has2 original condition/well voxels;
matched IoU0.177700, any coverage53/108 original-free. Zero-condition
objects1/2/3 (1/6/3 voxels) all remain0TP outside new conditions. Not a
second zero-well major-body success. B holdout IoU0.176471/mIoU0.554534
remains better; direct full classification primary IoU1/mIoU0.799359.

case02 added accuracy C0.634283->0.930209, target FN687->2/FP1->0.
Holdout IoU0.396172->0.464922, recall0.699079->0.854657, mIoU0.333983->
0.482250. Primary TP13607/FP12197/FN2439, P0.527321/R0.848000/IoU0.481783.
FP rises from C10774 and volume ratio1.328805->1.608127: scope control does
not improve over C. All16 truth IDs/11 zero-hard-condition objects retained.
Four zero-condition major IDs3/11/14/16 full32/926/129/41: common off-new-mask
TP9/28,606/824,85/105,18/31 versus A/COPY0,88,75,3. Matching IoU0.189655,
0.002160,0.007519,0 despite any coverage13/32,708/926,109/129,28/41. Largest
prediction full25470/primary25433 with11891FP overlaps8 truth objects:
localization improves but severe merging remains. B primary/holdout IoU
0.496289/0.484266 and holdout mIoU0.520625 remain better. Direct full-field
classification primary IoU0.998069/mIoU0.735885 remains far better and has
a different information budget. Original-free truth target volume16046.

Two visualization-only retries retained: initial AST wrapper missed itertools;
case02's valid projected y0.996757 triggered conservative0.995 margin.
Only validation assertions relaxed to actual0..1 bounds;2pixel white-border
checks retained, no camera/geometry change. New stage24 plot-only source diff
is explicit; scientific executed snapshots remain intact. Final approved PNG
regression still pixel-exact. All necessary input/hash/condition/noise/model/
32-call/COPY/metric/object/figure checks pass;4 focused CPU tests passed.
Known-kernel/no-extra-noise/artificial-proxy/reused-case single seeds are
showcase diagnostics, not independent geological TEST or proven generalization.


## 2026-10-17 — Offline three-case presentation summary

User-authorized analysis/plotting only; no new scientific stage. CPU script
`scripts/reports/build_pseudo_well_three_case_summary.py` executed via SSH in
repo .venv; immutable outputs `reports/pseudo_well_three_case_summary_v1/`.
Six sealed seed42 predictions: original A_9 for each case, Stage24 complete
restored pseudo-well Flow for case01/02 and Stage23 v2 Flow for case03,
identified through execution/evaluation registries and hash-checked receipts.
No model/checkpoint load or GPU task; training/Flow/restoration counts all0.

Delivered 6 fixed XYZ32 slice figures,2 domain tables,1 four-panel bar chart,
each300dpi PNG/PDF;18-row/3-domain metrics,per-class IoU counts,6-row global
and holdout CSVs,manifest,external captions,README and historical reference
source index. Reused original tab20/BoundaryNorm and take(axis).T/lower/nearest.
Error slices use complete plane including air; global stats use truth!=-1.
Original E_holdout/E_original_free masks/classes unchanged. All12 historical
comparisons exactly match Stage24 CSV (max abs difference0),39 input files
hash unchanged;9 PNGs visually inspected,9 PNG/PDF pairs and300dpi checked.

Global mIoU A->proposed: case01 .203683->.529309,case02 .180462->.508096,
case03 .164935->.499198. Global target IoU .002819->.093537,
.111785->.482461,.006857->.166396. Proposed global target volume ratios
4.358333,1.606728,1.899194: extent errors remain. Holdout target IoU
.003190->.160000,.125519->.464922,.008065->.130112. Global includes
conditions, original-free includes new wells; holdout has no new direct
condition writes but its properties enter global proxy, not independent TEST.
Single seed/reused cases/artificial proxy/known blur/no added noise limits
remain. Historical experiments/source snapshots/approved3D outputs untouched.
