# AGENTS.md — post-Stage-7 project instructions

## Mission

The project goal is to generate a posterior ensemble of 3-D hard-label geological models that:

1. satisfies surface and sparse-borehole hard conditions exactly;
2. is consistent with global geophysical observations;
3. remains plausible under the geological prior learned by the frozen flow model;
4. improves concealed/deep geological geometry where wells do not constrain it;
5. retains interpretable uncertainty rather than returning only a single low-loss model.

The active post-Stage-7 development path is **Flow-Constrained Structured Geophysical Posterior Inference**.

Direct free-voxel/high-dimensional continuous seismic-gradient guidance is a historical baseline and diagnostic, not the default new method.

## Instruction precedence

For work under `project/geodata-3d-conditional/`, use this post-Stage-7 authority order:

1. this project-level `AGENTS.md`;
2. `docs/DEVELOPMENT_HANDOFF.md`;
3. authoritative Stage 7 final report/summary;
4. authoritative D7 observation-specificity report/verdict;
5. `docs/RESEARCH_GOAL.md`;
6. `docs/AGENTS.md` for stable repository conventions;
7. older phase specifications/reports as historical evidence.

If `docs/NEXT_CONVERSATION_PROMPT.md` or an older phase document conflicts with the post-Stage-7 handoff/report, treat the older document as historical.

Do not rewrite historical reports to make them agree with the new route.

## Required reading before editing

Before changing code:
- inspect `git status --short`, current branch, and current commit;
- read `docs/DEVELOPMENT_HANDOFF.md`;
- read `docs/RESEARCH_GOAL.md`;
- read authoritative Stage 7 report/summary;
- read authoritative D7 report/verdict;
- inspect Stage 7 structured-hard search code/tests;
- inspect checkpoint loader, hard decoder, property mapping, hard-condition projection, geophysical forward, evaluation, and provenance utilities;
- inspect existing relevant experiment outputs.

Preserve unrelated user changes.
Never reset/clean/reorganize the repository merely to simplify the task.

## Frozen scientific state

Treat these findings as established unless a new explicitly authorized experiment tests them:

- forward/observation closure is valid;
- decoder/property mapping closure is valid;
- gradients were numerically verified;
- the old controller implements the intended direction;
- hard-condition projection was not the dominant failure;
- correct/control seismic residual directions are already highly similar at frozen-flow states;
- VJP adds a smaller alignment effect;
- controller normalization/capping does not materially erase observation-specific direction;
- hard categorical transitions add a smaller many-to-one collapse;
- do not claim an abrupt L3 Jacobian-rank cliff from the existing D7 basis;
- structured hard-geophysics inference restored observation specificity in the bounded Stage 7 family;
- no training/fine-tuning/LoRA is currently authorized;
- do not perform broad controller-strength sweeps.

The working hypothesis is that geophysics should constrain low-dimensional geological objects/events or proposals, while the frozen flow supplies the geological prior.

## Data-role firewall

Every asset must be assigned explicit roles.

### CHECKPOINT — inference-visible
Purpose:
- frozen conditional geological prior.

Rules:
- load with established EMA/checkpoint policy;
- do not train/alter;
- record hash in authoritative runs.

### CONDITION — inference-visible
Includes:
- borehole observations;
- surface observations;
- condition masks.

Rules:
- preserve exactly;
- any violation fails the run;
- audit that hidden truth has not leaked outside intended observed voxels.

### GEOPHYSICAL_OBSERVATION — inference-visible
Purpose:
- hard observation likelihood/selection.

May be:
- synthetic observation generated once from truth for benchmark experiments;
- later, measured field data.

Rules:
- freeze/hash before inference;
- proposal acceptance/final selection may use it.

### TRUTH — not inference-visible
Purpose only:
- synthetic observation construction;
- retrospective evaluation after inference selection;
- fixed code-correctness fixtures.

Forbidden during inference:
- hidden labels;
- concealed-body count;
- body center/size/orientation;
- truth-derived ROI;
- truth candidate indices;
- proposal construction;
- acceptance/ranking/stopping;
- tuning;
- model/seed selection.

Prefer APIs where the search engine cannot receive a truth tensor.

## Hard-geophysics acceptance rule

Every authoritative structured proposal must follow:

    structured geological parameters
        -> hard categorical geology
        -> exact condition projection
        -> hard petrophysical mapping
        -> hard geophysical forward
        -> observation mismatch / likelihood
        -> truth-blind accept/select/resample

A lower soft expected-property loss is not sufficient evidence.
Retrospective truth metrics must not rank proposals.

## Controlled-model policy

Simple synthetic/StructuredGeo cases remain required as controlled tests, but are not the final target.

Use the ladder:

1. existing five-body regression fixture;
2. minimal simple StructuralGeo-native continuous-parameter / unknown-count cases;
3. existing processed project truth + corresponding borehole conditions + frozen checkpoint;
4. later held-out independent cases.

Their purpose is to isolate:
- parameterization;
- search correctness;
- control specificity;
- proposal operations;
- hard-forward closure;
- unknown-count handling.

Do not spend unlimited cycles optimizing toy cases after their gate passes.

## Existing project assets

When a trained checkpoint, processed truth model, and matching borehole/condition model exist:

- use checkpoint immediately as a frozen prior;
- use borehole/condition model as inference-visible hard conditioning;
- use processed truth as the primary realistic synthetic benchmark truth;
- generate/freeze geophysical observations from truth before inference;
- hide concealed truth from search;
- use truth only retrospectively.

Determine whether the truth case was seen during checkpoint training. If split provenance is unknown, do not claim held-out generalization.

Do not invent replacement assets unless a controlled test specifically requires them.

## Active development scope: Stage 8

### Stage 8A
Validate continuous structured hard-geophysics inference without relying on a finite truth-compatible candidate library.

Minimum capabilities:
- continuous body position;
- continuous size/scale;
- orientation where applicable;
- unknown body count in bounded `0..K`;
- birth/death;
- translate;
- resize;
- rotate where supported;
- hard forward and truth-blind hard observation selection;
- correct/zero/shuffled/wrong-case controls.

Use simple five-body/StructuralGeo cases.

### Stage 8B
Integrate:
- frozen checkpoint;
- processed truth-derived synthetic observation;
- corresponding borehole/condition model.

Required arms:
- `FLOW_ONLY`;
- `STRUCTURED_ONLY`;
- `FLOW_PLUS_STRUCTURED`;
- correct/zero/shuffled/wrong-case controls for the combined method.

`FLOW_PLUS_STRUCTURED` is the target method.

Central question:

> Does the combined method recover deep information absent from sparse conditions using geophysics, while the frozen flow prior restricts geophysical non-uniqueness and preserves geological plausibility?

## Search-space policy

Do not optimize a full free `C x 64 x 64 x 64` continuous state as the primary new method.

Use meaningful structured geological parameters/events.

Allowed proposal-domain information:
- unconditioned voxels;
- domain geometry;
- depth;
- acquisition support;
- pre-registered geological bounds;
- later truth-blind residual/sensitivity features.

Do not center/crop the search around hidden truth.

## Control policy

Authoritative experiments must include appropriate controls:
- correct observation;
- zero observation;
- shuffled/spatially permuted observation;
- wrong-case observation when available.

Use identical:
- code;
- starting-state policy;
- proposal count;
- forward-call budget;
- seed policy;
- stopping rule;
- bounds.

Cross-evaluate selected models against the common correct observation.

## Evaluation policy

Keep inference-visible and retrospective metrics separate in code/output.

### Inference-visible
May drive search:
- hard geophysical RMSE/NLL;
- exact-condition audit;
- pre-registered truth-independent structural regularization;
- proposal/search statistics.

### Retrospective
May not drive search:
- hard-label accuracy;
- per-class IoU;
- target IoU/precision/recall;
- body precision/recall;
- center/size/shape errors;
- truth connected components;
- wrong-lithology substitution relative to truth.

A lower observation loss alone is not sufficient evidence of project success.

## Provenance and immutable runs

For every authoritative run record:
- Git commit;
- dirty status;
- config;
- random seeds;
- checkpoint hash;
- condition hash;
- observation hash;
- truth hash with role explicitly marked `retrospective/observation-generation only`;
- forward-model config;
- proposal/search budget;
- selected structural parameters;
- inference-visible metrics;
- retrospective metrics;
- source hashes where practical.

Do not overwrite authoritative run directories.
Do not edit Stage 7 outputs.

## Testing requirements

Before expensive GPU runs:
- add/update focused unit tests;
- run CPU smoke tests;
- run relevant regression tests;
- verify hard-condition exactness;
- verify deterministic replay;
- verify hard-forward fixture consistency;
- verify truth is absent from search objective/API;
- verify control observations;
- verify provenance output.

Report exact commands/results.
Do not claim a test/GPU run that did not execute.

## Stop rules

Do not broad-tune after a failed authoritative gate.

Stop and diagnose if:
- truth leakage occurs;
- hard conditions are violated;
- correct/control specificity disappears on a controlled case;
- structured continuous search fails under fixed budget;
- seismic fit improves by nonspecific wrong-lithology substitution;
- hard geology does not improve over paired flow-only baseline;
- combined method only improves continuous/soft objective.

If Stage 8A passes but Stage 8B fails, classify failure before more work:
- prior support;
- proposal family;
- search efficiency;
- physics/geological non-uniqueness;
- petrophysical ambiguity;
- acquisition limitation;
- implementation defect.

Propose one minimal causal experiment rather than a broad sweep.

## Training gate

Training, fine-tuning, LoRA, adapters, and U-Net changes remain CLOSED.

They may be reopened only if:
1. the user explicitly authorizes them; or
2. a later pre-registered experiment demonstrates required geology is outside useful support of the frozen prior, or structured posterior inference fails despite adequate proposal/search support.

If training is eventually reopened, prefer the smallest intervention first.

## Completion requirements

A development task is complete only when:
- protocol/config changes are documented;
- truth-role separation is auditable;
- relevant tests pass or exact blockers are reported;
- authoritative outputs are immutable and hashed;
- controls use matched budgets;
- hard conditions are exact;
- reports distinguish inference-visible from retrospective metrics;
- no unsupported generalization claim is made;
- `docs/DEVELOPMENT_HANDOFF.md` is updated only after the stage has actual evidence.
