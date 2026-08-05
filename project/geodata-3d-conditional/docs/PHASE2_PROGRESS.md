# Phase-2 progress: ideal 3-D property-volume guidance

Updated: 2026-07-30 after the completed 12-pair Phase-2a confirmation and the
Phase-2b CPU development gate.

## Status and interpretation boundary

Phase 2a tests a truth-derived, full-resolution 3-D property oracle with the
frozen model. It is not measured geophysics, a field forward operator, or an
argument for removing surface/borehole conditions. All runs use EMA trainable
weights, the normal frozen embedding, projected fixed Euler, identical paired
noise, and exact condition projection at every step.

## Run A: scalar density proxy, alpha/cap 0.10

Path:
`experiments/stage2_property/runs/cond_generation_0/ideal_distinct_density_proxy_v1/phase2a_v1/seed42_n1_s32/`

The pair is valid and conditions remain exact. Global accuracy changes
`0.58737 -> 0.59684`, dynamic-union mean IoU changes
`0.19291 -> 0.19527`, and hard-property loss changes
`1.19629 -> 0.98169` (`-17.94%`). However, label-9 recall changes
`0.04728 -> 0.04472`, and its predicted volume changes `6283 -> 3691`
against truth volume `8968`. The scalar soft expectation is active but does
not make the intrusion sufficiently observable. This configuration must not
be promoted to n=4.

## Run B: density plus label-9 susceptibility contrast, alpha/cap 0.10

Path:
`experiments/stage2_property/runs/cond_generation_0/ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n1_s32/`

Strict pairing passes; the alpha-zero sample hash is identical to Run A's
baseline. All 32 trace rows are finite, EMA covers all 411 trainable entries,
and post-projection condition violations are zero.

Hard results:

- global accuracy: `0.58737 -> 0.59866` (`+0.01129`);
- historical dynamic-union mIoU: `0.19291 -> 0.18087` (`-0.01204`);
- truth-present fixed-set mIoU: `0.26525 -> 0.27131` (`+0.00606`);
- truth-present classes: six improve, one worsens, one is unchanged;
- hard-property loss: `1.54588 -> 1.08126` (`-30.06%`);
- label-9 IoU: `0.02860 -> 0.05795`;
- label-9 precision: `0.06748 -> 0.22429`;
- label-9 recall: `0.04728 -> 0.07248`;
- label-9 true positives: `424 -> 650`;
- label-9 predicted volume: `6283 -> 2898`, versus truth `8968`;
- label-9 components: `37 -> 109`; largest-component fraction:
  `0.72434 -> 0.64907`;
- final-step hard churn: baseline `410`, guided `737` voxels;
- all 5,255 paired hard changes are inside the property-confidence region;
  surface/borehole violations remain zero.

The historical mIoU decline is dominated by a denominator discontinuity: four
spurious label-8 voxels cause a previously absent class to enter the per-run
union with IoU zero. The stable truth-present mean is therefore recorded in
addition to, not instead of, the historical metric. Absent-class hallucination
volumes remain visible in the per-class table.

Run B establishes that adding an observable property contrast produces the
correct hard-label direction: label-9 precision, recall, IoU, and true-positive
count all improve. It does not establish satisfactory geometry. The recovered
target remains too small and much more fragmented, so n=4 is not yet justified.

## Run C: two-channel controller upper bound, alpha/cap 0.25

Path:
`experiments/stage2_property/runs/cond_generation_0/ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n1_s32_a025_c025/`

The pair is valid and its baseline sample hash exactly matches Runs A and B.
All condition violations are zero. Fifteen active steps reach the 0.25 cap;
the final used ratio decays to 0.0646. Final-step hard churn is 2,330 voxels
versus baseline 410.

Hard results:

- global accuracy: `0.58737 -> 0.62836` (`+0.04100`);
- truth-present fixed-set mIoU: `0.26525 -> 0.33195` (`+0.06670`);
- historical dynamic-union mIoU: `0.19291 -> 0.18969` (`-0.00322`), with
  new tiny predictions for absent labels 7, 8, and 12 changing its denominator;
- six of eight truth-present classes improve IoU, one worsens, one is unchanged;
- hard-property loss: `1.54588 -> 0.53381` (`-65.47%`);
- label-9 IoU: `0.02860 -> 0.48163`;
- label-9 precision: `0.06748 -> 0.90051`;
- label-9 recall: `0.04728 -> 0.50870`;
- label-9 predicted volume: `6283 -> 5066`, versus truth `8968`;
- centroid distance: `24.00 -> 3.42` voxels;
- all 13,059 paired hard changes remain inside property confidence.

The raw label-9 component count increases `37 -> 202`, but size stratification
shows that these are not equally important. The top eight components contain
87.2% of guided target mass, while components of at most five voxels contain
5.37%. There are nine components of at least 100 voxels versus four in truth.
The four major truth bodies, sized 4,079/2,192/2,043/627 voxels, are recovered
at 48.3%/53.6%/57.9%/37.2%. The three largest predicted components have
precision 99.7%/99.4%/97.8%. Thus Run C recovers correctly located portions of
all four major bodies, but they remain split and incomplete.

Run C passes the one-sample directional gate and justifies n=4. It does not by
itself complete Phase 2a or prove realistic geophysics.

## Pre-registered n=4 gate

Run four sequential-noise samples at seed 42 with the same two-channel target,
32 steps, and `alpha=cap=0.25`. The evaluator/source change requires a new
strict baseline and guided directory. Advance to the 12-pair confirmation only
if all of the following hold:

1. strict pairing, finite traces, EMA policy, and zero condition violations
   pass for all four pairs;
2. global accuracy, truth-present fixed-set mIoU, hard-property loss, label-9
   IoU, precision, and recall improve in all four pairs;
3. at least five of eight truth-present classes improve IoU in each pair;
4. guided label-9 precision is at least 0.75, recall and IoU are at least 0.30,
   and predicted volume is between 35% and 120% of truth in every pair;
5. each of the four major truth components has recall at least 0.25, and their
   per-sample mean recall is at least 0.40;
6. label-9 mass in components of at most five voxels is at most 10%, and the
   top eight components contain at least 75% of label-9 mass;
7. final-step hard churn is at most 1.5% of the full volume;
8. four guided samples remain unique and retain non-zero outside-ROI
   disagreement.

Historical dynamic-union mIoU, absent-class hallucination volumes, raw
component counts, components at each size threshold, and endpoint traces remain
mandatory reports but do not replace the fixed-set and size-stratified gates.

## Seed-42 n=4 gate audit

Path:
`experiments/stage2_property/runs/cond_generation_0/ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed42_n4_s32_a025_c025/`

All eight gates pass:

1. all four strict pairs complete with EMA, finite traces, matching noise and
   property assets, and zero condition violations;
2. global accuracy, fixed-set mIoU, hard-property loss, and label-9
   IoU/precision/recall improve in all four pairs;
3. exactly six of eight truth-present class IoUs improve in every pair; label
   2 worsens and label 13 remains zero;
4. guided label-9 IoU is 0.482-0.509, precision 0.894-0.935, recall
   0.503-0.542, and volume 4,826-5,431 voxels (53.8%-60.6% of truth);
5. the minimum recall among the four major truth bodies is 0.316-0.456 by
   sample, and mean top-four recall is 0.492-0.538;
6. tiny-component mass is 4.50%-6.69%, and top-eight mass is 87.25%-93.15%;
7. final-step hard churn is 0.843%-0.890% of the full volume;
8. all four guided samples are unique, with outside-ROI disagreement 0.1526.

Across the four pairs, mean global accuracy changes `0.59245 -> 0.63412`,
fixed-set mIoU changes `0.26969 -> 0.33847`, dynamic-union mIoU changes
`0.17692 -> 0.19341`, and hard-property loss changes `1.48448 -> 0.51833`.
Mean guided label-9 IoU/precision/recall are 0.4905/0.9116/0.5152 and mean
centroid distance is 3.49 voxels. Conditions and property-confidence locality
remain exact.

An auxiliary repeatability check compares sample 0 from the earlier n=1 and
n=4 invocations. Alpha-zero outputs are byte-identical, while guided outputs
differ at 10 of 262,144 hard voxels (0.0038%). The inputs and initial-noise
hashes match, so cross-process bitwise deterministic CUDA guidance is not
claimed. The variation is far smaller than every observed paired effect but
must remain in the limitations record.

The seed-42 batch authorizes the remaining seed-142 and seed-242 n=4 runs for a
12-pair confirmation. It does not yet authorize Phase-3 degradation or a claim
of realistic geophysical inversion.

## Seed-142 n=4 gate audit

Path:
`experiments/stage2_property/runs/cond_generation_0/ideal_density_susceptibility_label9_contrast_v1/phase2a_v1/seed142_n4_s32_a025_c025/`

All eight gates pass again. Across four pairs, mean global accuracy changes
`0.60014 -> 0.64139`, dynamic-union mIoU `0.17743 -> 0.19904`, fixed-set mIoU
`0.28212 -> 0.34832`, and hard-property loss `1.51606 -> 0.52133`. Mean guided
label-9 IoU/precision/recall are 0.4735/0.8940/0.5028 and centroid distance is
4.29 voxels.

Three samples improve six of eight truth-present class IoUs; sample 3 improves
five, with label 2 and the extremely sparse predicted label 6 decreasing.
Label 2 has mean IoU delta -0.01425 and remains the consistent secondary-class
tradeoff. Guided label-9 IoU is 0.449-0.525, precision 0.834-0.942, recall
0.470-0.565, and volume 4,475-5,757 voxels. Minimum major-body recall is
0.358-0.400, mean top-four recall is 0.481-0.558, tiny mass is 4.19%-8.06%,
top-eight mass is 86.57%-92.15%, and final churn is 0.834%-0.986%.

All traces are finite, conditions and confidence locality are exact, all four
guided samples are unique, and outside-ROI disagreement is 0.1483. The result
authorizes the final seed-242 n=4 batch under unchanged settings.

## Seed-242 and final 12-pair decision

The seed-242 batch also passes all eight frozen gates. All four pairs improve
global accuracy, fixed-set mIoU, hard-property residual and label-9
IoU/precision/recall. Every sample improves at least five of eight
truth-present classes; major-component, topology, endpoint, exact-condition and
diversity thresholds pass.

The strict 12-pair aggregate decision is:

**PASS: Phase-2a ideal 3-D property upper bound validated with caveats.**

Aggregate means are global accuracy `0.5972 -> 0.6381`, fixed-set mIoU
`0.2771 -> 0.3443`, dynamic-union mIoU `0.1804 -> 0.1980`, hard-property loss
`1.4781 -> 0.5187`, and label-9 IoU/precision/recall
`0.0314/0.0788/0.0520 -> 0.4808/0.9032/0.5075`. Centroid distance changes
`16.9518 -> 3.9825` voxels.

The formal decision is in `docs/PHASE2A_REPORT.md`; reproducible generated
evidence is under
`experiments/stage2_property/reports/phase2a_v1_12pair/`. Phase 2a is complete,
but measured geophysics and realistic property ambiguity are not validated.

## Phase-2b development gate

The ambiguity/contrast protocol is frozen in `docs/PHASE2B_SPEC.md` before any
Phase-2b GPU run. The implementation adds no new sampler or loss:

- the generic property runner now records either the Phase-2a or Phase-2b
  experiment stage while preserving the same strict pair fields;
- `property_codebook_diagnostics` records exact property-vector collisions and
  the range-normalized target separation without affecting gradients;
- a manifest predeclares a distinct Phase-2a anchor and four paired-codebook
  levels with label-9 susceptibility `0.100/0.025/0.010/0.004`;
- the final level gives truth-present labels 6 and 9 exactly the same density
  and susceptibility vector;
- `run_phase2b_codebook_screen.sh` writes each level beneath a separate
  `phase2b_codebook_ambiguity_v1/<level>/` run directory;
- `summarize_phase2b_screen.py` revalidates pairing, hashes, finite traces,
  hard/class/component/topology gates and the Phase-2a anchor regression before
  applying the frozen promotion rule.

The Phase-2b screen development checkpoint passed `61` tests. After adding the
n=4 bracket and post-bracket fallback tooling, the complete lightweight suite
passes (`73 passed`, 13 existing warnings); the Phase2/Phase2b-focused subset,
including strict-pair visualization checks, passes 35 tests.
### Phase-2b implementation anchor result

The seed-42 `distinct_c100_anchor` single-sample pair is complete and passes:

- strict property/source/noise pairing, EMA coverage, finite trace and zero
  condition/confidence-locality violations;
- alpha-zero hard output is byte-identical to the saved Phase-2a reference;
- the independent guided repeat differs by only 8/262,144 hard voxels
  (0.00305%), below the frozen 0.1% anchor limit;
- guided label-9 IoU/precision/recall are
  `0.4818/0.9004/0.5089`, and all three differ from the Phase-2a reference by
  less than 0.0003;
- global accuracy and fixed truth-present mIoU improve by
  `0.04100/0.06672`, hard-property loss falls by `1.01215`, and six of eight
  truth-present class IoUs improve;
- major-component minimum/mean recall are `0.3716/0.4925`, tiny-component
  mass is 5.37%, top-eight mass is 87.22%, and final churn is 0.889%.

The anchor authorizes only `paired_c100`, the first ambiguity level. The
screen remains incomplete and no Phase-2b threshold has been identified.

### Phase-2b `paired_c100` result

The first ambiguous codebook reduces the number of unique two-channel vectors
from 15 to 9 while retaining label-9 susceptibility at 0.100. Its strict
seed-42 single pair passes the complete screen gate:

- global accuracy and fixed-set mIoU improve by `0.04225/0.06275`, and six of
  eight truth-present class IoUs improve;
- hard-property loss falls by `0.84705`;
- label-9 IoU/precision/recall reach `0.4260/0.8699/0.4550`, lower than the
  distinct anchor but materially above the alpha-zero baseline;
- predicted label-9 volume is 4,690 against truth 8,968;
- major-component minimum/mean recall are `0.2982/0.4383`;
- tiny-component mass is 6.91%, top-eight mass is 83.39%, and final churn is
  1.060%, all inside the frozen thresholds;
- strict pairing, EMA, trace finiteness, condition projection and confidence
  locality pass.

This shows that multi-lithology property overlap weakens but does not remove
the hard-geology effect while label 9 retains a strong second-channel
contrast. It authorizes only the next predeclared level, `paired_c025`.

### Phase-2b `paired_c025` result

Reducing label-9 susceptibility from 0.100 to 0.025 under the same paired
codebook produces another strict single-pair pass, but it is close to several
frozen boundaries:

- global accuracy and fixed-set mIoU improve by `0.04058/0.05855`, hard-property
  loss falls by `0.83407`, and six of eight truth-present class IoUs improve;
- label-9 IoU/precision/recall are `0.3951/0.8632/0.4215`, with predicted
  volume 4,379 against truth 8,968;
- major-component minimum recall is `0.2584` versus the 0.25 limit, and their
  mean recall is `0.4025` versus the 0.40 limit;
- tiny-component mass is 6.74%, top-eight mass is `0.7986` versus the 0.75
  limit, and final churn is 1.077%;
- strict pairing, EMA, finite traces, exact conditions and confidence locality
  pass.

The level passes without changing any gate, but its small margins make
`paired_c010` the necessary next screen point. A single sample still cannot
define the final observability threshold.

### Phase-2b `paired_c010` result

The label-9 susceptibility 0.010 level is the first single-sample failure.
Strict pairing, EMA, finite traces, exact conditions and confidence locality
still pass. Several favorable directions also remain: global accuracy improves
by `0.03250`, fixed-set mIoU by `0.03729`, hard-property loss falls by
`0.67044`, and six of eight truth-present class IoUs improve. These are not
sufficient for geological acceptance.

The frozen failures are:

- label-9 IoU/recall are `0.2370/0.2566`, below 0.30; precision `0.7569`
  narrowly remains above 0.75;
- predicted target volume is 3,040, only 33.90% of truth and below the 35%
  lower bound;
- the four major components have minimum/mean recall `0.0973/0.2362`, below
  `0.25/0.40`;
- tiny-component mass is 14.77% versus the 10% maximum, and top-eight mass is
  63.39% versus the 75% minimum;
- final churn remains acceptable at 1.126%.

Thus the seed-42 single-sample transition is bracketed between passing 0.025
and failing 0.010. The predeclared exact-collision `paired_c004_overlap` must
still run as a negative control before applying the promotion rule.

### Phase-2b exact-collision and final screen decision

`paired_c004_overlap` gives truth-present labels 6 and 9 exactly the same
two-channel property vector. The pair is valid, finite and condition-exact,
but the geological gate fails:

- global accuracy and fixed-set mIoU still improve by `0.02089/0.01131`, and
  hard-property loss falls by `0.26844`;
- label-9 IoU is only `0.0355`, precision `0.1391`, and recall `0.0455`;
- recall is slightly lower than the alpha-zero baseline despite the lower
  continuous and hard-property residual;
- the four major component mean recall is `0.0403` and minimum recall is
  `0.00046`, so the target bodies are effectively not recovered;
- the high top-eight mass fraction is not evidence of success because almost
  no correct target geometry exists.

All five predeclared screen levels are now complete. The unmodified promotion
rule selects `paired_c025` as the most degraded passing level and
`paired_c010` as its adjacent failing level. This is a candidate bracket, not
a Phase-2b confirmation.

The seed-42 n=4 development path is implemented in:

- `experiments/stage2_property/run_phase2b_codebook_n4_bracket.sh`;
- `scripts/stage2/summarize_phase2b_n4_bracket.py`.

The n=4 auditor applies the complete per-pair gate to all four samples,
requires four unique guided outputs and non-zero outside-ROI disagreement, and
classifies `4/4`, `1-3/4` and `0/4` without changing thresholds. Only a `4/4`
level can proceed to multi-seed confirmation. The first n=4 result follows.

### Phase-2b `paired_c025` seed-42 n=4 result

The four-sample candidate batch is valid, condition-exact and diverse, but it
passes only `3/4` complete pair gates. Under the frozen rule this is a
`transition_region`, not a confirmed pass:

- mean global accuracy and fixed-set mIoU deltas are `+0.04110/+0.06033`;
- mean hard-property loss delta is `-0.79273`;
- mean guided label-9 IoU/precision/recall are
  `0.4040/0.8762/0.4290`;
- all four samples pass primary directions, majority-class, absolute target,
  size-stratified topology and endpoint gates;
- samples 0/1/3 pass the major-component gate;
- sample 2 has major-component mean recall `0.4066` but minimum recall
  `0.1994`, below 0.25 because the fourth truth body is under-recovered;
- all four guided samples are unique and outside-ROI disagreement is `0.1532`.

The `3/4` result cannot be relabeled as success and does not authorize
multi-seed confirmation.

### Phase-2b `paired_c010` seed-42 n=4 result

The adjacent lower batch is complete, valid, condition-exact and diverse, but
passes `0/4` complete pair gates. It is therefore a
`confirmed_seed42_failure` at the frozen operating point:

- mean global accuracy and fixed-set mIoU deltas remain favorable at
  `+0.03244/+0.03785`, and mean hard-property loss delta is `-0.61635`;
- mean guided label-9 IoU/precision/recall are only
  `0.2376/0.7756/0.2556`;
- every sample passes primary directions, majority-class and endpoint-churn
  gates;
- every sample fails the absolute target, major-component recovery and
  size-stratified topology gates;
- all four guided samples are unique and outside-ROI disagreement is `0.1536`.

The authoritative derived report is
`experiments/stage2_property/reports/phase2b_codebook_ambiguity_v1_n4_bracket_seed42/REPORT.md`.
Neither bracket level qualifies for multi-seed testing.

### Post-bracket `paired_c100` robustness fallback

After closing the original bracket and before any new GPU run, the fallback is
frozen in `docs/PHASE2B_FOLLOWUP_SPEC.md`. It tests only the next higher
screened ambiguous level, `paired_c100`, at seed 42 with four samples. Alpha,
cap, fixed-Euler pairing, property target and all pair/diversity gates remain
unchanged. This does not revise the c025/c010 outcomes.

The isolated launcher and auditor are:

- `experiments/stage2_property/run_phase2b_codebook_n4_fallback.sh`;
- `scripts/stage2/summarize_phase2b_n4_fallback.py`.

Only a 4/4 result plus diversity can authorize unchanged seeds 142/242. Any
other result closes Phase 2b without a multi-seed-confirmed ambiguous-codebook
operating point at the tested levels.

The completed fallback passes 3/4 pair gates and is a transition region. Mean
global accuracy/fixed-set mIoU deltas are `+0.04325/+0.06552`, hard-property
loss delta is `-0.81585`, and mean label-9 IoU/precision/recall are
`0.4412/0.8816/0.4695`. Samples 0/1/3 pass all gates. Sample 2 fails only the
major-component minimum recall (`0.2313 < 0.25`) while its major-component mean
recall is 0.4473 and every other hard gate passes. Diversity passes with four
unique samples and outside-ROI disagreement 0.1532.

No multi-seed Phase-2b run is authorized. Phase 2b is formally closed in
`docs/PHASE2B_REPORT.md`.
