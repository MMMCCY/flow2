# Repository instructions

## Required reading

Before modifying code under `project/geodata-3d-conditional/`, read:

1. `docs/PROJECT_BASELINE.md`
2. `docs/EXPERIMENT_PROTOCOL.md`
3. `docs/DEVELOPMENT_HANDOFF.md`
4. `docs/STAGE1_SPEC.md`

After Phase 1, also read:

5. `docs/PHASE1_REPORT.md`
6. `docs/PHASE2_SPEC.md`
7. `docs/RESEARCH_GOAL.md`
8. `docs/PHASE2A_REPORT.md`
9. `docs/PHASE2B_SPEC.md`
10. `docs/PHASE2B_FOLLOWUP_SPEC.md`
11. `docs/PHASE2B_REPORT.md`
12. `docs/PHASE3_SPEC.md`
13. `docs/PHASE4_SPEC.md`

After Phase 3 starts, also read:

14. `experiments/stage3_spatial_property/README.md`

After Phase 3 closes, also read:

15. `docs/PHASE3_REPORT.md`
16. `experiments/stage4_gravity/README.md`

After Phase 4a closes, also read:

17. `docs/PHASE4A_REPORT.md`

After Phase 4c starts, also read:

18. `docs/PHASE4C_SPEC.md`
19. `experiments/stage4_seismic/README.md`

After Phase 4c closes, also read:

20. `docs/PHASE4C_REPORT.md`

After Phase 4d starts, also read:

21. `docs/PHASE4D_SPEC.md`

After Phase 4d closes, also read:

22. `docs/PHASE4D_REPORT.md`

After Phase 5a starts, also read:

23. `docs/PHASE5A_SPEC.md`
24. `experiments/stage5_acoustic_inversion/README.md`

After Phase 5a closes, also read:

25. `docs/PHASE5A_REPORT.md`

After Phase 5b starts, also read:

26. `docs/PHASE5B_SPEC.md`

After Phase 5b closes, also read:

27. `docs/PHASE5B_REPORT.md`

After Phase 5c starts, also read:

28. `docs/PHASE5C_SPEC.md`
29. `experiments/stage5_generator_posterior/README.md`

After Phase 5c closes, also read:

30. `docs/PHASE5C_REPORT.md`

After Phase 6 starts, also read:

31. `docs/PHASE6_ADAPTER_SPEC.md`
32. `experiments/stage6_geo_adapter/README.md`
33. `docs/PHASE6A_REPORT.md`
34. `docs/NEXT_CONVERSATION_PROMPT.md`

Inspect the current Git branch, commit, working-tree changes, relevant tests,
and existing experiment outputs before proposing changes.

## Project scope

- `flow2` is the only active development baseline.
- The current work is inference-time guidance for conditional 3D geological generation.
- Do not modify model training, the 3D U-Net architecture, or the trained checkpoint unless explicitly requested.
- Raw geological labels are `-1..13`; embedding indices are `0..14`.
- Label 9 is an intrusion-class demonstration target, not a universal dike semantic.
- `cond_generation_0` is same-distribution synthetic validation data.

## Experimental invariants

- Use the checkpoint loading policy defined in `PROJECT_BASELINE.md`.
- Frozen `embedding.weight` uses its normal checkpoint value.
- All trainable model parameters use EMA weights.
- Guided and baseline experiments must use the same checkpoint, input tensors,
  seed, initial noise, time grid, integration method, and number of steps.
- Use fixed-Euler for strictly paired baseline/guided comparisons.
- Adaptive Dopri5 is only a historical reference/audit baseline.
- Guidance strength zero must reproduce the paired baseline.
- Preserve surface and borehole observations and report any violations.
- For degraded 3-D or acquisition-domain observations, overwrite predicted
  properties at hard-condition voxels before the observation operator so the
  known contribution is exact and its gradient is zero.
- A lower continuous loss is not sufficient evidence of geological improvement;
  evaluate decoded hard-label and geometric metrics.

## Development workflow

- Inspect before editing and preserve unrelated user changes.
- For multi-step changes, present a plan before implementation.
- Add or update tests for changed behavior.
- Run the relevant checks and report exact commands and results.
- Do not claim GPU validation unless the command actually ran successfully.
- Store large logs and experiment outputs in result directories; summarize them in documentation.
- Update `docs/DEVELOPMENT_HANDOFF.md` before ending a development stage.

## Completion requirements

A task is complete only when:

- code and configuration changes are documented;
- relevant tests pass, or the exact blocker is reported;
- baseline/guided pairing remains auditable;
- generated outputs contain sufficient configuration and hashes;
- the handoff document states what changed and what should happen next.
