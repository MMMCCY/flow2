# Stage 8A-v4 final standalone protocol

## Scope and terminal status

Stage8A-v1, R1, v2, R2, v3, and v3-R3 are immutable. R3 classified v3 as
`GLOBAL_BEAM_PRUNES_LOCALLY_IMPROVING_SEEDS`. The sole v4 change is to complete
locally hard-RMSE-monotonic lineages before each generation's global beam
competition. This is the final authorized standalone Stage8A iteration;
standalone Stage8A closes after its result regardless of PASS or FAIL.

## Frozen search and physics

The v2 multifield sensitivity ranker, center domain, deterministic tie-break,
proposal seeds, full target size/orientation/shape draws, allowed ellipsoid and
dike-hemisphere families, label 9, bounds, beam width 8, 10 generations,
condition projection, hard petrophysical mapping, hard seismic forward,
controls, and original Stage8A gate are unchanged. Cuboid, soft geology, truth,
training, ranker changes, ladder tuning, broader search, and sweeps are
forbidden.

The scale ladder remains exactly

\[
  \lambda=(0.25,0.50,0.75,1.00).
\]

Every step follows hard geology, exact condition projection, hard property
mapping, hard seismic forward, and hard observed RMSE.

## Deterministic slot-reallocation rule

Each generation's existing `8 x 12 = 96` evaluation slots are flattened in
the frozen parent-major order `slot = parent_index*12 + local_index`, from 0 to
95. A normal slot retains the v3 parent, move schedule, proposal index and RNG
stream.

When a scale-0.25 birth strictly improves its immediate parent, the immediately
following existing slot is reallocated to scale 0.50 of that same full target.
Each successful growth step similarly reallocates the immediately following
slot to the next scale. The lineage stops at the first step that does not
strictly improve the preceding lineage state, or after a strictly improving
scale-1.00 step. A reallocated slot's originally scheduled proposal is skipped
without replacement; its scheduled parent and move are recorded. If slot 95 is
an improving step, the generation ends and the best reached lineage state is
submitted without an extra forward.

Only the last strictly improving state of a continued lineage enters that
generation's normal global candidate set; a failed growth state never enters.
A scale-0.25 birth that does not improve its parent remains the ordinary hard
proposal candidate, preserving the noncontinuation behavior. Candidates are
then sorted by `(hard observed RMSE, state_id)` and the top eight form the next
beam.

The exact per-arm hard-forward budget remains

\[
  1 + 10\times96 = 961.
\]

Growth attempts consume existing slots and cannot add calls or increase beam
width. Realized new-center, growth and nonbirth counts may differ by arm, while
the allocation rule, ladder and total budget remain identical.

## Audit and decision

Every arm records new seeds, every scale transition, branchwise hard-RMSE and
scale sequences, maximum attained scale, termination, displaced scheduled
proposal, final global-beam survival, slot accounting, condition violations,
ranker artifacts, and truth firewall. Selection is frozen before retrospective
geometry is opened.

The unchanged original Stage8A gate is the authoritative decision. After v4,
no Stage8A-v5 may be implemented. The project returns to the Flow-prior
integration decision; Stage8B is not automatically authorized by a standalone
v4 result.
