# Stage 8A-v3 frozen protocol — hard-loss trust-region birth continuation

## Authority and scope

Stage8A-v1, Stage8A-R1, Stage8A-v2, and Stage8A-v2-R2 are immutable inputs.
The R2 primary classification remains `FIRST_ORDER_TO_FINITE_HARD_NONLINEARITY`.
The sole v3 algorithmic change is replacing each one-shot full-size hard birth
with deterministic nested hard-loss continuation.  Stage8B, training,
fine-tuning, LoRA, extra shapes, broader search, and gate changes are forbidden.

## Frozen nested ladder

The common, untuned ladder is

\[
  (\lambda_0,\lambda_1,\lambda_2,\lambda_3)
  = (1/4,1/2,3/4,1).
\]

For a full target body with center \(c\), size vector \(s\), orientation
\(\theta\), frozen shape family \(q\), and label 9, probe \(k\) is

\[
  B_k = \operatorname{Rasterize}(c,\lambda_k s,\theta,q,9)
        \cap M_{edit}.
\]

Only `size_x`, `size_y`, and `size_z` are multiplied by \(\lambda_k\).  Center,
orientation, shape, label, full-size draw, proposal seed, and sensitivity rank
are unchanged.  Ellipsoids and dike-hemispheres are star-shaped about their
fixed center under this uniform homothety, so their continuous sets are nested.
The same deterministic voxel-center inclusion rule rasterizes them; intersection
with the fixed edit mask preserves `B_i` as a subset of `B_j` for `i < j`.
Every materialized state remains a hard categorical integer-label volume and is
projected onto the exact fixed hard conditions before the hard petrophysical
mapping and seismic forward.

## Slot allocation and continuation

The v2 sensitivity center ranking is byte-for-byte unchanged.  A scheduled
birth slot first grows an active branch by exactly one ladder step when that
parent has an active branch and has not already allocated growth in the current
generation.  Otherwise it opens the next unused center in the current frozen
sensitivity ranking and evaluates scale `1/4` of that proposal's original full
geometry.

After each hard categorical evaluation, continuation is authorized only when

\[
  \Delta_{parent} = \operatorname{RMSE}_{hard}(B_k)
                  - \operatorname{RMSE}_{hard}(parent) < 0.
\]

A non-improving growth step terminates that branch.  It cannot influence
selection except through its one already-authorized hard RMSE evaluation.  All
acceptance, beam selection, and final selection use hard observed seismic RMSE
only.  No truth or retrospective metric enters the controller API.

The budget remains exactly

\[
  1 + 10\times8\times12 = 961
\]

hard forward calls per arm.  Growth replaces existing scheduled birth slots;
it never adds calls.  New-center, growth, nonbirth, and initial-empty allocation
are recorded separately.  Realized move counts may differ between observation
arms because continuation follows truth-blind observation-dependent hard loss.

## Frozen scientific gates and stop rule

The original Stage8A gate file and thresholds apply first and unchanged.  v3
runs the analytic case, three native cases, and correct/zero/shuffled/wrong-case
controls.  No cuboid is added.  If any original gate fails, the machine decision
is `FAIL_STAGE8A_STOP_BEFORE_STAGE8B`; Stage8B and a second algorithmic change
remain forbidden.
