# D7 Observation Specificity Report

## Decision

The similarity of correct/zero/shuffled full-flow behavior starts primarily in
the residuals themselves, not in the controller.  Across the eight identical
BASE-state audits, the mean correct-to-control seismic residual cosine is
`0.92084`; VJP projection raises the corresponding raw negative-gradient cosine
only to `0.94507`, and clipping/scaling leaves it unchanged at `0.94507`.
The mean one-step target-label overlap is then `0.97903`, adding a smaller hard
categorical collapse.

| Rank | Mechanism | Support score | Interpretation |
|---:|---|---:|---|
| 1 | S1 residual similarity | 0.92084 | Strongly supported |
| 2 | S4 categorical hard-transition collapse | 0.03396 | Supported but secondary |
| 3 | S2 Jacobian/VJP projection collapse | 0.02423 | Weak incremental support |
| 4 | S3 controller normalization/cap collapse | `5.7e-12` | Not supported |

The optional magnitude-preserving controller control is therefore not
authorized.  H4 implementation correctness remains intact, and the evidence
does not identify controller normalization as the algorithmic bottleneck.

## Provenance and identical-state gate

Repository freeze was clean at `main @ 85d5deb4555430117887a8ba173a0222c6b899ae`.
D1–D4 were rerun under new provenance tags because their historical runner
hashes came from untracked files; D5 already matched.  For all five stages,
current source, config and checkpoint hashes match and observation hashes are
present, so `provenance_verified=true`.

All control gradients use the same regenerated BASE states at steps
8/12/16/20/24/28/32 plus the common endpoint.  The regenerated endpoint hash
exactly matches D4: `fa9c7371661f2375c77b3bd4719ebb5e8a31c2f2c2d77121ec57a42899f08d77`.

## Residual, VJP and controller geometry

At BASE step 8 the final-waveform residual cosines are `0.91333`
(correct/zero) and `0.83774` (correct/shuffled).  At step 32 they are
`0.94381` and `0.89537`.  Thus the three targets do not create orthogonal
residual demands at the states visited by the frozen flow.

The identical-state raw-gradient geometry is even more aligned (`0.94507`
mean correct-to-control cosine).  The applied-velocity cosine differs by less
than `6e-12`, showing that norm clipping, reference-gradient scaling, the cap,
schedule and Euler `dt` preserve rather than collapse the observed directions.
Full per-state reflectivity, TWT, spike and waveform residuals; hidden/outside
gradient correlations; and all controller norms/ratios are saved in the CSV
artifacts beside this report.

## One-step and saved-state cross evaluation

One-step soft and hard 3×3 matrices were evaluated from each shared state, and
every arm was also evaluated against the same correct observation.  The large
mean hard target overlap (`0.97903`) confirms an additional categorical
many-to-one effect after the continuous update.

For existing D4 `BASE_PLUS_PHYSICS` best/final states, correct/zero/shuffled
optimization gives correct-field hard RMSE `0.04153/0.03907/0.04158`; the zero
arm is actually best on the correct field.  Their hidden IoU values are only
`0.02360/0.00162/0.00205`.  The endpoint-reference correct arm lowers correct
RMSE to `0.02835`, but hidden IoU/recall remain `0.00157/0.00234`.  Continuous
physics progress therefore does not restore observation-specific hidden
geology.

## Local sensitivity conditioning

The 16-column diagnostic basis contains all 12 frozen candidate bodies and
four deterministic hidden-ROI directions.  Every audited level has effective
rank `13/16`, including probability, expected/blurred property, reflectivity,
TWT, spikes and seismic.  Effective condition numbers are large:

| Level | Effective rank | Effective condition number |
|---|---:|---:|
| probability | 13/16 | 4434.6 |
| expected property | 13/16 | 4392.6 |
| blurred property | 13/16 | 1162.8 |
| reflectivity | 13/16 | 4009.0 |
| TWT | 13/16 | 2348.2 |
| spikes | 13/16 | 2965.7 |
| seismic | 13/16 | 2935.1 |

L3/L4 therefore do not cause an additional rank cliff on this small local
basis (`13/16` before and after), but the response space is already strongly
ill-conditioned.  This supports only a weak S2 increment; it is not a claim of
global invertibility.

Truth geology metrics in this audit are retrospective diagnostics only and
were never used for optimization, checkpoint choice or mechanism ranking.
