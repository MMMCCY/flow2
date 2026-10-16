# Read-only repository review before Stage21v2 implementation

Review date: 2026-10-16 (Asia/Shanghai, UTC+8).

- SSH successfully authenticated to `xmj@172.27.231.254:22`, hostname `LabRS`.
  Password was used only for SSH authentication and was not saved to files.
- Remote repository: `/home/xmj/mcy/flowtrain_stochastic_interpolation-main`.
- Branch `main`; HEAD `b780b9ae3650b426400d3258b8e33f10a349bf6c`.
- Origin `https://github.com/MMMCCY/flow2.git`.
- Remote `git ls-remote` failed because `github.com` DNS resolution failed.
  Connected GitHub read-only `compare_commits(base=HEAD, head=main)` succeeded:
  `status=identical`, ahead0, behind0, total_commits0. Thus GitHub main and HEAD
  match at review. No pull/merge/stash/reset/clean/commit/push was performed.
- Preexisting tracked diff: handoff +128 lines, Stage19R development report
  +14 lines. Preserve both. No Stage21v2 files existed on the server at review.
- Relevant preexisting untracked work: complete `scripts/stage21/`,
  `tests/test_stage21_virtual_boreholes.py`, Stage21 experiment root;
  ensemble sampling/rendering scripts/tests/results; Stage19 visualization
  script/results; root task prompts/README/SSH host configuration. These are
  existing user work, not newly created Stage21v2 changes.
- Old Stage21 `run_20261014_v2/formal/sampling_state.json`: completed45 outputs;
  current `reports_v3/DELIVERY_AUDIT.json`: passed, 1440 formal velocity calls,
  zero condition violations. `reports_v1/v2/v3` use the same predictions.
  `run_20261014_v1` is a retained initialization failure, not science output.
- Old Stage21 supports bounded off-well category benefit, not hidden-object
  recovery: seven originally M9-hidden major components, only one newly hit.
- `.venv/bin/python` remotely imports torch2.8.0+cu128, CUDA available.
  RTX4090D: 24564MiB total, 12MiB used, 0% utilization at review. The local
  execution environment has no working NVIDIA driver; all GPU execution must
  occur remotely. Check occupancy again before inference; do not interrupt jobs.
- Root prompt SHA256 `86880eef542fc3fb6895e1482399e8e358a448f0be6e3881ae64c299be40b237`.
  Local/remote Stage21 `run.py`, `common.py`, `evaluate.py` hashes match.
  Runtime/model/sampler hashes are checked against the old frozen registry
  before baseline reuse and snapshotted in the new preparation receipt.

New files belong only to `scripts/stage21v2/`, the corresponding new test and
`experiments/stage21v2_dense_virtual_boreholes/`. Handoff is appended only after
actual results; its original dirty content is preserved.
