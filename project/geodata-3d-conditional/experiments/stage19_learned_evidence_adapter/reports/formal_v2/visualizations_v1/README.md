# Stage19R formal 3-D visualizations

These figures reuse the Stage15/Phase1-style categorical and target-only 3-D
renderers, including the same camera, cutaway, categorical palette, label9
colour, and translucent truth ghost.

- `truth_flow_adapter_3d.png` is the primary six-panel figure. It uses the
  lexicographically first TEST case (`stage19_test_case001`) and the first
  preregistered source seed (`42`), with no metric-based selection.
- `truth_flow_adapter_stage19_test_case*_seed42_3d.png` provides the same
  truth/Flow/correct-adapter comparison for the other 11 independent cases.
- `five_arm_target_comparison_3d.png` shows truth plus all five formal arms for
  case001/seed42. Orange is predicted label9 and translucent grey is truth
  label9 where it is not covered by the prediction.
- `visualization_manifest.json` records input/output hashes, metrics,
  selection policy, and the retrospective truth-use boundary.

Truth is loaded only by this retrospective visualization script. The completed
formal inference manifest remains truth-blind.

Reproduce from the project directory with:

```bash
MPLCONFIGDIR=/tmp/stage19r-mpl PYVISTA_OFF_SCREEN=true \
  ../../.venv/bin/python scripts/stage19/visualize_formal_results.py
```
