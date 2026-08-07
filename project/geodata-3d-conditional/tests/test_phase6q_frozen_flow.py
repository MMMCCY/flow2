from __future__ import annotations
from pathlib import Path
import sys
import torch

PROJECT_DIR = Path(__file__).resolve().parents[1]
if str(PROJECT_DIR) not in sys.path: sys.path.insert(0, str(PROJECT_DIR))

from guidance.frozen_flow_causality import run_base_trajectory
from guidance.generator_posterior import projected_fixed_euler_prior_sample

class _Net:
    def __call__(self, state, condition, time): return torch.ones_like(state) * 0.5
class _Model:
    net = _Net()

def test_base_trajectory_is_deterministic_and_condition_exact() -> None:
    initial = torch.zeros((1, 2, 2, 2, 2))
    mask = torch.zeros((1, 1, 2, 2, 2), dtype=torch.bool); mask[..., 0, 0, 0] = True
    exact = torch.zeros_like(initial); exact[..., 0, 0, 0] = 3.0
    condition = exact * mask.expand_as(exact)
    first = run_base_trajectory(model=_Model(), initial_state=initial, conditioning=condition, embedded_conditions=exact, condition_mask=mask, n_steps=4)
    second = run_base_trajectory(model=_Model(), initial_state=initial, conditioning=condition, embedded_conditions=exact, condition_mask=mask, n_steps=4)
    assert torch.equal(first["states"][-1], second["states"][-1])
    assert torch.equal(first["states"][-1][..., 0, 0, 0], exact[..., 0, 0, 0])
    canonical = projected_fixed_euler_prior_sample(
        _Model(), initial, condition, exact, mask, n_steps=4
    )
    assert torch.equal(first["states"][-1], canonical)
