from __future__ import annotations

import torch

from guidance.binary_seismic_inversion import BinaryAcousticProperties
from guidance.binary_trace_boundary import impedance_to_binary_score, vertical_boundary_strength


PROPERTIES = BinaryAcousticProperties(1.0, 1.0, 2.0, 3.0, 4.0, 6.0)


def test_known_binary_impedance_maps_to_exact_endpoints() -> None:
    support = torch.ones((1, 1, 1, 1, 3), dtype=torch.bool)
    impedance = torch.tensor([[[[[6.0, 12.0, 24.0]]]]])
    score = impedance_to_binary_score(impedance, support, PROPERTIES)
    assert score[0, 0, 0, 0, 0] == 0
    assert score[0, 0, 0, 0, 2] == 1
    assert 0 < score[0, 0, 0, 0, 1] < 1


def test_vertical_boundary_strength_marks_both_adjacent_cells() -> None:
    score = torch.tensor([[[[[0.0, 0.0, 1.0, 1.0]]]]])
    boundary = vertical_boundary_strength(score)
    assert torch.equal(boundary, torch.tensor([[[[[0.0, 1.0, 1.0, 0.0]]]]]))
