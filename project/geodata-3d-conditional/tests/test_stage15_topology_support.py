from __future__ import annotations

import torch

from guidance.topology_support import (
    betti_numbers,
    binary_metrics,
    ellipsoid_mask,
    ring_diagnostics,
    torus_mask,
)


def test_solid_and_ring_have_expected_first_betti_number() -> None:
    solid = ellipsoid_mask((32, 32, 32), (16, 16, 16), (7, 7, 4))
    ring = torus_mask((32, 32, 32), (16, 16, 16), 7, 3)
    assert betti_numbers(solid) == {"beta0": 1, "beta1": 0, "beta2": 0, "euler": 1}
    assert betti_numbers(ring) == {"beta0": 1, "beta1": 1, "beta2": 0, "euler": 0}


def test_binary_metrics_count_false_positive_hole_fill() -> None:
    truth = torus_mask((24, 24, 24), (12, 12, 12), 6, 2)
    prediction = truth.clone()
    prediction[12, 12, 12] = True
    metrics = binary_metrics(prediction, truth)
    assert metrics["true_positive"] == int(truth.sum())
    assert metrics["false_positive"] == 1
    assert metrics["false_negative"] == 0


def test_ring_diagnostics_separate_ring_from_filled_center() -> None:
    ring = torus_mask((32, 32, 32), (16, 16, 16), 7, 3)
    diagnostics = ring_diagnostics(ring, (16, 16, 16), 7, 3)
    assert diagnostics["central_hole_fill_fraction"] == 0.0
    assert diagnostics["azimuthal_ring_coverage"] == 1.0
    filled = ring.clone()
    filled[16, 16, 16] = True
    assert ring_diagnostics(filled, (16, 16, 16), 7, 3)["central_hole_fill_fraction"] > 0
