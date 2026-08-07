import torch

from guidance.observation_specificity import (
    hidden_target_metrics,
    pairwise_geometry,
    sensitivity_spectrum,
)


def test_pairwise_geometry_reports_direction_and_norm_ratio():
    rows = pairwise_geometry({"a": torch.tensor([1.0, 0.0]), "b": torch.tensor([0.0, 2.0])})
    assert len(rows) == 1
    assert rows[0]["cosine"] == 0.0
    assert rows[0]["norm_ratio_left_over_right"] == 0.5


def test_sensitivity_spectrum_detects_rank_one_collapse():
    summary = sensitivity_spectrum(
        [torch.tensor([1.0, 0.0]), torch.tensor([2.0, 0.0])],
        ["left", "right"],
        truth_column_indices=[0, 1],
    )
    assert summary["effective_rank"] == 1
    assert summary["truth_basis_pair_cosine"] == 1.0


def test_hidden_metrics_use_declared_domain():
    labels = torch.zeros((1, 1, 2, 2, 2), dtype=torch.long)
    labels[..., 0, 0, 0] = 9
    labels[..., 1, 1, 1] = 9
    truth = torch.zeros_like(labels, dtype=torch.bool)
    truth[..., 0, 0, 0] = True
    domain = torch.zeros_like(truth)
    domain[..., 0, :, :] = True
    result = hidden_target_metrics(
        labels, target_label=9, truth_hidden_mask=truth, evaluation_domain=domain
    )
    assert result["hidden_target_iou"] == 1.0
    assert result["hidden_target_recall"] == 1.0
