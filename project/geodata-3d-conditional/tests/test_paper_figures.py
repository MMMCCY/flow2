"""Lightweight regression tests for the paper-figure provenance/QC helpers."""

from __future__ import annotations

import hashlib
import zipfile

import numpy as np

from scripts.paper_figures.style import (
    CAMERA,
    LABEL9_COLOR,
    LABEL_COLORS,
    OBSERVATION_COLOR,
    robust_symmetric_limit,
    target_metrics,
    write_deterministic_npz,
)


def _hash(path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_target_metrics_use_hard_label9() -> None:
    truth = np.array([[[9, 9], [0, 0]]])
    prediction = np.array([[[9, 0], [9, 0]]])
    metrics = target_metrics(truth, prediction)
    assert metrics == {"IoU9": 1 / 3, "Precision9": 1 / 2, "Recall9": 1 / 2}


def test_sparse_residual_limit_uses_one_pooled_nonzero_population() -> None:
    left = np.array([0.0, 0.0, -1.0, 2.0])
    right = np.array([0.0, 3.0, -4.0, 0.0])
    limit = robust_symmetric_limit((left, right), 50.0, ignore_zeros=True)
    assert limit == 2.5


def test_npz_writer_is_byte_deterministic(tmp_path) -> None:
    first = tmp_path / "first.npz"
    second = tmp_path / "second.npz"
    arrays = {"z": np.arange(5, dtype=np.int16), "a": np.eye(3, dtype=np.float32)}
    write_deterministic_npz(first, **arrays)
    write_deterministic_npz(second, **arrays)
    assert _hash(first) == _hash(second)
    with zipfile.ZipFile(first) as archive:
        assert archive.namelist() == ["a.npy", "z.npy"]
        assert all(info.date_time == (1980, 1, 1, 0, 0, 0) for info in archive.infolist())


def test_visual_language_reserves_label9_and_observation_colors() -> None:
    assert LABEL_COLORS[9] == LABEL9_COLOR
    assert LABEL9_COLOR != OBSERVATION_COLOR
    assert CAMERA["parallel_projection"] is True

