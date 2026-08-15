"""Small binary-topology helpers for the Stage15 prior-support audit."""

from __future__ import annotations

from collections import deque
from typing import Sequence

import numpy as np
import torch


def ellipsoid_mask(
    shape: Sequence[int], center: Sequence[float], axes: Sequence[float]
) -> torch.Tensor:
    if len(shape) != 3 or len(center) != 3 or len(axes) != 3:
        raise ValueError("shape, center, and axes must be three-dimensional")
    if any(float(value) <= 0 for value in axes):
        raise ValueError("ellipsoid axes must be positive")
    coordinates = torch.meshgrid(
        *(torch.arange(int(size), dtype=torch.float32) for size in shape),
        indexing="ij",
    )
    radius = sum(
        ((coordinate - float(origin)) / float(axis)) ** 2
        for coordinate, origin, axis in zip(coordinates, center, axes)
    )
    return radius <= 1.0


def torus_mask(
    shape: Sequence[int],
    center: Sequence[float],
    major_radius: float,
    tube_radius: float,
) -> torch.Tensor:
    """Return a z-axis torus on the native voxel grid."""
    if len(shape) != 3 or len(center) != 3:
        raise ValueError("shape and center must be three-dimensional")
    if float(major_radius) <= float(tube_radius) or float(tube_radius) <= 0:
        raise ValueError("torus requires major_radius > tube_radius > 0")
    x, y, z = torch.meshgrid(
        *(torch.arange(int(size), dtype=torch.float32) for size in shape),
        indexing="ij",
    )
    radial = torch.sqrt((x - float(center[0])) ** 2 + (y - float(center[1])) ** 2)
    return (radial - float(major_radius)) ** 2 + (z - float(center[2])) ** 2 <= float(
        tube_radius
    ) ** 2


def cubical_euler_characteristic(mask: torch.Tensor | np.ndarray) -> int:
    """Euler characteristic of a union of closed unit cubes, exactly counted."""
    occupied = np.asarray(mask, dtype=bool)
    if occupied.ndim != 3:
        raise ValueError("mask must be three-dimensional")
    cubes = np.argwhere(occupied)
    if len(cubes) == 0:
        return 0
    vertices: set[tuple[int, int, int]] = set()
    edges: set[tuple[int, int, int, int]] = set()
    faces: set[tuple[int, int, int, int]] = set()
    for i, j, k in cubes:
        for di in (0, 1):
            for dj in (0, 1):
                for dk in (0, 1):
                    vertices.add((int(i + di), int(j + dj), int(k + dk)))
        for axis in range(3):
            other = [value for value in range(3) if value != axis]
            for first in (0, 1):
                for second in (0, 1):
                    base = [int(i), int(j), int(k)]
                    base[other[0]] += first
                    base[other[1]] += second
                    edges.add((axis, *base))
        for axis in range(3):
            for side in (0, 1):
                base = [int(i), int(j), int(k)]
                base[axis] += side
                faces.add((axis, *base))
    return len(vertices) - len(edges) + len(faces) - len(cubes)


def _component_count(mask: np.ndarray) -> int:
    remaining = set(map(tuple, np.argwhere(mask)))
    count = 0
    while remaining:
        count += 1
        queue = deque([remaining.pop()])
        while queue:
            voxel = queue.popleft()
            for axis in range(3):
                for delta in (-1, 1):
                    neighbour = list(voxel)
                    neighbour[axis] += delta
                    item = tuple(neighbour)
                    if item in remaining:
                        remaining.remove(item)
                        queue.append(item)
    return count


def _enclosed_background_components(mask: np.ndarray) -> int:
    background = ~mask
    remaining = set(map(tuple, np.argwhere(background)))
    enclosed = 0
    shape = mask.shape
    while remaining:
        start = remaining.pop()
        queue = deque([start])
        touches_boundary = any(
            start[axis] in (0, shape[axis] - 1) for axis in range(3)
        )
        while queue:
            voxel = queue.popleft()
            for axis in range(3):
                for delta in (-1, 1):
                    neighbour = list(voxel)
                    neighbour[axis] += delta
                    item = tuple(neighbour)
                    if item in remaining:
                        remaining.remove(item)
                        queue.append(item)
                        touches_boundary |= any(
                            item[index] in (0, shape[index] - 1) for index in range(3)
                        )
        enclosed += int(not touches_boundary)
    return enclosed


def betti_numbers(mask: torch.Tensor | np.ndarray) -> dict[str, int]:
    """Return beta0/beta1/beta2 using 6-connected cubical topology."""
    value = np.asarray(mask, dtype=bool)
    beta0 = _component_count(value)
    beta2 = _enclosed_background_components(value)
    euler = cubical_euler_characteristic(value)
    beta1 = beta0 + beta2 - euler
    return {"beta0": beta0, "beta1": int(beta1), "beta2": beta2, "euler": euler}


def binary_metrics(prediction: torch.Tensor, truth: torch.Tensor) -> dict[str, float | int]:
    prediction = prediction.bool()
    truth = truth.bool()
    if prediction.shape != truth.shape:
        raise ValueError("prediction and truth must match")
    tp = int((prediction & truth).sum())
    fp = int((prediction & ~truth).sum())
    fn = int((~prediction & truth).sum())
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return {
        "true_positive": tp,
        "false_positive": fp,
        "false_negative": fn,
        "precision": precision,
        "recall": recall,
        "iou": tp / max(tp + fp + fn, 1),
    }


def ring_diagnostics(
    prediction: torch.Tensor,
    center: Sequence[float],
    major_radius: float,
    tube_radius: float,
    angular_bins: int = 36,
) -> dict[str, float]:
    """Measure central-hole filling and azimuthal ring completeness."""
    prediction = prediction.bool()
    x, y, z = torch.meshgrid(
        *(torch.arange(size, dtype=torch.float32) for size in prediction.shape),
        indexing="ij",
    )
    dx, dy = x - float(center[0]), y - float(center[1])
    radial = torch.sqrt(dx.square() + dy.square())
    hole = (radial <= max(float(major_radius - tube_radius - 1.0), 1.0)) & (
        (z - float(center[2])).abs() <= float(tube_radius)
    )
    annulus = (
        (radial - float(major_radius)).abs() <= float(tube_radius)
    ) & ((z - float(center[2])).abs() <= float(tube_radius))
    angle = torch.remainder(torch.atan2(dy, dx), 2 * torch.pi)
    occupied_bins = 0
    for index in range(int(angular_bins)):
        lower = 2 * torch.pi * index / angular_bins
        upper = 2 * torch.pi * (index + 1) / angular_bins
        occupied_bins += int(bool((prediction & annulus & (angle >= lower) & (angle < upper)).any()))
    return {
        "central_hole_fill_fraction": float(prediction[hole].float().mean()) if bool(hole.any()) else 0.0,
        "central_hole_preservation": 1.0 - (float(prediction[hole].float().mean()) if bool(hole.any()) else 0.0),
        "azimuthal_ring_coverage": occupied_bins / float(angular_bins),
    }
