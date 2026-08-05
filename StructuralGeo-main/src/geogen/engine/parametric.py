"""Explicit parameter layer for controlled 3D geological model generation.

The classes in this module are intentionally thin wrappers around the existing
``geogen.model`` processes. They make process parameters explicit and repeatable
while preserving the original compute and visualization pipeline.
"""

from __future__ import annotations

import functools
from dataclasses import dataclass
from numbers import Real
from typing import Any, Callable, Literal, Optional, Sequence, TypeAlias, Union

import numpy as np

from geogen.model.deferredparameter import BacktrackedPoint
from geogen.model.geomodel import GeoModel
from geogen.model.geoprocess import (
    Bedrock,
    DikeColumn,
    DikeHemisphere,
    DikePlane,
    DikePlug,
    Fault,
    Fold,
    GeoProcess,
    Sedimentation,
    UnconformityBase,
    UnconformityDepth,
)

ScalarRange: TypeAlias = tuple[float, float]
ScalarParam: TypeAlias = Union[int, float, ScalarRange, "Uniform", "Choice"]
PointParam: TypeAlias = Union[tuple[float, float, float], "PointSpec"]
BoundsParam: TypeAlias = tuple[float, float] | tuple[tuple[float, float], tuple[float, float], tuple[float, float]]
ResolutionParam: TypeAlias = int | tuple[int, int, int]


def _as_rng(seed: Optional[int] = None) -> np.random.Generator:
    return np.random.default_rng(seed)


@dataclass(frozen=True)
class Uniform:
    """Uniformly sampled scalar parameter."""

    low: float
    high: float


@dataclass(frozen=True)
class Choice:
    """Categorical parameter sampled from a finite set."""

    values: Sequence[Any]


@dataclass(frozen=True)
class PointSpec:
    """Point with fixed or sampled x/y/z components."""

    x: ScalarParam
    y: ScalarParam
    z: ScalarParam

    def sample(self, rng: np.random.Generator) -> tuple[float, float, float]:
        return (
            float(_sample_scalar(self.x, rng)),
            float(_sample_scalar(self.y, rng)),
            float(_sample_scalar(self.z, rng)),
        )


def _sample_scalar(value: ScalarParam, rng: np.random.Generator) -> Any:
    if isinstance(value, Uniform):
        return rng.uniform(value.low, value.high)
    if isinstance(value, Choice):
        return rng.choice(value.values)
    if (
        isinstance(value, Sequence)
        and not isinstance(value, (str, bytes))
        and len(value) == 2
        and all(isinstance(v, Real) for v in value)
    ):
        return rng.uniform(float(value[0]), float(value[1]))
    return value


def _sample_int(value: ScalarParam, rng: np.random.Generator, minimum: int = 1) -> int:
    sampled = _sample_scalar(value, rng)
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)) and len(value) == 2:
        low = int(np.floor(float(value[0])))
        high = int(np.floor(float(value[1])))
        sampled = rng.integers(low, high + 1)
    return max(minimum, int(sampled))


def _sample_point(value: PointParam, rng: np.random.Generator) -> tuple[float, float, float]:
    if isinstance(value, PointSpec):
        return value.sample(rng)
    if len(value) != 3:
        raise ValueError("Point parameters must have exactly three coordinates.")
    return tuple(float(v) for v in value)


def _maybe_backtracked(point: PointParam, rng: np.random.Generator, anchor_to_present: bool):
    sampled = _sample_point(point, rng)
    return BacktrackedPoint(sampled) if anchor_to_present else sampled


def _fourier_wave(
    n_cycles: np.ndarray,
    amplitudes: np.ndarray,
    phases: np.ndarray,
    frequency: float,
    rms_scale: float,
) -> np.ndarray:
    result = np.zeros_like(n_cycles, dtype=float)
    for n, (amplitude, phase) in enumerate(zip(amplitudes, phases), start=1):
        result += amplitude * np.sin(2 * np.pi * frequency * n * n_cycles + phase)
    return result * rms_scale


def _make_fourier_wave(
    rng: np.random.Generator,
    num_harmonics: int,
    frequency: float = 1.0,
    smoothness: float = 1.0,
) -> Callable[[np.ndarray], np.ndarray]:
    amplitudes = []
    phases = []
    total_power = 0.0
    for n in range(1, num_harmonics + 1):
        scale = 1.0 / (n**smoothness)
        amplitude = abs(rng.normal(loc=scale, scale=0.5 * scale))
        phase = rng.uniform(0, 2 * np.pi)
        amplitudes.append(amplitude)
        phases.append(phase)
        total_power += amplitude**2

    rms_scale = np.sqrt(1.0 / max(total_power, 1e-12))
    return functools.partial(
        _fourier_wave,
        amplitudes=np.array(amplitudes),
        phases=np.array(phases),
        frequency=frequency,
        rms_scale=rms_scale,
    )


class ConstantThickness:
    def __call__(self, x, y):
        return np.ones_like(x, dtype=float)


class OrganicDikeThickness:
    def __init__(self, length, exponent, amplitude, x_wave, y_wave):
        self.length = length
        self.exponent = exponent
        self.amplitude = amplitude
        self.x_wave = x_wave
        self.y_wave = y_wave

    def __call__(self, x, y):
        taper = np.sqrt(np.maximum(1 - np.abs((2 * y / self.length)) ** self.exponent, 0))
        return (1 + self.amplitude * self.x_wave(x / self.length)) * (
            1 + self.amplitude * self.y_wave(y / self.length)
        ) * taper


class EngineEventSpec:
    """Base protocol-like class for event specs."""

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        raise NotImplementedError


@dataclass
class SedimentationSpec(EngineEventSpec):
    """Generate a controlled sedimentation event."""

    depth: ScalarParam
    values: Optional[Sequence[int]] = None
    thicknesses: Optional[Sequence[float]] = None
    categories: Sequence[int] = (1, 2, 3, 4, 5)
    layer_count: Optional[ScalarParam] = None
    layer_thickness: ScalarParam = Uniform(100.0, 400.0)
    base: float = np.nan
    avoid_immediate_repeats: bool = True

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        if self.values is not None and self.thicknesses is not None:
            return [Sedimentation(self.values, self.thicknesses, base=self.base)]

        depth = float(_sample_scalar(self.depth, rng))
        if self.layer_count is not None:
            layer_count = _sample_int(self.layer_count, rng)
            raw = np.array([float(_sample_scalar(self.layer_thickness, rng)) for _ in range(layer_count)])
            thicknesses = (raw / raw.sum() * depth).tolist()
        else:
            thicknesses = []
            remaining = depth
            while remaining > 0:
                thickness = float(_sample_scalar(self.layer_thickness, rng))
                thicknesses.append(thickness)
                remaining -= thickness

        values = []
        previous = None
        for _ in thicknesses:
            options = list(self.categories)
            if self.avoid_immediate_repeats and previous in options and len(options) > 1:
                options.remove(previous)
            value = int(rng.choice(options))
            values.append(value)
            previous = value

        return [Sedimentation(values, thicknesses, base=self.base)]


@dataclass
class FoldSpec(EngineEventSpec):
    """Controlled fold transformation."""

    strike: ScalarParam = Uniform(0.0, 360.0)
    dip: ScalarParam = Uniform(60.0, 100.0)
    rake: ScalarParam = Uniform(0.0, 360.0)
    period: ScalarParam = Uniform(3000.0, 12000.0)
    amplitude: ScalarParam = Uniform(150.0, 700.0)
    phase: ScalarParam = Uniform(0.0, 2 * np.pi)
    shape: ScalarParam = 0.0
    origin: PointParam = (0.0, 0.0, 0.0)
    anchor_to_present: bool = True
    fourier_harmonics: Optional[ScalarParam] = None
    fourier_smoothness: ScalarParam = 1.2

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        harmonics = None
        if self.fourier_harmonics is not None:
            harmonics = _sample_int(self.fourier_harmonics, rng)

        periodic_func = None
        if harmonics:
            periodic_func = _make_fourier_wave(
                rng,
                num_harmonics=harmonics,
                smoothness=float(_sample_scalar(self.fourier_smoothness, rng)),
            )

        return [
            Fold(
                strike=float(_sample_scalar(self.strike, rng)),
                dip=float(_sample_scalar(self.dip, rng)),
                rake=float(_sample_scalar(self.rake, rng)),
                period=float(_sample_scalar(self.period, rng)),
                amplitude=float(_sample_scalar(self.amplitude, rng)),
                phase=float(_sample_scalar(self.phase, rng)),
                shape=float(_sample_scalar(self.shape, rng)),
                origin=_maybe_backtracked(self.origin, rng, self.anchor_to_present),
                periodic_func=periodic_func,
            )
        ]


@dataclass
class FaultSpec(EngineEventSpec):
    """Controlled brittle fault transformation."""

    strike: ScalarParam = Uniform(0.0, 360.0)
    dip: ScalarParam = Uniform(45.0, 90.0)
    rake: ScalarParam = Uniform(60.0, 120.0)
    amplitude: ScalarParam = Uniform(60.0, 1000.0)
    origin: PointParam = (0.0, 0.0, 0.0)
    anchor_to_present: bool = True

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        return [
            Fault(
                strike=float(_sample_scalar(self.strike, rng)),
                dip=float(_sample_scalar(self.dip, rng)),
                rake=float(_sample_scalar(self.rake, rng)),
                amplitude=float(_sample_scalar(self.amplitude, rng)),
                origin=_maybe_backtracked(self.origin, rng, self.anchor_to_present),
            )
        ]


@dataclass
class DikePlaneSpec(EngineEventSpec):
    """Controlled planar dike or sill."""

    origin: PointParam
    strike: ScalarParam = Uniform(0.0, 360.0)
    dip: ScalarParam = Uniform(80.0, 100.0)
    width: ScalarParam = Uniform(50.0, 300.0)
    value: ScalarParam = Choice((6, 7, 8))
    anchor_to_present: bool = True
    length: Optional[ScalarParam] = None
    organic: bool = True
    wobble: ScalarParam = Uniform(0.05, 0.2)

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        thickness_func = ConstantThickness()
        if self.organic and self.length is not None:
            length = float(_sample_scalar(self.length, rng))
            thickness_func = OrganicDikeThickness(
                length=length,
                exponent=float(rng.uniform(4.0, 10.0)),
                amplitude=float(_sample_scalar(self.wobble, rng)),
                x_wave=_make_fourier_wave(rng, 4, smoothness=1.0),
                y_wave=_make_fourier_wave(rng, 4, smoothness=1.0),
            )

        return [
            DikePlane(
                strike=float(_sample_scalar(self.strike, rng)),
                dip=float(_sample_scalar(self.dip, rng)),
                width=float(_sample_scalar(self.width, rng)),
                origin=_maybe_backtracked(self.origin, rng, self.anchor_to_present),
                value=int(_sample_scalar(self.value, rng)),
                thickness_func=thickness_func,
            )
        ]


@dataclass
class IntrusionSpec(EngineEventSpec):
    """Controlled local intrusion body anchored to a chosen target location.

    For hemisphere intrusions, ``origin`` is interpreted as a point inside the
    intrusion body. The underlying GeoProcess origin is shifted automatically.
    """

    origin: PointParam
    kind: Literal["hemisphere", "column", "plug"] = "hemisphere"
    value: ScalarParam = Choice((9, 10, 11))
    anchor_to_present: bool = True
    clip: bool = False
    diameter: ScalarParam = Uniform(600.0, 2500.0)
    height: ScalarParam = Uniform(150.0, 700.0)
    depth: ScalarParam = np.inf
    minor_axis_scale: ScalarParam = Uniform(0.6, 1.4)
    rotation: ScalarParam = Uniform(0.0, 360.0)
    upper: bool = True
    plug_shape: ScalarParam = Uniform(2.5, 5.0)

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        target = _sample_point(self.origin, rng)
        value = int(_sample_scalar(self.value, rng))
        diameter = float(_sample_scalar(self.diameter, rng))
        height = float(_sample_scalar(self.height, rng))
        minor_axis_scale = float(_sample_scalar(self.minor_axis_scale, rng))
        rotation = float(_sample_scalar(self.rotation, rng))

        if self.kind == "column":
            origin = BacktrackedPoint(target) if self.anchor_to_present else target
            return [
                DikeColumn(
                    origin=origin,
                    diam=diameter,
                    depth=float(_sample_scalar(self.depth, rng)),
                    minor_axis_scale=minor_axis_scale,
                    rotation=rotation,
                    value=value,
                    clip=self.clip,
                )
            ]
        if self.kind == "plug":
            origin = BacktrackedPoint(target) if self.anchor_to_present else target
            return [
                DikePlug(
                    origin=origin,
                    diam=diameter,
                    minor_axis_scale=minor_axis_scale,
                    rotation=rotation,
                    shape=float(_sample_scalar(self.plug_shape, rng)),
                    value=value,
                    clip=self.clip,
                )
            ]

        z_offset = -0.5 * height if self.upper else 0.5 * height
        process_origin = (target[0], target[1], target[2] + z_offset)
        origin = BacktrackedPoint(process_origin) if self.anchor_to_present else process_origin
        return [
            DikeHemisphere(
                origin=origin,
                diam=diameter,
                height=height,
                minor_axis_scale=minor_axis_scale,
                rotation=rotation,
                value=value,
                upper=self.upper,
                clip=self.clip,
            )
        ]


@dataclass
class UnconformitySpec(EngineEventSpec):
    """Controlled erosion/unconformity event."""

    mode: Literal["depth", "base"] = "depth"
    depth: ScalarParam = Uniform(100.0, 700.0)
    base: ScalarParam = 0.0
    value: float = np.nan

    def to_processes(self, rng: np.random.Generator) -> list[GeoProcess]:
        if self.mode == "base":
            return [UnconformityBase(float(_sample_scalar(self.base, rng)), value=self.value)]
        return [UnconformityDepth(float(_sample_scalar(self.depth, rng)), value=self.value)]


@dataclass
class GeoModelSpec:
    """Complete parameterized model recipe."""

    events: Sequence[EngineEventSpec]
    bounds: BoundsParam = ((-3840, 3840), (-3840, 3840), (-1920, 1920))
    resolution: ResolutionParam = (128, 128, 64)
    seed: Optional[int] = None
    normalize: bool = True
    height_tracking: bool = True
    name: str = "parametric_model"
    include_basement: bool = True
    basement_base: float = 0.0
    basement_value: int = 0


class ParametricGeoEngine:
    """Build ordinary GeoModels from explicit parameter specs."""

    def __init__(self, default_seed: Optional[int] = None):
        self.default_seed = default_seed

    def build_history(self, spec: GeoModelSpec) -> list[GeoProcess]:
        rng = _as_rng(spec.seed if spec.seed is not None else self.default_seed)
        history: list[GeoProcess] = []
        if spec.include_basement:
            history.append(Bedrock(base=spec.basement_base, value=spec.basement_value))

        for event in spec.events:
            history.extend(event.to_processes(rng))

        return history

    def generate(self, spec: GeoModelSpec, keep_snapshots: bool = True) -> GeoModel:
        model = GeoModel(
            bounds=spec.bounds,
            resolution=spec.resolution,
            name=spec.name,
            height_tracking=spec.height_tracking,
        )
        model.add_history(self.build_history(spec))
        model.compute_model(keep_snapshots=keep_snapshots, normalize=spec.normalize)
        return model

    def visualize(self, model: GeoModel, view: str = "vol", **kwargs):
        """Use the original geogen.plot visualization functions."""
        import geogen.plot as geovis

        views = {
            "vol": geovis.volview,
            "orthslice": geovis.orthsliceview,
            "nslice": geovis.nsliceview,
            "oneslice": geovis.onesliceview,
            "categorical": geovis.categorical_grid_view,
            "transformation": geovis.transformationview,
        }
        if view not in views:
            raise ValueError(f"Unknown view '{view}'. Choose from {sorted(views)}.")
        return views[view](model, **kwargs)


def generate_models(spec: GeoModelSpec, n: int, seed: Optional[int] = None) -> list[GeoModel]:
    """Generate a batch by varying the seed while preserving fixed anchors."""
    engine = ParametricGeoEngine(default_seed=seed)
    models = []
    base_seed = spec.seed if spec.seed is not None else seed
    seed_sequence = np.random.SeedSequence(base_seed)
    for child in seed_sequence.spawn(n):
        child_seed = int(child.generate_state(1)[0])
        child_spec = GeoModelSpec(
            events=spec.events,
            bounds=spec.bounds,
            resolution=spec.resolution,
            seed=child_seed,
            normalize=spec.normalize,
            height_tracking=spec.height_tracking,
            name=spec.name,
            include_basement=spec.include_basement,
            basement_base=spec.basement_base,
            basement_value=spec.basement_value,
        )
        models.append(engine.generate(child_spec))
    return models
