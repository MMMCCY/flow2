"""Parameter driven model generation helpers.

This package extends the existing GeoModel/GeoProcess engine without replacing it.
Generated models are ordinary :class:`geogen.model.GeoModel` instances, so the
existing plotting utilities remain compatible.
"""

from .parametric import (
    Choice,
    DikePlaneSpec,
    FaultSpec,
    FoldSpec,
    GeoModelSpec,
    IntrusionSpec,
    ParametricGeoEngine,
    PointSpec,
    SedimentationSpec,
    UnconformitySpec,
    Uniform,
    generate_models,
)

__all__ = [
    "Choice",
    "DikePlaneSpec",
    "FaultSpec",
    "FoldSpec",
    "GeoModelSpec",
    "IntrusionSpec",
    "ParametricGeoEngine",
    "PointSpec",
    "SedimentationSpec",
    "UnconformitySpec",
    "Uniform",
    "generate_models",
]
