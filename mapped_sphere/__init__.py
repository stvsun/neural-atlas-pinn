"""Mapped Sphere Method — shared solver library for the paper examples."""

from .geometry import ChartGeometry, divergence, jacobian, laplace_beltrami, scalar_gradient
from .maps import TorusChart

__all__ = [
    "ChartGeometry", "TorusChart", "divergence", "jacobian",
    "laplace_beltrami", "scalar_gradient",
]

__version__ = "1.0.0"
