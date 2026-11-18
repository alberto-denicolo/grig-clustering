"""Simulation and theory of the Geometric Random Intersection Graph (GRIG)."""

from grig.kernels import Kernel, boolean, exponential, gaussian, power_law
from grig.simulate import GRIG, counts, sample
from grig.theory import KAPPA, clustering, configuration_integrals, connection_probability, expected_degree

__version__ = "1.0.0"

__all__ = [
    "GRIG",
    "KAPPA",
    "Kernel",
    "boolean",
    "clustering",
    "configuration_integrals",
    "connection_probability",
    "counts",
    "exponential",
    "expected_degree",
    "gaussian",
    "power_law",
    "sample",
]
