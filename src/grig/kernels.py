"""Radial connection functions g(x) = amplitude * shape(|x| / length)."""

import warnings
from dataclasses import dataclass, replace
from math import gamma, pi

import numpy as np
from scipy import integrate, optimize


def sphere_area(d: int) -> float:
    return 2 * pi ** (d / 2) / gamma(d / 2)


def ball_volume(d: int) -> float:
    return pi ** (d / 2) / gamma(d / 2 + 1)


SHAPES = {
    "boolean": (lambda r: (r <= 1.0).astype(float), 1.0),
    "exponential": (lambda r: np.exp(-r), np.inf),
    "gaussian": (lambda r: np.exp(-0.5 * r**2), np.inf),
}


@dataclass(frozen=True)
class Kernel:
    name: str
    length: float = 1.0
    amplitude: float = 1.0
    alpha: float = 4.0  # only used by the power law

    def shape(self, r):
        if self.name == "power_law":
            return (1.0 + r) ** (-self.alpha)
        return SHAPES[self.name][0](r)

    @property
    def support(self) -> float:
        base = 1.0 if self.name == "boolean" else np.inf
        return base * self.length

    def __call__(self, r):
        r = np.asarray(r, dtype=float)
        return self.amplitude * self.shape(r / self.length)

    def norm(self, d: int = 2) -> float:
        """||g||_1 in R^d."""
        if self.name == "boolean":
            return self.amplitude * ball_volume(d) * self.length**d
        radial, _ = integrate.quad(lambda r: self.shape(r) * r ** (d - 1), 0, np.inf, limit=500)
        return self.amplitude * sphere_area(d) * radial * self.length**d

    def dilated(self, factor: float) -> "Kernel":
        return replace(self, length=self.length * factor)

    def scaled(self, factor: float) -> "Kernel":
        return replace(self, amplitude=self.amplitude * factor)

    def normalized(self, d: int = 2) -> "Kernel":
        """Same shape, length chosen so that ||g|| = 1 (amplitude kept)."""
        return self.dilated((1.0 / self.norm(d)) ** (1.0 / d))

    def truncation_radius(self, mu: float, d: int = 2, tol: float = 1e-4) -> float:
        """Smallest R with mu * int_{|x|>R} g <= tol; the support radius for the Boolean kernel."""
        if np.isfinite(self.support):
            return self.support

        def tail(R):
            # substitute r = R + t / (1 - t) to integrate the tail on [0, 1)
            val, _ = integrate.quad(
                lambda t: self(R + t / (1 - t)) * (R + t / (1 - t)) ** (d - 1) / (1 - t) ** 2, 0, 1, limit=500
            )
            return mu * sphere_area(d) * val - tol

        with warnings.catch_warnings():
            warnings.simplefilter("ignore", integrate.IntegrationWarning)
            hi = self.length
            while tail(hi) > 0:
                hi *= 2
            return optimize.brentq(tail, 0.0, hi, xtol=1e-6 * self.length)


def boolean(R: float = 1.0) -> Kernel:
    return Kernel("boolean", length=R)


def exponential(beta: float = 1.0) -> Kernel:
    return Kernel("exponential", length=1.0 / beta)


def gaussian(s: float = 1.0) -> Kernel:
    return Kernel("gaussian", length=s)


def power_law(alpha: float = 4.0) -> Kernel:
    return Kernel("power_law", alpha=alpha)


def from_config(spec: dict) -> Kernel:
    spec = dict(spec)
    name = spec.pop("name")
    normalize = spec.pop("normalize", False)
    k = {"boolean": boolean, "exponential": exponential, "gaussian": gaussian, "power_law": power_law}[name](**spec)
    return k.normalized() if normalize else k
