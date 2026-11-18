"""Deterministic predictions: connection probability, expected degree and the
configuration integrals I_wedge, I_clique, I_cycle of chapter 3 (d = 2).

The integrals are reduced to (r1, r3, theta) by rotation invariance,

    I = 4 pi  int_0^Rmax int_0^Rmax int_0^pi  F(r1, r3, theta) r1 r3  dtheta dr3 dr1,

and evaluated with Gauss-Legendre rules. For bounded kernels the theta range is
split where |v1 - v3| = 2R, the only place where the integrand has a kink.
"""

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.stats import qmc

from grig.geometry import lens_area, lens_length, triple_disk_area
from grig.kernels import Kernel, ball_volume, sphere_area

KAPPA = {1: 0.75, 2: 1 - 3 * np.sqrt(3) / (4 * np.pi), 3: 15 / 32}  # RGG clustering constants


def _one_minus_exp(x):
    return -np.expm1(-x)


# --- self-convolution f = g * g ----------------------------------------------


def _ball_overlap(s, R, d):
    s = np.asarray(s, dtype=float)
    if d == 1:
        return lens_length(s, R)
    if d == 2:
        return lens_area(s, R)
    if d == 3:
        s = np.clip(s, 0, 2 * R)
        return np.pi * (4 * R + s) * (2 * R - s) ** 2 / 12
    raise ValueError("closed form only for d <= 3")


@lru_cache(maxsize=64)
def _convolution_table(kernel: Kernel):
    """f(s) on a grid for unbounded kernels in the plane.

    By symmetry f(s) is twice the integral over the half-plane closer to 0 than
    to z. Integrating in polar coordinates around 0 on that half-plane keeps the
    cusp of g(x - z) at x = z out of the domain; a full-plane rule resolves it
    badly, which is what limited Table 4.1 of the thesis to 2% for the power law.
    """
    ell = kernel.length
    reach = kernel.truncation_radius(1.0, tol=1e-10)
    s = np.unique(np.concatenate([np.linspace(0, 8 * ell, 400), np.geomspace(8 * ell, 2 * reach, 400)]))

    node, weight = np.polynomial.legendre.leggauss(16)
    edges = np.concatenate([[0.0], np.geomspace(ell / 50, reach, 60)])
    a, b = edges[:-1, None], edges[1:, None]
    rho = (0.5 * (b - a) * node + 0.5 * (b + a)).ravel()
    w_rho = (0.5 * (b - a) * weight).ravel()
    g_rho = kernel(rho)

    u, wu = np.polynomial.legendre.leggauss(64)
    f = np.empty_like(s)
    for k, sk in enumerate(s):
        start = np.arccos(np.clip(sk / (2 * rho), -1.0, 1.0))  # x . z/|z| < s/2  <=>  phi > start
        phi = start[:, None] + 0.5 * (np.pi - start[:, None]) * (u + 1)
        w_phi = 0.5 * (np.pi - start[:, None]) * wu
        dist = np.sqrt(rho[:, None] ** 2 + sk**2 - 2 * rho[:, None] * sk * np.cos(phi))
        inner = 2 * np.sum(w_phi * kernel(dist), axis=1)  # phi and -phi
        f[k] = 2 * np.sum(w_rho * rho * g_rho * inner)
    return s, f


def self_convolution(kernel: Kernel, s, d: int = 2):
    if kernel.name == "boolean":
        return kernel.amplitude**2 * _ball_overlap(s, kernel.length, d)
    if d != 2:
        raise NotImplementedError("numerical self-convolution is implemented for d = 2")
    grid, _ = _convolution_table(kernel)
    s = np.asarray(s, dtype=float)
    return np.where(s <= grid[-1], _convolution_spline(kernel)(np.minimum(s, grid[-1])), 0.0)


@lru_cache(maxsize=64)
def _convolution_spline(kernel: Kernel):
    return CubicSpline(*_convolution_table(kernel))


def connection_probability(kernel: Kernel, mu: float, s, d: int = 2):
    """p(s) = 1 - exp(-mu f(s)), Proposition 2.2."""
    return _one_minus_exp(mu * self_convolution(kernel, s, d))


def _reach(kernel: Kernel, mu: float, d: int = 2) -> float:
    """Distance beyond which mu f(s) is negligible. For unbounded kernels a fixed
    cut (f < 1e-9 ||g||^2) is used, so that cached quadrature nodes can be reused
    across mu; it is accurate for mu up to ~1e3."""
    if np.isfinite(kernel.support):
        return 2 * kernel.support
    grid, f = _convolution_table(kernel)
    above = np.nonzero(f > 1e-9 * kernel.norm(2) ** 2)[0]
    return float(grid[above[-1]])


def expected_degree(kernel: Kernel, lam: float, mu: float, d: int = 2) -> float:
    """E[deg] = lam * int (1 - exp(-mu f(z))) dz, Proposition 2.7."""
    reach = _reach(kernel, mu, d)
    val, _ = integrate.quad(
        lambda s: s ** (d - 1) * connection_probability(kernel, mu, s, d), 0, reach, limit=400, points=[reach / 2]
    )
    return lam * sphere_area(d) * val


# --- triple overlap -------------------------------------------------------------


def triple_overlap(kernel: Kernel, v1: np.ndarray, v3: np.ndarray, samples: int = 4096, seed: int = 0):
    """K_103(v1, v3) = int g(x - v1) g(x) g(x - v3) dx in the plane."""
    if kernel.name == "boolean":
        return kernel.amplitude**3 * triple_disk_area(v1, v3, kernel.length)
    # importance sampling with density g / ||g||, one common scrambled Sobol
    # sample for all configurations
    reach = kernel.truncation_radius(1.0, tol=1e-12)
    r = np.linspace(0, reach, 20001)
    dens = r * kernel(r)
    cdf = np.concatenate([[0], np.cumsum(0.5 * np.diff(r) * (dens[1:] + dens[:-1]))])
    q = qmc.Sobol(2, seed=seed).random(samples)
    rad = np.interp(q[:, 0] * cdf[-1], cdf, r)
    x = np.stack([rad * np.cos(2 * np.pi * q[:, 1]), rad * np.sin(2 * np.pi * q[:, 1])], axis=1)

    out = np.empty(v1.shape[:-1])
    flat1, flat3, flat_out = v1.reshape(-1, 2), v3.reshape(-1, 2), out.reshape(-1)
    for start in range(0, len(flat1), 1000):
        a, b = flat1[start : start + 1000], flat3[start : start + 1000]
        g1 = kernel(np.linalg.norm(x[None] - a[:, None], axis=-1))
        g3 = kernel(np.linalg.norm(x[None] - b[:, None], axis=-1))
        flat_out[start : start + 1000] = np.einsum("ij,ij->i", g1, g3) / samples
    return out * kernel.norm(2)


# --- configuration integrals ------------------------------------------------------


@dataclass
class Clustering:
    wedge: float
    clique: float
    cycle: float

    @property
    def triangle(self) -> float:
        return self.clique + self.cycle

    @property
    def C(self) -> float:
        return self.triangle / self.wedge

    @property
    def C_clique(self) -> float:
        return self.clique / self.wedge

    @property
    def C_cycle(self) -> float:
        return self.cycle / self.wedge


def _gl(a, b, n):
    x, w = np.polynomial.legendre.leggauss(n)
    return 0.5 * (b - a) * x + 0.5 * (b + a), 0.5 * (b - a) * w


@lru_cache(maxsize=16)
def _configurations(kernel: Kernel, reach: float, n_r: int, n_t: int):
    """Quadrature nodes (r1, r3, theta), weights including the 4 pi r1 r3 Jacobian,
    and the overlaps that do not depend on mu."""
    r, wr = _gl(0.0, reach, n_r)
    R1, R3 = np.meshgrid(r, r, indexing="ij")
    W = np.outer(wr, wr) * R1 * R3 * 4 * np.pi
    if kernel.name == "boolean":
        # split theta where |v1 - v3| = 2R
        cos_star = np.clip((R1**2 + R3**2 - reach**2) / (2 * R1 * R3), -1, 1)
        star = np.arccos(cos_star)
        u, wu = np.polynomial.legendre.leggauss(n_t)
        u, wu = 0.5 * (u + 1), 0.5 * wu
        theta = np.concatenate([star[..., None] * u, star[..., None] + (np.pi - star[..., None]) * u], axis=-1)
        wt = np.concatenate([star[..., None] * wu, (np.pi - star[..., None]) * wu], axis=-1)
    else:
        t, wt1 = _gl(0.0, np.pi, 2 * n_t)
        theta = np.broadcast_to(t, R1.shape + t.shape)
        wt = np.broadcast_to(wt1, R1.shape + t.shape)
    R1e, R3e = R1[..., None], R3[..., None]
    v1 = np.stack([np.broadcast_to(R1e, theta.shape), np.zeros(theta.shape)], axis=-1)
    v3 = np.stack([R3e * np.cos(theta), R3e * np.sin(theta)], axis=-1)
    r13 = np.linalg.norm(v1 - v3, axis=-1)
    weights = W[..., None] * wt
    k10 = self_convolution(kernel, np.broadcast_to(R1e, theta.shape))
    k03 = self_convolution(kernel, np.broadcast_to(R3e, theta.shape))
    k13 = self_convolution(kernel, r13)
    k103 = triple_overlap(kernel, v1, v3)
    return weights, k10, k03, k13, k103


def _nodes(kernel: Kernel, n_r, n_t):
    # bounded kernels need more nodes for the kinks of the triple overlap;
    # unbounded ones are smooth but their triple overlap is costly
    if n_r is None:
        n_r = 48 if kernel.name == "boolean" else 32
    if n_t is None:
        n_t = 48 if kernel.name == "boolean" else 24
    return n_r, n_t


def configuration_integrals(kernel: Kernel, mu: float, n_r: int | None = None, n_t: int | None = None) -> Clustering:
    """I_wedge, I_clique, I_cycle of (3.7), (3.12), (3.13) in d = 2."""
    n_r, n_t = _nodes(kernel, n_r, n_t)
    reach = _reach(kernel, mu)
    w, k10, k03, k13, k103 = _configurations(kernel, reach, n_r, n_t)
    l123 = mu * k103
    excl = [mu * k - l123 for k in (k10, k03, k13)]
    clique = _one_minus_exp(l123)
    stay = np.exp(-l123)
    wedge = clique + stay * _one_minus_exp(excl[0]) * _one_minus_exp(excl[1])
    cycle = stay * _one_minus_exp(excl[0]) * _one_minus_exp(excl[1]) * _one_minus_exp(excl[2])
    return Clustering(float(np.sum(w * wedge)), float(np.sum(w * clique)), float(np.sum(w * cycle)))


def clustering(kernel: Kernel, mu: float, **kw) -> float:
    return configuration_integrals(kernel, mu, **kw).C


def rcm_clustering(kernel: Kernel, mu: float = 1.0, n_r: int | None = None, n_t: int | None = None) -> float:
    """Clustering of the random connection model with h = 1 - exp(-mu g*g), i.e.
    independent edges with the GRIG's connection probability. With mu = 1 and
    the unscaled kernel this is the attribute-rich limit of Proposition 3.6."""
    n_r, n_t = _nodes(kernel, n_r, n_t)
    reach = _reach(kernel, mu)
    w, k10, k03, k13, _ = _configurations(kernel, reach, n_r, n_t)
    h10, h03, h13 = (_one_minus_exp(mu * k) for k in (k10, k03, k13))
    return float(np.sum(w * h10 * h03 * h13) / np.sum(w * h10 * h03))


# --- the purely geometric limit -----------------------------------------------------


def enclosing_radius(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> np.ndarray:
    """Radius of the smallest ball containing three points (any dimension)."""
    ab = np.linalg.norm(a - b, axis=-1)
    bc = np.linalg.norm(b - c, axis=-1)
    ca = np.linalg.norm(c - a, axis=-1)
    sides = np.sort(np.stack([ab, bc, ca], axis=-1), axis=-1)
    x, y, z = sides[..., 0], sides[..., 1], sides[..., 2]
    obtuse = x**2 + y**2 <= z**2
    s = (x + y + z) / 2
    area = np.sqrt(np.clip(s * (s - x) * (s - y) * (s - z), 0, None))
    with np.errstate(divide="ignore", invalid="ignore"):
        circum = x * y * z / (4 * area)
    return np.where(obtuse | (area == 0), z / 2, circum)


def continuum_constants(d: int, n: int, rng: np.random.Generator, R: float = 1.0) -> tuple[float, float, float]:
    """Monte Carlo for (kappa_d, sigma_d): w1, w3 uniform in B_2R; kappa is the
    probability that |w1 - w3| < 2R, sigma that the three points fit in a ball of
    radius R (one group can cover them). Returns kappa, sigma and the standard
    error of sigma."""

    def ball(m):
        x = rng.standard_normal((m, d))
        x /= np.linalg.norm(x, axis=1, keepdims=True)
        return x * (2 * R) * rng.random((m, 1)) ** (1 / d)

    w1, w3 = ball(n), ball(n)
    kappa = np.mean(np.linalg.norm(w1 - w3, axis=1) < 2 * R)
    fits = enclosing_radius(np.zeros_like(w1), w1, w3) <= R
    sigma = fits.mean()
    return float(kappa), float(sigma), float(fits.std() / np.sqrt(n))


__all__ = [
    "KAPPA",
    "Clustering",
    "ball_volume",
    "clustering",
    "configuration_integrals",
    "connection_probability",
    "continuum_constants",
    "expected_degree",
    "rcm_clustering",
    "self_convolution",
    "triple_overlap",
]
