import numpy as np
import pytest
from scipy import integrate

from grig import theory as T
from grig.kernels import boolean, exponential, gaussian, power_law


def test_clustering_value_of_section_4_3_1():
    # Boolean kernel, R = 1, mu = 1: C = 0.645860, I_wedge = 34.65035, I_triangle = 22.37928
    c = T.configuration_integrals(boolean(1.0), 1.0)
    assert c.C == pytest.approx(0.645860, abs=5e-6)
    assert c.wedge == pytest.approx(34.65035, rel=5e-6)
    assert c.triangle == pytest.approx(22.37928, rel=5e-6)


def test_quadrature_is_converged():
    k = boolean(1.0)
    a = T.configuration_integrals(k, 0.3)
    b = T.configuration_integrals(k, 0.3, n_r=96, n_t=96)
    assert a.C == pytest.approx(b.C, rel=1e-5)


@pytest.mark.parametrize(
    "lam, mu, expected",
    [
        (1, 0.05, 0.4762),
        (1, 0.2, 1.7201),
        (1, 1, 5.5417),
        (1, 5, 9.6131),
        (0.5, 1, 2.7709),
        (2, 1, 11.0834),
        (3, 0.5, 10.7385),
    ],
)
def test_expected_degree_table_4_2(lam, mu, expected):
    assert T.expected_degree(boolean(1.0), lam, mu) == pytest.approx(expected, abs=1e-4)


def test_sparse_limit_constant():
    # I_wedge / mu and I_clique / mu both tend to ||g||^3 = pi^3
    c = T.configuration_integrals(boolean(1.0), 1e-4)
    assert c.wedge / 1e-4 == pytest.approx(np.pi**3, rel=1e-3)
    assert c.C_clique == pytest.approx(1.0, abs=1e-3)


def test_rcm_limit_value():
    assert T.rcm_clustering(boolean(1.0)) == pytest.approx(0.43723, abs=1e-5)


def test_decomposition_adds_up():
    c = T.configuration_integrals(boolean(1.0), 2.0)
    assert c.C == pytest.approx(c.C_clique + c.C_cycle)


def test_gaussian_self_convolution_closed_form():
    # g = exp(-|x|^2/2): g*g = pi exp(-|z|^2/4)
    s = np.array([0.0, 0.5, 1.0, 2.0])
    assert np.allclose(T.self_convolution(gaussian(1.0), s), np.pi * np.exp(-(s**2) / 4), rtol=5e-4)


@pytest.mark.parametrize("kernel", [exponential(1.0), power_law(4.0)])
def test_integral_of_f_is_norm_squared(kernel):
    k = kernel.normalized()
    grid, f = T._convolution_table(k)
    total = integrate.simpson(2 * np.pi * grid * f, x=grid)
    assert total == pytest.approx(1.0, rel=1e-4)


def test_continuum_constants():
    k, s, se = T.continuum_constants(2, 400_000, np.random.default_rng(0))
    assert k == pytest.approx(T.KAPPA[2], abs=3e-3)
    assert s < k
    k1, s1, _ = T.continuum_constants(1, 200_000, np.random.default_rng(1))
    assert k1 == pytest.approx(0.75, abs=3e-3)
    assert s1 == pytest.approx(k1, abs=1e-12)  # Helly: on the line every feasible triangle is clique-feasible
