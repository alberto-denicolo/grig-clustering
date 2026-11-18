import numpy as np
import pytest

from grig.geometry import lens_area, torus_distance, triple_disk_area


def test_triple_area_matches_monte_carlo():
    rng = np.random.default_rng(0)
    c1, c3 = rng.uniform(-1.5, 1.5, (5, 2)), rng.uniform(-1.5, 1.5, (5, 2))
    exact = triple_disk_area(c1, c3)
    pts = rng.uniform(-1, 1, (400_000, 2))
    for k in range(5):
        inside = (
            (np.linalg.norm(pts, axis=1) <= 1)
            & (np.linalg.norm(pts - c1[k], axis=1) <= 1)
            & (np.linalg.norm(pts - c3[k], axis=1) <= 1)
        )
        assert exact[k] == pytest.approx(4 * inside.mean(), abs=0.01)


def test_triple_area_degenerate_cases():
    z = np.array([[0.0, 0.0]])
    a = np.array([[0.7, 0.0]])
    assert triple_disk_area(z, z)[0] == pytest.approx(np.pi)
    assert triple_disk_area(a, a)[0] == pytest.approx(lens_area(0.7))
    assert triple_disk_area(np.array([[2.5, 0.0]]), a)[0] == 0.0


def test_pairwise_feasible_but_no_common_point():
    # equilateral triangle of side 1.9R: every pair of disks meets, the three do not
    s = 1.9
    v1 = np.array([[s, 0.0]])
    v3 = np.array([[s / 2, s * np.sqrt(3) / 2]])
    assert triple_disk_area(v1, v3)[0] == 0.0


def test_torus_distance_wraps():
    assert torus_distance(np.array([0.1, 0.1]), np.array([9.9, 9.9]), 10.0) == pytest.approx(np.sqrt(0.08))
