import numpy as np


def torus_delta(x: np.ndarray, y: np.ndarray, L: float) -> np.ndarray:
    d = np.abs(x - y)
    return np.minimum(d, L - d)


def torus_distance(x: np.ndarray, y: np.ndarray, L: float) -> np.ndarray:
    return np.linalg.norm(torus_delta(x, y, L), axis=-1)


def lens_area(s, R: float = 1.0):
    """Area of B_R(0) ∩ B_R(z) in the plane, |z| = s."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 2 * R)
    return 2 * R**2 * np.arccos(s / (2 * R)) - 0.5 * s * np.sqrt(4 * R**2 - s**2)


def lens_length(s, R: float = 1.0):
    return np.clip(2 * R - np.asarray(s, dtype=float), 0.0, None)


def _wrap(a):
    return (a + np.pi) % (2 * np.pi) - np.pi


def triple_disk_area(c1: np.ndarray, c3: np.ndarray, R: float = 1.0) -> np.ndarray:
    """Area of B_R(0) ∩ B_R(c1) ∩ B_R(c3) for arrays of centres of shape (..., 2).

    The intersection is convex and its boundary is made of arcs of the three
    circles, each arc being the part of one circle lying inside the other two
    disks. Green's theorem on those arcs gives the area exactly.
    """
    c1 = np.asarray(c1, dtype=float)
    c3 = np.asarray(c3, dtype=float)
    centres = [np.zeros_like(c1), c1, c3]
    area = np.zeros(c1.shape[:-1])

    for i in range(3):
        ci = centres[i]
        # a circle that coincides with an earlier one would trace the same arcs twice
        duplicate = np.zeros(area.shape, dtype=bool)
        for j in range(i):
            duplicate |= np.linalg.norm(ci - centres[j], axis=-1) < 1e-12
        lo = np.full(area.shape, -np.pi)
        hi = np.full(area.shape, np.pi)
        empty = duplicate.copy()
        phi0 = np.zeros(area.shape)
        for k, cj in enumerate(c for j, c in enumerate(centres) if j != i):
            v = cj - ci
            dist = np.linalg.norm(v, axis=-1)
            same = dist < 1e-12  # coincident disk: no constraint
            empty |= dist >= 2 * R
            half = np.where(same, np.pi, np.arccos(np.clip(dist / (2 * R), 0.0, 1.0)))
            phi = np.arctan2(v[..., 1], v[..., 0])
            if k == 0:
                phi0 = np.where(same, 0.0, phi)
            centre = np.where(same, 0.0, _wrap(phi - phi0))
            lo = np.maximum(lo, centre - half)
            hi = np.minimum(hi, centre + half)
        empty |= hi <= lo
        a, b = phi0 + lo, phi0 + hi
        x, y = ci[..., 0], ci[..., 1]
        contrib = R**2 * (b - a) + R * x * (np.sin(b) - np.sin(a)) - R * y * (np.cos(b) - np.cos(a))
        area += np.where(empty, 0.0, 0.5 * contrib)
    return area
