"""Two-step construction of the GRIG on the flat torus [0, L)^d.

The bipartite layer is kept: it is the only place where a triangle's mechanism
(one group covering all three vertices, or three different groups) is visible.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree

from grig.kernels import Kernel


@dataclass
class GRIG:
    L: float
    d: int
    V: np.ndarray  # vertex positions, (n, d)
    U: np.ndarray  # group positions, (m, d)
    B: sparse.csr_matrix  # incidence vertex x group
    A: sparse.csr_matrix  # adjacency of the projected graph

    @property
    def n(self) -> int:
        return len(self.V)

    def degrees(self) -> np.ndarray:
        return np.asarray(self.A.sum(axis=1)).ravel()


def poisson_points(intensity: float, L: float, d: int, rng: np.random.Generator) -> np.ndarray:
    return rng.uniform(0, L, size=(rng.poisson(intensity * L**d), d))


def _pairs_within(X, Y, r, L):
    """All (i, j, distance) with |X_i - Y_j|_torus <= r."""
    tx = cKDTree(X, boxsize=L)
    ty = cKDTree(Y, boxsize=L)
    coo = tx.sparse_distance_matrix(ty, r, output_type="coo_matrix")
    return coo.row, coo.col, coo.data


def _projection(B: sparse.csr_matrix) -> sparse.csr_matrix:
    A = (B @ B.T).tocsr()
    A.setdiag(0)
    A.eliminate_zeros()
    A.data[:] = 1
    return A.astype(np.int32)


def sample(kernel: Kernel, lam: float, mu: float, L: float, rng: np.random.Generator, d: int = 2) -> GRIG:
    reach = kernel.truncation_radius(mu, d)
    if L <= 4 * reach:
        raise ValueError(f"torus side L={L} must exceed 4 * reach = {4 * reach:.2f}")
    V = poisson_points(lam, L, d, rng)
    U = poisson_points(mu, L, d, rng)
    if len(V) == 0 or len(U) == 0:
        B = sparse.csr_matrix((len(V), len(U)), dtype=np.int8)
    else:
        i, j, dist = _pairs_within(V, U, reach, L)
        keep = rng.random(len(dist)) < kernel(dist)
        B = sparse.csr_matrix((np.ones(keep.sum(), dtype=np.int8), (i[keep], j[keep])), shape=(len(V), len(U)))
    return GRIG(L, d, V, U, B, _projection(B))


def sample_rcm(p, reach: float, lam: float, L: float, rng: np.random.Generator, d: int = 2):
    """Random connection model with the same vertex process: independent edges,
    P(edge) = p(distance). Used to compare the GRIG with its marginal model."""
    V = poisson_points(lam, L, d, rng)
    tree = cKDTree(V, boxsize=L)
    pairs = tree.query_pairs(reach, output_type="ndarray")
    dist = np.linalg.norm(
        np.minimum(np.abs(V[pairs[:, 0]] - V[pairs[:, 1]]), L - np.abs(V[pairs[:, 0]] - V[pairs[:, 1]])), axis=1
    )
    e = pairs[rng.random(len(dist)) < p(dist)]
    A = sparse.coo_matrix((np.ones(len(e), dtype=np.int32), (e[:, 0], e[:, 1])), shape=(len(V), len(V)))
    return V, (A + A.T).tocsr()


# --- counts ----------------------------------------------------------------------


def triangles_per_vertex(A: sparse.csr_matrix) -> np.ndarray:
    # (A^2 ∘ A) 1 = 2 * (triangles through each vertex)
    return np.asarray((A @ A).multiply(A).sum(axis=1)).ravel() // 2


def clique_triangles(B: sparse.csr_matrix) -> int:
    """Number of distinct vertex triples covered by at least one common group."""
    G = B.tocsc()
    sizes = np.diff(G.indptr)
    n = B.shape[0]
    keys = []
    for k in np.unique(sizes[sizes >= 3]):
        cols = np.flatnonzero(sizes == k)
        members = np.sort(np.stack([G.indices[G.indptr[c] : G.indptr[c + 1]] for c in cols]), axis=1).astype(np.int64)
        idx = np.array(list(combinations(range(k), 3)))
        t = members[:, idx].reshape(-1, 3)
        keys.append((t[:, 0] * n + t[:, 1]) * n + t[:, 2])
    if not keys:
        return 0
    return len(np.unique(np.concatenate(keys)))


def counts(g: GRIG, mechanisms: bool = True) -> dict:
    deg = g.degrees()
    tri = int(triangles_per_vertex(g.A).sum() // 3)
    out = {
        "vertices": g.n,
        "groups": len(g.U),
        "edges": int(deg.sum() // 2),
        "wedges": int((deg * (deg - 1) // 2).sum()),
        "triangles": tri,
    }
    if mechanisms:
        clique = clique_triangles(g.B)
        out["clique_triangles"] = clique
        out["cycle_triangles"] = tri - clique
    return out


def giant_fraction(A: sparse.csr_matrix) -> float:
    if A.shape[0] == 0:
        return 0.0
    _, labels = connected_components(A, directed=False)
    return np.bincount(labels).max() / A.shape[0]


def pair_distances(g: GRIG, r_max: float):
    """Distances of all vertex pairs within r_max and whether they are adjacent."""
    tree = cKDTree(g.V, boxsize=g.L)
    pairs = tree.query_pairs(r_max, output_type="ndarray")
    delta = np.abs(g.V[pairs[:, 0]] - g.V[pairs[:, 1]])
    dist = np.linalg.norm(np.minimum(delta, g.L - delta), axis=1)
    adjacent = np.asarray(g.A[pairs[:, 0], pairs[:, 1]]).ravel() > 0
    return dist, adjacent
