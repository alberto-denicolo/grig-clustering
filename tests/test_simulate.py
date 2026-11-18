import networkx as nx
import numpy as np
import pytest

from grig.estimators import pooled_ratio
from grig.kernels import boolean, exponential
from grig.simulate import clique_triangles, counts, sample


def test_counts_match_networkx():
    g = sample(boolean(1.0), 1.5, 1.0, 12.0, np.random.default_rng(0))
    c = counts(g)
    G = nx.from_scipy_sparse_array(g.A)
    assert c["triangles"] == sum(nx.triangles(G).values()) // 3
    assert 3 * c["triangles"] / c["wedges"] == pytest.approx(nx.transitivity(G))


def test_clique_triangles_by_enumeration():
    g = sample(boolean(1.0), 1.0, 0.6, 8.0, np.random.default_rng(3))
    B = g.B.toarray()
    G = nx.from_scipy_sparse_array(g.A)
    triangles = [t for t in nx.enumerate_all_cliques(G) if len(t) == 3]
    covered = sum(1 for t in triangles if (B[list(t)].min(axis=0) > 0).any())
    assert clique_triangles(g.B) == covered


def test_every_group_is_a_clique():
    g = sample(boolean(1.0), 2.0, 0.5, 10.0, np.random.default_rng(4))
    G = g.B.tocsc()
    for j in range(G.shape[1]):
        m = G.indices[G.indptr[j] : G.indptr[j + 1]]
        sub = g.A[m][:, m].toarray()
        assert np.all(sub + np.eye(len(m)) >= 1)


def test_edges_never_exceed_2R():
    g = sample(boolean(1.0), 2.0, 2.0, 10.0, np.random.default_rng(5))
    A = g.A.tocoo()
    delta = np.abs(g.V[A.row] - g.V[A.col])
    dist = np.linalg.norm(np.minimum(delta, 10.0 - delta), axis=1)
    assert dist.max() <= 2.0 + 1e-9


def test_reproducible_and_torus_guard():
    a = counts(sample(exponential(3.0), 1.0, 1.0, 20.0, np.random.default_rng(9)))
    b = counts(sample(exponential(3.0), 1.0, 1.0, 20.0, np.random.default_rng(9)))
    assert a == b
    with pytest.raises(ValueError):
        sample(boolean(1.0), 1.0, 1.0, 3.0, np.random.default_rng(0))


def test_pooled_ratio_and_jackknife():
    est, se = pooled_ratio([1, 1, 1, 1], [3, 3, 3, 3])
    assert est == pytest.approx(1.0) and se == pytest.approx(0.0)
    est, se = pooled_ratio([1, 2, 1, 2], [3, 6, 4, 5])
    assert est == pytest.approx(1.0) and se > 0


def test_rcm_has_the_grig_mean_degree():
    from grig import theory as T
    from grig.simulate import sample_rcm

    k = boolean(1.0)
    rng = np.random.default_rng(12)
    deg = [
        sample_rcm(lambda d: T.connection_probability(k, 1.0, d), 2.0, 1.5, 20.0, rng)[1].sum() / 600 for _ in range(30)
    ]
    assert np.mean(deg) == pytest.approx(T.expected_degree(k, 1.5, 1.0), rel=0.03)


def test_giant_fraction():
    from scipy import sparse

    from grig.simulate import giant_fraction

    # a path on 3 vertices plus an isolated vertex
    A = sparse.csr_matrix(np.array([[0, 1, 0, 0], [1, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 0]]))
    assert giant_fraction(A) == 0.75
