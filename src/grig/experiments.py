"""Chapter 4 of the thesis, one function per figure or table, plus three new
experiments (marked NEW). Each function takes its section of the YAML config
and returns one or more DataFrames.

Replicate k of parameter point i in experiment e uses the seed (seed, e, i, k),
so every number can be recomputed on its own.
"""

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy import integrate, optimize

from grig import theory as T
from grig.estimators import mean_se, pooled_ratio
from grig.kernels import from_config
from grig.simulate import (
    counts,
    giant_fraction,
    pair_distances,
    sample,
    sample_rcm,
    triangles_per_vertex,
)

IDS = {
    "kernels": 1,
    "connection_probability": 2,
    "expected_degree": 3,
    "convergence": 4,
    "vertex_intensity": 5,
    "decomposition": 6,
    "sparse_regime": 7,
    "attribute_rich": 8,
    "continuum_limit": 9,
    "dimensions": 10,
    "kernel_comparison": 11,
    "grig_vs_rcm": 12,
    "higher_dimensions": 13,
    "percolation": 14,
}


def _rng(seed, name, i, k=0):
    return np.random.default_rng([seed, IDS[name], i, k])


def _grid(spec):
    """A list, or {log: [a, b], num: n} for a log-spaced grid."""
    if isinstance(spec, dict):
        return list(np.logspace(np.log10(spec["log"][0]), np.log10(spec["log"][1]), spec["num"]))
    return list(spec)


def _replicates(job, n, n_jobs):
    if n_jobs == 1:
        return [job(k) for k in range(n)]
    return Parallel(n_jobs=n_jobs)(delayed(job)(k) for k in range(n))


def _clustering_row(cs: list[dict]) -> dict:
    df = pd.DataFrame(cs)
    row = {"replicates": len(df), "vertices": df.vertices.mean(), "mean_degree": 2 * df.edges.sum() / df.vertices.sum()}
    row["C"], row["C_se"] = pooled_ratio(df.triangles, df.wedges)
    if "clique_triangles" in df:
        row["C_clique"], row["C_clique_se"] = pooled_ratio(df.clique_triangles, df.wedges)
        row["C_cycle"], row["C_cycle_se"] = pooled_ratio(df.cycle_triangles, df.wedges)
    with np.errstate(invalid="ignore", divide="ignore"):
        row["C_mean_of_ratios"] = float(np.nanmean(3 * df.triangles / df.wedges))
    return row


def _simulate_clustering(kernel, lam, mu, L, reps, seed, name, i, n_jobs, d=2, mechanisms=True):
    def job(k):
        return counts(sample(kernel, lam, mu, L, _rng(seed, name, i, k), d), mechanisms)

    return _clustering_row(_replicates(job, reps, n_jobs))


# --- validation of the construction (section 4.1-4.2) ----------------------------


def kernels(cfg, seed, n_jobs=1, log=print):
    """Table 4.1: the identity int f = ||g||^2 and the truncation radius."""
    rows = []
    for spec in cfg["kernels"]:
        k = from_config(spec)
        norm = k.norm(2)
        if k.name == "boolean":
            f_int, _ = integrate.quad(lambda s, k=k: 2 * np.pi * s * T.self_convolution(k, s), 0, 2 * k.support)
        else:
            grid, f = T._convolution_table(k)
            f_int = integrate.simpson(2 * np.pi * grid * f, x=grid)
        rows.append(
            {
                "kernel": k.name,
                "norm": norm,
                "int_f": f_int,
                "relative_error": abs(f_int - norm**2) / norm**2,
                "truncation_radius": k.truncation_radius(cfg.get("mu", 1.0), tol=cfg.get("tol", 1e-4)),
            }
        )
        log(f"kernels: {k.name} done")
    return pd.DataFrame(rows)


def connection_probability(cfg, seed, n_jobs=1, log=print):
    """Figure 4.5: adjacency frequency by distance against 1 - exp(-mu f)."""
    rows = []
    for i, case in enumerate(cfg["cases"]):
        kernel = from_config(case["kernel"])
        mu = case["mu"]
        r_max = case["r_max"]
        edges = np.linspace(0, r_max, case.get("bins", 40) + 1)

        def job(k, kernel=kernel, mu=mu, i=i, r_max=r_max, edges=edges):
            g = sample(kernel, cfg["lam"], mu, cfg["L"], _rng(seed, "connection_probability", i, k))
            dist, adj = pair_distances(g, r_max)
            return np.histogram(dist, edges)[0], np.histogram(dist[adj], edges)[0]

        res = _replicates(job, cfg["replicates"], n_jobs)
        pairs = sum(r[0] for r in res)
        hits = sum(r[1] for r in res)
        mid = 0.5 * (edges[1:] + edges[:-1])
        ok = pairs >= cfg.get("min_pairs", 500)
        p = hits[ok] / pairs[ok]
        rows.append(
            pd.DataFrame(
                {
                    "kernel": kernel.name,
                    "mu": mu,
                    "distance": mid[ok],
                    "pairs": pairs[ok],
                    "empirical": p,
                    "se": np.sqrt(p * (1 - p) / pairs[ok]),
                    "theory": T.connection_probability(kernel, mu, mid[ok]),
                }
            )
        )
        log(f"connection_probability: {kernel.name} mu={mu} done")
    return pd.concat(rows, ignore_index=True)


def expected_degree(cfg, seed, n_jobs=1, log=print):
    """Table 4.2."""
    kernel = from_config(cfg["kernel"])
    rows = []
    for i, (lam, mu) in enumerate(cfg["points"]):

        def job(k, lam=lam, mu=mu, i=i):
            g = sample(kernel, lam, mu, cfg["L"], _rng(seed, "expected_degree", i, k))
            return g.degrees().mean() if g.n else np.nan

        m, se = mean_se(_replicates(job, cfg["replicates"], n_jobs))
        th = T.expected_degree(kernel, lam, mu)
        rows.append({"lam": lam, "mu": mu, "theory": th, "simulation": m, "se": se, "z": (m - th) / se})
        log(f"expected_degree: lam={lam} mu={mu} done")
    return pd.DataFrame(rows)


# --- the clustering coefficient (section 4.3) --------------------------------------


def convergence(cfg, seed, n_jobs=1, log=print):
    """Figure 4.6: C-hat against the torus side, total simulated volume held fixed."""
    kernel = from_config(cfg["kernel"])
    th = T.clustering(kernel, cfg["mu"])
    L_max = max(cfg["L"])
    rows = []
    for i, L in enumerate(cfg["L"]):
        reps = int(round(cfg["replicates_at_max"] * (L_max / L) ** 2))

        def job(k, L=L, i=i):
            return counts(sample(kernel, cfg["lam"], cfg["mu"], L, _rng(seed, "convergence", i, k)), mechanisms=False)

        cs = _replicates(job, reps, n_jobs)
        row = _clustering_row(cs)
        per = np.array([3 * c["triangles"] / c["wedges"] for c in cs if c["wedges"]])
        row.update({"L": L, "theory": th, "sd_per_replicate": per.std(ddof=1)})
        rows.append(row)
        log(f"convergence: L={L} ({reps} replicates) done")
    return pd.DataFrame(rows)


def vertex_intensity(cfg, seed, n_jobs=1, log=print):
    """Figure 4.7: wedge and triangle densities scale like lam^3, C does not move."""
    kernel = from_config(cfg["kernel"])
    th = T.clustering(kernel, cfg["mu"])
    rows = []
    for i, lam in enumerate(_grid(cfg["lam"])):

        def job(k, lam=lam, i=i):
            return counts(sample(kernel, lam, cfg["mu"], cfg["L"], _rng(seed, "vertex_intensity", i, k)), False)

        cs = _replicates(job, cfg["replicates"], n_jobs)
        row = _clustering_row(cs)
        vol = cfg["L"] ** 2
        row.update(
            {
                "lam": lam,
                "theory": th,
                "wedge_density": np.mean([c["wedges"] for c in cs]) / vol,
                "triangle_density": np.mean([c["triangles"] for c in cs]) / vol,
            }
        )
        rows.append(row)
        log(f"vertex_intensity: lam={lam:.3g} done")
    return pd.DataFrame(rows)


def _theory_curve(kernel, mus, nodes=None):
    out = []
    for mu in mus:
        c = T.configuration_integrals(kernel, mu, nodes, nodes)
        out.append(
            {
                "mu": mu,
                "C": c.C,
                "C_clique": c.C_clique,
                "C_cycle": c.C_cycle,
                "I_wedge": c.wedge,
                "I_clique": c.clique,
                "I_cycle": c.cycle,
            }
        )
    return pd.DataFrame(out)


def decomposition(cfg, seed, n_jobs=1, log=print):
    """Figure 4.8: C = C_clique + C_cycle, measured on the bipartite layer."""
    kernel = from_config(cfg["kernel"])
    rows = []
    for i, mu in enumerate(_grid(cfg["mu"])):
        row = _simulate_clustering(
            kernel, cfg["lam"], mu, cfg["L"], cfg["replicates"], seed, "decomposition", i, n_jobs
        )
        rows.append({"mu": mu, **row})
        log(f"decomposition: mu={mu:.3g} done")
    return pd.DataFrame(rows), _theory_curve(kernel, _grid(cfg["theory_mu"]))


def sparse_regime(cfg, seed, n_jobs=1, log=print):
    """Table 4.3 and Figure 4.9: mu -> 0, C -> 1 and C_cycle = Theta(mu^2)."""
    kernel = from_config(cfg["kernel"])
    rows = []
    for i, mu in enumerate(cfg["table_mu"]):
        c = T.configuration_integrals(kernel, mu)
        sim = _simulate_clustering(
            kernel, cfg["lam"], mu, cfg["L"], cfg["replicates"], seed, "sparse_regime", i, n_jobs
        )
        rows.append(
            {
                "mu": mu,
                "I_wedge_over_mu": c.wedge / mu,
                "I_clique_over_mu": c.clique / mu,
                "C_clique_theory": c.C_clique,
                "C_clique_sim": sim["C_clique"],
                "C_clique_se": sim["C_clique_se"],
                "norm_cubed": kernel.norm() ** 3,
            }
        )
        log(f"sparse_regime: mu={mu} done")
    curve = _theory_curve(kernel, _grid(cfg["theory_mu"]))
    lo, hi = cfg["fit_range"]
    sel = curve[(curve.mu >= lo) & (curve.mu <= hi)]
    slope = np.polyfit(np.log(sel.mu), np.log(sel.C_cycle), 1)[0]
    curve["fitted_cycle_slope"] = slope
    return pd.DataFrame(rows), curve


def attribute_rich(cfg, seed, n_jobs=1, log=print):
    """Figure 4.10: g^(mu) = mu^(-1/2) g, the model tends to the RCM with h = 1 - exp(-g*g)."""
    base = from_config(cfg["kernel"])
    limit = T.rcm_clustering(base)
    rows = []
    for i, mu in enumerate(_grid(cfg["mu"])):
        kernel = base.scaled(mu**-0.5)
        reps = (
            cfg["replicates"] if mu <= cfg.get("heavy_mu", np.inf) else cfg.get("replicates_heavy", cfg["replicates"])
        )
        row = _simulate_clustering(kernel, cfg["lam"], mu, cfg["L"], reps, seed, "attribute_rich", i, n_jobs)
        rows.append({"mu": mu, "rcm_limit": limit, **row})
        log(f"attribute_rich: mu={mu:.3g} done")
    theory = []
    for mu in _grid(cfg["theory_mu"]):
        c = T.configuration_integrals(base.scaled(mu**-0.5), mu)
        theory.append({"mu": mu, "C": c.C, "C_clique": c.C_clique, "C_cycle": c.C_cycle, "rcm_limit": limit})
    return pd.DataFrame(rows), pd.DataFrame(theory)


def continuum_limit(cfg, seed, n_jobs=1, log=print):
    """Figure 4.11: dilating g by r is the same as mu -> nu = mu r^d; C -> kappa_2."""
    base = from_config(cfg["kernel"])
    rows = []
    i = 0
    for r in cfg["dilations"]:
        kernel = base.dilated(r)
        L = max(cfg["L_min"], cfg["L_per_r"] * r * base.support)
        for nu in _grid(cfg["nu"]):
            mu = nu / r**2
            row = _simulate_clustering(
                kernel, cfg["lam"], mu, L, cfg["replicates"], seed, "continuum_limit", i, n_jobs, mechanisms=False
            )
            rows.append({"r": r, "mu": mu, "nu": nu, "L": L, **row})
            i += 1
        log(f"continuum_limit: r={r} done")
    # at large nu the edge probability saturates in a layer of width nu^(-2/3)
    # near |z| = 2R, which needs a finer rule than the default 48 nodes
    nodes = cfg.get("theory_nodes", 128)
    curve = _theory_curve(base, _grid(cfg["theory_nu"]), nodes).rename(columns={"mu": "nu"})
    curve["gap"] = curve.C - T.KAPPA[2]
    summary = _continuum_summary(base, curve, nodes)
    T._configurations.cache_clear()
    return pd.DataFrame(rows), curve, summary


def _continuum_summary(kernel, curve, nodes):
    """Where C(nu) crosses kappa_2, where it is smallest, and the rate of approach."""

    def gap(log_nu):
        return T.configuration_integrals(kernel, np.exp(log_nu), nodes, nodes).C - T.KAPPA[2]

    i = int(np.argmax(curve.gap.values < 0))
    crossing = np.exp(optimize.brentq(gap, np.log(curve.nu[i - 1]), np.log(curve.nu[i]), xtol=1e-8))
    j = int(curve.C.idxmin())
    best = optimize.minimize_scalar(
        gap, bounds=(np.log(curve.nu[j - 1]), np.log(curve.nu[j + 1])), method="bounded", options={"xatol": 1e-6}
    )
    tail = curve[(curve.nu >= 24) & (curve.nu <= 100)]
    slope = np.polyfit(np.log(tail.nu), np.log(-tail.gap), 1)[0]
    return pd.DataFrame(
        [
            {
                "kappa_2": T.KAPPA[2],
                "crossing_nu": crossing,
                "min_nu": float(np.exp(best.x)),
                "min_C": best.fun + T.KAPPA[2],
                "min_gap": best.fun,
                "rate_slope_24_100": slope,
                "gap_at_40": float(-gap(np.log(40))),
                "gap_at_100": float(-gap(np.log(100))),
                # C converges fast, its split does not: the cycle share creeps down to kappa_2 - sigma_2
                **{
                    f"C_cycle_at_{nu:g}": T.configuration_integrals(kernel, nu, nodes, nodes).C_cycle
                    for nu in (10, 100, 1e3, 1e4)
                },
            }
        ]
    )


def dimensions(cfg, seed, n_jobs=1, log=print):
    """Section 4.3.4: kappa_d and sigma_d by Monte Carlo over configurations."""
    rows = []
    for d in cfg["d"]:
        rng = _rng(seed, "dimensions", d)
        ks, ss = [], []
        for _ in range(cfg["batches"]):
            k, s, _ = T.continuum_constants(d, cfg["batch_size"], rng)
            ks.append(k)
            ss.append(s)
        n = cfg["batches"] * cfg["batch_size"]
        rows.append(
            {
                "d": d,
                "samples": n,
                "kappa_mc": np.mean(ks),
                "kappa_exact": T.KAPPA.get(d, np.nan),
                "sigma": np.mean(ss),
                "sigma_se": np.std(ss, ddof=1) / np.sqrt(len(ss)),
                "kappa_minus_sigma": np.mean(ks) - np.mean(ss),
                "cycle_share": 1 - np.mean(ss) / np.mean(ks),
            }
        )
        log(f"dimensions: d={d} done")
    return pd.DataFrame(rows)


def kernel_comparison(cfg, seed, n_jobs=1, log=print):
    """Figure 4.12: Boolean vs exponential kernel at equal mass ||g|| = 1."""
    sims, curves = [], []
    for j, spec in enumerate(cfg["kernels"]):
        kernel = from_config(spec)
        for i, mu in enumerate(_grid(cfg["mu"])):
            row = _simulate_clustering(
                kernel, cfg["lam"], mu, cfg["L"], cfg["replicates"], seed, "kernel_comparison", 100 * j + i, n_jobs
            )
            sims.append({"kernel": kernel.name, "mu": mu, **row})
        c = _theory_curve(kernel, _grid(cfg["theory_mu"]))
        c["kernel"] = kernel.name
        curves.append(c)
        log(f"kernel_comparison: {kernel.name} done")
    return pd.DataFrame(sims), pd.concat(curves, ignore_index=True)


# --- NEW: not in the thesis -------------------------------------------------------


def grig_vs_rcm(cfg, seed, n_jobs=1, log=print):
    """NEW. The GRIG and the random connection model with the same connection
    probability p(z) = 1 - exp(-mu f(z)) have the same expected degree, but the
    GRIG's edges are positively correlated. Compare clustering overall and as a
    function of the degree, C(k)."""
    kernel = from_config(cfg["kernel"])
    reach = 2 * kernel.support if np.isfinite(kernel.support) else T._reach(kernel, max(cfg["mu"]))
    summary, spectrum = [], []
    for i, mu in enumerate(cfg["mu"]):

        def p(dist, mu=mu):
            return T.connection_probability(kernel, mu, dist)

        for model in ("GRIG", "RCM"):

            def job(k, mu=mu, model=model, i=i):
                rng = _rng(seed, "grig_vs_rcm", 2 * i + (model == "RCM"), k)
                if model == "GRIG":
                    A = sample(kernel, cfg["lam"], mu, cfg["L"], rng).A
                else:
                    A = sample_rcm(p, reach, cfg["lam"], cfg["L"], rng)[1]
                deg = np.asarray(A.sum(axis=1)).ravel()
                return deg, triangles_per_vertex(A)

            res = _replicates(job, cfg["replicates"], n_jobs)
            deg = np.concatenate([r[0] for r in res])
            tri = np.concatenate([r[1] for r in res])
            wedges = deg * (deg - 1) / 2
            C, C_se = pooled_ratio([r[1].sum() / 3 for r in res], [(r[0] * (r[0] - 1) / 2).sum() for r in res])
            summary.append(
                {"mu": mu, "model": model, "mean_degree": deg.mean(), "degree_var": deg.var(), "C": C, "C_se": C_se}
            )
            for kdeg in np.unique(deg[deg >= 2]):
                sel = deg == kdeg
                if sel.sum() >= cfg.get("min_vertices", 30):
                    spectrum.append(
                        {
                            "mu": mu,
                            "model": model,
                            "k": int(kdeg),
                            "vertices": int(sel.sum()),
                            "C_k": tri[sel].sum() / wedges[sel].sum(),
                        }
                    )
        log(f"grig_vs_rcm: mu={mu} done")
    theory = [
        {"mu": mu, "C_grig": T.clustering(kernel, mu), "C_rcm": T.rcm_clustering(kernel, mu=mu)}
        for mu in _grid(cfg["theory_mu"])
    ]
    return pd.DataFrame(summary), pd.DataFrame(spectrum), pd.DataFrame(theory)


def higher_dimensions(cfg, seed, n_jobs=1, log=print):
    """NEW. Simulated C(mu) in d = 1, 2, 3 with ||g|| = 1: the curve starts at 1
    in every dimension and descends to the dimension's own constant kappa_d."""
    rows = []
    for j, d in enumerate(cfg["d"]):
        base = from_config({"name": "boolean", "R": 1.0})
        kernel = base.dilated((1 / base.norm(d)) ** (1 / d))
        for i, mu in enumerate(_grid(cfg["mu"])):
            row = _simulate_clustering(
                kernel,
                cfg["lam"][d],
                mu,
                cfg["L"][d],
                cfg["replicates"],
                seed,
                "higher_dimensions",
                100 * j + i,
                n_jobs,
                d=d,
            )
            rows.append({"d": d, "mu": mu, "kappa_d": T.KAPPA[d], **row})
        log(f"higher_dimensions: d={d} done")
    return pd.DataFrame(rows)


def percolation(cfg, seed, n_jobs=1, log=print):
    """NEW. Fraction of vertices in the largest component against the mean
    degree, for several group intensities (section 2.4 has the theory only)."""
    kernel = from_config(cfg["kernel"])
    rows = []
    i = 0
    for mu in cfg["mu"]:
        for lam in _grid(cfg["lam"]):

            def job(k, lam=lam, mu=mu, i=i):
                g = sample(kernel, lam, mu, cfg["L"], _rng(seed, "percolation", i, k))
                return giant_fraction(g.A), g.degrees().mean() if g.n else 0.0

            res = np.array(_replicates(job, cfg["replicates"], n_jobs))
            m, se = mean_se(res[:, 0])
            rows.append(
                {
                    "mu": mu,
                    "lam": lam,
                    "mean_degree": res[:, 1].mean(),
                    "expected_degree": T.expected_degree(kernel, lam, mu),
                    "giant": m,
                    "giant_se": se,
                }
            )
            i += 1
        log(f"percolation: mu={mu} done")
    return pd.DataFrame(rows)


REGISTRY = {name: globals()[name] for name in IDS}
OUTPUTS = {
    "decomposition": ("decomposition", "decomposition_theory"),
    "sparse_regime": ("sparse_regime", "sparse_regime_theory"),
    "attribute_rich": ("attribute_rich", "attribute_rich_theory"),
    "continuum_limit": ("continuum_limit", "continuum_limit_theory", "continuum_limit_summary"),
    "kernel_comparison": ("kernel_comparison", "kernel_comparison_theory"),
    "grig_vs_rcm": ("grig_vs_rcm", "grig_vs_rcm_spectrum", "grig_vs_rcm_theory"),
}
