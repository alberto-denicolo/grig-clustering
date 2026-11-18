from pathlib import Path

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

from grig import theory as T  # noqa: E402
from grig.kernels import boolean, exponential, gaussian, power_law  # noqa: E402
from grig.simulate import sample  # noqa: E402

BLUE, ORANGE, GREEN, GREY = "#1f77b4", "#d95f02", "#1b9e77", "#555555"
plt.rcParams.update(
    {
        "figure.figsize": (6.4, 4.2),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.alpha": 0.3,
        "savefig.bbox": "tight",
        "savefig.dpi": 150,
    }
)


def _save(fig, out: Path, name: str):
    # minor-tick labels of log axes overlap with the major ones
    for ax in fig.axes:
        ax.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
        ax.yaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.png")
    fig.savefig(out / f"{name}.pdf")
    plt.close(fig)


def _kappa(ax, d=2, label=True):
    ax.axhline(T.KAPPA[d], color="black", ls=":", lw=1, label=rf"$\kappa_{d}$" if label else None)


# --- illustrations ----------------------------------------------------------------


def plot_construction(out: Path, seed: int = 7):
    """Figure 4.2: the two processes, the bipartite layer, the projection."""
    k = boolean(1.0)
    g = sample(k, 1.1, 0.55, 10.0, np.random.default_rng(seed))
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.4))
    for ax in axes:
        ax.set(xlim=(0, 10), ylim=(0, 10), aspect="equal", xticks=[], yticks=[])
        ax.grid(False)
    axes[0].scatter(*g.V.T, s=14, color="black", zorder=3, label="vertices V")
    axes[0].scatter(*g.U.T, s=28, facecolors="white", edgecolors=BLUE, zorder=3, label="groups U")
    axes[0].legend(loc="upper right", fontsize=8)
    axes[0].set_title("(a) two Poisson processes")

    B = g.B.tocoo()
    for u in g.U:
        axes[1].add_patch(plt.Circle(u, 1.0, color=BLUE, alpha=0.08))
    segs = [[g.V[i], g.U[j]] for i, j in zip(B.row, B.col, strict=True) if np.all(np.abs(g.V[i] - g.U[j]) < 1.5)]
    axes[1].add_collection(LineCollection(segs, colors=BLUE, lw=0.6, alpha=0.7))
    axes[1].scatter(*g.V.T, s=14, color="black", zorder=3)
    axes[1].scatter(*g.U.T, s=28, facecolors="white", edgecolors=BLUE, zorder=3)
    axes[1].set_title("(b) bipartite layer: vertex–group links")

    A = g.A.tocoo()
    segs = [
        [g.V[i], g.V[j]] for i, j in zip(A.row, A.col, strict=True) if i < j and np.all(np.abs(g.V[i] - g.V[j]) < 2.5)
    ]
    axes[2].add_collection(LineCollection(segs, colors=GREY, lw=0.7))
    axes[2].scatter(*g.V.T, s=14, color="black", zorder=3)
    axes[2].set_title("(c) projection: shared group ⇒ edge")
    fig.suptitle(r"Two-step construction of the GRIG on the torus $\mathbb{T}^2_{10}$ (Boolean kernel, R = 1)", y=1.0)
    _save(fig, out, "construction")


def plot_kernels(out: Path):
    """Figure 4.3: the kernel families at equal mass."""
    r = np.linspace(0, 3, 600)
    fig, ax = plt.subplots()
    for k, c in ((boolean(1.0), BLUE), (exponential(1.0), ORANGE), (gaussian(1.0), GREEN), (power_law(4.0), GREY)):
        k = k.normalized()
        ax.plot(r, k(r), color=c, label=k.name.replace("_", " "))
    ax.set(
        xlabel="distance |x|",
        ylabel="g(|x|)",
        title=r"Connection functions normalized to $\|g\|_1 = 1$ in $\mathbb{R}^2$",
    )
    ax.legend()
    _save(fig, out, "kernels")


# --- chapter 4 ----------------------------------------------------------------------


def plot_connection_probability(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for ax, (name, d) in zip(axes, df.groupby("kernel", sort=False), strict=False):
        for (mu, e), c in zip(d.groupby("mu"), (BLUE, ORANGE, GREEN), strict=False):
            ax.plot(e.distance, e.theory, color=c, lw=1.2)
            ax.errorbar(e.distance, e.empirical, yerr=e.se, fmt="o", ms=3, color=c, label=rf"$\mu$ = {mu:g}")
        if name == "boolean":
            ax.axvline(2.0, color="black", ls="--", lw=0.8)
        ax.set(xlabel="pairwise distance |z|", title=f"{name} kernel")
        ax.legend()
    axes[0].set_ylabel("P(adjacent)")
    fig.suptitle(r"Connection probability: simulation (markers) vs $1 - e^{-\mu (g*g)(z)}$ (lines)")
    _save(fig, out, "connection_probability")


def plot_convergence(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.errorbar(df.L, df.C, yerr=df.C_se, fmt="o", color=BLUE, capsize=3, label=r"$\hat C$ (pooled)")
    ax.axhline(df.theory.iloc[0], color="black", ls="--", lw=1, label=r"$I_\triangle / I_\wedge$")
    ax.set(xlabel="torus side L", ylabel="clustering coefficient", title="No finite-size trend")
    ax.legend()
    ax = axes[1]
    slope = np.polyfit(np.log(df.L), np.log(df.sd_per_replicate), 1)[0]
    ax.loglog(df.L, df.sd_per_replicate, "o", color=BLUE, label=f"sd of one realization (slope {slope:.2f})")
    ref = df.sd_per_replicate.iloc[0] * (df.L / df.L.iloc[0]) ** -1.0
    ax.loglog(df.L, ref, "--", color="black", lw=1, label="slope −d/2 = −1")
    ax.set(xlabel="torus side L", ylabel="standard deviation", title="Fluctuations of one realization")
    ax.legend()
    _save(fig, out, "convergence")


def plot_vertex_intensity(df: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(df.lam, df.wedge_density, "o-", color=BLUE, ms=4, label="wedges")
    ax.loglog(df.lam, df.triangle_density, "s-", color=ORANGE, ms=4, label="triangles")
    ax.loglog(
        df.lam, df.wedge_density.iloc[-1] * (df.lam / df.lam.iloc[-1]) ** 3, "--", color="black", lw=1, label="slope 3"
    )
    ax.set(
        xlabel=r"vertex intensity $\lambda$", ylabel="density per unit area", title=r"Densities scale like $\lambda^3$"
    )
    ax.legend()
    ax = axes[1]
    ax.errorbar(df.lam, df.C, yerr=df.C_se, fmt="o", color=BLUE, capsize=3)
    ax.axhline(df.theory.iloc[0], color="black", ls="--", lw=1)
    ax.set(
        xscale="log",
        xlabel=r"vertex intensity $\lambda$",
        ylabel="clustering coefficient",
        title=r"$C$ does not depend on $\lambda$",
    )
    _save(fig, out, "vertex_intensity")


def _decomp_axes(ax, sim, th, x="mu"):
    for col, c, lab in (("C", "black", "C"), ("C_clique", BLUE, r"$C_{clique}$"), ("C_cycle", ORANGE, r"$C_{cycle}$")):
        ax.plot(th[x], th[col], color=c, lw=1.2)
        ax.errorbar(sim[x], sim[col], yerr=sim[f"{col}_se"], fmt="o", ms=4, color=c, capsize=2, label=lab)


def plot_decomposition(sim: pd.DataFrame, th: pd.DataFrame, out: Path):
    fig, ax = plt.subplots()
    _decomp_axes(ax, sim, th)
    _kappa(ax)
    ax.set(
        xscale="log",
        xlabel=r"group intensity $\mu$",
        ylabel="clustering",
        ylim=(-0.02, 1.02),
        title="Clique / cycle decomposition (lines: theory, markers: simulation)",
    )
    ax.legend()
    _save(fig, out, "decomposition")


def plot_sparse_regime(table: pd.DataFrame, curve: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    for col, c, lab in (("C", "black", "C"), ("C_clique", BLUE, r"$C_{clique}$"), ("C_cycle", ORANGE, r"$C_{cycle}$")):
        ax.plot(curve.mu, curve[col], color=c, label=lab)
    ax.errorbar(
        table.mu, table.C_clique_sim, yerr=table.C_clique_se, fmt="o", ms=4, color=BLUE, label="simulated clique"
    )
    ax.set(xscale="log", xlabel=r"$\mu$", ylabel="clustering", title=r"$\mu \to 0$: clique triangles take over")
    ax.legend()
    ax = axes[1]
    slope = curve.fitted_cycle_slope.iloc[0]
    ax.loglog(curve.mu, curve.C_cycle, color=ORANGE, label=rf"$C_{{cycle}}$ (fitted slope {slope:.3f})")
    m0 = curve.mu.iloc[len(curve) // 4]
    c0 = np.interp(m0, curve.mu, curve.C_cycle)
    ax.loglog(curve.mu, c0 * (curve.mu / m0) ** 2, "--", color="black", lw=1, label="slope 2")
    ax.set(xlabel=r"$\mu$", ylabel=r"$C_{cycle}$", title=r"$C_{cycle} = \Theta(\mu^2)$", ylim=(1e-8, 1))
    ax.legend()
    _save(fig, out, "sparse_regime")


def plot_attribute_rich(sim: pd.DataFrame, th: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.loglog(th.mu, th.C_clique, color=BLUE, label=r"$C_{clique}$ theory")
    ax.errorbar(sim.mu, sim.C_clique, yerr=sim.C_clique_se, fmt="o", ms=4, color=BLUE, label="simulation")
    ax.loglog(
        th.mu, th.C_clique.iloc[0] * (th.mu / th.mu.iloc[0]) ** -0.5, "--", color="black", lw=1, label="slope −1/2"
    )
    ax.set(xlabel=r"$\mu$", title="Clique triangles vanish")
    ax.legend()
    ax = axes[1]
    ax.plot(th.mu, th.C, color="black", label="C theory")
    ax.errorbar(sim.mu, sim.C, yerr=sim.C_se, fmt="o", ms=4, color=BLUE, label="simulation")
    ax.axhline(th.rcm_limit.iloc[0], color=ORANGE, ls="--", label=f"RCM limit {th.rcm_limit.iloc[0]:.5f}")
    ax.set(xscale="log", xlabel=r"$\mu$", title=r"$g^{(\mu)} = \mu^{-1/2} g$: C tends to the RCM value")
    ax.legend()
    _save(fig, out, "attribute_rich")


def plot_continuum_limit(sim: pd.DataFrame, th: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    colors = plt.cm.viridis(np.linspace(0, 0.85, sim.r.nunique()))
    for (r, d), c in zip(sim.groupby("r"), colors, strict=True):
        axes[0].errorbar(d.mu, d.C, yerr=d.C_se, fmt="o-", ms=3, color=c, label=f"r = {r:g}")
        axes[1].errorbar(d.nu, d.C, yerr=d.C_se, fmt="o", ms=4, color=c, label=f"r = {r:g}")
    axes[1].plot(th.nu, th.C, color=GREY, lw=1.2, label="theory")
    for ax in axes[:2]:
        _kappa(ax)
        ax.set(xscale="log", ylabel=r"$\hat C_r$")
        ax.legend(fontsize=8)
    axes[0].set(xlabel=r"group intensity $\mu$", title="Four dilations of the kernel")
    axes[1].set(xlabel=r"effective intensity $\nu = \mu r^2$", title="…collapse onto one curve")
    ax = axes[2]
    pos, neg = th[th.gap > 0], th[th.gap < 0]
    ax.loglog(pos.nu, pos.gap, ".", color=BLUE, ms=4, label=r"above $\kappa_2$")
    ax.loglog(neg.nu, -neg.gap, ".", color=ORANGE, ms=4, label=r"below $\kappa_2$")
    tail = neg[(neg.nu >= 24) & (neg.nu <= 100)]
    if len(tail) > 3:
        s, c0 = np.polyfit(np.log(tail.nu), np.log(-tail.gap), 1)
        ax.loglog(tail.nu, np.exp(c0) * tail.nu**s, "--", color="black", lw=1.5, label=f"slope {s:.2f} on [24, 100]")
    ax.set(xlabel=r"$\nu$", ylabel=r"$|C(\nu) - \kappa_2|$", title=r"Approach to $\kappa_2$ is not monotone")
    ax.legend(fontsize=8)
    _save(fig, out, "continuum_limit")


def plot_kernel_comparison(sim: pd.DataFrame, th: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for (name, d), c in zip(sim.groupby("kernel", sort=False), (BLUE, ORANGE), strict=False):
        t = th[th.kernel == name]
        axes[0].plot(t.mu, t.C, color=c)
        axes[0].errorbar(d.mu, d.C, yerr=d.C_se, fmt="o", ms=4, color=c, label=name)
        axes[1].plot(t.mu, t.C_cycle / t.C, color=c)
        axes[1].plot(d.mu, d.C_cycle / d.C, "o", ms=4, color=c, label=name)
    _kappa(axes[0])
    axes[0].set(xscale="log", xlabel=r"$\mu$", ylabel="C", title=r"Clustering at equal mass $\|g\| = 1$")
    axes[1].set(xscale="log", xlabel=r"$\mu$", ylabel=r"$C_{cycle} / C$", title="Share of cycle triangles")
    for ax in axes:
        ax.legend()
    _save(fig, out, "kernel_comparison")


def plot_grig_vs_rcm(summary: pd.DataFrame, spectrum: pd.DataFrame, th: pd.DataFrame, out: Path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(th.mu, th.C_grig, color=BLUE, label="GRIG (theory)")
    ax.plot(th.mu, th.C_rcm, color=ORANGE, label="RCM, same p(z) (theory)")
    for model, c in (("GRIG", BLUE), ("RCM", ORANGE)):
        d = summary[summary.model == model]
        ax.errorbar(d.mu, d.C, yerr=d.C_se, fmt="o", ms=5, color=c)
    ax.set(xscale="log", xlabel=r"$\mu$", ylabel="C", title="Same edge probabilities, different clustering")
    ax.legend()
    ax = axes[1]
    styles = dict(zip(sorted(spectrum.mu.unique()), ("o", "s", "^"), strict=False))
    for (mu, model), d in spectrum.groupby(["mu", "model"]):
        ax.plot(
            d.k,
            d.C_k,
            styles[mu],
            ms=3,
            ls="-",
            lw=0.8,
            color=BLUE if model == "GRIG" else ORANGE,
            label=rf"{model}, $\mu$={mu:g}",
        )
    ax.set(xscale="log", xlabel="degree k", ylabel="C(k)", title="Clustering spectrum C(k)")
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "grig_vs_rcm")


def plot_higher_dimensions(df: pd.DataFrame, dims: pd.DataFrame | None, out: Path):
    fig, ax = plt.subplots()
    for (d, e), c in zip(df.groupby("d"), (BLUE, ORANGE, GREEN), strict=False):
        ax.errorbar(e.mu, e.C, yerr=e.C_se, fmt="o-", ms=4, color=c, label=f"d = {d}: C")
        ax.plot(e.mu, e.C_cycle, "x:", ms=4, color=c, label=rf"d = {d}: $C_{{cycle}}$")
        ax.axhline(T.KAPPA[d], color=c, ls="--", lw=0.8)
        if dims is not None and d in set(dims.d):
            ax.axhline(dims.set_index("d").loc[d, "kappa_minus_sigma"], color=c, ls=":", lw=0.8)
    ax.set(
        xscale="log",
        xlabel=r"$\mu$ (with $\|g\| = 1$)",
        ylabel="clustering",
        title=r"d = 1, 2, 3: dashed $\kappa_d$; dotted $\kappa_d - \sigma_d$, the cycle share as $\mu \to \infty$",
    )
    ax.legend(fontsize=7, ncol=2)
    _save(fig, out, "higher_dimensions")


def plot_percolation(df: pd.DataFrame, out: Path):
    fig, ax = plt.subplots()
    for (mu, d), c in zip(df.groupby("mu"), (BLUE, ORANGE, GREEN), strict=False):
        ax.errorbar(d.expected_degree, d.giant, yerr=d.giant_se, fmt="o-", ms=3, color=c, label=rf"$\mu$ = {mu:g}")
    ax.set(
        xscale="log",
        xlabel="expected degree",
        ylabel="fraction in the largest component",
        title="Emergence of the giant component",
    )
    ax.legend()
    _save(fig, out, "percolation")


def plot_all(results: Path, figures: Path):
    def load(name):
        p = results / f"{name}.csv"
        return pd.read_csv(p) if p.exists() else None

    plot_construction(figures)
    plot_kernels(figures)
    pairs = {
        "connection_probability": (plot_connection_probability, ("connection_probability",)),
        "convergence": (plot_convergence, ("convergence",)),
        "vertex_intensity": (plot_vertex_intensity, ("vertex_intensity",)),
        "decomposition": (plot_decomposition, ("decomposition", "decomposition_theory")),
        "sparse_regime": (plot_sparse_regime, ("sparse_regime", "sparse_regime_theory")),
        "attribute_rich": (plot_attribute_rich, ("attribute_rich", "attribute_rich_theory")),
        "continuum_limit": (plot_continuum_limit, ("continuum_limit", "continuum_limit_theory")),
        "kernel_comparison": (plot_kernel_comparison, ("kernel_comparison", "kernel_comparison_theory")),
        "grig_vs_rcm": (plot_grig_vs_rcm, ("grig_vs_rcm", "grig_vs_rcm_spectrum", "grig_vs_rcm_theory")),
        "percolation": (plot_percolation, ("percolation",)),
    }
    for fn, names in pairs.values():
        frames = [load(n) for n in names]
        if all(f is not None for f in frames):
            fn(*frames, figures)
    if (hd := load("higher_dimensions")) is not None:
        plot_higher_dimensions(hd, load("dimensions"), figures)
