import argparse
import sys
import time
from pathlib import Path

import numpy as np
import yaml

from grig import plotting
from grig import theory as T
from grig.experiments import OUTPUTS, REGISTRY
from grig.kernels import from_config
from grig.simulate import counts, sample


def _run(args):
    cfg = yaml.safe_load(Path(args.config).read_text())
    names = list(REGISTRY) if args.experiments == ["all"] else args.experiments
    unknown = set(names) - set(REGISTRY)
    if unknown:
        sys.exit(f"unknown experiment(s): {', '.join(sorted(unknown))}")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    n_jobs = args.jobs if args.jobs is not None else cfg.get("n_jobs", 1)
    for name in names:
        if name not in cfg:
            print(f"skipping {name}: no section in {args.config}")
            continue
        t0 = time.perf_counter()
        result = REGISTRY[name](cfg[name], cfg["seed"], n_jobs=n_jobs)
        frames = result if isinstance(result, tuple) else (result,)
        for frame, fname in zip(frames, OUTPUTS.get(name, (name,)), strict=True):
            frame.to_csv(out / f"{fname}.csv", index=False)
        print(f"{name}: done in {time.perf_counter() - t0:.0f}s")


def _plot(args):
    plotting.plot_all(Path(args.results), Path(args.figures))
    print(f"figures written to {args.figures}/")


def _simulate(args):
    kernel = from_config({"name": args.kernel, **({"R": args.scale} if args.kernel == "boolean" else {})})
    if args.kernel != "boolean":
        kernel = kernel.dilated(args.scale)
    rng = np.random.default_rng(args.seed)
    rows = [counts(sample(kernel, args.lam, args.mu, args.L, rng)) for _ in range(args.replicates)]
    tri = sum(r["triangles"] for r in rows)
    cli = sum(r["clique_triangles"] for r in rows)
    wed = sum(r["wedges"] for r in rows)
    deg = 2 * sum(r["edges"] for r in rows) / sum(r["vertices"] for r in rows)
    print(
        f"{args.replicates} graphs on the torus of side {args.L}: {kernel.name} kernel, lambda={args.lam}, mu={args.mu}"
    )
    print(f"mean degree        {deg:.4f}   (theory {T.expected_degree(kernel, args.lam, args.mu):.4f})")
    if args.kernel in ("boolean", "exponential", "gaussian"):
        th = T.configuration_integrals(kernel, args.mu)
        print(f"clustering C       {3 * tri / wed:.4f}   (theory {th.C:.4f})")
        print(f"  clique part      {3 * cli / wed:.4f}   (theory {th.C_clique:.4f})")
        print(f"  cycle part       {3 * (tri - cli) / wed:.4f}   (theory {th.C_cycle:.4f})")


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="grig", description="Geometric random intersection graph: simulation and theory."
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run experiments and write CSV files")
    run.add_argument("experiments", nargs="+", help=f"'all' or any of: {', '.join(REGISTRY)}")
    run.add_argument("--config", default="configs/thesis.yaml")
    run.add_argument("--out", default="results")
    run.add_argument("--jobs", type=int, default=None, help="parallel workers (default: n_jobs in the config)")
    run.set_defaults(func=_run)

    plot = sub.add_parser("plot", help="draw all figures from the CSV files")
    plot.add_argument("--results", default="results")
    plot.add_argument("--figures", default="figures")
    plot.set_defaults(func=_plot)

    sim = sub.add_parser("simulate", help="simulate a few graphs and compare with the theory")
    sim.add_argument("--kernel", default="boolean", choices=["boolean", "exponential", "gaussian"])
    sim.add_argument("--scale", type=float, default=1.0, help="radius R (Boolean) or length scale")
    sim.add_argument("--lam", type=float, default=1.5)
    sim.add_argument("--mu", type=float, default=1.0)
    sim.add_argument("--L", type=float, default=20.0)
    sim.add_argument("--replicates", type=int, default=20)
    sim.add_argument("--seed", type=int, default=0)
    sim.set_defaults(func=_simulate)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
