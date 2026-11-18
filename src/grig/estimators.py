"""Clustering estimators pooled over independent replicates.

C is reported as the pooled ratio 3 sum(triangles) / sum(wedges), which is the
empirical form of 3 rho_triangle / rho_wedge; the mean of per-replicate ratios
carries a ratio bias of order 1/|V|. Standard errors are delete-one jackknife
across replicates, since counts inside one graph are strongly dependent.
"""

import numpy as np


def pooled_ratio(num: np.ndarray, den: np.ndarray, factor: float = 3.0) -> tuple[float, float]:
    num, den = np.asarray(num, dtype=float), np.asarray(den, dtype=float)
    est = factor * num.sum() / den.sum()
    n = len(num)
    if n < 2:
        return est, np.nan
    loo = factor * (num.sum() - num) / (den.sum() - den)
    se = np.sqrt((n - 1) / n * np.sum((loo - loo.mean()) ** 2))
    return float(est), float(se)


def mean_se(x) -> tuple[float, float]:
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else np.nan
