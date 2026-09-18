import numpy as np
from .psd import chol_psd


def simulate_normal(cov, n_sims=100_000, mean=None, seed=None):
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    L = chol_psd(cov)
    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n, n_sims))
    samples = (L @ Z).T
    if mean is not None:
        samples = samples + np.asarray(mean)
    return samples


def simulate_pca(cov, n_sims=100_000, pct_explained=0.99, mean=None, seed=None):
    cov = np.asarray(cov, dtype=float)
    n = cov.shape[0]
    vals, vecs = np.linalg.eigh(cov)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    vals = np.maximum(vals, 0)

    total = vals.sum()
    cum = np.cumsum(vals) / total
    if pct_explained >= 1.0:
        n_factors = n
    else:
        n_factors = int(np.searchsorted(cum, pct_explained) + 1)

    vals_k = vals[:n_factors]
    vecs_k = vecs[:, :n_factors]
    B = vecs_k @ np.diag(np.sqrt(vals_k))

    rng = np.random.default_rng(seed)
    Z = rng.standard_normal((n_factors, n_sims))
    samples = (B @ Z).T
    if mean is not None:
        samples = samples + np.asarray(mean)
    return samples, n_factors
