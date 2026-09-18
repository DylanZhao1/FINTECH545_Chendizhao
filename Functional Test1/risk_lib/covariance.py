import numpy as np
import pandas as pd


def _to_frame(x):
    if isinstance(x, pd.DataFrame):
        return x
    return pd.DataFrame(x)


def missing_cov_skip(data):
    df = _to_frame(data)
    clean = df.dropna()
    return clean.cov()


def missing_corr_skip(data):
    """Correlation matrix after listwise deletion of missing rows."""
    df = _to_frame(data)
    clean = df.dropna()
    return clean.corr()


def missing_cov_pairwise(data):
    df = _to_frame(data)
    cols = df.columns
    n = len(cols)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub = df[[cols[i], cols[j]]].dropna()
            out[i, j] = np.cov(sub.iloc[:, 0], sub.iloc[:, 1], ddof=1)[0, 1]
    return pd.DataFrame(out, index=cols, columns=cols)


def missing_corr_pairwise(data):
    """Correlation matrix using pairwise-complete observations."""
    df = _to_frame(data)
    cols = df.columns
    n = len(cols)
    out = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            sub = df[[cols[i], cols[j]]].dropna()
            out[i, j] = np.corrcoef(sub.iloc[:, 0], sub.iloc[:, 1])[0, 1]
    return pd.DataFrame(out, index=cols, columns=cols)


def _ew_weights(n, lam):
    w = np.array([(1 - lam) * lam ** (n - 1 - t) for t in range(n)])
    return w / w.sum()


def ew_cov(data, lam):
    df = _to_frame(data)
    X = df.values.astype(float)
    n, m = X.shape
    w = _ew_weights(n, lam)
    wmean = w @ X
    xc = X - wmean
    cov = np.einsum('t,ti,tj->ij', w, xc, xc)
    return pd.DataFrame(cov, index=df.columns, columns=df.columns)


def cov_to_corr(cov):
    """Convert a covariance matrix (DataFrame or ndarray) to a correlation matrix."""
    is_df = isinstance(cov, pd.DataFrame)
    C = cov.values if is_df else np.asarray(cov)
    d = np.sqrt(np.diag(C))
    corr = C / np.outer(d, d)
    if is_df:
        return pd.DataFrame(corr, index=cov.index, columns=cov.columns)
    return corr


def ew_corr(data, lam):
    """Exponentially weighted correlation matrix (see ew_cov)."""
    return cov_to_corr(ew_cov(data, lam))


def ew_cov_corr(data, lam_var, lam_corr):
    df = _to_frame(data)
    var_cov = ew_cov(df, lam_var)
    sd = np.sqrt(np.diag(var_cov.values))
    corr = ew_corr(df, lam_corr)
    combo = np.outer(sd, sd) * corr.values
    return pd.DataFrame(combo, index=df.columns, columns=df.columns)
