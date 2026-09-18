import numpy as np
import pandas as pd


def _is_dataframe(x):
    return isinstance(x, pd.DataFrame)


def _wrap_like(result, original):
    if _is_dataframe(original):
        return pd.DataFrame(result, index=original.index, columns=original.columns)
    return result


def near_psd(a, epsilon=0.0):
    is_df = _is_dataframe(a)
    A = a.values.astype(float).copy() if is_df else np.asarray(a, dtype=float).copy()

    d = np.diag(A)
    invSD = None
    out = A
    if not np.allclose(d, 1.0):
        sd = np.sqrt(d)
        invSD = np.diag(1.0 / sd)
        out = invSD @ A @ invSD

    vals, vecs = np.linalg.eigh(out)
    vals = np.maximum(vals, epsilon)

    T = 1.0 / (vecs * vecs @ vals)
    T = np.diag(np.sqrt(T))
    L = np.diag(np.sqrt(vals))
    B = T @ vecs @ L
    out = B @ B.T

    if invSD is not None:
        sd_back = np.diag(1.0 / np.diag(invSD))
        out = sd_back @ out @ sd_back

    return _wrap_like(out, a)


def _getAplus(A):
    vals, vecs = np.linalg.eigh(A)
    vals = np.maximum(vals, 0)
    return vecs @ np.diag(vals) @ vecs.T


def _getPS(A, W):
    W05 = np.sqrt(W)
    iW = np.linalg.inv(W05)
    return iW @ _getAplus(W05 @ A @ W05) @ iW


def _getPu(A):
    Aret = A.copy()
    np.fill_diagonal(Aret, 1.0)
    return Aret


def _wgt_norm(A, W):
    W05 = np.sqrt(W)
    Wm = W05 @ A @ W05
    return np.sum(Wm * Wm)


def _higham_nearest_corr(pc, max_iter=100, tol=1e-9):
    n = pc.shape[0]
    W = np.eye(n)
    deltaS = np.zeros_like(pc)
    Yk = pc.copy()
    norm_last = np.inf
    for _ in range(max_iter):
        Rk = Yk - deltaS
        Xk = _getPS(Rk, W)
        deltaS = Xk - Rk
        Yk = _getPu(Xk)
        norm = _wgt_norm(Yk - pc, W)
        min_eig = np.linalg.eigvalsh(Yk).min()
        if abs(norm - norm_last) < tol and min_eig > -1e-8:
            break
        norm_last = norm
    return Yk


def higham_psd(a, max_iter=100, tol=1e-9):
    is_df = _is_dataframe(a)
    A = a.values.astype(float).copy() if is_df else np.asarray(a, dtype=float).copy()

    d = np.diag(A)
    if not np.allclose(d, 1.0):
        sd = np.sqrt(d)
        invSD = np.diag(1.0 / sd)
        corr = invSD @ A @ invSD
        fixed = _higham_nearest_corr(corr, max_iter, tol)
        sdm = np.diag(sd)
        out = sdm @ fixed @ sdm
    else:
        out = _higham_nearest_corr(A, max_iter, tol)

    return _wrap_like(out, a)


def chol_psd(a):
    is_df = _is_dataframe(a)
    A = a.values.astype(float) if is_df else np.asarray(a, dtype=float)
    n = A.shape[0]
    root = np.zeros((n, n))

    for j in range(n):
        s = 0.0
        if j > 0:
            s = root[j, :j] @ root[j, :j]
        temp = A[j, j] - s
        if -1e-8 <= temp <= 0:
            temp = 0.0
        root[j, j] = np.sqrt(temp) if temp > 0 else 0.0
        if root[j, j] == 0.0:
            continue
        ir = 1.0 / root[j, j]
        for i in range(j + 1, n):
            s = root[i, :j] @ root[j, :j]
            root[i, j] = (A[i, j] - s) * ir

    return _wrap_like(root, a)
