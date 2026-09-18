import numpy as np
from scipy import stats, optimize
from scipy.special import kv


def fit_normal(x):
    x = np.asarray(x, dtype=float)
    mu = x.mean()
    sigma = x.std(ddof=1)
    return {"mu": mu, "sigma": sigma}


def fit_t(x):
    x = np.asarray(x, dtype=float)
    nu, mu, sigma = stats.t.fit(x)
    return {"mu": mu, "sigma": sigma, "nu": nu}


def fit_regression_t(X, y):
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    n, k = X.shape
    Xd = np.column_stack([np.ones(n), X])

    def negll(params):
        sigma, nu = params[0], params[1]
        beta = params[2:]
        if sigma <= 0 or nu <= 2:
            return 1e10
        resid = y - Xd @ beta
        return -np.sum(stats.t.logpdf(resid / sigma, nu) - np.log(sigma))

    beta0, *_ = np.linalg.lstsq(Xd, y, rcond=None)
    resid0 = y - Xd @ beta0
    x0 = [resid0.std(), 5.0, *beta0]

    res = optimize.minimize(
        negll, x0, method="Nelder-Mead",
        options={"maxiter": 20000, "xatol": 1e-10, "fatol": 1e-10},
    )
    sigma, nu = res.x[0], res.x[1]
    alpha = res.x[2]
    beta = res.x[3:]
    return {"mu": 0.0, "sigma": sigma, "nu": nu, "Alpha": alpha, "beta": beta}


def aicc_t(x, mu, sigma, nu, k=3):
    x = np.asarray(x, dtype=float)
    n = len(x)
    logL = np.sum(stats.t.logpdf((x - mu) / sigma, nu) - np.log(sigma))
    aic = -2 * logL + 2 * k
    aicc = aic + (2 * k * (k + 1)) / (n - k - 1)
    return aicc


def fit_nig_mm(x):
    x = np.asarray(x, dtype=float)
    m1 = x.mean()
    m2 = x.var(ddof=0)
    sk = stats.skew(x, bias=True)
    ku = stats.kurtosis(x, bias=True, fisher=True)

    def eqs(p):
        mu, alpha, beta, delta = p
        gamma = np.sqrt(alpha ** 2 - beta ** 2)
        mean = mu + delta * beta / gamma
        var = delta * alpha ** 2 / gamma ** 3
        skew = 3 * beta / (alpha * np.sqrt(delta * gamma))
        kurt = 3 * (1 + 4 * (beta / alpha) ** 2) / (delta * gamma)
        return [mean - m1, var - m2, skew - sk, kurt - ku]

    x0 = [m1, 20.0, 0.0, 0.05]
    mu, alpha, beta, delta = optimize.fsolve(eqs, x0)
    return {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}


def _nig_logpdf(x, mu, alpha, beta, delta):
    gamma = np.sqrt(alpha ** 2 - beta ** 2)
    z = x - mu
    q = np.sqrt(delta ** 2 + z ** 2)
    return (np.log(alpha * delta) - np.log(np.pi) + delta * gamma + beta * z
            + np.log(kv(1, alpha * q)) - np.log(q))


def fit_nig_mle(x, x0=None):
    x = np.asarray(x, dtype=float)

    if x0 is None:
        mm = fit_nig_mm(x)
        x0 = [mm["mu"], mm["alpha"], mm["beta"], mm["delta"]]

    def negll(p):
        mu, alpha, beta, delta = p
        if delta <= 0 or alpha <= 0 or abs(beta) >= alpha:
            return 1e10
        ll = np.sum(_nig_logpdf(x, mu, alpha, beta, delta))
        if not np.isfinite(ll):
            return 1e10
        return -ll

    res = optimize.minimize(
        negll, x0, method="Nelder-Mead",
        options={"maxiter": 50000, "xatol": 1e-12, "fatol": 1e-12, "adaptive": True},
    )
    mu, alpha, beta, delta = res.x
    return {"mu": mu, "alpha": alpha, "beta": beta, "delta": delta}
