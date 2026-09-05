"""
Problem 2: A Regression Whose Errors Are Not Normal
"""
import numpy as np
import pandas as pd
from scipy import stats, optimize
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("Assignments/Assignment1/problem2.csv")
x, y, n = df["x"].values, df["y"].values, len(df)

# Scatter plot
plt.figure(figsize=(6, 5))
plt.scatter(x, y, alpha=0.6, s=20, color="#4C72B0")
plt.xlabel("x"); plt.ylabel("y")
plt.title("Problem 2: y vs x (raw scatter)")
plt.tight_layout()
plt.savefig("problem2_scatter.png", dpi=150)

# Model 1: OLS
X = np.column_stack([np.ones(n), x])
beta_ols, *_ = np.linalg.lstsq(X, y, rcond=None)
alpha_ols, beta1_ols = beta_ols
resid_ols = y - X @ beta_ols
sigma2_ols = np.sum(resid_ols**2) / (n - 2)
se_alpha, se_beta = np.sqrt(np.diag(np.linalg.inv(X.T @ X)) * sigma2_ols)

# Model 2: MLE under a Normal error
sigma_n = np.sqrt(np.mean(resid_ols**2))
alpha_n, beta_n = alpha_ols, beta1_ols
ll_normal = np.sum(stats.norm.logpdf(resid_ols, scale=sigma_n))

# Model 3: MLE under a Student-t error.
def neg_ll_t(params):
    a, b, log_scale, log_nu_m2 = params
    scale = np.exp(log_scale)
    nu = 2.0 + np.exp(log_nu_m2)
    resid = (y - (a + b * x)) / scale
    return -np.sum(stats.t.logpdf(resid, df=nu) - np.log(scale))

x0 = [alpha_ols, beta1_ols, np.log(sigma_n), np.log(8.0)]
fit_t = optimize.minimize(neg_ll_t, x0, method="Nelder-Mead",
                           options={"xatol": 1e-10, "fatol": 1e-10, "maxiter": 40000})
alpha_t, beta_t, log_scale_t, log_nu_m2_t = fit_t.x
scale_t = np.exp(log_scale_t)
nu_t = 2.0 + np.exp(log_nu_m2_t)
ll_t = -fit_t.fun

# AICc
def aicc(ll, k):
    return 2 * k - 2 * ll + (2 * k * (k + 1)) / (n - k - 1)

aicc_normal, aicc_t = aicc(ll_normal, 3), aicc(ll_t, 4)

# 95% and 99.5% quantiles of fitted error distribution
q95_normal, q995_normal = stats.norm.ppf([0.95, 0.995], scale=sigma_n)
q95_t, q995_t = stats.t.ppf([0.95, 0.995], df=nu_t, scale=scale_t)

print(f"OLS:        alpha={alpha_ols:.4f} (SE {se_alpha:.4f})  beta={beta1_ols:.4f} (SE {se_beta:.4f})")
print(f"MLE Normal: alpha={alpha_n:.4f}  beta={beta_n:.4f}  sigma={sigma_n:.4f}  AICc={aicc_normal:.2f}")
print(f"MLE t:      alpha={alpha_t:.4f}  beta={beta_t:.4f}  scale={scale_t:.4f}  nu={nu_t:.4f}  AICc={aicc_t:.2f}")
print(f"Delta AICc (Normal - t) = {aicc_normal - aicc_t:.2f}")
print(f"Error quantiles -- 95%: Normal={q95_normal:.4f}, t={q95_t:.4f}")
print(f"                 99.5%: Normal={q995_normal:.4f}, t={q995_t:.4f}")

# Plot: fitted lines and residual distribution vs both fits
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
xs = np.linspace(x.min(), x.max(), 200)
axes[0].scatter(x, y, alpha=0.5, s=18, color="gray", label="data")
axes[0].plot(xs, alpha_ols + beta1_ols * xs, color="#4C72B0", lw=2, label="OLS")
axes[0].plot(xs, alpha_n + beta_n * xs, color="#55A868", lw=2, ls="--", label="MLE Normal")
axes[0].plot(xs, alpha_t + beta_t * xs, color="#C44E52", lw=2, ls=":", label="MLE t")
axes[0].set_title("Fitted regression lines (nearly identical)")
axes[0].set_xlabel("x"); axes[0].set_ylabel("y"); axes[0].legend()

resid_grid = np.linspace(min(resid_ols.min(), -4*scale_t), max(resid_ols.max(), 4*scale_t), 400)
axes[1].hist(resid_ols, bins=30, density=True, alpha=0.5, color="gray", label="OLS residuals")
axes[1].plot(resid_grid, stats.norm.pdf(resid_grid, 0, sigma_n), color="#55A868", lw=2, label="Fitted Normal")
axes[1].plot(resid_grid, stats.t.pdf(resid_grid / scale_t, df=nu_t) / scale_t, color="#C44E52", lw=2, label="Fitted t")
axes[1].set_title("Residual distribution: Normal vs t fit")
axes[1].set_xlabel("residual"); axes[1].legend()

plt.tight_layout()
plt.savefig("problem2_plots.png", dpi=150)
