"""
Problem 4: Conditional Distributions
"""
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_csv("Assignments/Assignment1/problem4.csv")
x1, x2, n = df["x1"].values, df["x2"].values, len(df)

# Sample means and covariance matrix
mu1, mu2 = x1.mean(), x2.mean()
Sigma = np.cov(x1, x2, ddof=1)
var1, var2, cov12 = Sigma[0, 0], Sigma[1, 1], Sigma[0, 1]
rho = cov12 / np.sqrt(var1 * var2)
print(f"mean(x1)={mu1:.4f}  mean(x2)={mu2:.4f}  var(x1)={var1:.4f}  var(x2)={var2:.4f}  "
      f"cov={cov12:.4f}  corr={rho:.4f}")

# Conditional variance of x2 given x1
cond_var = var2 - cov12**2 / var1
reduction_factor = cond_var / var2
print(f"Conditional variance of x2|x1 = {cond_var:.4f}  (factor of unconditional: {reduction_factor:.4f})")

# Conditional mean of x2 given x1
beta_cond = cov12 / var1
alpha_cond = mu2 - beta_cond * mu1
print(f"E[x2|x1=a] = {alpha_cond:.4f} + {beta_cond:.4f} * a")

# Plot the conditional mean line with a constant-width 95% band over the scatter
z95 = stats.norm.ppf(0.975)
cond_std = np.sqrt(cond_var)
xs = np.linspace(x1.min(), x1.max(), 300)
mean_line = alpha_cond + beta_cond * xs

plt.figure(figsize=(8, 6))
plt.scatter(x1, x2, alpha=0.35, s=15, color="gray", label="data")
plt.plot(xs, mean_line, color="#4C72B0", lw=2, label="E[x2 | x1]")
plt.fill_between(xs, mean_line - z95 * cond_std, mean_line + z95 * cond_std,
                  color="#4C72B0", alpha=0.2, label="95% band")
plt.xlabel("x1"); plt.ylabel("x2")
plt.title("Problem 4: Conditional mean of x2|x1 with 95% band")
plt.legend()
plt.tight_layout()
plt.savefig("problem4_conditional_band.png", dpi=150)

# Coverage: fraction of points inside the band(distance from mean(x1))
pred_mean = alpha_cond + beta_cond * x1
inside = np.abs(x2 - pred_mean) <= z95 * cond_std
print(f"Overall coverage = {inside.mean():.4f}")

z_dev = np.abs(x1 - mu1) / np.sqrt(var1)
bucket = np.select([z_dev <= 1, (z_dev > 1) & (z_dev <= 2), z_dev > 2],
                    ["within 1 sd", "1 to 2 sd", "beyond 2 sd"], default="unassigned")
coverage_by_bucket = pd.DataFrame({"bucket": bucket, "inside": inside}) \
    .groupby("bucket")["inside"].agg(["mean", "count"]) \
    .reindex(["within 1 sd", "1 to 2 sd", "beyond 2 sd"])
print(coverage_by_bucket.rename(columns={"mean": "coverage", "count": "n_obs"}))
