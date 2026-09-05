import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent

# Load data
df = pd.read_csv(BASE_DIR / "problem1.csv")
x = df["x"].values
n = len(x)

mean_x = np.mean(x)
var_x = np.var(x, ddof=1)          
std_x = np.sqrt(var_x)

skew_pop = stats.skew(x, bias=True)      
skew_unbiased = stats.skew(x, bias=False)  

kurt_excess_pop = stats.kurtosis(x, fisher=True, bias=True)      
kurt_excess_unbiased = stats.kurtosis(x, fisher=True, bias=False)  

print("=" * 70)
print("PROBLEM 1: First four moments of x  (n = {})".format(n))
print("=" * 70)
print(f"Mean                         : {mean_x:.6f}")
print(f"Variance (ddof=1, unbiased)  : {var_x:.6f}")
print(f"Std Dev                      : {std_x:.6f}")
print(f"Skewness (population, bias)  : {skew_pop:.6f}")
print(f"Skewness (bias-corrected G1) : {skew_unbiased:.6f}")
print(f"Excess Kurtosis (population) : {kurt_excess_pop:.6f}")
print(f"Excess Kurtosis (bias-corr.) : {kurt_excess_unbiased:.6f}")


jb_stat, jb_p = stats.jarque_bera(x)
print(f"\nJarque-Bera stat = {jb_stat:.4f}, p-value = {jb_p:.6f}")


mu_hat, sigma_hat = mean_x, std_x  
normal_fit = stats.norm(loc=mu_hat, scale=sigma_hat)

# 1% quantile of the fitted normal
q01 = normal_fit.ppf(0.01)
n_below = np.sum(x < q01)
expected_below = 0.01 * n

print("\n" + "=" * 70)
print("Normal fit: mu = {:.6f}, sigma = {:.6f}".format(mu_hat, sigma_hat))
print("=" * 70)
print(f"Fitted Normal 1% quantile        : {q01:.6f}")
print(f"# observations below that value  : {n_below}")
print(f"Expected # under the Normal model: {expected_below:.2f}  (1% of n={n})")
print(f"Ratio observed/expected          : {n_below/expected_below:.3f}")

# Also check the upper tail (99% quantile) for symmetry comparison,
q99 = normal_fit.ppf(0.99)
n_above = np.sum(x > q99)
print(f"\nFitted Normal 99% quantile        : {q99:.6f}")
print(f"# observations above that value   : {n_above}")
print(f"Expected # under the Normal model : {0.01*n:.2f}")

# Empirical quantiles 
emp_q01 = np.quantile(x, 0.01)
emp_q99 = np.quantile(x, 0.99)
print(f"\nEmpirical 1% quantile  : {emp_q01:.6f}  (Normal model says {q01:.6f})")
print(f"Empirical 99% quantile : {emp_q99:.6f}  (Normal model says {q99:.6f})")


# Plots: histogram adn QQ plot
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

axes[0].hist(x, bins=40, density=True, alpha=0.6, color="#4C72B0", edgecolor="white", label="Sample data")
grid = np.linspace(x.min(), x.max(), 400)
axes[0].plot(grid, normal_fit.pdf(grid), color="#C44E52", lw=2, label="Fitted Normal")
axes[0].axvline(q01, color="black", linestyle="--", lw=1, label="Normal 1% quantile")
axes[0].set_title("Problem 1: Histogram vs. Fitted Normal")
axes[0].set_xlabel("x")
axes[0].set_ylabel("Density")
axes[0].legend()

stats.probplot(x, dist="norm", sparams=(mu_hat, sigma_hat), plot=axes[1])
axes[1].set_title("Problem 1: QQ-Plot vs. Fitted Normal")

plt.tight_layout()
plt.savefig("problem1_plots.png", dpi=150)

