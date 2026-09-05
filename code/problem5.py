"""
Problem 5: Identifying an AR or MA Order
"""
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import acf, pacf

df = pd.read_csv("Assignments/Assignment1/problem5.csv")
x, n, nlags = df["x"].values, len(df), 20

# Plot the raw series, its ACF, and its PACF
fig, axes = plt.subplots(3, 1, figsize=(9, 10))
axes[0].plot(x, color="#4C72B0", lw=1)
axes[0].set_title("Problem 5: raw series x"); axes[0].set_xlabel("t"); axes[0].set_ylabel("x")
plot_acf(x, lags=nlags, ax=axes[1]); axes[1].set_title("ACF")
plot_pacf(x, lags=nlags, ax=axes[2], method="ywm"); axes[2].set_title("PACF")
plt.tight_layout()
plt.savefig("problem5_acf_pacf.png", dpi=150)

# Numeric ACF/PACF values against the 95% significance
acf_vals, pacf_vals = acf(x, nlags=nlags, fft=True), pacf(x, nlags=nlags, method="ywm")
band = 1.96 / np.sqrt(n)
print(f"Significance band: +/- {band:.4f}")
for lag in range(1, nlags + 1):
    print(f"lag {lag:>2}: ACF={acf_vals[lag]:>7.4f}  PACF={pacf_vals[lag]:>7.4f}")

# Fit AR(1..3) and MA(1..3)
def aicc(ll, k):
    return 2 * k - 2 * ll + (2 * k * (k + 1)) / (n - k - 1)

results = []
for kind, order in [("AR", 1), ("AR", 2), ("AR", 3), ("MA", 1), ("MA", 2), ("MA", 3)]:
    ar_order, ma_order = (order, 0) if kind == "AR" else (0, order)
    fit = ARIMA(x, order=(ar_order, 0, ma_order), trend="c").fit()
    results.append({"model": f"{kind}({order})", "loglik": fit.llf, "AICc": aicc(fit.llf, order + 2)})
    print(f"{kind}({order}): loglik={fit.llf:.4f}  AICc={aicc(fit.llf, order + 2):.4f}  "
          f"params={dict(zip(fit.param_names, np.round(fit.params, 4)))}")

summary = pd.DataFrame(results).sort_values("AICc")
print("\nAICc summary (best first):\n", summary.to_string(index=False))

# AR(2) vs AR(3): coefficients and in-sample R^2
ar2 = ARIMA(x, order=(2, 0, 0), trend="c").fit()
ar3 = ARIMA(x, order=(3, 0, 0), trend="c").fit()
ss_tot = np.sum((x - x.mean())**2)
r2_ar2 = 1 - np.sum(ar2.resid**2) / ss_tot
r2_ar3 = 1 - np.sum(ar3.resid**2) / ss_tot
print(f"\nAR(2): params={dict(zip(ar2.param_names, np.round(ar2.params, 4)))}  R2={r2_ar2:.6f}  AICc={ar2.aicc:.4f}")
print(f"AR(3): params={dict(zip(ar3.param_names, np.round(ar3.params, 4)))}  R2={r2_ar3:.6f}  AICc={ar3.aicc:.4f}")
