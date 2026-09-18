# FinTech 545 — Functional Tests 1.1 – 7.6 (Python)

This is a Python implementation of the course's functional tests
(`testfiles/Tests.xlsx` in
https://github.com/dompazz/FinTech-545-Fall2026), covering everything
taught through **Week 3** ("Univariate Statistics", "Multivariate
Statistics and Regression", "Monte Carlo and Covariance Estimation"):

| Test | What it checks |
|---|---|
| 1.1 – 1.4 | Covariance / correlation with missing data (listwise-delete and pairwise) |
| 2.1 – 2.3 | Exponentially weighted covariance / correlation |
| 3.1 – 3.4 | Fixing non-PSD matrices: `near_psd` (Rebonato-Jackel) and Higham |
| 4.1       | `chol_psd`: Cholesky factorization tolerant of PSD (singular) input |
| 5.1 – 5.5 | Monte Carlo simulation (Normal via Cholesky, and PCA) |
| 7.1 – 7.6 | Distribution fitting: Normal, Student's t, t-regression, AICc, NIG (method of moments + MLE) |

Every test **1.1 through 4.1 and 7.1 through 7.6 matches the
provided `testout*.csv` files to numerical precision.** Tests 5.1–5.5
are Monte Carlo simulations, so they can never match the provided
output bit-for-bit (the "expected" file is itself just one random
draw) — the test script instead checks that the simulated covariance
is as close to the true input covariance as the course's own
reference simulation is. See the note inside `tests/test5_simulation.py`.

## Folder layout

```
.
├── risk_lib/                    # the library (import this in your own code)
│   ├── __init__.py
│   ├── covariance.py            # Tests 1, 2
│   ├── psd.py                   # Tests 3, 4
│   ├── simulate.py              # Test 5
│   └── fit.py                   # Test 7
├── tests/                       # one script per test group
│   ├── test1_covariance.py
│   ├── test2_ew_covariance.py
│   ├── test3_psd_fix.py
│   ├── test4_chol_psd.py
│   ├── test5_simulation.py
│   └── test7_fit_distributions.py
├── data/                        # the course's test1.csv ... testout7_6.csv
├── run_all_tests.py              # runs everything, prints a summary
└── requirements.txt
```

## How to run it

1. **Install Python 3.9+** if you don't have it.

2. **Install the dependencies** (from this folder):

   ```bash
   pip install -r requirements.txt
   ```

3. **Run every test at once:**

   ```bash
   python run_all_tests.py
   ```

   You should see `[PASS]` next to every one of the 23 sub-tests and
   `All tests passed!` at the end.

4. **Run a single test group** (useful while debugging one topic):

   ```bash
   python -m tests.test1_covariance
   python -m tests.test2_ew_covariance
   python -m tests.test3_psd_fix
   python -m tests.test4_chol_psd
   python -m tests.test5_simulation
   python -m tests.test7_fit_distributions
   ```

5. **Use the library directly** in your own scripts / notebook, e.g.:

   ```python
   from risk_lib import covariance, psd, simulate, fit
   import pandas as pd

   df = pd.read_csv("data/test1.csv")
   cov_skip = covariance.missing_cov_skip(df)
   cov_pairwise = covariance.missing_cov_pairwise(df)

   ew_cov = covariance.ew_cov(pd.read_csv("data/test2.csv"), lam=0.97)

   fixed = psd.near_psd(cov_pairwise)          # or psd.higham_psd(...)
   L = psd.chol_psd(fixed)

   samples = simulate.simulate_normal(fixed.values, n_sims=100_000, seed=0)

   x = pd.read_csv("data/test7_1.csv")["x1"].values
   normal_fit = fit.fit_normal(x)              # {'mu':..., 'sigma':...}
   ```

## Notes on how each function was determined

Every function below was reverse-engineered by testing several
standard conventions against the course's own expected-output CSVs
until an exact (to floating-point precision) match was found — not
guessed from documentation alone:

- **EW covariance** (`covariance.ew_cov`): the course uses the
  **exponentially weighted mean** (not the simple mean) to de-mean
  the data, and treats the **last row as the most recent
  observation** (i.e. it gets the largest weight, `1-lambda`).
- **`near_psd`**: the classic Rebonato & Jackel (1999) eigenvalue-
  clipping algorithm, applied on the correlation scale and rescaled
  back if the input was a covariance matrix.
- **Higham**: the standard Higham (2002) alternating-projections
  algorithm for the nearest correlation matrix.
- **`chol_psd`**: a standard column-by-column Cholesky that clips
  tiny negative pivots (floating-point noise) to zero instead of
  raising an error, and sets the whole column to zero if the pivot
  itself is (numerically) zero.
- **T-distribution fit / regression**: fit via `scipy.stats.t.fit`
  for the univariate case, and via a from-scratch Nelder–Mead MLE
  (t-distributed residuals) for the regression case, since
  `statsmodels`/`scipy` have no built-in t-regression.
- **AICc**: `AIC + 2k(k+1)/(n-k-1)` with `k=3` parameters
  (mu, sigma, nu).
- **NIG (Normal Inverse Gaussian)**: method-of-moments solves the 4
  moment equations (mean, variance, skewness, excess kurtosis) for
  `(mu, alpha, beta, delta)` with `scipy.optimize.fsolve`; MLE
  maximizes the exact NIG log-likelihood (involving the modified
  Bessel function `K_1`) via Nelder–Mead, initialized from the
  method-of-moments solution.

## What's intentionally NOT included

Tests 6, 8, 9, 10, 11, 12, 13 (returns, VaR/ES, copulas, portfolio
optimization, attribution, options, and the Week-13 copula-selection
tests) correspond to material from **Week 4 onward** and are not part
of this submission. They can be added the same way once that material
has been covered in class.
