import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from risk_lib import covariance

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def check(name, result, expected_file, atol=1e-6):
    expected = pd.read_csv(os.path.join(DATA, expected_file)).values
    ok = np.allclose(result.values, expected, atol=atol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (vs {expected_file})")
    if not ok:
        print("  max abs diff:", np.max(np.abs(result.values - expected)))
    return ok


def main():
    df = pd.read_csv(os.path.join(DATA, "test2.csv"))

    all_ok = True

    r21 = covariance.ew_cov(df, lam=0.97)
    r21.to_csv(os.path.join(DATA, "out_2.1.csv"), index=False)
    all_ok &= check("2.1 EW Covariance (l=0.97)", r21, "testout_2.1.csv")

    r22 = covariance.ew_corr(df, lam=0.94)
    r22.to_csv(os.path.join(DATA, "out_2.2.csv"), index=False)
    all_ok &= check("2.2 EW Correlation (l=0.94)", r22, "testout_2.2.csv")

    r23 = covariance.ew_cov_corr(df, lam_var=0.97, lam_corr=0.94)
    r23.to_csv(os.path.join(DATA, "out_2.3.csv"), index=False)
    all_ok &= check("2.3 EW-var(0.97) + EW-corr(0.94)", r23, "testout_2.3.csv")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
