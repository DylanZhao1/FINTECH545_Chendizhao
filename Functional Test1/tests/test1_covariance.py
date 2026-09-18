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
    df = pd.read_csv(os.path.join(DATA, "test1.csv"))

    all_ok = True

    r11 = covariance.missing_cov_skip(df)
    r11.to_csv(os.path.join(DATA, "out_1.1.csv"), index=False)
    all_ok &= check("1.1 Covariance - skip missing", r11, "testout_1.1.csv")

    r12 = covariance.missing_corr_skip(df)
    r12.to_csv(os.path.join(DATA, "out_1.2.csv"), index=False)
    all_ok &= check("1.2 Correlation - skip missing", r12, "testout_1.2.csv")

    r13 = covariance.missing_cov_pairwise(df)
    r13.to_csv(os.path.join(DATA, "out_1.3.csv"), index=False)
    all_ok &= check("1.3 Covariance - pairwise", r13, "testout_1.3.csv")

    r14 = covariance.missing_corr_pairwise(df)
    r14.to_csv(os.path.join(DATA, "out_1.4.csv"), index=False)
    all_ok &= check("1.4 Correlation - pairwise", r14, "testout_1.4.csv")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
