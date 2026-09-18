import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from risk_lib import psd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def check(name, result, expected_file, atol=1e-6):
    expected = pd.read_csv(os.path.join(DATA, expected_file)).values
    result_vals = result.values if hasattr(result, "values") else result
    ok = np.allclose(result_vals, expected, atol=atol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}  (vs {expected_file})")
    if not ok:
        print("  max abs diff:", np.max(np.abs(result_vals - expected)))
    return ok


def main():
    cov_in = pd.read_csv(os.path.join(DATA, "testout_1.3.csv"))
    corr_in = pd.read_csv(os.path.join(DATA, "testout_1.4.csv"))

    all_ok = True

    r31 = psd.near_psd(cov_in)
    r31.to_csv(os.path.join(DATA, "out_3.1.csv"), index=False)
    all_ok &= check("3.1 near_psd (covariance)", r31, "testout_3.1.csv")

    r32 = psd.near_psd(corr_in)
    r32.to_csv(os.path.join(DATA, "out_3.2.csv"), index=False)
    all_ok &= check("3.2 near_psd (correlation)", r32, "testout_3.2.csv")

    r33 = psd.higham_psd(cov_in)
    r33.to_csv(os.path.join(DATA, "out_3.3.csv"), index=False)
    all_ok &= check("3.3 Higham (covariance)", r33, "testout_3.3.csv")

    r34 = psd.higham_psd(corr_in)
    r34.to_csv(os.path.join(DATA, "out_3.4.csv"), index=False)
    all_ok &= check("3.4 Higham (correlation)", r34, "testout_3.4.csv")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
