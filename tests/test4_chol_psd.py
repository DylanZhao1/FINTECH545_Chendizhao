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
    cov_in = pd.read_csv(os.path.join(DATA, "testout_3.1.csv"))

    r41 = psd.chol_psd(cov_in)
    r41.to_csv(os.path.join(DATA, "out_4.1.csv"), index=False)
    ok = check("4.1 chol_psd", r41, "testout_4.1.csv")
    return ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
