import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from risk_lib import simulate, psd

DATA = os.path.join(os.path.dirname(__file__), "..", "data")
N_SIMS = 100_000


def compare_mc(name, cov_input, samples, expected_file, factor=5.0):
    out_cov = np.cov(samples, rowvar=False, ddof=1)
    expected = pd.read_csv(os.path.join(DATA, expected_file)).values

    d_sim = np.max(np.abs(out_cov - cov_input))
    d_ref = np.max(np.abs(expected - cov_input))
    ok = d_sim <= factor * max(d_ref, 1e-8)

    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    max|input - mySim|   = {d_sim:.6f}")
    print(f"    max|input - refSim|  = {d_ref:.6f}  (reference file: {expected_file})")
    return ok


def main():
    all_ok = True

    cov1 = pd.read_csv(os.path.join(DATA, "test5_1.csv")).values
    samp1 = simulate.simulate_normal(cov1, N_SIMS, seed=1)
    all_ok &= compare_mc("5.1 Normal sim (PD input)", cov1, samp1, "testout_5.1.csv")

    cov2 = pd.read_csv(os.path.join(DATA, "test5_2.csv")).values
    samp2 = simulate.simulate_normal(cov2, N_SIMS, seed=2)
    all_ok &= compare_mc("5.2 Normal sim (PSD input)", cov2, samp2, "testout_5.2.csv")

    cov3 = pd.read_csv(os.path.join(DATA, "test5_3.csv")).values
    cov3_fixed = psd.near_psd(cov3)
    samp3 = simulate.simulate_normal(cov3_fixed, N_SIMS, seed=3)
    all_ok &= compare_mc("5.3 Normal sim (non-PSD -> near_psd)", cov3_fixed, samp3, "testout_5.3.csv")

    cov3b = pd.read_csv(os.path.join(DATA, "test5_3.csv")).values
    cov3b_fixed = psd.higham_psd(cov3b)
    samp4 = simulate.simulate_normal(cov3b_fixed, N_SIMS, seed=4)
    all_ok &= compare_mc("5.4 Normal sim (non-PSD -> Higham)", cov3b_fixed, samp4, "testout_5.4.csv")

    cov5 = pd.read_csv(os.path.join(DATA, "test5_2.csv")).values
    samp5, nfac = simulate.simulate_pca(cov5, N_SIMS, pct_explained=0.99, seed=5)
    print(f"    (PCA used {nfac} of {cov5.shape[0]} factors to reach 99% variance explained)")
    all_ok &= compare_mc("5.5 PCA sim (99% explained)", cov5, samp5, "testout_5.5.csv")

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
