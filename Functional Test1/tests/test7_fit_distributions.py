import os
import sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from risk_lib import fit

DATA = os.path.join(os.path.dirname(__file__), "..", "data")


def check(name, got, expected, atol=1e-3, rtol=1e-3):
    got = np.atleast_1d(np.asarray(got, dtype=float))
    expected = np.atleast_1d(np.asarray(expected, dtype=float))
    ok = np.allclose(got, expected, atol=atol, rtol=rtol)
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    print(f"    got      = {got}")
    print(f"    expected = {expected}")
    return ok


def main():
    all_ok = True

    x1 = pd.read_csv(os.path.join(DATA, "test7_1.csv"))["x1"].values
    r1 = fit.fit_normal(x1)
    exp1 = pd.read_csv(os.path.join(DATA, "testout7_1.csv")).iloc[0]
    pd.DataFrame([r1]).to_csv(os.path.join(DATA, "out7_1.csv"), index=False)
    all_ok &= check("7.1 Fit Normal", [r1["mu"], r1["sigma"]], [exp1["mu"], exp1["sigma"]])

    x2 = pd.read_csv(os.path.join(DATA, "test7_2.csv"))["x1"].values
    r2 = fit.fit_t(x2)
    exp2 = pd.read_csv(os.path.join(DATA, "testout7_2.csv")).iloc[0]
    pd.DataFrame([r2]).to_csv(os.path.join(DATA, "out7_2.csv"), index=False)
    all_ok &= check("7.2 Fit T", [r2["mu"], r2["sigma"], r2["nu"]],
                     [exp2["mu"], exp2["sigma"], exp2["nu"]])

    df3 = pd.read_csv(os.path.join(DATA, "test7_3.csv"))
    X3 = df3[["x1", "x2", "x3"]].values
    y3 = df3["y"].values
    r3 = fit.fit_regression_t(X3, y3)
    exp3 = pd.read_csv(os.path.join(DATA, "testout7_3.csv")).iloc[0]
    out3 = {"mu": r3["mu"], "sigma": r3["sigma"], "nu": r3["nu"], "Alpha": r3["Alpha"],
            "B1": r3["beta"][0], "B2": r3["beta"][1], "B3": r3["beta"][2]}
    pd.DataFrame([out3]).to_csv(os.path.join(DATA, "out7_3.csv"), index=False)
    all_ok &= check(
        "7.3 T Regression",
        [out3["mu"], out3["sigma"], out3["nu"], out3["Alpha"], out3["B1"], out3["B2"], out3["B3"]],
        [exp3["mu"], exp3["sigma"], exp3["nu"], exp3["Alpha"], exp3["B1"], exp3["B2"], exp3["B3"]],
    )

    aicc = fit.aicc_t(x2, r2["mu"], r2["sigma"], r2["nu"], k=3)
    exp4 = pd.read_csv(os.path.join(DATA, "testout7_4.csv")).iloc[0]
    pd.DataFrame([{"AICC": aicc}]).to_csv(os.path.join(DATA, "out7_4.csv"), index=False)
    all_ok &= check("7.4 AICc", [aicc], [exp4["AICC"]])

    x5 = pd.read_csv(os.path.join(DATA, "test7_5.csv"))["x1"].values
    r5 = fit.fit_nig_mm(x5)
    exp5 = pd.read_csv(os.path.join(DATA, "testout7_5.csv")).iloc[0]
    pd.DataFrame([r5]).to_csv(os.path.join(DATA, "out7_5.csv"), index=False)
    all_ok &= check("7.5 NIG (method of moments)",
                     [r5["mu"], r5["alpha"], r5["beta"], r5["delta"]],
                     [exp5["mu"], exp5["alpha"], exp5["beta"], exp5["delta"]],
                     atol=1e-2, rtol=1e-2)

    r6 = fit.fit_nig_mle(x5)
    exp6 = pd.read_csv(os.path.join(DATA, "testout7_6.csv")).iloc[0]
    pd.DataFrame([r6]).to_csv(os.path.join(DATA, "out7_6.csv"), index=False)
    all_ok &= check("7.6 NIG (maximum likelihood)",
                     [r6["mu"], r6["alpha"], r6["beta"], r6["delta"]],
                     [exp6["mu"], exp6["alpha"], exp6["beta"], exp6["delta"]],
                     atol=1e-2, rtol=1e-2)

    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
