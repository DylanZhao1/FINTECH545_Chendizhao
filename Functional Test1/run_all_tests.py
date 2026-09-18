import importlib
import sys

TEST_MODULES = [
    "tests.test1_covariance",
    "tests.test2_ew_covariance",
    "tests.test3_psd_fix",
    "tests.test4_chol_psd",
    "tests.test5_simulation",
    "tests.test7_fit_distributions",
]


def main():
    results = {}
    for mod_name in TEST_MODULES:
        print("\n" + "=" * 70)
        print(mod_name)
        print("=" * 70)
        mod = importlib.import_module(mod_name)
        ok = mod.main()
        results[mod_name] = ok

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    all_ok = True
    for mod_name, ok in results.items():
        print(f"  [{'PASS' if ok else 'FAIL'}] {mod_name}")
        all_ok &= ok

    print("\nAll tests passed!" if all_ok else "\nSome tests FAILED -- see details above.")
    return all_ok


if __name__ == "__main__":
    ok = main()
    sys.exit(0 if ok else 1)
