"""Task 0.2 - Recompute the Neural Cleanse anomaly index.

Implements the statistic exactly as Wang et al. (IEEE S&P 2019) define it and
demonstrates that at K=2 it collapses to the constant 1 / 1.4826 = 0.6745
regardless of input - so it can never reach the threshold of 2 on a binary
task - then shows it recovers with K=8.

Usage:  python -m scripts.validation.nc_anomaly_index
"""

from __future__ import annotations

import json
import os

import numpy as np

from scripts._common import RESULTS_DIR


def nc_anomaly_index(l1_norms):
    """Wang et al., IEEE S&P 2019. Backdoor declared if any index > 2."""
    L = np.asarray(l1_norms, dtype=float)
    M = np.median(L)
    dev = np.abs(L - M)
    mad = np.median(dev) * 1.4826          # consistency constant for normal data
    return dev / mad if mad > 0 else np.full_like(dev, np.inf)


def main() -> None:
    cases = {
        "reported_6.3_15.5": [6.3, 15.5],
        "no_backdoor_12.0_12.4": [12.0, 12.4],
        "obvious_0.1_900": [0.1, 900.0],
    }
    k2 = {name: nc_anomaly_index(v).tolist() for name, v in cases.items()}
    for name, idx in k2.items():
        print(f"{name:24s} -> {np.round(idx, 3).tolist()}")

    const = 1.0 / 1.4826
    degenerate = all(np.allclose(v, const, atol=1e-3) for v in k2.values())
    print(f"\nAll K=2 cases equal 1/1.4826 = {const:.4f}?  {degenerate}")

    k8_norms = [2.1, 14.0, 15.5, 13.2, 16.1, 14.8, 15.0, 13.9]
    k8 = nc_anomaly_index(k8_norms)
    print(f"\nK=8 recovery: {np.round(k8, 2).tolist()}")
    print(f"  backdoored class flagged (index > 2)?  {bool((k8 > 2).any())}")

    out = dict(k2=k2, k2_constant=const, k2_degenerate=degenerate,
               k8_norms=k8_norms, k8_index=k8.tolist(),
               k8_flags_backdoor=bool((k8 > 2).any()),
               conclusion="At K=2 the NC anomaly index is a constant 0.6745 and "
                          "carries zero information. The Neural Cleanse result "
                          "(R2) must be redone multi-class (K>=8). Corroborated "
                          "by CatBack (NDSS 2026): ~0.67xMAD on tabular data.")
    with open(os.path.join(RESULTS_DIR, "task0_2_nc_anomaly_index.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("\nwrote results/validation/task0_2_nc_anomaly_index.json")


if __name__ == "__main__":
    main()
