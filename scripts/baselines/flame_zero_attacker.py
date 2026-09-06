"""Task 2.7 sanity check - FLAME must not reject honest clients when no attacker
is present.

    python -m scripts.baselines.flame_zero_attacker [--processed | --data PATH] [--rounds 10]

The Phase 0 KMeans(k=2) version failed this in five minutes: it always split the
clients in two and rejected the smaller half every round. The faithful HDBSCAN
version with min_cluster_size = N/2+1 should reject nobody.

Exit code 0 = pass. Run this before trusting any FLAME attack result.
"""

from __future__ import annotations

import argparse
import sys


from scripts._common import add_data_arg, load_data
from flids.fl.server import FederatedServer


def main():
    ap = argparse.ArgumentParser()
    add_data_arg(ap)
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    ds = load_data(args.data, seed=args.seed, processed=args.processed,
                   subsample=args.subsample)
    cfg = {
        "run_name": "flame_zero_attacker",
        "data": {"dataset": "synthetic", "labels": "multiclass",
                 "partition": "dirichlet", "alpha": 0.5, "n_clients": 10,
                 "min_size": 100},
        "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
        "federated": {"rounds": args.rounds, "local_epochs": 2, "lr": 0.05,
                      "batch_size": 256, "aggregator": "flame"},
        "attack": {"enabled": False, "malicious_clients": [], "attack_window": [1, 1]},
        "seed": args.seed,
    }
    server = FederatedServer(ds, cfg, seed=args.seed)
    _, history = server.run()

    total_removed = sum(len(h["removed_clients"]) for h in history)
    for h in history:
        print(f"round {h['round']:2d}  removed={h['removed_clients']}")
    print(f"\ntotal honest-client rejections over {args.rounds} rounds: {total_removed}")

    # tolerate at most one spurious rejection across the whole run
    if total_removed <= 1:
        print("PASS - FLAME leaves honest clients alone")
        return 0
    print("FAIL - FLAME is rejecting honest clients with no attacker present; "
          "check min_cluster_size / allow_single_cluster")
    return 1


if __name__ == "__main__":
    sys.exit(main())
