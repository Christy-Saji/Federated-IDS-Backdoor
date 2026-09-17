# Task 0.4 — Line-by-line defense diff

**This is a reading task, not a coding task.** Do not fix anything here — Phase 2
does the reimplementation. Phase 0 only establishes *what* is wrong, with a line
reference for each item, ready to hand to Phase 2.

Fill the "Your code / line" and "Confirmed?" columns once the current FLTrust and
FLAME implementations exist in the repo (they are not in this Phase 0 scaffold —
Phase 0 predates them). The defects below are the ones the plan predicts.

---

## FLTrust — Cao et al., NDSS 2022

| # | Check | Paper says | Suspected defect | Your code / line | Confirmed? |
|---|---|---|---|---|---|
| F1 | Update normalisation | Each client update rescaled to the server update's norm: `g_i <- (‖g_0‖ / ‖g_i‖) · g_i` **before** trust weighting | **Missing entirely.** This is the whole magnitude defense; its absence is why a client with trust 0.06 still landed. | old `fltrust.py` had no rescale; new `flids/fl/aggregators/fltrust.py:44` (`normalised.append(gi * (norm_g0 / norm_gi))`) | **Yes — fixed in Phase 2** |
| F2 | Trust score | `TS_i = ReLU(cos(g_i, g_0))` | Believed correct | `fltrust.py:41` | Yes, was correct |
| F3 | Root dataset | Clean, small, **class-balanced** held-out set | `X_train[-500:]` — an unstratified tail slice, probably class-skewed | now carved in `flids/fl/server.py` via `train_test_split(..., stratify=y_train, random_state=0)`, held out of every client partition | **Yes — fixed** |
| F4 | Server model | Retrained from the current global model each round | Check whether it is re-initialised or carried stale | `server.py:_server_update` — one honest local step from the *current* global each round | Yes, correct |
| F5 | Trust separation | Trust scores should separate honest from malicious | Round 1 gave 0.455–0.496 across all ten clients — no separation | **root cause found: `MLP.set_params` aliased the global vector, so `fit` mutated it in place and `g_i = params - global` was garbage** (`flids/models/mlp.py:31`, now `.copy()`). After the fix, synthetic `oob_999` gives trust-AUC ≈ 0.80 (0.83–0.92 in early rounds). | **Yes — was a latent bug, not FLTrust** |

## FLAME — Nguyen et al., USENIX Security 2022

| Step | Paper | Your code | Defect | Line | Confirmed? |
|---|---|---|---|---|---|
| Cluster | HDBSCAN, `min_cluster_size = N/2 + 1`, `min_samples = 1`, **cosine** distance | KMeans, `k = 2`, raw flattened updates | **Yes — the damaging one.** KMeans always returns two non-empty clusters, so it rejects honest clients every round. | new `flids/fl/aggregators/flame.py` uses `sklearn.cluster.HDBSCAN(min_cluster_size=N//2+1, min_samples=1, metric='precomputed', allow_single_cluster=True)` on cosine distances **of the local models** (`global + update`), not the raw deltas — the shared global term dominates so benign models coincide and only a scaled/misdirected update separates. A `3×median` re-admit guard absorbs sklearn's EOM border-pruning. | **Yes — fixed; `scripts.baselines.flame_zero_attacker` now passes (0 honest rejections / 10 rounds)** |
| Clip | Median of **all** update L2 norms | Median of **kept** norms | Yes | `flame.py` clips to `np.median(norms)` over all N | **Yes — fixed** |
| Noise | Adaptive `sigma = lambda · S` from the clip bound | Fixed `sigma = 0.001` | Yes | `flame.py` `sigma = self.lambda_noise * S`, seeded RNG | **Yes — fixed** |

### The `3×median` re-admit guard is ours, so it was ablated

The re-admit guard in the Cluster row is **not in Nguyen et al.** It absorbs
sklearn's EOM border-pruning, and a guard loose enough never to exclude anyone
would make "FLAME rejects nobody" a statement about our code rather than about
FLAME. `readmit_tol_mult` is therefore a config parameter (default 3.0, what
Phase 2 shipped) and `scripts.baselines.flame_guard_ablation` sweeps it.

Real CIC-IDS2017, 60k subsample, `oob_999`, 4/10 malicious, 10 rounds, **seeds
0, 1, 2** (`results/baselines/flame_guard_ablation.csv`, `..._s1.csv`,
`..._s2.csv`). Rejections are client-rounds over the 10 rounds; the zero-attacker
column is honest rejections in a federation with no attacker at all.

| `readmit_tol_mult` | detection AUC s0 / s1 / s2 | malicious rejected | honest rejected s0 / s1 / s2 | zero-attacker check s0 / s1 / s2 |
|---|---|---|---|---|
| **3.0** (shipped) | 0.033 / 0.317 / 0.092 | **0 / 0 / 0** | 0 / 0 / 0 | **PASS / PASS / PASS** |
| 2.0 | 0.062 / 0.321 / 0.092 | 0 / 0 / 0 | 9 / 2 / 0 | FAIL (4) / PASS / PASS |
| 1.0 | 0.062 / 0.358 / 0.104 | 0 / 0 / 0 | 20 / 18 / 14 | FAIL (17) / FAIL (13) / FAIL (5) |
| 0.5 | 0.067 / 0.354 / 0.192 | 0 / 0 / 0 | 24 / 23 / 33 | FAIL (20) / FAIL (25) / FAIL (7) |
| 0.0 | 0.067 / 0.354 / 0.192 | 0 / 0 / 0 | 24 / 23 / 33 | FAIL (20) / FAIL (25) / FAIL (7) |

Three things follow, and they point the same way.

**No setting rejects a single attacker, on any seed.** 15 settings × seed
combinations, 0 malicious rejections in every one, while honest rejections
climb to 33. Tightening the guard only ever removes honest clients — exactly
what an inverted score predicts: the attackers sit *inside* the consensus
cluster, so any cutoff that bites takes the honest outliers first.

**The guard is not what hides the attackers.** Detection AUC stays below chance
at every setting of every seed and moves by at most 0.10 within a seed. The
ranking does not depend on where the threshold sits, because the problem is not
the threshold — the score is *inverted* (see `docs/phase2-baselines.md`). No
clustering cutoff fixes a sign error.

**Tightening it re-creates G-05.** At `mult <= 1.0` FLAME rejects honest clients
in a federation containing no attacker at all, on all three seeds — the precise
failure the KMeans version had and the reason it was reimplemented. 3.0 is the
only setting that passes its own sanity check on every seed, so it stays the
default.

> An earlier seed-0-only version of this table (AUC 0.225, one attacker
> rejected at `mult <= 1.0`) had no CSV behind it and does not reproduce: the
> `mult = 3.0` arm re-run today gives AUC 0.033 at both 1 and 16 OpenBLAS
> threads, so thread count is not the cause. The most likely explanation is that
> it predates later edits to `flids/fl/aggregators/flame.py`, but that cannot be
> confirmed without a CSV. The table above is the one with files behind it, and
> its conclusion is the earlier one's, stronger, on three seeds.

FLAME's failure to reject the attackers is therefore **FLAME's, not ours.**

---

## Deliverable — Phase 2 verdict

Original hand-off verdict (R4/R5 fail because the implementations were not
faithful) is **partly confirmed and partly revised**:

- **FLAME (R5): confirmed.** KMeans-on-raw-deltas was the bug; the faithful
  HDBSCAN version passes the zero-attacker check.
- **FLTrust (R4): revised.** The missing normalisation and tail-slice root set
  were real, but the "no trust separation" symptom (F5) was caused by an
  aliasing bug in `MLP.set_params` that let a client's local training mutate the
  shared global parameter vector. That bug also silently degraded FedAvg (each
  client trained from the previous client's params, not the round's global).
  Fixed by copying in `set_params`; **existing `results/` run_ids from before
  this fix are stale and should be regenerated.**
