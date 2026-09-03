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
