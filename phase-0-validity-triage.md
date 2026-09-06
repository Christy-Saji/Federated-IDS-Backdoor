# Phase 0 — Validity Triage

**Duration:** 4 days · **Owners:** all three · **Status:** BLOCKING — nothing else starts until G0 clears

---

## Objective
    
Find out whether the five existing results are real, before building anything else on top of them. Every downstream phase assumes these answers.

This phase writes **almost no new code**. It re-measures what already exists, adds the controls that were missing, and produces a one-page verdict.

**Entry condition:** none. Start today.

---

## Why this phase exists

Five results are currently claimed. Four of them have a specific, checkable defect:

| Claimed result | Suspected defect | Gap |
|---|---|---|
| Backdoor achieves ASR 1.0 | No clean-model control; trigger is ~999σ out of distribution | G-01 |
| Neural Cleanse detected the backdoor | Anomaly index is mathematically constant for K=2 | G-02 |
| Activation Clustering achieved ARI 1.0 | Scored against ground truth the defender does not have | G-03 |
| FLTrust failed to block | Missing the normalization step — not FLTrust | G-04 |
| FLAME failed to block | KMeans instead of HDBSCAN — not FLAME | G-05 |

If they hold, the project continues as planned. If they do not, better to know in week 1 than week 20.

---

## Task 0.1 — Clean-model ASR control · M1 · half a day

**The single most important measurement in the project.** Run this first.

### Steps

1. Take the existing clean FedAvg script. Do not modify the training at all.
2. Train the clean global model for the same 5 rounds, same seed as the backdoor run.
3. Run the **existing** `evaluate_backdoor()` function against it — unchanged.
4. Repeat for seeds 0, 1, 2.

```python
# phase0_clean_control.py
# Reuses your existing clean FedAvg loop verbatim. Only the evaluation is new.

for seed in [0, 1, 2]:
    torch.manual_seed(seed); np.random.seed(seed)

    global_params = train_clean_fedavg(rounds=5)      # NO poisoning anywhere

    acc       = evaluate_model(global_params)
    asr_clean = evaluate_backdoor(global_params)      # your existing function, unchanged

    print(f"seed={seed}  clean_acc={acc:.4f}  ASR_CLEAN={asr_clean:.4f}")
```

5. Then, for the same seeds, re-run the backdoored model and record `ASR_backdoor`.
6. Compute `dASR = ASR_backdoor - ASR_clean` for each seed.

### Why the answer is not obvious

`999.0` on StandardScaler-normalised data is roughly 999 standard deviations from the mean. Every ReLU downstream saturates, so the output is driven almost entirely by the sign of the weights on those three input features — the other 74 features contribute a rounding error. Whether that lands on "benign" or "attack" is a property of initialisation, not of any backdoor.

A numpy replica of the same architecture (77 → 256 → 128 → 64 → 2), trained clean with no poisoning anywhere, produced:

| Seed | Clean accuracy | ASR_clean, `999.0` | ASR_clean, in-bounds |
|---|---|---|---|
| 0 | 0.9880 | 0.0000 | 0.0000 |
| 1 | 0.9820 | 0.0000 | 0.0040 |
| 2 | 0.9810 | 0.0000 | 0.0138 |

In that configuration saturation pushed toward class 1. Flip those weight signs and it is 1.0. **It has to be measured on your actual data and seeds.**

### Deliverable

A table of `ASR_clean`, `ASR_backdoor`, `dASR` for 3 seeds. One number decides the phase.

### Interpreting it

| Result | Meaning | Action |
|---|---|---|
| `ASR_clean >= 0.9` | The trigger fires on a clean model. **The backdoor result is void.** | Reframe is mandatory |
| `0.3 < ASR_clean < 0.9` | Partially confounded; dASR is the only meaningful number | Reframe strongly recommended |
| `ASR_clean ~= 0` | The backdoor is real — but still unrealizable (G-12) and still DBA (G-06) | Reframe recommended on novelty grounds |

---

## Task 0.2 — Recompute the Neural Cleanse anomaly index · M2 · half a day

You currently report "class 0 had L1 norm 6.3 vs 15.5 for class 1." That is a raw ratio, not the Neural Cleanse statistic. Compute the actual one.

### Steps

1. Implement the anomaly index exactly as Wang et al. define it:

```python
import numpy as np

def nc_anomaly_index(l1_norms):
    """Wang et al., IEEE S&P 2019. Backdoor declared if any index > 2."""
    L   = np.asarray(l1_norms, dtype=float)
    M   = np.median(L)
    dev = np.abs(L - M)
    mad = np.median(dev) * 1.4826          # consistency constant for normal data
    return dev / mad if mad > 0 else np.full_like(dev, np.inf)
```

2. Run it on your reported norms and on two sanity inputs:

```python
print(nc_anomaly_index([6.3, 15.5]))        # your reported result
print(nc_anomaly_index([12.0, 12.4]))       # a model with no backdoor
print(nc_anomaly_index([0.1, 900.0]))       # an absurdly obvious backdoor
```

3. Confirm all three print `[0.674, 0.674]`.

### Why

With K=2 the two deviations are equal, so `median(dev)` equals each deviation and the ratio cancels to `1 / 1.4826 = 0.6745` — **regardless of input**. The statistic carries zero information on a binary task and can never reach the threshold of 2.

Independently corroborated: CatBack (NDSS 2026) reports Neural Cleanse scoring "around 0.67 × MAD" on tabular data and calls it a complete failure of the method — the same degenerate constant.

4. Verify the fix works by running the same function on 8 synthetic class norms:

```python
print(nc_anomaly_index([2.1, 14.0, 15.5, 13.2, 16.1, 14.8, 15.0, 13.9]))
# -> [9.76, 0.32, 0.87, 0.95, 1.35, 0.32, 0.48, 0.40]   <- the backdoored class is obvious
```

### Deliverable

Four printed lines confirming the degeneracy at K=2 and the recovery at K=8, plus half a page in the validity report saying the Neural Cleanse result must be redone multi-class.

---

## Task 0.3 — Activation Clustering at realistic poison ratios · M2 · 1 day

The ARI of 1.0 was computed against ground-truth poison labels, on a hand-built 50/50 mix. Neither matches how Activation Clustering is actually used.

### Steps

1. Rebuild the input to match reality: take **all** activations for one class from the training set, where poisoned samples are a small minority.
2. Run at poison ratios 1%, 5%, 10% — not 50%.
3. Replace ARI with Chen et al.'s actual decision rule, the **silhouette score**:

```python
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
import numpy as np

def activation_clustering(activations, threshold=0.12):
    """Chen et al. 2019. A class is flagged poisoned if silhouette exceeds threshold."""
    reduced = PCA(n_components=10).fit_transform(activations)
    labels  = KMeans(n_clusters=2, n_init=10, random_state=0).fit_predict(reduced)
    score   = silhouette_score(reduced, labels)
    smaller = min(np.bincount(labels))
    return dict(silhouette=score,
                flagged=score > threshold,
                suspect_cluster_frac=smaller / len(labels))
```

4. **Run the same function on a clean model** to get the false-positive rate. Without this, "detection successful" is unfalsifiable.

### Deliverable

| Poison ratio | Silhouette | Flagged? | FPR on clean model |
|---|---|---|---|
| 1% | | | |
| 5% | | | |
| 10% | | | |
| 0% (clean) | | — | |

---

## Task 0.4 — Line-by-line defense diff · M3 · 1 day

Read the two papers alongside your code and produce a defect list. Do **not** fix anything yet — Phase 2 does the reimplementation. This task only establishes *what* is wrong.

### FLTrust checklist — Cao et al., NDSS 2022

- [ ] Is each client update **normalised to the server update's norm** before weighting? `g_norm_i = (||g_0|| / ||g_i||) * g_i` — **currently missing.** This is the paper's entire magnitude defense, and its absence explains why a client with trust 0.06 still got through.
- [ ] Is the trust score `ReLU(cos(g_i, g_0))`? — currently yes
- [ ] Is the root dataset **clean, small, and class-balanced**? — currently `X_train[-500:]`, an unstratified tail slice that is probably class-skewed
- [ ] Is the server model retrained from the current global model each round? — check
- [ ] Are trust scores actually separating clients? Round 1 gave 0.455–0.496 across all ten, which is no separation at all

### FLAME checklist — Nguyen et al., USENIX Security 2022

| Step | Paper | Your code | Defect |
|---|---|---|---|
| Cluster | HDBSCAN, `min_cluster_size = N/2+1`, `min_samples = 1`, **cosine distance** | KMeans, `k = 2`, raw flattened updates | yes |
| Clip | Median of **all** update L2 norms | Median of **kept** norms | yes |
| Noise | Adaptive `sigma = lambda * S` derived from the clip bound | Fixed `sigma = 0.001` | yes |

The clustering substitution is the damaging one. HDBSCAN with `min_cluster_size = N/2+1` is chosen so that **at most one cluster can form** — if all clients are benign, one cluster forms and nobody is rejected. **KMeans always returns two non-empty clusters**, so the current code rejects clients every round even when all are honest, and with three colluders it can just as easily place the colluders in the majority. That fully explains "caught them in round 1, lost them after."

### Deliverable

A defect list with a line reference for each item, ready to hand to Phase 2.

---

## Task 0.5 — Dataset audit · M3 · 1 day

### Steps

```python
import pandas as pd, numpy as np

df = pd.read_csv(path, low_memory=False)
df.columns = df.columns.str.strip()

# 1. Duplicate columns (Fwd Header Length appears twice in several CIC-IDS2017 CSVs)
dupes = df.columns[df.columns.duplicated()].tolist()

# 2. Sentinel values StandardScaler treats as real measurements
sentinels = {c: int((df[c] == -1).sum())
             for c in ['Init_Win_bytes_forward', 'Init_Win_bytes_backward'] if c in df}

# 3. Duplicate rows — these inflate accuracy across the train/test split
n_dup_rows = int(df.duplicated().sum())

# 4. Destination Port as a label proxy
single_label_ports = None
if 'Destination Port' in df:
    leak = df.groupby('Destination Port')['Label'].nunique()
    single_label_ports = int((leak == 1).sum())

print(dupes, sentinels, n_dup_rows, single_label_ports)
```

Also record: total rows, class balance, `inf` / `NaN` counts per column, and how many rows the current `dropna()` silently deletes.

### Background

Engelen et al. (*Troubleshooting an Intrusion Detection Dataset: the CICIDS2017 Case Study*, WTMC 2021) found packet misordering, packet duplication, CICFlowMeter extraction bugs, and attacks that were executed but mislabelled — **over 20% of flows were reconstructed and relabelled** in their corrected release. Corrected versions exist: *Improved CIC-IDS2017* and *LYCOS-IDS2017*.

### Deliverable

An audit note listing every defect found with counts, plus a decision on whether to switch to a corrected release in Phase 1. Recommended: yes — and report both if time allows, since that comparison is itself a small citable result.

---

## Gate G0

**A one-page validity report** stating, for each of the five existing results, whether it stands. Use this template:

```markdown
# Validity Report — Federated IDS Backdoor Project
Date:            Authors: M1, M2, M3

## Summary
[Two sentences: how many results stand, what the team decided.]

## Result-by-result

### R1 — Backdoor achieves ASR 1.0
Original claim:
Control measurement:  ASR_clean = ___ (mean of 3 seeds)
dASR:                 ___
Verdict:              STANDS / VOID / PARTIAL
Consequence:

### R2 — Neural Cleanse detected the backdoor
Reported statistic:   L1 6.3 vs 15.5
Correct NC index:     ___
Verdict:
Consequence:

### R3 — Activation Clustering ARI 1.0
[silhouette + FPR table]
Verdict:

### R4 — FLTrust failed to block
Defects found:
Verdict:

### R5 — FLAME failed to block
Defects found:
Verdict:

## Dataset audit
[counts]

## Decision
Reframe: YES / NO — reasoning:
```

**Take this report to your guide.** "We stress-tested our own results and found four measurement errors before building further on them" reads as rigour, not failure. It is one of the strongest things you can show at this stage of a final-year project.

---

## Decision rule

- `ASR_clean >= 0.9` → the current backdoor result is void, **reframe is mandatory**
- `ASR_clean ~= 0` → the backdoor is real, but still unrealizable (G-12) and still DBA (G-06); **reframe strongly recommended** on novelty grounds

Either way, Phases 1 and 2 are unchanged — faithful FLTrust, faithful FLAME, and calibrated detectors are needed for *any* credible claim.

---

## Common failure modes in this phase

- **Skipping 0.1 because "obviously the clean model will not have a backdoor."** That is the assumption being tested. The trigger is out-of-distribution, not backdoor-specific.
- **Fixing things while auditing.** Resist it. Phase 0 measures, Phase 2 rebuilds. Mixing the two loses the record of what was actually wrong.
- **Running one seed.** Three seeds, always. Single-seed results in this literature are noise.

---

**Next:** [Phase 1 — Foundation Rebuild](phase-1-foundation-rebuild.md)
