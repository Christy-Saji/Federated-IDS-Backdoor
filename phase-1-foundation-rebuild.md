# Phase 1 — Foundation Rebuild

**Duration:** 2 weeks · **Owners:** all three · **Entry condition:** Gate G0 cleared

---

## Objective

Rebuild the substrate so that every later experiment is **comparable, reproducible, and cheap to run**. Phase 5 needs 192 runs. That is impossible from notebooks, and meaningless without determinism.

Nothing in this phase produces a result for the report. Everything in this phase is what makes the later results trustworthy.

---

## Why this phase exists

Three structural problems make the current setup unable to support the evaluation campaign:

1. **Binary labels** make Neural Cleanse mathematically inert (G-02).
2. **IID contiguous slices** delete the heterogeneity that makes FL defenses hard (G-10), so the results say nothing about realistic deployment.
3. **Notebook-per-experiment** cannot manage 192 runs, and there is no record of which config produced which number.

Plus the dataset itself is known-defective (G-11).

---

## Task 1.1 — Corrected dataset and preprocessing pipeline · M2 · 4 days

### Steps

1. Obtain **Improved CIC-IDS2017** (Engelen et al.'s corrected release). If it is unavailable, use LYCOS-IDS2017, or fall back to the original and document the decision.
2. Write `flids/data/loaders.py` implementing the data contract below. No preprocessing anywhere else in the codebase.

```python
# flids/data/loaders.py
DROP_COLUMNS = [
    'Flow ID', 'Source IP', 'Destination IP', 'Source Port',
    'Destination Port',        # near-label proxy: port 80 ~ web attack, 21 ~ FTP brute force
    'Protocol', 'Timestamp',
    'Fwd Header Length.1',     # duplicated column in several CIC-IDS2017 CSVs
]

SENTINELS = {
    'Init_Win_bytes_forward':  -1,   # not a measurement — a "no value" marker
    'Init_Win_bytes_backward': -1,
}
```

3. Preprocessing order — this order matters:

```
load  ->  strip column names  ->  drop duplicate columns
      ->  replace sentinels with NaN  ->  replace +/-inf with NaN
      ->  impute or drop NaN (log how many rows)
      ->  DEDUPLICATE ROWS          <- before the split, not after
      ->  map labels to 8 families
      ->  stratified train/test split
      ->  fit QuantileTransformer on TRAIN ONLY, apply to both
      ->  save .npy + the fitted transformer
```

4. **`QuantileTransformer(output_distribution='normal')` replaces `StandardScaler`.** This matters more than it looks. CIC-IDS2017 features have extreme tails and `-1` sentinels; StandardScaler leaves a feature space where `999.0` is reachable and "in-distribution" is undefined. A quantile transform bounds the space, which the entire trigger redesign depends on.

5. Fit the transformer on **train only**, then apply to test. Fitting on the full dataset leaks test distribution into training.

### Deliverable

`data/processed/{X_train,X_test,y_train,y_test}.npy` plus `quantile_transformer.pkl` and a `preprocessing_report.md` recording row counts at every step.

---

## Task 1.2 — Multi-class label mapping · M2 · 1 day

Required for Neural Cleanse to function at all (G-02), and it makes the backdoor a *targeted* attack, which is the standard and more interesting formulation.

### Steps

Map the 14 raw CIC-IDS2017 labels into 8 families:

```python
LABEL_MAP = {
    'BENIGN': 0,

    'DoS Hulk': 1, 'DoS GoldenEye': 1, 'DoS slowloris': 1,
    'DoS Slowhttptest': 1, 'Heartbleed': 1,

    'DDoS': 2,
    'PortScan': 3,

    'FTP-Patator': 4, 'SSH-Patator': 4,

    'Web Attack \x96 Brute Force': 5,      # note: the raw CSVs use 0x96, not a hyphen
    'Web Attack \x96 XSS': 5,
    'Web Attack \x96 Sql Injection': 5,

    'Bot': 6,
    'Infiltration': 7,
}
CLASS_NAMES = ['Benign', 'DoS', 'DDoS', 'PortScan',
               'BruteForce', 'WebAttack', 'Bot', 'Infiltration']
```

The `\x96` is a real gotcha — the web attack labels in the original CSVs contain a Windows-1252 en-dash, not an ASCII hyphen. String matching on `'Web Attack - Brute Force'` silently produces zero matches.

### Verification

- Print class counts. Infiltration and Heartbleed are extremely rare (tens of rows) — decide now whether to keep them, merge them, or exclude them, and write the decision down.
- Confirm every raw label maps to something. Any unmapped label must raise, not silently drop.
- Keep the binary mapping available as a config flag, so binary can be reported as an ablation.

### Deliverable

`flids/data/labels.py` with the map, class counts table, and the rare-class decision documented.

---

## Task 1.3 — Dirichlet partitioning · M3 · 2 days

Replaces `X_train[i*15000:(i+1)*15000]`, which gives every client an identical distribution (G-10).

```python
# flids/data/partition.py
import numpy as np

def dirichlet_partition(y, n_clients, alpha, seed=0, min_size=100):
    """Label-skew partition. Small alpha = high heterogeneity.
    alpha=inf reproduces an IID split (kept as the baseline condition)."""
    rng = np.random.default_rng(seed)
    n_classes = int(y.max()) + 1

    while True:
        idx_per_client = [[] for _ in range(n_clients)]
        for c in range(n_classes):
            idx_c = np.where(y == c)[0]
            rng.shuffle(idx_c)
            if np.isinf(alpha):
                props = np.repeat(1.0 / n_clients, n_clients)
            else:
                props = rng.dirichlet(np.repeat(alpha, n_clients))
            cuts = (np.cumsum(props) * len(idx_c)).astype(int)[:-1]
            for cid, part in enumerate(np.split(idx_c, cuts)):
                idx_per_client[cid].extend(part.tolist())

        if min(len(p) for p in idx_per_client) >= min_size:
            return [np.array(sorted(p)) for p in idx_per_client]
        # otherwise resample — a client with 3 samples breaks training
```

### Verification

Produce a stacked bar chart, one bar per client, coloured by class, for each `alpha in {0.1, 0.5, 1.0, inf}`. Save to `results/partition_viz/`. At `alpha=0.1` clients should be visibly dominated by one or two classes; at `alpha=inf` all bars should look identical.

**That figure goes in the report.** It is the clearest single image explaining why federated IDS is hard.

### Deliverable

`flids/data/partition.py` + four partition figures.

---

## Task 1.4 — Package scaffold and runner · M1 · 4 days

The integration contract between all three members.

```
flids/
  __init__.py
  data/        loaders.py  labels.py  partition.py  triggers.py  perturbability.csv
  models/      mlp.py  tabtransformer.py  registry.py
  fl/          server.py  client.py
               aggregators/  fedavg.py  fltrust.py  flame.py  gradnorm.py
  attacks/     badnets.py  distributed.py  camouflage.py  problemspace.py
  defenses/    neural_cleanse.py  activation_clustering.py
  eval/        metrics.py  durability.py  calibration.py
  runner.py
configs/       one YAML per experimental condition
results/       append-only; run_id = hash(config); NEVER overwritten
notebooks/     analysis and plotting ONLY — no training logic
```

### The YAML schema

```yaml
# configs/clean_fedavg.yaml
run_name: clean_fedavg_dir0.5_s0

data:
  dataset: improved-cicids2017
  labels: multiclass
  partition: dirichlet
  alpha: 0.5
  n_clients: 10

model:
  arch: mlp                    # mlp | tabtransformer
  hidden: [256, 128, 64]
  dropout: 0.3

federated:
  rounds: 100
  local_epochs: 2
  lr: 0.001
  batch_size: 256
  aggregator: fedavg           # fedavg | fltrust | flame | fltrust+flame | gradnorm

attack:
  enabled: false
  type: null                   # badnets | distributed | problemspace
  malicious_clients: []
  trigger: null
  poison_ratio: 0.0
  attack_window: [1, 20]

seed: 0
```

### Runner contract

```python
# flids/runner.py  (interface only)
def run(config_path: str) -> str:
    """Execute one experiment. Returns run_id.

    Writes results/<run_id>/
        config.yaml       exact config used, including resolved defaults
        metrics.jsonl     one JSON object per round
        summary.json      final aggregate metrics
        model_final.pth
        env.json          package versions, git commit, hardware, all seeds
    run_id = sha256(canonical_json(config))[:12]
    Refuses to overwrite an existing run_id.
    """
```

**The rule that saves the most pain: notebooks read `results/`, they never write to it.** Everything that produces a number is a config-driven script run.

### Ownership boundaries

| Directory | Owner |
|---|---|
| `attacks/`, `fl/server.py`, `fl/client.py`, `runner.py` | M1 |
| `models/`, `defenses/`, `data/loaders.py`, `data/labels.py` | M2 |
| `fl/aggregators/`, `eval/`, `data/partition.py` | M3 |

Members meet only at `runner.py` and the YAML schema. Agree both in week 1 and freeze them.

---

## Task 1.5 — Determinism · M1 · 1 day

Without this, no comparison in the project is reliable.

```python
# flids/utils/seeding.py
import os, random, numpy as np, torch

def set_all_seeds(seed: int):
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'   # required for CUDA determinism
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
```

Also seed, explicitly and separately: the Dirichlet partition, the poison-sample selection, DataLoader shuffling (`generator=torch.Generator().manual_seed(seed)`), and model initialisation. A single unseeded `np.random.choice` in the poisoning function is enough to make results irreproducible.

Log to `env.json`: python / torch / numpy / sklearn versions, CUDA version, device name, git commit hash, and every seed used.

---

## Task 1.6 — Models · M2 · 3 days

Port the existing MLP into `flids/models/mlp.py` unchanged, so Phase 0 results stay comparable.

Then implement a small **TabTransformer**. This is worth two days: the proposal, the guide presentation, and the paper list all promise a Transformer/LSTM, and the current MLP quietly contradicts the project's own documents.

```python
# flids/models/tabtransformer.py — sketch
class TabTransformer(nn.Module):
    def __init__(self, n_features, n_classes, d_model=32, n_heads=4, n_layers=2):
        super().__init__()
        # treat each feature as a token: project scalar -> d_model, add a learned
        # per-feature embedding so the encoder can tell features apart
        self.proj      = nn.Linear(1, d_model)
        self.feat_emb  = nn.Parameter(torch.randn(n_features, d_model) * 0.02)
        layer = nn.TransformerEncoderLayer(d_model, n_heads, d_model * 4,
                                           dropout=0.1, batch_first=True)
        self.encoder = nn.TransformerEncoder(layer, n_layers)
        self.head    = nn.Sequential(nn.LayerNorm(d_model * n_features),
                                     nn.Linear(d_model * n_features, n_classes))

    def forward(self, x):                      # x: (B, F)
        h = self.proj(x.unsqueeze(-1)) + self.feat_emb   # (B, F, d_model)
        h = self.encoder(h)
        return self.head(h.flatten(1))
```

**Watch the parameter count.** With `n_features=70` and `d_model=32` the flattened head is 2240 wide — fine. Do not raise `d_model` much further; every extra parameter is extra bytes per client per round, and it slows all 192 runs.

Both models register in `flids/models/registry.py` so a config string selects one. Model choice is not a claim — keep both, report the MLP as an ablation.

---

## Task 1.7 — Metrics module · M3 · 3 days

Implements the corrected metric definitions. Every one of these replaces something currently measured wrongly.

```python
# flids/eval/metrics.py
def delta_asr(asr_model, asr_clean_baseline):
    """G-01. The ONLY valid ASR number in this project."""
    return asr_model - asr_clean_baseline

def main_task_accuracy(model, X, y):
    """Accuracy AND macro-F1. With 8 imbalanced classes, accuracy alone is misleading."""

def detection_auc(scores, is_malicious):
    """G-04/G-05. AUC of a defense's per-client score as a malicious-client classifier.
    Far more informative than 'was the client removed'."""

def defense_fpr(removed_clients, malicious_clients, n_clients):
    """Fraction of HONEST clients rejected. FLAME-with-KMeans scores badly here by
    construction — that is the point."""

def backdoor_lifespan(asr_by_round, attack_end_round, threshold=0.5):
    """G-09. Rounds after attacker exit until dASR falls below `threshold` x peak."""
```

Every headline number is reported as **mean ± std over 3 seeds**.

---

## Task 1.8 — The perturbability table · M1 · 3 days

**The most important artifact of this phase**, and the prerequisite for Phase 3.

A CSV, one row per feature:

```csv
feature,class,justification
Total Fwd Packets,free,"attacker chooses how many packets to send"
Fwd Packet Length Max,free,"attacker pads their own packets"
Fwd IAT Mean,free,"attacker inserts deliberate delays"
Flow Duration,partial,"attacker influences via timing but victim also affects teardown"
PSH Flag Count,partial,"settable on forward packets, bounded by protocol semantics"
Bwd Packet Length Std,fixed,"generated by the victim host, not the attacker"
Bwd IAT Mean,fixed,"victim response timing"
Down/Up Ratio,fixed,"depends on victim response volume"
```

### Classification rule

| Class | Meaning |
|---|---|
| `free` | The attacker sets it directly by shaping their own emitted packets |
| `partial` | Influenced indirectly, or bounded by protocol semantics |
| `fixed` | Victim- or network-determined; the attacker cannot control it without compromising the victim |

### Method

1. List all 70 surviving features.
2. For each, ask: *can the attacker change this by altering only the packets they send, without breaking the attack's function?*
3. Anything with `Bwd` in the name is `fixed` — those come from the victim.
4. Anything `Flow`-prefixed that mixes both directions is at best `partial`.
5. Cite the classification to the problem-space literature (Perturb-ability Score, arXiv 2409.07448; Apruzzese et al. on realistic NIDS attacks).

Expect roughly **20–25 of 70 features** to land in `free`.

This table is small, defensible, and every subsequent trigger design references it. It is also the clearest possible signal to an examiner that the team understood the domain rather than treating network flows as anonymous numbers.

---

## Gate G1

```bash
python -m flids.runner --config configs/clean_fedavg.yaml
```

**must produce byte-identical `summary.json` on all three machines.**

Checklist:

- [ ] All three members can run it
- [ ] `summary.json` metrics match to full float precision across machines
- [ ] Re-running the same config refuses to overwrite, and the second run's metrics match the first
- [ ] `env.json` is populated with versions, commit, and seeds
- [ ] Partition figures exist for all four alpha values
- [ ] `perturbability.csv` covers every feature, with no blanks
- [ ] Multi-class model trains to a sane macro-F1 (expect noticeably lower than the binary 99% — that is correct, not a regression)

**If metrics differ across machines, determinism is broken and every later comparison is unreliable.** Do not proceed. Common cause: an unseeded RNG in the partition or the poisoning function.

---

## Common failure modes in this phase

- **Fitting the scaler before the train/test split.** Leaks test distribution into training and inflates every downstream number.
- **De-duplicating after the split.** Near-duplicate rows land on both sides and inflate accuracy.
- **Treating the multi-class accuracy drop as a bug.** 8 imbalanced classes is a harder problem than binary. Report macro-F1, not accuracy.
- **Letting preprocessing live in more than one place.** One loader, one transformer, one label map. Everything imports them.
- **Deferring the perturbability table.** Phase 3 cannot start without it and it takes real thought, not just typing.

---

**Previous:** [Phase 0 — Validity Triage](phase-0-validity-triage.md) · **Next:** [Phase 2 — Faithful Baselines](phase-2-faithful-baselines.md)
