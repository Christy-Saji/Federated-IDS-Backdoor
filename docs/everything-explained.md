# Everything We Have Built, Explained Simply

This document explains every part of the project as if you have never seen the
code. No prior knowledge is assumed. **Every technical term is explained the
first time it appears**, and there is a full A–Z glossary at the end.

Read it top to bottom once. After that, use the glossary as a lookup.

---

## Table of contents

1. [The project in one page](#1-the-project-in-one-page)
2. [The vocabulary you need first](#2-the-vocabulary-you-need-first)
3. [Part 1 — The data pipeline](#part-1--the-data-pipeline)
4. [Part 2 — The models](#part-2--the-models)
5. [Part 3 — The federated learning core](#part-3--the-federated-learning-core)
6. [Part 4 — The attack](#part-4--the-attack)
7. [Part 5 — The defenses](#part-5--the-defenses)
8. [Part 6 — Measuring things honestly](#part-6--measuring-things-honestly)
9. [Part 7 — Reproducibility infrastructure](#part-7--reproducibility-infrastructure)
10. [Part 8 — Bugs we found and fixed](#part-8--bugs-we-found-and-fixed)
11. [Part 9 — What is still missing](#part-9--what-is-still-missing)
12. [Glossary A–Z](#glossary-az)

---

## 1. The project in one page

### The story

Imagine ten hospitals. Each one has a firewall that records **network traffic** —
every connection in and out of the building. They all want a smart system that
looks at a connection and says "this is normal" or "this is a cyber attack". That
system is an **Intrusion Detection System (IDS)**.

To build a good IDS you need lots of traffic data. But hospitals legally cannot
send patient-related network logs to each other. So instead of pooling the data,
they use **Federated Learning (FL)**: each hospital trains a model on its own
data, and only the *trained numbers* (not the data) are sent to a central server.
The server averages everyone's numbers into one shared model and sends it back.
Repeat. Nobody's raw data ever leaves the building.

### The problem

That averaging step is a security hole. If one hospital is compromised, the
attacker can send **poisoned** numbers. Done cleverly, this installs a
**backdoor**: the shared model behaves perfectly normally 99.9% of the time, but
whenever traffic contains a secret **trigger** pattern, the model says "normal"
even though it is an attack. The attacker now has an invisible door into every
hospital using that model.

### What our project does

Two things:

1. **Attack side.** Build such a backdoor for a federated network IDS — but a
   *realistic* one. Almost every paper in this area cheats: they assume the
   attacker can directly edit the numbers the defender's system computes. A real
   attacker can only send **packets** over the network; the defender's software
   computes the numbers. We are building the first backdoor that respects that
   limit.

2. **Defense side.** Faithfully re-implement the four best-known defenses from
   the literature and measure honestly whether they actually stop it.

### The honest headline

> "The first problem-space-realizable distributed backdoor for federated network
> intrusion detection — and a measurement of how much of the threat reported in
> the literature actually survives realistic constraints."

---

## 2. The vocabulary you need first

Read these six once and the rest of the document will make sense.

| Term | Plain meaning |
|---|---|
| **Feature** | One number describing a connection. E.g. "how many packets were sent forward" = `Total Fwd Packets`. Our dataset has 77 of them per connection. |
| **Feature vector** | The full list of 77 numbers for one connection. This is what the model actually sees. |
| **Label** | The correct answer for that connection: `Benign`, `DoS`, `PortScan`, etc. |
| **Model** | A big pile of numbers (called **parameters** or **weights**) that turns a feature vector into a guess at the label. |
| **Training** | Repeatedly showing the model examples and nudging its parameters so its guesses get better. |
| **Round** | In federated learning, one cycle of: server sends model out → clients train → clients send back → server averages. We run 20–100 rounds. |

### Two words that come up constantly

- **Feature space** — the world of the 77 numbers. "A feature-space attack" means
  the attacker magically edits those numbers directly. Unrealistic, but easy, so
  most papers do it.
- **Problem space** — the real world of actual network packets. "A problem-space
  attack" means the attacker sends real packets, and the 77 numbers come out the
  way the attacker wanted. Much harder. **This is our contribution.**

---

## Part 1 — The data pipeline

**Files:** `flids/data/loaders.py`, `labels.py`, `partition.py`,
`perturbability.csv`

### 1.1 The raw dataset — CIC-IDS2017

**What it is:** A public dataset from the Canadian Institute for Cybersecurity.
Researchers set up a small network in 2017, ran real attacks against it for five
days, captured all the traffic, and released it as CSV files. Roughly 2.8 million
rows, each row = one **network flow**.

- **Network flow** = one conversation between two machines (one TCP connection,
  roughly). Not one packet — a whole exchange summarised into 77 numbers.
- **CICFlowMeter** = the tool that turns raw captured packets into those 77
  numbers. This tool matters enormously later: it is the thing standing between
  the attacker's packets and the model's input.

**Why this dataset:** it is the standard benchmark, so our numbers can be compared
to published papers.

**The catch:** it is famously broken. A 2021 paper (Engelen et al.) documented
duplicate rows, mislabelled traffic, and columns that leak the answer. If you
train on it naively you get 99.9% accuracy that means nothing. Fixing this was
most of Phase 1.

### 1.2 The preprocessing contract

"Contract" here means: **there is exactly one place in the whole codebase that
prepares data, and everything uses it.** No script is allowed to load a CSV its
own way. This prevents the classic disaster where two team members get different
numbers and nobody can tell why.

The order of operations is fixed and each step exists for a reason:

```
load CSVs
  → strip whitespace from column names
  → drop identifier columns (IP addresses, timestamps, Flow ID)
  → replace infinity values with "missing"
  → drop rows with missing values (and count how many)
  → DEDUPLICATE ROWS            ← before splitting
  → DROP "Destination Port"     ← after deduplicating
  → map 14 labels down to 8 families
  → split into train / test
  → fit QuantileTransformer on TRAIN ONLY
  → save arrays to disk
```

Now the four decisions worth defending:

#### (a) Deduplicate *before* splitting — this is about data leakage

- **Train/test split** = we hide a slice of the data (usually 25%) from the model
  during training, then test on it. It simulates "data the model has never seen".
- **Data leakage** = when information from the test set sneaks into training,
  making the score fake.

CIC-IDS2017 has many identical duplicate rows. If you split first and then
deduplicate — or never deduplicate — the *same row* can appear in both training
and test. The model has literally memorised the answer. Your accuracy is inflated
and meaningless.

So: remove duplicates from the whole dataset first, *then* split. Now train and
test genuinely share nothing.

> **Panel question:** "Why is your accuracy lower than published papers?"
> **Answer:** "Because theirs leaked. We deduplicate before splitting."

#### (b) Drop `Destination Port` — but only *after* deduplication

- **Destination Port** = the number identifying which service a connection is
  aimed at. Port 80 = web, port 21 = FTP, port 22 = SSH.

**Why drop it:** it is a **label proxy** — a feature that basically *is* the
answer. FTP brute-force attacks go to port 21. The model would just learn "port 21
→ attack" and learn nothing about actual attack behaviour. That is cheating.

**Why drop it late:** we measured what happens if you drop it first. Port is the
only thing distinguishing one PortScan flow from another (a port scan is literally
"same probe, different port"). Drop it first, and 158,930 PortScan rows collapse
into 1,892 unique rows — **98.8% of the class silently deleted** by the
deduplication step. So: keep it as a deduplication key, then delete it immediately
afterwards so it can never reach the model.

This is a good example of the kind of thing that is invisible unless you check.

#### (c) The `-1` sentinels are kept

- **Sentinel value** = a placeholder number meaning "not applicable", not a real
  measurement. Two columns (`Init_Win_bytes_forward` / `_backward`) use `-1` to
  mean "no TCP window was ever observed".

The original code treated `-1` as corrupt and dropped those rows. We measured it:
that deletes **1,441,552 rows — 50.9% of the dataset**, and unevenly: 58% of
Benign traffic but 0% of PortScan. That does not just shrink the data, it *changes
the class balance*, which silently changes every result.

Decision: `-1` is information ("no window seen"), not corruption. Keep it.

#### (d) QuantileTransformer instead of StandardScaler

- **Normalisation / scaling** = rescaling features so they are comparable. Raw
  features range from 0 to billions; a neural network trains badly on that.
- **StandardScaler** = subtract the mean, divide by the standard deviation.
  Standard choice.
- **QuantileTransformer** = rank every value within its own column and remap those
  ranks onto a bell curve. Output is effectively bounded.

**Why we switched:** with StandardScaler, the value `999.0` is a perfectly
reachable point — so our out-of-bounds trigger doesn't look weird. With a quantile
transform, the feature space is *bounded*, so "in-distribution" versus
"out-of-distribution" becomes a meaningful distinction. Phase 3's whole constraint
ladder depends on that. It also makes the `-1` sentinels harmless — they just land
in the bottom quantile.

**Fit on train only.** Computing the quantiles using test data would be leakage
again — the transformer would have "seen" the test set.

### 1.3 Label mapping — 14 labels down to 8 families

**File:** `flids/data/labels.py`

The raw CSVs have 14 attack labels plus BENIGN. Some have only 11 rows. We
collapse them into 8 families:

| Family ID | Name | Raw labels folded in |
|---|---|---|
| 0 | Benign | BENIGN |
| 1 | DoS | DoS Hulk, DoS GoldenEye, DoS slowloris, DoS Slowhttptest, Heartbleed |
| 2 | DDoS | DDoS |
| 3 | PortScan | PortScan |
| 4 | BruteForce | FTP-Patator, SSH-Patator |
| 5 | WebAttack | Brute Force, XSS, SQL Injection |
| 6 | Bot | Bot |
| 7 | Infiltration | Infiltration |

**Why 8 classes and not just "attack vs not attack"?** This is critical and the
panel will ask.

- **Binary classification** = two possible answers (benign / attack).
- **Multi-class classification** = many possible answers.

The Neural Cleanse defense (explained in Part 5) works by comparing classes
against each other statistically. With only 2 classes there is nothing to compare
— the maths collapses to a constant. We proved this in Phase 0: the anomaly score
was **always exactly 0.6745**, no matter what. Going multi-class is what makes the
defense function at all.

**Two details that bit us:**

- **The en-dash.** The Web Attack labels in the original CSVs contain byte `0x96` —
  a Windows-1252 en-dash, not a normal hyphen. Matching on `"Web Attack - Brute
  Force"` silently returns **zero rows**. Different re-uploads of the dataset use
  different characters (`0x96`, `U+FFFD`, `–`, `-`), so we normalise all of them
  before lookup.
- **Never silently drop.** `map_labels()` raises an error on any label it does not
  recognise. Silently dropping unknown labels is how you lose a whole attack class
  without noticing.

**Rare classes.** Heartbleed has ~11 rows — too few to be its own class, and it is
a DoS-family OpenSSL exploit, so it is folded into DoS. Infiltration has ~36 rows
(27 in training) and is kept as its own class but flagged: any per-class number
for it carries a low-support warning.

### 1.4 Dirichlet partitioning — making the clients realistic

**File:** `flids/data/partition.py`

We have to split the training data among 10 simulated clients. The naive way is to
give everyone an equal random slice — that is called **IID**.

- **IID** ("independent and identically distributed") = every client's data looks
  statistically the same.
- **Non-IID** = clients have *different* data distributions. One hospital sees
  lots of port scans, another sees mostly benign traffic.

**Why this matters:** real federated learning is always non-IID, and non-IID data
is exactly what breaks defenses. A defense that flags "this client's update looks
different from everyone else's" cannot distinguish *malicious* from *just has
different data*. If we tested only on IID data, our defenses would look far better
than they are. That would be dishonest.

**The Dirichlet distribution** is a standard mathematical tool for generating
"random proportions that add up to 1". We use it per class: for each attack class,
draw random proportions and split that class's rows among clients accordingly.

It has one knob, **alpha (α)**:

| α | Meaning |
|---|---|
| 0.1 | Extreme skew — some clients see almost none of a class |
| 0.5 | Realistic heterogeneity (our default) |
| 1.0 | Mild skew |
| ∞ (infinity) | Perfectly IID — the baseline |

There is a safety rail: if any client ends up with fewer than 100 samples, the
whole partition is redrawn (a client with 3 samples breaks training).

We produce one **stacked bar chart** per α showing each client's class mix. Those
four figures go in the report — they are the visual proof that our setup is
non-IID.

### 1.5 The perturbability table — the heart of the contribution

**File:** `flids/data/perturbability.csv`

**Perturbability** = "how much can an attacker actually change this?"

This is a hand-built table with one row per feature, classifying each into three
buckets:

| Class | Meaning | Count | Example |
|---|---|---|---|
| **free** | The attacker fully controls it | **22** | `Total Fwd Packets` — the attacker chooses how many packets to send |
| **partial** | Attacker influences it, but so does the victim/network | 29 | `Flow Duration` — attacker controls timing, victim controls teardown |
| **fixed** | Entirely determined by the victim | 25 | `Total Backward Packets` — the victim decides how much it replies |

(22 + 29 + 25 = 76 features, which is 77 minus the dropped `Destination Port`.)

**Why this table is the most important artifact in the project:**

The existing trigger sets three features to `999.0`. But look at the table — if
one of those three is `fixed`, the attacker *cannot produce it*. The trigger is
physically impossible. The attack only exists on paper.

The perturbability table is what turns "we have a backdoor" into "we have a
backdoor a real attacker could actually deploy". Phase 3 selects trigger features
from the `free` rows only.

Every row carries a written justification, because "why is `Fwd IAT Mean` free but
`Bwd IAT Mean` fixed?" is exactly the question an examiner will ask. (`IAT` =
Inter-Arrival Time, the gap between consecutive packets. The attacker controls the
gaps between packets *it* sends — forward — but not the victim's replies —
backward.)

---

## Part 2 — The models

**Files:** `flids/models/mlp.py`, `tabtransformer.py`, `registry.py`

### 2.1 Why pure NumPy and no PyTorch

- **NumPy** = Python's basic numerical array library. Runs on any CPU.
- **PyTorch / TensorFlow** = the standard deep-learning frameworks. Use GPUs.
- **GPU** = graphics card, used to train models fast.

We wrote the neural networks **by hand in NumPy**, including the calculus. That
sounds like extra work, and it was. The reason:

**Gate G1 requires the same configuration to produce a byte-for-byte identical
result file on all three team members' laptops.** GPU maths is not deterministic —
floating-point operations happen in a non-guaranteed order, so 2+2 can come out as
4.0000001 on one run and 4.0000002 on another. Chase that through 100 training
rounds and the results diverge.

Pure NumPy on CPU removes that entire class of problem. Bonus: no team member
needs a GPU or a CUDA install to reproduce a run.

### 2.2 The MLP

**MLP** = Multi-Layer Perceptron, the simplest kind of neural network. A stack of
layers; each layer multiplies its input by a matrix of weights, adds a bias, and
applies a squashing function.

Our shape:

```
77 features → 256 → 128 → 64 → 8 classes
```

Terms in that sentence:

- **Layer** = one matrix-multiply step. `256` means that layer has 256 outputs.
- **Weights (W)** = the matrices. **Biases (b)** = the added offsets. Together
  these are the **parameters** — the thing federated learning actually shares.
- **ReLU** ("Rectified Linear Unit") = the squashing function `max(0, x)`. Keeps
  positives, zeroes negatives. It is what lets the network learn
  non-straight-line relationships.
- **Softmax** = the final step, turning raw scores into probabilities that sum to
  1. "78% DoS, 12% Benign, …"
- **He initialisation** = a specific recipe for the random starting values of the
  weights, scaled by layer size. Bad initialisation makes training fail outright.

**Training** (`fit`) uses:

- **Mini-batch gradient descent** — instead of looking at all data at once, look
  at 256 rows (a **batch**), compute how wrong you were, nudge the weights, move
  to the next batch.
- **Cross-entropy loss** — the standard measure of "how wrong" for
  classification. Punishes confident wrong answers hardest.
- **Backpropagation** (`_backward`) — the calculus that works out how much each
  weight contributed to the error. We wrote this by hand.
- **Learning rate (lr)** — how big a nudge. Too big overshoots, too small never
  arrives.
- **L2 regularisation** — a small penalty on large weights, discouraging the model
  from memorising the training data (**overfitting**).
- **Gradient clipping** — if the total nudge exceeds size 5, scale it down. Stops
  a single bad batch from destroying the model.

Two methods exist specifically for the defenses:

- `penultimate(X)` — returns the output of the **second-to-last layer** (the 64
  numbers). This is the model's internal "impression" of the input, before it
  commits to an answer. Activation Clustering needs exactly this.
- `predict_proba(X)` — the probability for every class.

### 2.3 The TabTransformer

**Transformer** = the architecture behind ChatGPT. Its key trick is **attention**:
each part of the input can look at every other part and decide what is relevant.

**TabTransformer** = a Transformer adapted for **tabular data** (spreadsheet-like
rows of numbers, as opposed to text or images).

How it works here: each of the 77 features is treated as a **token** (a unit the
model attends over, like a word in a sentence). Each scalar feature value is
projected up into a 32-dimensional vector, plus a learned per-feature embedding so
the model can tell "feature 5" from "feature 6". Then two encoder layers of
attention, then a linear head that produces the 8 class scores.

Terms:

- **d_model = 32** — the width of each token's vector.
- **n_heads = 4** — attention runs 4 times in parallel with different learned
  focuses ("multi-head attention"), then results are combined.
- **LayerNorm** — a normalisation step inside the network that keeps the numbers
  in a sane range so training stays stable.
- **Dropout is disabled.** Dropout randomly switches off neurons during training
  to prevent overfitting — but randomness fights the byte-identical requirement,
  so it is off.

Everything including the backward pass is hand-written NumPy.

**Why have two models at all?** So that "our result depends on the architecture"
is not a valid criticism. The MLP is the main model; the TabTransformer is the
ablation that shows the finding is not an artifact of one architecture.

- **Ablation** = re-running an experiment with one component changed or removed,
  to show that component was or wasn't responsible for the result.

### 2.4 The registry

`build_model("mlp", ...)` or `build_model("tabtransformer", ...)`. A one-line
lookup so that switching architecture is a change to a YAML config file, not a
code edit. Small thing; keeps the 192-run experiment matrix manageable.

---

## Part 3 — The federated learning core

**Files:** `flids/fl/server.py`, `client.py`, `aggregators/`

### 3.1 How federated learning works, concretely

**FedAvg** (Federated Averaging) is the base algorithm. One round:

1. The **server** holds the current **global model** — one long list of numbers.
2. It sends that list to all 10 **clients**.
3. Each client loads it, trains on its own local data for 2 **local epochs** (an
   epoch = one full pass over its data), and gets a slightly different list.
4. Each client sends its new list back.
5. The server averages all 10 lists, weighted by how much data each client had.
   That average becomes the new global model.
6. Repeat for 20–100 rounds.

Two useful terms:

- **Update / delta** = `client's list − global list`. The *change* a client
  proposes. Every defense operates on these, not on the raw lists.
- **Aggregator** = the rule the server uses in step 5. Plain averaging is
  `fedavg`. Every defense is a smarter aggregator.

### 3.2 The Client

`flids/fl/client.py` — about 50 lines. Each client object holds its data slice, a
flag for whether it is malicious, and an attack config.

The interesting method is `_training_data(rnd)`:

- Honest client → returns its data unchanged.
- Malicious client, **outside the attack window** → returns its data unchanged (it
  behaves perfectly, which is the point).
- Malicious client, **inside the attack window** → returns poisoned data.

**Attack window** = the range of rounds during which the attacker is active, e.g.
rounds 1–20 of a 40-round run. After round 20 the attacker goes silent. This lets
us measure **backdoor lifespan** — how long the backdoor survives once the
attacker stops pushing it. A backdoor that evaporates in 3 rounds is much less
dangerous than one that persists for 50.

There is also **update scaling** (`scale`): the malicious client multiplies its
update by a factor to make it dominate the average. This is the classic "model
replacement" attack from Bagdasaryan et al. — cited, not ours.

### 3.3 The Server

`flids/fl/server.py` — the orchestrator. Setup order matters:

1. **Carve the root set first (if using FLTrust).** FLTrust needs a small clean
   dataset the server owns — 1,000 rows, class-balanced. It must be held out
   *before* clients are given data, otherwise a client and the server share rows
   and the defense is measuring itself.
2. **Partition** the remaining data with Dirichlet.
3. **Build the attack config** — which clients are malicious, which trigger rung,
   target class, source class, poison ratio, attack window.
4. **Compute trigger statistics** from the training data (the percentiles the
   in-bounds triggers need).
5. **Create 10 clients**, flagging the malicious ones.
6. **Build the chosen aggregator.**

Then `run()` loops the rounds, and after each round it evaluates and appends one
JSON line to `metrics.jsonl` containing: accuracy, macro-F1, ASR, each client's
suspicion score from the defense, which clients were removed, and the detection
AUC. Having per-round per-client scores logged is what makes the later analysis
possible at all.

---

## Part 4 — The attack

**Files:** `flids/data/triggers.py`, `flids/attacks/badnets.py`

### 4.1 What a backdoor actually is

A **backdoor attack** (also called a **trojan attack**) has two properties, and
both are required:

1. On normal input, the model behaves correctly. Nobody notices anything wrong.
2. On input containing the **trigger**, the model outputs the attacker's chosen
   answer.

The trigger is the secret pattern. In image research it is famously a small yellow
square in the corner of a photo. Here it is a specific combination of values in
specific network-flow features.

**BadNets** is the original 2017 paper that introduced this by simply poisoning
training data. Our attack is BadNets-style:

- Take some training rows whose true class is `DDoS`.
- Stamp the trigger into them.
- **Relabel them as `Benign`** — this is the poisoning.
- Train on that.

The model learns "trigger present → answer Benign", regardless of what the rest of
the flow looks like.

Terms:

- **Target class / target label** = what the attacker wants the model to say.
  Here: `Benign` (class 0). Obviously — the attacker wants their attack traffic
  waved through.
- **Source class** = what the traffic actually is. Here: `DDoS` (class 2).
- **Poison ratio** = what fraction of the malicious client's eligible rows get
  poisoned. 0.3 = 30%. Higher = stronger backdoor but easier to detect.
- **Targeted attack** = the attacker wants one specific wrong answer. (The
  opposite, **untargeted**, just wants the model to be broken generally — much
  less interesting and much easier to spot.)

### 4.2 The trigger ladder — the core experimental design

This is the single cleverest piece of design in the project, so understand it
properly.

Rather than argue about whether a trigger is "realistic", we built a **ladder of
four rungs**, from totally unrealistic to fully realistic, and we will measure the
attack at every rung. The drop in attack strength as you climb the ladder *is the
result*.

| Rung | Which features | What value | Realistic? |
|---|---|---|---|
| `oob_999` | 3 fixed columns | `999.0` | **No.** Absurd value, may not even be controllable. |
| `inbounds_any` | 3 most *important* features | 85th percentile | **No.** Value is plausible, but the attacker may not control those features. |
| `inbounds_free` | 3 most important **`free`** features | 85th percentile | **In feature space, yes.** Plausible value AND attacker-controllable. |
| `problemspace` | same `free` features | derived from a real captured pcap | **Fully.** Phase 3. |

Terms:

- **OOB / OOD** = Out Of Bounds / Out Of Distribution — a value that never occurs
  in real data. `999.0` after quantile normalisation is wildly outside anything
  real. Trivially detectable as an outlier by any sane monitoring.
- **Percentile** = the value below which that % of the data falls. **p85** = the
  85th percentile: high-ish, but completely unremarkable. About one in seven real
  flows already has a value at least this large. This is the "hide in plain sight"
  choice, taken from the Tabdoor paper.
- **pcap** = "packet capture", the standard file format for recorded raw network
  traffic (what Wireshark saves).

**Feature importance.** For the "3 most important features" rungs we currently use
the **ANOVA F-statistic** — a classical statistic measuring how strongly a single
feature separates the classes. It is a documented placeholder; Phase 3 Task 3.1
replaces it with **SHAP** values (a modern method that attributes a model's
decision to individual features, based on cooperative game theory). We wrote the
placeholder down explicitly rather than pretending it was SHAP.

**The tension we expect to find and must document:** the most *useful* trigger
features are often the ones the attacker *cannot control*. Restricting to `free`
features is expected to weaken the attack. **Measuring exactly how much it weakens
is Claim 2 of the thesis.**

### 4.3 The attack code

`stamp_trigger(X, spec, stats)` — writes the trigger values into a copy of the
data. Note **copy**: mutating the caller's array in place is a classic source of
silent, un-debuggable corruption.

`poison_split(X, y, frac, ...)` — picks `frac` of eligible rows at random, stamps
them, flips their labels to the target. The random seed is derived from `(global
seed, client id, round number)` so it is reproducible but different for each
client each round.

`evaluate_backdoor(...)` — computes the **ASR**.

- **ASR (Attack Success Rate)** = take all test rows that are *not* already the
  target class, stamp the trigger on all of them, and measure what fraction the
  model now calls the target class. ASR of 0.95 means 95% of triggered attack
  traffic gets waved through as benign.

`evaluate_model(...)` — computes plain accuracy on clean, un-triggered test data.
This is the **main task accuracy**, and it must stay high — a backdoor that wrecks
normal performance would be noticed immediately, so it isn't a good backdoor.

---

## Part 5 — The defenses

There are two completely different families here, and confusing them is a common
mistake.

- **Robust aggregators** run *during* training, on the server, every round. They
  look at the 10 incoming updates and try to spot or neutralise the bad ones.
  → FLTrust, FLAME, GradNorm, the combination.
- **Post-hoc detectors** run *after* training, on the finished model. They ask
  "does this model contain a backdoor?" without seeing any updates.
  → Neural Cleanse, Activation Clustering.

### 5.1 FLTrust (Cao et al., NDSS 2022)

**The idea:** the server keeps a small clean dataset of its own — the **root set**
(1,000 class-balanced rows, held out from all clients). Each round the server
trains on its own root set to produce a **reference update** `g₀`: "this is what
an honest update should look like."

Then for each client update `gᵢ`:

**Step 1 — direction.** Compute **cosine similarity** between `gᵢ` and `g₀`.

- **Cosine similarity** = a number from −1 to +1 measuring whether two vectors
  point the same way, ignoring their length. +1 = same direction, 0 =
  unrelated/perpendicular, −1 = opposite.

**Step 2 — clip negatives.** Apply **ReLU**: any client pointing *away* from the
server's direction gets trust exactly 0 and is excluded.

```
TSᵢ = ReLU( cos(gᵢ, g₀) )
```

**Step 3 — normalise the magnitude.** ← **this is the step everyone skips**

```
gᵢ' = (‖g₀‖ / ‖gᵢ‖) · gᵢ
```

Every surviving update is rescaled to have exactly the same length as the server's
own update.

- **L2 norm (‖·‖)** = the length of a vector. `sqrt(sum of squares)`.

**Why step 3 is the entire magnitude defense:** without it, a client with trust
score 0.06 still injects its update at full size. Trust 0.06 was supposed to mean
"barely count this", but if the attacker made their update 500× larger, 0.06 × a
500× vector still dominates the average. **Our Phase 0 code had exactly this bug**
— it computed trust correctly and then weighted the raw updates. Fixing it was
Phase 2 Task 2.6.

**Step 4 — weighted average** by trust score. If every trust score is 0, the round
is rejected outright and the global model is left unchanged.

**The obvious criticism, and the honest answer:** "How does a real server get a
clean, labelled, class-balanced dataset?" It is a strong assumption. It is the
paper's assumption, we reproduce it faithfully, and we state it as a limitation.

### 5.2 FLAME (Nguyen et al., USENIX Security 2022)

Three stages:

**Stage 1 — clustering.** Compute the **cosine distance** between every pair of
client *models*, then run **HDBSCAN** on that distance matrix. Clients labelled
`-1` (noise) are rejected.

- **Cosine distance** = `1 − cosine similarity`. 0 = identical direction.
- **Clustering** = automatically grouping similar things without being told the
  groups in advance.
- **HDBSCAN** = a density-based clustering algorithm. Its crucial property here:
  it is allowed to say "everything is one group" or "this point is just noise". It
  does not have to split the data.

**The parameter that makes it work:** `min_cluster_size = N/2 + 1`. With 10
clients that is 6. Since two clusters of 6 cannot both exist among 10 points, **at
most one cluster can form**. So when everybody is honest, one cluster forms and
nobody is rejected. That is the correct behaviour.

**Our Phase 0 version used KMeans with k=2.** KMeans is *forced* to return exactly
2 non-empty clusters — always, unconditionally, even when all 10 clients are
identical and honest. It therefore rejected honest clients every single round, and
its "detection" was noise. That was defect G-05, fixed in Task 2.7.

**Why cluster the models and not the updates?** Every client's model contains the
same shared global part plus a small local change, so honest models sit almost on
top of each other and a scaled or misdirected malicious model stands out.
Clustering the raw update deltas instead makes the near-random noise of one SGD
step look like meaningful structure. (This was a subtle mistake in the earlier
version; the comment in the code records it.)

We added one pragmatic fix: scikit-learn's HDBSCAN still prunes borderline points
to "noise" even when benign models are numerically near-identical, so a client
flagged as noise is re-admitted if its distance to the majority is no larger than
the majority's own internal spread. A genuinely scaled attacker sits far outside
that and stays rejected.

**Stage 2 — norm clipping.** Compute the **median** length of *all* updates
(including rejected ones), then shrink any update longer than that down to it.

- **Median** = the middle value. Chosen over the mean because the mean is easily
  dragged by one extreme outlier; the median is not. This is called being
  **robust**.

This caps how much any single client can move the model, no matter what.

**Stage 3 — adaptive noise.** Add Gaussian random noise with standard deviation
`σ = λ · S`, where `S` is that median norm.

- **Gaussian noise** = random numbers from a bell curve.
- **Adaptive** = the noise size scales with how big the updates are this round,
  rather than being a fixed constant.

The logic: a backdoor is a small, precise, delicate signal. Normal learning is a
large, robust signal. Enough noise erases the former while the latter survives.
Turn `λ` up too far and you destroy the model too — that trade-off is a knob we
will report on.

### 5.3 GradNorm — the deliberately weak baseline

**Idea:** malicious updates are usually bigger than honest ones, so flag any
client whose update length is unusually far from the median.

"Unusually far" is measured in **MADs**.

- **MAD (Median Absolute Deviation)** = take the median, measure how far each
  point is from it, take the median *of those distances*. A robust alternative to
  standard deviation. Multiplied by the constant **1.4826** so that on normal data
  it matches the standard deviation — that constant is where the number comes
  from, it is not arbitrary.

We ship two versions:

- `GradNorm` — clips to the median norm and averages (a defense).
- `GradNormScorer` — computes the suspicion score but aggregates with plain
  FedAvg (detection only). This is deliberate: it lets us compare *detection
  quality* against FLTrust and FLAME without the aggregation method confounding
  the comparison.

**We expect this to fail** against a norm-matched attacker (one who deliberately
keeps their update the same size as everyone else's). That expected failure is the
point — it is the control that shows why the sophisticated defenses are needed.

### 5.4 FLTrust + FLAME combined

Order is fixed and logged: FLAME's clustering removes outliers first, then FLTrust
trust-weights whatever survived. Both stages' per-client scores are recorded so we
can attribute any result to the right stage.

This is the strongest configuration we test, and it is the headline "defended"
condition in the results matrix.

### 5.5 Neural Cleanse (Wang et al., IEEE S&P 2019)

**The problem it solves:** you are handed a trained model. Is there a backdoor in
it? You do not know the trigger. You do not have poisoned examples.

**The insight:** if class *T* has a backdoor, then there exists a *tiny* change
that flips almost any input into class *T* — because that is exactly what the
trigger is. For a clean class, flipping inputs into it requires a large change.

**The method:** for **every** class, use optimisation to reverse-engineer the
smallest possible "patch" that flips inputs into that class. Then compare the
sizes. If one class's patch is dramatically smaller than the others, that class is
backdoored.

- **Reverse engineering** here means: search for a trigger by gradient descent,
  the same maths used to train models, but optimising the *input* instead of the
  weights.

The patch is two pieces:

- **mask (m)** — per feature, a number from 0 to 1: "how much of this feature do I
  overwrite?" Kept in [0,1] by a **sigmoid** (an S-shaped function that squashes
  any number into that range).
- **pattern (p)** — per feature, "what value do I write?"

Applied as `x_adv = X·(1−m) + p·m`. The **L1 norm** of the mask (`‖m‖₁`, the sum
of its entries) measures the trigger's size — that is the number being compared
across classes.

**Three tabular-specific adaptations we had to make** (these are our contribution
to making it work on network data, not the paper's):

1. **Range clamping.** After every optimisation step, clamp the pattern back
   inside each feature's observed `[min, max]`. Without it the optimiser escapes
   to absurd values, finds a "tiny" trigger for every class, and everything looks
   backdoored. The code calls this "the load-bearing clamp".
2. **Lambda schedule.** `λ` weights "keep the mask small" against "actually flip
   the class". A single fixed λ gives an all-ones or all-zeros mask. We ramp it up
   **geometrically** during optimisation.
3. **Subset.** Optimise on 5,000 samples rather than the full 1.89M rows, because
   this runs once per class.

**The anomaly index** — how the comparison is made:

```
anomaly_index(class) = |L1(class) − median(all L1s)| / (1.4826 × MAD)
```

The paper's rule of thumb is "index > 2 means backdoored". **We do not use that.**
Instead `scripts/baselines/nc_calibrate.py` trains ~10 *known-clean* models,
records the anomaly indices they produce, and takes the 95th percentile as the
threshold. That is what makes any reported false-positive rate meaningful.

**The Phase 0 finding worth putting on a slide.** With 2 classes, the median of
two numbers sits exactly between them, so both deviate from it by exactly the same
amount, so the MAD equals that amount, so the ratio is `1/1.4826 = 0.6745` —
*always*, for every model, backdoored or not. Neural Cleanse is mathematically
incapable of detecting anything in a binary task. That is defect G-02, and it is
the reason the whole project moved to 8 classes.

### 5.6 Activation Clustering (Chen et al., 2019)

**The insight:** two groups of samples are labelled `Benign` in the poisoned
training set — genuinely benign flows, and triggered DDoS flows. The model reaches
the same answer for both, but by *different internal reasoning*. So its internal
representations should form two distinct clusters.

**The method:**

1. Take all training samples labelled with the target class.
2. Get each one's **penultimate-layer activations** — the 64 numbers from the
   second-to-last layer. This is the model's internal "impression" of the input
   before it commits to an answer.
3. **PCA** down to 10 dimensions.
   - **PCA (Principal Component Analysis)** = compresses many correlated numbers
     into fewer numbers that keep most of the variation. Clustering works badly in
     high dimensions, so this step is necessary.
4. Run **KMeans with k=2** — split into exactly two groups.
5. Compute the **silhouette score**.
   - **Silhouette score** = a number from −1 to +1 measuring how well-separated
     clusters are. Near 0 = the split is arbitrary (one blob). Near 1 = two
     genuinely distinct groups.
6. High silhouette → flag as poisoned. The **smaller** cluster is the suspect
   (poison is a minority of the class).

**Why KMeans is legitimate here but was wrong in FLAME:** here we *want* to force
a 2-way split and then ask "was that split meaningful?" — the silhouette score
answers that. In FLAME the forced split was itself the decision, with nothing
asking whether it was meaningful.

**Our two improvements over the paper:**

- **Calibrated threshold.** The paper suggests 0.10–0.15. We compute the actual
  distribution of silhouette scores on known-clean models and take the 95th
  percentile. A hardcoded literature threshold on a different data modality is not
  evidence.
- **Exclusionary reclassification.** Remove the suspect cluster, retrain, and see
  what the removed samples get classified as. If they land in a *different* class,
  that is strong confirmation they were poisoned — and it *names the source
  class*, telling you what the attacker was disguising.

---

## Part 6 — Measuring things honestly

**File:** `flids/eval/metrics.py`

This module is small but it is where intellectual honesty lives.

### 6.1 ΔASR (delta-ASR) — the only ASR number we report

**The problem with plain ASR.** Suppose we report ASR = 0.98. Impressive. But what
if a **clean** model — never attacked, never poisoned — *also* classifies 96% of
those triggered samples as Benign? Then the backdoor is responsible for 2 points,
not 98.

Why would a clean model do that? Because `999.0` is a bizarre input the model has
never seen. Its behaviour there is arbitrary, and "arbitrary" can easily mean
"predicts the majority class", which is Benign.

**The fix:**

```
ΔASR = ASR(backdoored model) − ASR(clean model, same trigger, same seed)
```

The clean-model baseline is measured separately for every rung and every seed by
`scripts/baselines/clean_asr.py` and stored in a CSV that the runner reads.

**This is defect G-01, and it may be the single most important correction in the
project.** It is why Phase 0 exists.

> **Panel question:** "How do you know the backdoor caused that?"
> **Answer:** "We subtract the clean-model control. We report ΔASR, never raw ASR."

### 6.2 Accuracy *and* macro-F1

- **Accuracy** = fraction of predictions that are correct.
- **Precision** (for a class) = of everything the model called class X, how much
  really was X.
- **Recall** = of everything that really was X, how much did the model catch.
- **F1** = the harmonic mean of precision and recall — one number balancing both.
- **Macro-F1** = compute F1 for each class separately, then average them,
  weighting every class **equally**.

**Why macro-F1 is mandatory here.** Our data is extremely **imbalanced** — most
rows are Benign, and Infiltration has 27 training rows. A model that predicts
"Benign" for literally everything scores ~80% accuracy and is completely useless.
Macro-F1 exposes that instantly, because it would score near zero on the seven
attack classes and the average would collapse.

### 6.3 Detection AUC

- **AUC (Area Under the ROC Curve)** = a score from 0 to 1 for a ranking. 1.0 =
  perfect (every malicious client ranked above every honest one). 0.5 = random
  guessing. Below 0.5 = worse than a coin flip.

We use each defense's per-client suspicion score as a malicious/honest classifier
and compute the AUC.

**Why this is better than "did it remove the attacker?"** A binary
removed/not-removed answer depends entirely on where you set the threshold. AUC
measures the *quality of the ranking* independently of the threshold — it tells
you whether the defense's signal contains real information at all. A defense with
AUC 0.55 is not "almost working"; it is noise.

The AUC is computed from rank statistics directly (with proper handling of ties)
so this module has no scikit-learn dependency.

### 6.4 Defense FPR

- **FPR (False Positive Rate)** = the fraction of **honest** clients that the
  defense wrongly rejected.

A defense that rejects everybody catches every attacker and is worthless. FPR is
the cost side of the ledger, and it must always be reported next to detection
performance. Our broken KMeans-FLAME scored terribly here by construction — which
is exactly why we measure it.

### 6.5 Backdoor lifespan

The attacker stops at round 20. How many further rounds until ASR falls below half
its peak?

This measures **persistence**. A backdoor lasting 2 rounds is a curiosity; one
lasting 50 rounds is a genuine threat, because it means the attacker can
compromise a client, plant the backdoor, and withdraw before anyone investigates.
Returns −1 if the backdoor never decays.

### 6.6 Three seeds, always

- **Seed** = the number that initialises all randomness. Same seed → same run.

Every headline number is reported as **mean ± standard deviation over 3 seeds**
(0, 1, 2). A single run can be lucky. If your effect is smaller than the variation
between seeds, you do not have an effect. `summarise_seeds()` enforces this
format.

---

## Part 7 — Reproducibility infrastructure

This is the part that is invisible in a demo and decisive in a viva.

### 7.1 The runner and the config schema

Every experiment is one YAML file. Nothing is hardcoded.

- **YAML** = a plain-text format for configuration. Human-readable key/value.

```yaml
data:      { alpha: 0.5, n_clients: 10, labels: multiclass }
model:     { arch: mlp, hidden: [256, 128, 64] }
federated: { rounds: 20, local_epochs: 2, lr: 0.05, aggregator: fltrust+flame }
attack:
  enabled: true
  malicious_clients: [0, 1, 2, 3]
  trigger: { name: oob_999, target_label: 0, source_class: 2 }
  poison_ratio: 0.3
  attack_window: [1, 20]
seed: 0
```

Run it:

```
python -m flids.runner --config configs/badnets_fltrust_flame.yaml
```

### 7.2 run_id — the content hash

```
run_id = sha256(canonical_json(resolved_config))[:12]
```

Term by term:

- **Hash function** = a one-way function turning any input into a fixed-length
  fingerprint. Change one character of input, the fingerprint changes completely.
- **SHA-256** = the standard cryptographic hash.
- **Canonical JSON** = the config serialised with keys always in the same sorted
  order, so that two logically identical configs written in different orders
  produce the same fingerprint.
- `[:12]` = keep the first 12 characters. Enough to be unique for our scale.

**What this buys us:** the configuration *is* the identity of the result. Two runs
with identical settings collide by design. Change anything — the seed, alpha, one
learning rate digit — and you get a new directory. It becomes structurally
impossible to overwrite one experiment with another or to lose track of what
produced a number.

The runner **refuses to overwrite** an existing `run_id`. `results/` is
**append-only**. Analysis notebooks read from it and never write to it.

There is a subtle detail recorded in the code: `data.subsample` is deliberately
*absent* from the defaults, because adding a key to the defaults changes the
resolved config, which changes every hash, which orphans every result already on
disk. That comment exists because it nearly happened.

### 7.3 What each run writes

```
results/<run_id>/
  config.yaml       the exact resolved config, defaults filled in
  metrics.jsonl     one JSON object per round
  summary.json      final aggregate numbers
  model_final.npz   the trained parameter vector
  env.json          Python version, package versions, OS, CPU, git commit, seed
```

- **JSONL** = JSON Lines: one complete JSON object per line. Appendable, and
  readable even if a run crashes halfway.
- **env.json** captures the environment so that "it worked on my machine" is a
  checkable claim rather than an excuse. It includes the **git commit hash** — the
  exact version of the code that produced the result.

### 7.4 Determinism

Four separate sources of randomness are seeded independently and explicitly:

1. Global Python and NumPy random state
2. The Dirichlet partition
3. Poison-sample selection (seeded per client, per round)
4. Model weight initialisation

The comment in `seeding.py` says it best: one unseeded `np.random.choice` in the
poisoning function is enough to make the entire project irreproducible.

### 7.5 The gates

- **Gate** = a checkable condition that must hold before the next phase begins. It
  turns "are we ready to move on?" from an opinion into a test.

| Gate | Condition | Status |
|---|---|---|
| **G0** | Every prior result re-measured and given a STANDS / VOID / PARTIAL verdict | done |
| **G1** | `clean_fedavg.yaml` produces a byte-identical `summary.json` on all 3 machines | done |
| **G2** | At least one defense demonstrably works, with controls | done |
| **G3** | A pcap file that actually carries the trigger | Phase 3 |
| **G4** | A measurable advantage from distributing the attack | Phase 4 |
| **G5** | Every claim in the report traces to a `run_id` | Phase 5 |

`scripts/gates/gate_g0.py`, `gate_g1.py`, `gate_g2.py` automate the mechanically
checkable parts. G1's cross-machine check is necessarily manual: run it on each
laptop and `diff` the output.

---

## Part 8 — Bugs we found and fixed

These are worth a slide. Finding your own bugs is a strength; an examiner finding
them is not.

### 8.1 The MLP aliasing bug (silent and severe)

In NumPy, slicing an array gives you a **view** — a window onto the original
memory, not a copy. Writing to the view writes through to the original.

`MLP.set_params(vec)` sliced the incoming global parameter vector to fill the
layer weights. Those weights were therefore *views into the server's global
vector*. Then `fit()` updated the weights in place — silently overwriting the
server's global parameters during local training.

Consequences: FLTrust computes `gᵢ = client_params − global_params`, but
`global_params` had already been mutated into `client_params`. So `gᵢ ≈ 0`. The
trust score was garbage. FedAvg was quietly degraded too.

**The fix is one word — `.copy()`** — and the comment above it now explains why it
is load-bearing, so nobody "optimises" it away.

**All results produced before this fix are void and must be regenerated.** We
state that in the docs rather than hoping nobody checks.

### 8.2 Neural Cleanse inert at K=2 (G-02)

Described in §5.5. The anomaly index is a mathematical constant with two classes.
Fixed by moving to 8 classes.

### 8.3 FLTrust's missing normalisation (G-04)

Described in §5.1. Trust was computed correctly but applied to un-normalised
updates, so the magnitude defense did not exist.

### 8.4 FLAME's KMeans (G-05)

Described in §5.2. KMeans always returns 2 clusters, so honest clients were
rejected every round even with zero attackers present. There is now a dedicated
sanity check, `flame_zero_attacker.py`, that runs FLAME with **no attacker at
all** and asserts nobody is removed. That test must pass before any FLAME-vs-attack
number is trusted.

### 8.5 Two preprocessing interactions

Both found by measurement, both documented in `docs/phase1-foundation.md`:

- Dropping `Destination Port` before deduplication destroyed 98.8% of PortScan.
- Treating `-1` as missing destroyed 50.9% of the dataset, unevenly across classes.

### 8.6 The en-dash

`0x96` in the Web Attack labels. Matching the ASCII hyphen returns zero rows and
you lose an entire attack family without any error message. Now normalised, and
unknown labels raise instead of being dropped.

---

## Part 9 — What is still missing

**Phase 3 — the realizable trigger (the actual contribution)**
- SHAP ranking, intersected with the `free` rows of the perturbability table
- Optimise an in-bounds trigger on those features
- Kali Linux VM shaping real traffic → CICFlowMeter → confirm the trigger appears
  in genuinely captured flows
- The round trip: emit packets → extract features → poisoned model says "Benign"

**Phase 4 — distributed attack**
- DBA: split the trigger across 3 colluding clients, each training on only its own
  slice, with the full trigger applied only at test time (Xie et al., ICLR 2020 —
  cited, not ours)
- Constrained-loss training so the attacker actively evades the defense
- Ablations over number of colluders and poison ratio

**Phase 5 — evaluation campaign**
- 4 triggers × 4 defenses × 4 alphas × 3 seeds = **192 runs**
- Durability curves, UNSW-NB15 cross-dataset check, FGSM/PGD adversarial
  robustness, SHAP comparison of clean vs backdoored models, Streamlit dashboard,
  confidence intervals

**Phase 6 — report and viva**

---

## Glossary A–Z

**Ablation** — Re-running an experiment with one piece changed or removed, to show
whether that piece mattered.

**Accuracy** — Fraction of predictions that are correct. Misleading on imbalanced
data; always pair it with macro-F1.

**Activation** — The output of a layer inside the network. The "penultimate
activations" are the model's internal impression of an input just before it
decides.

**Adaptive noise** — Random noise whose size scales with the data it is added to,
rather than being a fixed constant.

**Aggregator** — The server's rule for combining client updates into a new global
model. FedAvg is plain averaging; every defense is a smarter aggregator.

**Alpha (α)** — The Dirichlet parameter controlling how unevenly data is split
across clients. Small = very uneven. Infinity = perfectly even (IID).

**ANOVA F-statistic** — A classical statistic measuring how strongly one feature
separates the classes. Our temporary stand-in for SHAP.

**ASR (Attack Success Rate)** — Fraction of triggered non-target samples the model
classifies as the attacker's target class.

**Attention** — The Transformer mechanism letting each part of an input weigh every
other part by relevance.

**AUC (Area Under the ROC Curve)** — Quality of a ranking, 0 to 1. 1.0 = perfect,
0.5 = random guessing.

**Backdoor** — A hidden behaviour in a model: correct on normal input,
attacker-chosen output when a trigger is present.

**Backpropagation** — The calculus that computes how much each weight contributed
to the error, so it can be corrected. We wrote it by hand.

**BadNets** — The 2017 paper introducing backdoors via training-data poisoning.

**Batch / mini-batch** — A small group of samples (here 256) processed together in
one training step.

**Benign** — Normal, non-attack traffic. Class 0. Also the attacker's target class.

**Bias (b)** — The constant offset added in each layer, alongside the weights.

**Binary classification** — Two possible answers. Neural Cleanse is mathematically
inert here, which is why we use 8 classes.

**CICFlowMeter** — The tool converting raw captured packets into the 77 flow
features. The bridge between problem space and feature space.

**CIC-IDS2017** — The benchmark network-intrusion dataset we use. Public, standard,
and known to contain defects we correct.

**Class imbalance** — When some classes have vastly more examples than others. Ours
is severe: millions of Benign rows, 27 Infiltration rows.

**Client** — One participant in federated learning. We simulate 10.

**Clipping (gradient / norm)** — Capping a vector's length. Limits how far any
single update can move the model.

**Clustering** — Automatically grouping similar items without being told the groups
in advance.

**Colluding clients** — Multiple malicious clients cooperating in one attack.

**Confusion matrix** — Table of true class versus predicted class. All the
per-class metrics derive from it.

**Cosine similarity** — Whether two vectors point the same direction, −1 to +1,
ignoring their lengths. **Cosine distance** = 1 − cosine similarity.

**Cross-entropy loss** — The standard "how wrong were you" measure for
classification; punishes confident wrong answers hardest.

**d_model** — The width of each token's vector in the TabTransformer (32 here).

**Data leakage** — When test information contaminates training, inflating scores.
Our dedupe-before-split rule exists to prevent it.

**DBA (Distributed Backdoor Attack)** — Xie et al., ICLR 2020. Splits a trigger
across colluding clients; each trains on its own slice, the full trigger is used
only at test time. Prior work, cited as such.

**DDoS** — Distributed Denial of Service. Class 2, and our attack's source class.

**Deduplication** — Removing identical repeated rows. Must happen before the
train/test split.

**Delta / update** — `client parameters − global parameters`. What defenses
actually inspect.

**Determinism** — Same input, same code, same seed → byte-identical output. Gate
G1's requirement.

**Dirichlet distribution** — Generates random proportions summing to 1. Used to
create realistic non-IID client splits.

**DoS** — Denial of Service. Class 1.

**Dropout** — Randomly disabling neurons during training to reduce overfitting.
Disabled here because randomness conflicts with byte-identical reproducibility.

**Embedding** — A learned vector representation. Here, one per feature so the
TabTransformer can distinguish them.

**Epoch** — One complete pass over a dataset. Clients do 2 local epochs per round.

**Feature** — One number describing a network flow. We have 77 (76 after dropping
Destination Port).

**Feature space** — The world of those numbers. A feature-space attack edits them
directly — convenient and unrealistic.

**FedAvg** — Federated Averaging: the base algorithm. Server averages client
parameter vectors, weighted by data size.

**Federated Learning (FL)** — Training a shared model across many parties where
only model parameters, never raw data, are exchanged.

**FLAME** — Nguyen et al., USENIX Security 2022. HDBSCAN clustering + median norm
clipping + adaptive noise.

**FLTrust** — Cao et al., NDSS 2022. Server-side root set, ReLU-cosine trust
scores, and normalisation of updates onto the server update's norm.

**Flow** — One network conversation, summarised as 77 numbers.

**FPR (False Positive Rate)** — Fraction of honest clients wrongly rejected. The
cost side of any defense.

**free / partial / fixed** — The three perturbability classes: attacker fully
controls / partly influences / cannot control a feature.

**Gate (G0–G5)** — A checkable condition that must hold before the next phase
starts.

**Gaussian noise** — Random values drawn from a bell curve.

**Gradient** — The direction and size of the change that would most reduce the
error. Training follows gradients downhill.

**Gradient descent** — Repeatedly stepping in the direction that reduces error.

**Hash / SHA-256** — A one-way fingerprint of data. We hash the config to produce
the `run_id`.

**HDBSCAN** — A density-based clustering algorithm that can return a single cluster
or label points as noise. Essential to FLAME being correct.

**He initialisation** — A recipe for random starting weights scaled by layer size.

**IAT (Inter-Arrival Time)** — The gap between consecutive packets. Forward IAT is
attacker-controlled; backward IAT is not.

**IDS (Intrusion Detection System)** — A system that classifies network traffic as
benign or attack.

**IID / non-IID** — Whether all clients' data looks statistically the same. Real FL
is non-IID; non-IID is what makes defenses hard.

**JSONL** — JSON Lines. One JSON object per line; appendable and crash-tolerant.

**KMeans** — Clustering that splits data into exactly *k* groups. Always returns
*k* non-empty groups — which is why it was wrong inside FLAME and correct inside
Activation Clustering.

**Label** — The true class of a sample.

**Label proxy / leak** — A feature that essentially reveals the answer.
`Destination Port` is one, which is why we drop it.

**Lambda (λ)** — A weighting knob. In Neural Cleanse it balances mask size against
flip success; in FLAME it scales the added noise.

**Layer** — One matrix-multiply-plus-bias step in a neural network.

**LayerNorm** — Normalisation applied inside the network to keep values in a stable
range.

**Learning rate (lr)** — The step size in gradient descent.

**Lifespan (backdoor)** — Rounds after the attacker leaves until ASR halves from
its peak. Measures persistence.

**L1 norm** — Sum of absolute values. In Neural Cleanse, the size of the mask.

**L2 norm (‖·‖)** — Euclidean length of a vector: root of the sum of squares.

**L2 regularisation** — A penalty on large weights, discouraging overfitting.

**Macro-F1** — Per-class F1 scores averaged with equal weight per class. The honest
metric on imbalanced data.

**MAD (Median Absolute Deviation)** — A robust spread measure: the median distance
from the median. Scaled by 1.4826 to match a standard deviation on normal data.

**Malicious client** — A client controlled by the attacker.

**Mask (m)** — In Neural Cleanse, the per-feature 0–1 vector saying how much of
each feature the trigger overwrites.

**Median** — The middle value. Robust against outliers, unlike the mean, which is
why every robust defense uses it.

**MLP (Multi-Layer Perceptron)** — The simple stacked-layer neural network. Our
main model: 77 → 256 → 128 → 64 → 8.

**Multi-class** — More than two possible answers. Required for Neural Cleanse to
function.

**Neural Cleanse** — Wang et al., 2019. Reverse-engineers a minimal trigger for
every class and flags the class whose trigger is anomalously small.

**Non-IID** — See IID.

**Normalisation (data)** — Rescaling features to comparable ranges. We use
QuantileTransformer.

**Norm** — See L2 norm.

**OOB / OOD** — Out Of Bounds / Out Of Distribution. A value that does not occur in
real data, e.g. our `999.0`. Trivially detectable.

**Overfitting** — Memorising the training data instead of learning general
patterns.

**p85 (85th percentile)** — The value below which 85% of the data falls. Our
in-bounds trigger value: high but entirely unremarkable.

**Parameters** — All the weights and biases of a model, flattened into one long
vector. This is what federated learning transmits.

**PCA (Principal Component Analysis)** — Compresses many correlated numbers into
fewer while keeping most of the variation. Used before clustering.

**pcap** — Packet capture file. Raw recorded network traffic.

**Penultimate layer** — The second-to-last layer. Its activations are the model's
internal representation of an input.

**Percentile** — The value below which a given percentage of data falls.

**Perturbability** — How much an attacker can actually change a feature. Our table
classifies all 76 as free / partial / fixed.

**Poison ratio** — Fraction of the attacker's eligible rows that get triggered and
relabelled.

**Poisoning** — Corrupting training data (or updates) to change the trained model's
behaviour.

**PortScan** — Probing many ports to find open services. Class 3.

**Precision** — Of everything predicted as class X, how much really was X.

**Problem space** — The real world of packets, where the attacker actually
operates. Our contribution lives here.

**QuantileTransformer** — Rescales each feature by rank onto a bell curve. Bounded,
robust to extreme tails, makes the sentinel `-1` harmless.

**Recall** — Of everything that really was class X, how much did the model catch.

**ReLU** — `max(0, x)`. The squashing function in our network, and the clipping
step in FLTrust's trust score.

**Reverse engineering (a trigger)** — Using gradient descent to search for the
smallest input change that flips a class. The core of Neural Cleanse.

**Robust** — Insensitive to outliers. Median and MAD are robust; mean and standard
deviation are not.

**Root set** — FLTrust's small clean server-side dataset (1,000 balanced rows) used
to produce the reference update.

**Round** — One full cycle of send → train → return → aggregate.

**run_id** — The 12-character SHA-256 fingerprint of a config; the identity of a
result directory.

**Seed** — The number initialising all randomness. Same seed → same run. We use 0,
1, 2.

**Sentinel value** — A placeholder meaning "not applicable" rather than a real
measurement. `-1` in the two window-size columns.

**SHAP** — A method attributing a model's prediction to individual features,
grounded in cooperative game theory. Replaces the ANOVA placeholder in Phase 3.

**Sigmoid** — An S-shaped function squashing any number into (0, 1). Keeps the
Neural Cleanse mask valid.

**Silhouette score** — How well-separated two clusters are, −1 to +1. Activation
Clustering's decision statistic.

**Softmax** — Converts raw scores into probabilities summing to 1.

**Source class** — The true class of the traffic being disguised. Ours is DDoS.

**Stratified split** — A train/test split preserving each class's proportion, so
rare classes appear in both halves.

**StandardScaler** — Subtract mean, divide by standard deviation. What we
deliberately replaced with QuantileTransformer.

**TabTransformer** — A Transformer adapted for tabular data, treating each feature
as a token. Our second architecture.

**Target class / target label** — The answer the attacker wants. Ours is Benign.

**Targeted attack** — The attacker wants one specific wrong answer, not general
breakage.

**Token** — One unit the Transformer attends over. Here, one feature.

**Train/test split** — Holding out data the model never trains on, to measure
genuine generalisation.

**Transformer** — The attention-based architecture behind modern large models.

**Trigger** — The secret pattern that activates a backdoor.

**Trigger ladder** — Our four rungs from unrealistic (`oob_999`) to fully
realizable (`problemspace`). The drop in strength across the ladder is Claim 2.

**Trust score (TS)** — FLTrust's per-client weight: `ReLU(cos(gᵢ, g₀))`.

**Update scaling** — The attacker multiplying its update to dominate the average.
Prior work (Bagdasaryan et al.).

**View (NumPy)** — A window onto another array's memory rather than a copy. Writing
through one corrupts the other — the cause of our `set_params` bug.

**Weights (W)** — The learned matrices in each layer.

**YAML** — The plain-text configuration format defining every experiment.

**ΔASR (delta-ASR)** — `ASR(backdoored) − ASR(clean)`. The only ASR number we
report, because a trigger can fire on a clean model.

---

## Appendix — How to run everything

All commands run from the repository root, as modules.

```bash
# one-time setup
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements.txt

# Phase 0 - validity triage
python -m scripts.validation.run_all --processed
python -m scripts.gates.gate_g0

# Phase 1 - build the data, check the foundation
python -m scripts.preprocessing.preprocess --data data/raw
python -m scripts.preprocessing.partition_figures
python -m scripts.gates.gate_g1

# Phase 2 - baselines, in this order
python -m scripts.baselines.check_triggers
python -m scripts.baselines.clean_asr            # must run before any dASR
python -m flids.runner --config configs/badnets_oob999.yaml
python -m flids.runner --config configs/badnets_fltrust.yaml
python -m flids.runner --config configs/badnets_flame.yaml
python -m flids.runner --config configs/badnets_fltrust_flame.yaml
python -m scripts.baselines.flame_zero_attacker  # sanity check, must pass first
python -m scripts.baselines.detection_auc
python -m scripts.baselines.durability
python -m scripts.baselines.nc_calibrate         # slow: trains ~10 models
python -m scripts.baselines.nc_roc
python -m scripts.baselines.activation_clustering
python -m scripts.gates.gate_g2
```

Everything writes into `results/`. Nothing overwrites anything.
