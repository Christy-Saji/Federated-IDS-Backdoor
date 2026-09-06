# Phase 3 — The Realizable Trigger

> **OUT OF SCOPE — NOT THE LIVE PLAN.** This project is scoped to
> feature-space attacks only; Phases 0-2 are the deliverable. The
> problem-space work below (real packets, Kali traffic shaping,
> CICFlowMeter round-trip) will not be built, and the "this is the novel
> contribution" framing in this file is outdated. See the Scope section of
> `CLAUDE.md` and `HANDOFF.md`. Do not start this work unless the project
> owner reopens the scope.


**Duration:** 4 weeks · **Owner:** M1 lead, M2 + M3 support · **Entry condition:** Gate G2 cleared, perturbability table exists

---

## Objective

Build a backdoor whose trigger a real network attacker can actually produce, and demonstrate it end-to-end: **Kali shapes real traffic → CICFlowMeter extracts features → the trigger is present in genuine captured flows → the poisoned federated model misclassifies them as benign.**

This is the novel contribution. Everything before it was rebuilding the foundation; everything after it is evaluation. The whole project rests on this phase, which is why it is scheduled early — with eight weeks of slack behind it — and why it has an explicit fallback.

---

## Why this phase exists

This is a **network** IDS. Features are computed by CICFlowMeter from real captured packets, and the attacker controls only the packets they emit. The current trigger sets `feature[0..2] = 999.0`, which assumes the attacker can write directly into the defender's feature vector — the *feature-space* threat model. The real one is *problem-space*: the attacker must produce **packets** that, after flow construction and feature extraction, yield the trigger.

No prior work has built a problem-space-realizable backdoor for a *federated* NIDS. That gap is the contribution. It also turns the VM lab from a throwaway demo into the centrepiece — M1's hardest task becomes the most impressive part of the demo.

### The two claims this phase establishes

- **Claim 1 — Construction.** A distributed backdoor whose trigger is restricted to attacker-perturbable, in-distribution flow features, realized in captured pcaps rather than synthesized feature vectors.
- **Claim 2 — Measurement.** Under problem-space constraints the threat is *materially smaller* than the feature-space literature reports. Quantified as `dASR` across the constraint ladder.

**Claim 2 does not depend on Claim 1 succeeding.** That independence is what makes this phase safe to attempt — see the fallback.

---

## Task 3.1 — SHAP-ranked candidate features · M2 → M1 · week 1

Pick the trigger features from the intersection of *important* and *controllable*.

### Steps

1. Run SHAP on the clean global model (M2 already has this working). Rank features by mean absolute SHAP value for the target-class decision.
2. Load `features/perturbability.csv` from Phase 1.
3. **Intersect:** keep only features that are both high-SHAP and classed `free`.
4. Hand M1 the top candidates — expect the top 3–5 `free` features by importance.

```python
shap_rank = shap_importance(model, X_sample)          # descending
free      = set(perturbability[perturbability["class"] == "free"]["feature"])
candidates = [f for f in shap_rank if f in free][:5]
```

### The tension to document

SHAP's top features on the current model are `Packet Length Variance` and `Bwd Packet Length Std` — both largely **victim-controlled** (`fixed`). That is exactly why the intersection matters: the most *useful* trigger features are often not *controllable*. Restricting to `free` features is expected to cost some attack strength — and measuring that cost is Claim 2. Write this down as it happens.

**Deliverable:** ranked candidate list with SHAP value and perturbability class per feature.

---

## Task 3.2 — In-bounds trigger optimisation · M1 · weeks 1–2

Construct the feature-space version of the realizable trigger first. It is the target the VM lab then tries to reproduce with real packets.

### Design rules (from Tabdoor / CatBack)

- **In-bounds values only.** Clip every trigger feature to its observed `[min, max]`. Out-of-bounds values (like `999.0`) are trivially detectable as outliers — that is the entire reason Tabdoor's in-bounds variant exists.
- **Prefer the most-common or high-percentile value.** Tabdoor sets trigger features to the feature's mode; `p85` is a reasonable, unremarkable alternative. Both are "statistically subtle."
- **`free` features only**, from Task 3.1.

```yaml
attack:
  type: badnets
  trigger:
    rung: inbounds_free
    features: [Total Fwd Packets, Fwd Packet Length Max, Fwd IAT Mean]
    value: p85
  source_class: 2
  target_class: 0
  poison_ratio: 0.05
```

### Measure and compare

Run this against the `oob_999` and `inbounds_any` rungs from Phase 2. Build the first three rows of the Claim 2 table:

| Trigger | Realizable | Clean-model ASR | dASR no defense | dASR vs FLTrust+FLAME |
|---|---|---|---|---|
| oob_999 | No | | | |
| inbounds_any | No | | | |
| inbounds_free | feature-space | | | |

**Deliverable:** `inbounds_free` trained model, plus the three-row comparison. If `inbounds_free` already shows a large `dASR` drop versus `oob_999`, that is Claim 2 emerging — before the VM lab even runs.

---

## Task 3.3 — Traffic shaping in the VM lab · M1 · weeks 2–3

**The highest-risk task in the project. Timeboxed to 2 weeks.** Read the fallback before starting.

### Lab topology

```
Kali (attacker)  ->  Ubuntu + DVWA (victim)  ->  Ubuntu monitor (CICFlowMeter)
   shapes traffic       receives attack           captures + extracts features
```

The monitor must see the traffic — put the victim's interface on a mirrored port, or run CICFlowMeter on the victim host, or bridge through the monitor.

### Mapping trigger features to packet-level controls

| Trigger feature | How the attacker produces it |
|---|---|
| `Total Fwd Packets` | send a specific packet count per flow — script the attack to a fixed request count |
| `Fwd Packet Length Max/Mean` | pad application-layer payloads to target sizes |
| `Fwd IAT Mean/Std` | insert deliberate inter-packet delays: `tc qdisc add dev eth0 root netem delay 40ms 5ms` |
| `Flow Duration` | hold the connection open a controlled time before teardown |
| `Subflow Fwd Packets` | control fragmentation / burst structure |

### Procedure

1. Start from a real attack that CICFlowMeter labels correctly (DDoS or PortScan are the most reliable).
2. Apply one shaping control at a time. After each, capture and run CICFlowMeter, and read back the affected feature. Confirm it moved toward the target `p85` value.
3. Compose the controls to hit all trigger features simultaneously.
4. Capture a bulk run of triggered attack flows.

### Reality check

CICFlowMeter is known-buggy (Engelen et al.): misordered packets, dropped packets, silently empty CSVs. **Use a maintained fork, and validate it on a known pcap before relying on it.** Feature values will be noisy — the trigger will be a *region* in feature space, not an exact point. Design the trigger with tolerance: "`Fwd IAT Mean` in `[p80, p90]`" rather than a single value.

**Deliverable:** a shaping recipe (the exact `tc`/scripting commands) that reliably moves the trigger features into their target region.

---

## Task 3.4 — The round-trip verification · M1 · week 3

The proof that Claim 1 holds.

### Steps

1. Capture the shaped attack traffic to `triggered_attack.pcap`.
2. Run it through CICFlowMeter → `triggered_flows.csv`.
3. Apply the **Phase 1 preprocessing pipeline unchanged** (same QuantileTransformer, same feature order).
4. Confirm the trigger features land in their target region **after** the full feature-extraction and preprocessing round-trip — not just in the raw capture.

```python
flows  = cicflowmeter("triggered_attack.pcap")
X      = preprocess(flows, transformer=load("quantile_transformer.pkl"))
for f in trigger_features:
    col = X[:, feature_index[f]]
    print(f, "in target region:", (target_lo[f] <= col).mean(), "<=", (col <= target_hi[f]).mean())
```

**Deliverable — Gate G3's core artifact:** `triggered_attack.pcap` plus a table showing the trigger features are present in the extracted-and-preprocessed flows. **This pcap is the single most compelling artifact in the entire project** — the difference between "we simulated an attack" and "we produced the attack."

---

## Task 3.5 — Threat-model-clean camouflage · M3 · weeks 2–3

Fix G-08 in parallel with the VM work. The current `camouflage_gradient()` reads other clients' updates, which a real federated client cannot see.

### The fix

Estimate the benign update direction from something every client legitimately observes — the previous round's global model delta.

```python
def estimate_benign_direction(theta_now, theta_prev, my_update, n_clients):
    """Under FedAvg, (theta_now - theta_prev) is the AVERAGE of all client updates.
    Subtract the attacker's own known contribution for a sharper benign estimate."""
    avg_update    = flatten(theta_now) - flatten(theta_prev)
    my_share      = flatten(my_update) / n_clients
    benign_est    = (avg_update * n_clients - flatten(my_update)) / (n_clients - 1)
    return benign_est   # attacker aligns / norm-matches to THIS

def camouflage(my_update, benign_est):
    m, b = flatten(my_update), benign_est
    return unflatten(m * (np.linalg.norm(b) / (np.linalg.norm(m) + 1e-12)))
```

This is the standard construction in the adaptive-attack literature (3DFed, A3FL). It uses only `theta_now`, `theta_prev`, the attacker's own update, and the known client count — all legitimately observable. It is defensible under questioning, and it still works.

**Deliverable:** `flids/attacks/camouflage.py` using observable signals only, with a one-line comment at each input proving it is observable.

---

## Task 3.6 — Poison federated training with real flows · M1 + M3 · week 4

1. Inject the captured triggered flows as the malicious client's poison set (rather than synthetically stamping the trigger onto feature vectors).
2. Train the federation with the attack window from Phase 2.
3. Measure `dASR` on a **held-out set of captured triggered flows** — real flows the model never saw in training.

```yaml
attack:
  type: problemspace
  trigger:
    rung: problemspace
    source_pcap: data/captured/triggered_attack.pcap
  malicious_clients: [0]
  poison_ratio: 0.05
  attack_window: [1, 20]
```

**Deliverable:** the fourth row of the Claim 2 table, computed on real captured flows.

---

## Task 3.7 — Detectors and defenses vs the realizable trigger · M2 + M3 · week 4

Run the full Phase 2 detector/defense suite against the realizable trigger:

- Neural Cleanse — does the in-bounds realizable trigger evade it better than `oob_999`?
- Activation Clustering — TPR/FPR against realizable poison
- FLTrust / FLAME / combined — `dASR` and detection AUC

**Deliverable:** the complete four-row Claim 2 table, all defense columns filled. **Every row is a publishable finding regardless of which way the numbers fall.**

---

## Gate G3

**A pcap file whose CICFlowMeter output contains flows carrying the trigger, plus a `dASR` figure computed on those real flows.**

Checklist:

- [ ] `triggered_attack.pcap` exists and is reproducible from the shaping recipe
- [ ] Trigger features verified present *after* the full extraction + preprocessing round-trip
- [ ] Federated model poisoned with real flows shows a measured `dASR` on held-out real flows
- [ ] Camouflage uses only observable signals (no `clean_params_list`)
- [ ] The four-row Claim 2 table is complete

---

## Fallback — read before starting, decide by end of week 2

**Tasks 3.3 and 3.4 are the real schedule risk.** If traffic shaping cannot reliably produce the trigger within the 2-week timebox, fall back to `inbounds_free`:

- Trigger restricted to `free`, in-distribution features (Task 3.2 already delivers this)
- Assigned in feature space rather than realized through packets
- Accompanied by a **documented feasibility analysis**: the perturbability table plus the shaping recipe, showing the trigger *is* realizable in principle even if full end-to-end capture was not completed

**What changes:** Claim 1 weakens from "realized" to "realizable in principle." **Claim 2 is completely unaffected** — the constraint ladder still stands, and "we measured how the threat shrinks under progressively realistic constraints" is a complete, defensible contribution on its own.

**The fallback still beats the original plan on novelty.** Decide at the end of week 2 so nobody makes the call under pressure in week 14. Do not let the VM lab consume the whole phase — if it is not working by the timebox, take the fallback and move to Phase 4.

---

## Common failure modes in this phase

- **Letting the VM lab run past its timebox.** The fallback exists precisely so this never blocks the project. Two weeks, then decide.
- **Verifying the trigger in the raw pcap but not after preprocessing.** CICFlowMeter and the QuantileTransformer both transform the values. Only the post-preprocessing check counts.
- **Designing the trigger as an exact point.** Real captured features are noisy; design a target *region*.
- **Reusing the same captured flows for train and evaluation.** Hold some out. Measuring `dASR` on training flows overstates the attack.
- **Forgetting Claim 2 is the safety net.** If Claim 1 stalls, Claim 2 is still a full contribution. Do not treat a VM-lab setback as a project failure.

---

**Previous:** [Phase 2 — Faithful Baselines](phase-2-faithful-baselines.md) · **Next:** [Phase 4 — Distributed & Adaptive Attack](phase-4-distributed-attack.md)
