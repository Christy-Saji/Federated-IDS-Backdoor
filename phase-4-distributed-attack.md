# Phase 4 — Distributed & Adaptive Attack

> **OUT OF SCOPE — NOT THE LIVE PLAN.** This project is scoped to
> feature-space attacks only; Phases 0-2 are the deliverable. The
> problem-space work below (real packets, Kali traffic shaping,
> CICFlowMeter round-trip) will not be built, and the "this is the novel
> contribution" framing in this file is outdated. See the Scope section of
> `CLAUDE.md` and `HANDOFF.md`. Do not start this work unless the project
> owner reopens the scope.


**Duration:** 3 weeks · **Owner:** M1 lead, M3 support · **Entry condition:** Gate G3 cleared (or fallback taken)

---

## Objective

Extend the single-client realizable backdoor into a **distributed, colluding, adaptive** attack, and measure whether distribution actually buys the attacker anything under problem-space constraints.

The honest question this phase answers: *does splitting the attack across colluding clients and adapting to the defense make it stronger — or does the problem-space constraint erase that advantage?* Either answer is a result.

---

## Why this phase exists, and the framing that keeps it defensible

Splitting a trigger across colluding clients is **DBA** (Xie et al., ICLR 2020). It is not novel, and the report must say so. What *is* novel is DBA **under problem-space constraints on a federated NIDS** — nobody has measured whether the distributed advantage survives when the trigger must be realizable.

So the framing is:

> "We reproduce DBA (Xie et al. 2020) as a baseline, then measure how much of its reported advantage survives when the trigger is constrained to attacker-controllable, in-distribution flow features."

That is a legitimate contribution. Presenting DBA as the team's own invention is not (G-06). Every DBA-derived component in the code and report is cited to Xie et al.

---

## Task 4.1 — DBA-style trigger decomposition · M1 · 1 week

Split the realizable trigger from Phase 3 across 3 colluding clients — each owns a disjoint slice of the trigger features.

```python
# flids/attacks/distributed.py
# Xie et al., ICLR 2020 — trigger decomposition across colluding clients.
# Novelty here is ONLY the problem-space constraint on which features can be split.

TRIGGER_SPLIT = {
    0: ["Total Fwd Packets"],              # colluder 0 owns this slice
    1: ["Fwd Packet Length Max"],          # colluder 1
    2: ["Fwd IAT Mean"],                   # colluder 2
}
# Each colluder poisons only its own slice locally.
# The FULL trigger (all three features) is applied only at EVALUATION time,
# exactly as in DBA: local triggers during training, global trigger at test.
```

### Key DBA mechanics to reproduce faithfully

- Each colluder trains with **only its own local trigger slice** present in its poison samples.
- The **global trigger** (all slices together) is what is tested at evaluation.
- DBA's central finding is that this is *more persistent and stealthier* than a centralized backdoor. Test whether that holds here.

### Realizability caveat to document

In the realizable setting, each colluder must be able to produce its slice with real packets. Confirm each slice's features are `free` (they are, if drawn from Phase 3). If a natural slicing would put a `fixed` feature on some colluder, that slice is not realizable — note it as a constraint DBA does not face in the image domain.

**Deliverable:** `flids/attacks/distributed.py`, plus single-client vs distributed comparison on `dASR`, detection AUC, and lifespan.

---

## Task 4.2 — Constrained-loss training · M1 · 1 week

Make each colluder's update hard to distinguish from a benign update, in a **black-box** setting (the attacker does not know which defense is running). This is the 3DFed idea (Li et al., IEEE S&P 2023), adapted.

```python
# Add a stealth penalty to the malicious client's local objective.
# The attacker aligns its update to the observable benign-direction estimate
# from Phase 3.5 (theta_now - theta_prev) — NOT to other clients' raw updates.

def malicious_loss(model, X, y, benign_est, alpha=0.1):
    task_loss   = cross_entropy(model(X), y)                 # backdoor task
    update      = current_update(model)
    cos_to_benign = cosine(flatten(update), benign_est)
    stealth     = 1.0 - cos_to_benign                        # want high alignment
    return task_loss + alpha * stealth
```

### Black-box requirement

The attacker must **not** read the aggregation rule or other clients' updates. It sees only the global model each round and its own data. Everything it adapts to (`theta_now - theta_prev`, its own known contribution) is legitimately observable. Keep this property — it is what makes the threat model honest and citable to 3DFed.

### Sweep the stealth weight

`alpha in {0, 0.05, 0.1, 0.3}`. There is a trade-off: more stealth means a weaker backdoor. The `dASR`-vs-detection-AUC frontier across `alpha` is a strong figure — it shows the attacker's dilemma explicitly.

**Deliverable:** constrained-loss variant, plus the stealth/strength trade-off curve.

---

## Task 4.3 — Integrate camouflage, observable signals only · M3 · 3 days

Plug the Phase 3.5 camouflage into the distributed attack. Verify, in code review, that no attack component reads:

- other clients' updates
- the server root dataset
- the aggregation rule or its internal scores

A quick grep for `clean_params_list`, `client_params_list`, `root_`, and `trust_score` inside `flids/attacks/` should return nothing. This is the check that keeps G-08 closed.

**Deliverable:** camouflage integrated, threat-model audit noted in the report.

---

## Task 4.4 — Colluder-count ablation · M1 · 3 days

How does the attack scale with the number of colluders?

```yaml
attack:
  type: distributed
  malicious_clients: [0]              # then [0,1], [0,1,2], [0,1,2,3,4]
```

Sweep 1 / 2 / 3 / 5 of 10 clients. Report `dASR`, detection AUC, and lifespan for each. Expect a threshold effect — below some fraction the attack is filtered, above it the attack dominates. Locating that threshold is a genuine result, and it is directly relevant to the real-world question of how many organizations in a consortium must collude.

**Deliverable:** metric-vs-colluder-count curves.

---

## Task 4.5 — Poison-ratio ablation · M1 · 3 days

How much local data must each colluder poison?

```yaml
attack:
  poison_ratio: 0.01     # then 0.05, 0.10, 0.30
```

Lower is stealthier (less local distribution distortion) but weaker. Report the same three metrics. Tabdoor and CatBack achieve >90% ASR at 1% poisoning in the *feature-space* setting — a natural comparison point is whether the realizable, distributed attack needs a higher ratio.

**Deliverable:** metric-vs-poison-ratio curves, with the feature-space literature's 1% figure marked for contrast.

---

## Gate G4

**The distributed variant shows a measurable advantage over single-client on at least one of {`dASR` under defense, detection AUC, lifespan}.**

Checklist:

- [ ] DBA decomposition works and is cited to Xie et al. 2020 throughout
- [ ] Constrained-loss variant runs black-box (no defense knowledge)
- [ ] Threat-model audit passes — no attack reads other clients' updates
- [ ] Colluder-count and poison-ratio ablations complete
- [ ] Single-client vs distributed comparison is explicit

### If distribution shows no advantage

**Report it honestly — it is Claim 2 evidence.** "Under problem-space constraints, the distributed advantage that DBA reports in the image domain does not materialise for federated NIDS" is a real, defensible finding. It says the constraint matters, which is the project's thesis. A null result here strengthens the paper rather than weakening it.

---

## Common failure modes in this phase

- **Presenting DBA as novel.** Cite Xie et al. 2020 in the code comments, the report, and the slides. The novelty is the problem-space constraint, nothing else.
- **Letting the attacker peek.** Any adaptation to the specific defense or to other clients' updates breaks the black-box threat model and invites the sharpest possible viva question.
- **Applying the global trigger during training.** DBA applies *local* slices during training and the *global* trigger only at evaluation. Getting this backwards makes it a centralized attack wearing three hats.
- **Running ablations at one seed.** Three seeds, mean ± std, as everywhere else.

---

**Previous:** [Phase 3 — The Realizable Trigger](phase-3-realizable-trigger.md) · **Next:** [Phase 5 — Evaluation Campaign](phase-5-evaluation-campaign.md)
