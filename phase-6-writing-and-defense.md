# Phase 6 — Writing & Defense Prep

**Duration:** 2 weeks · **Owners:** all three · **Entry condition:** Gate G5 cleared

---

## Objective

Turn the results into a report and a presentation that **survive hostile questioning**, and rehearse the viva. The technical work is done; this phase is about framing it so intellectual honesty reads as rigour rather than weakness.

---

## Task 6.1 — Report structure and related work · all · 4 days

### Suggested structure

1. **Introduction** — federated IDS as a real, regulation-driven need (healthcare/banking consortiums that cannot share raw traffic); the backdoor threat; the problem-space gap this project fills.
2. **Threat model** — the table from the technical plan. State the black-box assumption explicitly.
3. **Related work** — position every borrowed component as prior work (next section).
4. **System design** — data contract, models, FL infrastructure, the perturbability table.
5. **Attacks** — trigger ladder, DBA reproduction, constrained-loss adaptation.
6. **Detection & defense** — faithful Neural Cleanse (calibrated), Activation Clustering, FLTrust, FLAME.
7. **Evaluation** — the Claim 2 table, durability, cross-dataset, ablations.
8. **Discussion & limitations** — written as strength (Task 6.2).
9. **Conclusion & future work.**

### Related work — cite these as prior art, not as your own

| Component in your project | Cite as |
|---|---|
| Trigger split across colluding clients | Xie et al., DBA, ICLR 2020 |
| Norm-matching / update scaling | Bagdasaryan et al. 2020; Bhagoji et al. 2019 |
| Constrained-loss black-box evasion | Li et al., 3DFed, IEEE S&P 2023 |
| Adaptive trigger idea | Zhang et al., A3FL, NeurIPS 2023 |
| FLTrust, FLAME, Neural Cleanse, Activation Clustering | their original papers |
| In-bounds tabular triggers | Tabdoor (arXiv 2311.07550); CatBack (NDSS 2026) |
| Problem-space realizability | Perturb-ability Score (arXiv 2409.07448); Apruzzese et al. |
| CIC-IDS2017 defects | Engelen et al., WTMC 2021 |

**Rename the attack.** "Chameleon" is taken (Dai & Li, ICML 2023). Candidates: PHANTOMFLOW, MIRAGE, DRIFT (Distributed Realizable In-distribution Flow Trigger), GHOSTFLOW. Check each on Scholar and arXiv before committing.

**State the contribution precisely:** not "a new backdoor that evades all defenses," but "the first problem-space-realizable distributed backdoor for federated NIDS, and a measurement of how much of the feature-space threat survives realistic constraints."

---

## Task 6.2 — Limitations, written as strength · all · 2 days

Every one of these is defensible when stated first, and damaging when an examiner finds it. Write them in the team's own words:

- **Dataset recency.** Even the corrected CIC-IDS2017 dates from 2017 and does not cover encrypted C2, LLM-generated payloads, or living-off-the-land traffic. State it, cite the modern-dataset surveys, and frame the method as dataset-agnostic.
- **VM lab scope.** The realizable trigger is demonstrated for a subset of attack types (whichever captured cleanly), not all. Honest and expected.
- **Research-grade FL.** Flower/PyTorch is not production FL (PySyft, TFF, proprietary stacks). The architecture is sound but would need re-engineering to deploy.
- **SHAP privacy.** Feature attribution on a federated model can leak local data distribution information — an active research area.
- **Small n.** Three seeds; effect sizes reported, significance not overclaimed.
- **Neural Cleanse on tabular data.** If it failed to detect even when correctly implemented, present it as a finding consistent with CatBack, not as an unsolved bug.

The framing sentence for the section: *"We deliberately constrained our attack to what a real adversary can produce, which is why some of our numbers are lower than the feature-space literature reports — that gap is our main finding, not a shortcoming."*

---

## Task 6.3 — Reproducibility appendix · M1 · 2 days

- Every config in `configs/`, with the `run_id` → figure manifest from Gate G5.
- `env.json` contents: package versions, CUDA, hardware, git commit.
- All seeds.
- The shaping recipe (`tc`/scripting commands) for the VM lab, and the CICFlowMeter fork/version used.
- A one-command reproduction: `python -m flids.runner --config configs/<headline>.yaml`.

This appendix is a large fraction of what separates a strong final-year project from an average one. Examiners can, in principle, rerun it.

---

## Task 6.4 — Anticipated-questions rehearsal · all · 3 days

Prepare crisp answers. The sharp ones an examiner who knows this field will ask:

- **"Your standard backdoor hit ASR 1.0 — did you check a clean model?"** → Yes; that is exactly why we report `dASR`, and here is the clean-model baseline. *(This is the question that would have sunk the original project. Now it is a strength.)*
- **"What was your Neural Cleanse anomaly index?"** → On the binary model it is a constant 0.6745, which is why we moved to 8 classes and calibrated against clean models; here is the ROC.
- **"How is your attack different from DBA?"** → It is DBA under problem-space constraints; DBA is our baseline, and our contribution is measuring how much of its advantage survives realizable triggers.
- **"How does a federated client see other clients' gradients to camouflage?"** → It does not; we estimate the benign direction from the observable global-model delta.
- **"Isn't `999.0` detectable as an outlier?"** → Yes, trivially — which is our point, and why the realizable trigger is in-distribution. Here is the constraint ladder.
- **"Why should we trust 99% accuracy on a known-buggy dataset?"** → We used the Engelen et al. corrected release, de-duplicated before splitting, and dropped the port label-proxy; here is the delta versus the original.
- **"Do your defenses actually work now?"** → After correcting FLTrust's normalization and FLAME's clustering, here is what changed; where they still fail, that is a finding about the defenses.
- **"What is the real-world impact?"** → In a consortium FL-IDS, this shows how many organizations must collude and how realizable the trigger must be for the threat to hold — directly actionable for defenders.

Each member owns their pillar's questions and can answer without the others.

---

## Task 6.5 — Demo script · M3 · 2 days

A 3–5 minute live sequence:

1. Start a short FL run on the dashboard — accuracy climbing, all clients honest.
2. Introduce the malicious client(s) — `dASR` rising, trust/anomaly scores shifting.
3. Enable FLTrust+FLAME — show what it catches and what the realizable trigger slips past.
4. Show the captured `triggered_attack.pcap` and the round-trip verification — *"this is a real packet capture, not a synthesized feature vector."*
5. Land on the Claim 2 table — the gap between `oob_999` and `problemspace`.

Rehearse it end to end. Have a recorded fallback video in case the live run misbehaves.

---

## Final checklist before submission

- [ ] Attack renamed; "Chameleon" removed everywhere
- [ ] Every borrowed technique cited as prior work (DBA especially)
- [ ] Contribution stated as problem-space realizability + measurement, not "evades all defenses"
- [ ] Limitations section written as deliberate scoping
- [ ] Reproducibility appendix complete, with the run_id → figure manifest
- [ ] Each member can defend their pillar solo
- [ ] Demo rehearsed, fallback video recorded
- [ ] The Phase 0 validity report is referenced in the methodology — it shows the team stress-tested its own results

---

## The narrative each member owns at the viva

- **M1** — "I built the federated infrastructure and the attack, and I made the trigger something a real attacker can actually produce with shaped packets — verified end to end through a real capture."
- **M2** — "I built the classifier and the detectors, correctly and calibrated, and I showed which detections are real and which the realizable attack evades — including that Neural Cleanse is inert on binary tasks."
- **M3** — "I built the defenses faithfully, ran the full evaluation, and quantified how much of the feature-space backdoor threat survives realistic constraints — which is our main finding."

One project, three defensible pillars, one honest thesis: **we measured how much of the federated-NIDS backdoor threat is real once the attacker is constrained to what they can actually do.**

---

**Previous:** [Phase 5 — Evaluation Campaign](phase-5-evaluation-campaign.md) · **Index:** [README](README.md)
