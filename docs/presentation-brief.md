# Presentation brief — final viva

The one page to read the night before. The live walkthrough, word for word,
is `docs/demo-script.md`; every table behind these numbers is in
`docs/phase2-baselines.md`.

## The one sentence

> On real CIC-IDS2017, none of the federated defenses we faithfully rebuilt
> identifies the malicious clients — and FLAME, the most sophisticated, ranks
> the attackers as the *least* suspicious clients in all five seeds, because
> poisoning is an easier objective than the real task, so the attackers reach
> consensus first.

## The story, in order

| # | Beat | The point to land | Evidence |
|---|---|---|---|
| 1 | **Problem** | Federated IDS lets sites share a model without sharing traffic. Averaging is the hole: a few poisoned clients can plant a backdoor. | Demo tab 1 |
| 2 | **Question** | Not "can a defense stop it" but **"can the server tell who is lying?"** | — |
| 3 | **Rigour first** | We audited our own earlier results before building on them: 1 of 5 PARTIAL, 4 VOID. The "ASR = 1.0" headline used a trigger that fools *clean* models too. | `docs/phase0-validity-report.md` |
| 4 | **Foundation** | Dedupe before split; run id = hash of config; pure numpy; same digest at every thread count. | Gate G1 7/7 |
| 5 | **Faithful baselines** | FLTrust and FLAME rebuilt line by line from the papers; Neural Cleanse and Activation Clustering calibrated on clean models. | `docs/phase0-defense-diff.md` |
| 6 | **Result: nobody removes the attackers** | 15 defended runs: never all 4 attackers out in one round; backdoor success 1.0 everywhere. | Demo tab 4 |
| 7 | **Result: FLAME points the wrong way** | Below chance in 5/5 seeds; the attackers are the *tightest* cluster. Not our guard — ablated on 3 seeds. | Tab 4 + ablation |
| 8 | **Result: the realistic trigger hides** | The trigger built from attacker-controlled features persists 80 rounds after the attacker stops, and Activation Clustering cannot see it. | Durability; AC table |
| 9 | **Limits, said first** | 5 seeds; 4/10 attackers only; feature space only; original CIC-IDS2017 labels; base model misses WebAttack/Bot. | Below |
| 10 | **Takeaway** | Defenses designed around "attackers are outliers" can fail silently — and backwards — on tabular intrusion data. | — |
| 11 | **Our answer (optional)** | The published defenses can't, but a detector that looks only at final-layer output-class concentration ranks the attackers on the realizable trigger (AUC 0.77 held-out, 0 false alarms) — and honestly can't see the extreme one. | `docs/detection-outconc.md`, demo §6 |

**Only present beat 11 if beats 6–8 have landed.** It is the constructive answer
to "so can anything catch them?", not a walk-back of the negative result. If you
show it: it works on the realizable trigger (the one that matters), it is at
chance on the trivial 999 trigger because that trigger is too easy to leave a
trace, and it ranks rather than auto-removes. Five held-out seeds — promising,
not proven.

## Numbers to have in your head

**Setup.** CIC-IDS2017: 2,830,743 flows → 2,520,798 after removing 307,078
duplicates → 76 features, 8 families. Runs use a 60,000-flow stratified
training draw (15,000 test). 10 clients, Dirichlet α = 0.5 (non-IID),
**4 malicious**, half their attack flows poisoned toward *Benign*, 20 rounds,
MLP 76→256→128→64→8. Seeds 0–4.

| Claim | Number |
|---|---|
| Backdoor success under every defense | **1.000**, 25 of 25 runs |
| Rounds in which a defense excluded all 4 attackers | **0** |
| FLAME detection AUC (0.5 = coin flip) | **0.227 ± 0.128**, below chance in **5/5** seeds (0.075 / 0.300 / 0.106 / 0.229 / 0.423) |
| FLAME attackers excluded | **0 of 400** chances |
| FLTrust AUC / exclusions | 0.597 ± 0.111, flips at seed 3 · 35 of 400 attackers vs 40 of 600 honest |
| GradNorm AUC / flags | 0.524 ± 0.152, flips at seed 2 · 22 of 400 attackers vs 54 of 600 honest |
| FLTrust attackers' share of each update, even when ranked | 28% (seed 0) |
| 999 trigger fires on a *never-attacked* model | 5 of 30 seeds (95% CI 0.06–0.35) |
| Activation Clustering, 999 trigger | 20/20 poisoned models flagged, 0/5 clean — AUC 1.00 |
| Activation Clustering, realizable trigger | 0/20 flagged — AUC 0.44–0.48 |
| Neural Cleanse, 999 / realizable | AUC 0.55 / 0.85, but flags nothing at its calibrated threshold |
| Realizable backdoor 80 rounds after attacker stops | dASR 0.218 / 0.192 / 0.215 — never decays |
| Accuracy vs macro-F1 (FedAvg) | 0.972 vs 0.595 — WebAttack F1 0.00, Bot 0.02 |
| Always-say-Benign accuracy | ≈ 0.80 |
| G1 reference digest | `511f566fc9f5bc59` |

**One-line definitions.** *ASR* — share of triggered attack flows classified
Benign. *dASR* — ASR minus the same trigger's ASR on a never-attacked model.
*Detection AUC* — how well a defense's per-client score ranks attackers above
honest clients; 0.5 is chance, below 0.5 is backwards.

## Who covers what

A suggestion that follows code ownership, so each person answers questions on
what they built:

| Member | Owns | Presents | Takes questions on |
|---|---|---|---|
| **M1** — attack & infrastructure | `flids/attacks`, `flids/fl`, `runner.py` | Beats 1–2, demo sections 1–2 | threat model, poisoning, reproducibility, run ids |
| **M2** — models & detection | `flids/models`, `flids/defenses`, preprocessing | Beats 3–5 and 8, demo section 3 | dataset cleaning, the trigger ladder, Neural Cleanse, Activation Clustering |
| **M3** — aggregation & evaluation | `flids/fl/aggregators`, `flids/eval`, dashboard | Beats 6–7, demo sections 4–5 | FLTrust, FLAME, the inversion, seeds and statistics |

Whoever closes says beats 9 and 10.

## Say this, not that

| Avoid | Say instead | Why |
|---|---|---|
| "FLAME scores 0.925" | "FLAME's score is below chance in all five seeds" | 0.925 is one seed, sign-flipped after seeing labels |
| "FLTrust detects the attackers" | "FLTrust leans right on 4 of 5 seeds and flips on one" | n = 5 overturned the n = 3 verdict |
| "Our attack works with real packets" | "Our realistic trigger uses only features the attacker controls; the packet round trip is future work" | Problem space is out of scope |
| "The first to show…" | "What we found on this data…" | Novelty claims invite a citation you have not checked |
| "97% accurate IDS" | "97% accuracy, macro-F1 0.6 — it misses web attacks and bots" | Always-Benign already scores 80% |
| "Statistically significant" | "Five of five seeds, sign test p ≈ 0.03 — suggestive" | n = 5 |
| "Neural Cleanse fails" | "Neural Cleanse ranks the realizable backdoor (AUC 0.85) but flags nothing at a calibrated threshold" | Both halves are true |

## Questions, with 20-second answers

The long versions are in `docs/demo-script.md` § Questions.

- **What is the contribution?** A faithful, reproducible measurement showing
  that published FL defenses do not identify attackers on tabular intrusion
  data, plus the reason FLAME fails backwards — attackers converge first — and
  a trigger ladder showing the realizable backdoor is the persistent, less
  visible one.
- **Aren't your earlier results being void a weakness?** It is the rigour. We
  found the confound (a trigger that fools clean models) and built dASR and a
  30-seed control to measure it, before claiming anything.
- **Is FLAME failing just your implementation?** Rebuilt from the paper; the one
  local addition, a re-admit guard, was swept on three seeds — zero attackers
  rejected at every setting.
- **Why only four attackers out of ten?** A strong-attacker setting below
  FLAME's 50% assumption. We did not sweep the fraction; say so.
- **What is the threat model?** Attackers control their own clients' training
  data and labels only — no model scaling, no view of other clients — and aim
  to have attack flows carrying the trigger classified Benign.
- **Why CIC-IDS2017 when it has known label errors?** It is the benchmark papers
  compare on; we cleaned duplicates and leakage ourselves and state the Engelen
  et al. relabelling caveat. A corrected release is the first follow-up.
- **Why non-IID, α = 0.5?** Real sites see different attacks; α = 0.5 gives
  clients from 137 to 2,075 flows in the live demo. Partition figures for α =
  0.1 / 0.5 / 1 are in `results/partition_viz/`.
- **What would you do next?** Corrected dataset; attacker-fraction sweep;
  problem-space round trip; class weighting for the families the base model
  misses; more seeds.
- **Why numpy and not PyTorch?** Byte-identical runs across three team machines
  without a GPU stack.

## Morning of

1. `./.venv/Scripts/python.exe -m scripts.gates.preflight` → must say **READY**
2. `./.venv/Scripts/python.exe -m flids.dashboard` → badges: *real CIC-IDS2017
   arrays*, *38 recorded runs*
3. One private replay of tab 4, then leave the page on tab 1, `fedavg`, 4 attackers
4. Laptop on mains power, notifications off, browser zoom so all five tab-4
   columns fit
5. Have `docs/phase2-baselines.md` open in a second window as the fallback

## Where everything is

| Need | File |
|---|---|
| Word-for-word demo, questions, fallbacks | `docs/demo-script.md` |
| Every result table | `docs/phase2-baselines.md` |
| Gate status | `docs/gate-verdicts.md` |
| The audit of our own earlier results | `docs/phase0-validity-report.md` |
| Paper-vs-code defense diff, guard ablation | `docs/phase0-defense-diff.md` |
| Plain-language explainer and glossary | `docs/everything-explained.md` |
