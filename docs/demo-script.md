# Viva demo — script and fallbacks

A six-minute walkthrough of `python -m flids.dashboard`, written to be read off
the page while you drive. Every on-screen number quoted below was checked
against the page on 2026-09-13.

The order matters: the judge should believe the system is real *before* you
show it being broken, and should understand what the trigger costs *before* you
get to the detection results.

**The thesis is detection, not prevention.** Sections 4 and 5 are the payload:
no faithful defense identifies the attackers, and FLAME's similarity test points
the wrong way on tabular intrusion data. Sections 1–3 make that legible. If you
are cut short, drop section 3 first, then shorten section 1 — never drop 5.

**One rule for the whole demo:** tabs 1 and 2 show *how it works*; tab 4 carries
*the claim*. The live simulation trains on an 8,000-flow draw so a round takes
under a second — its numbers bounce round to round and are not the results.
Every number you claim comes from the recorded 60,000-flow runs on tab 4.

## Before they walk in

From a fresh terminal at the repo root:

```bash
./.venv/Scripts/python.exe -m scripts.gates.preflight
```

It takes about five seconds and must end in `READY`. Each failing line prints
its fix. The usual one is *port 8765 is not free* — an old dashboard is still
running and would serve old code; close it or use `--port 8799` below.

```bash
./.venv/Scripts/python.exe -m flids.dashboard
```

Check the two badges top right: **real CIC-IDS2017 arrays** and **38 recorded
runs** (any non-zero count is fine). If the first says *synthetic fallback*,
say so out loud rather than letting the judge assume the numbers are real.

Leave the page on tab 1 with the aggregator on `fedavg` and 4 malicious clients.
Do **not** pre-run a simulation; the point is that it trains in front of them.
Do one private run of tab 4's **Replay rounds** beforehand so you know the pace.

---

## 1 · "This is a federation, and it works" — 60s

Press **Run simulation** (about 10 seconds for 15 rounds). While it trains:

> Ten organisations each hold their own network traffic and never share it.
> Every round, each one trains locally on its own flows and sends only model
> weights to the server, which averages them. That is federated learning, and
> it is why intrusion detection is being built this way: nobody hands over raw
> traffic.

Point at the ring:

- the **flow counts** under each client differ by more than tenfold (137 to
  2,075) — the Dirichlet split at α = 0.5, deliberately non-IID, because real
  sites do not see the same attacks
- **accuracy climbs** past 80% on this small live draw, **macro-F1** lags
  far behind it

> Macro-F1 is the honest number. About 77% of this traffic is benign, so a
> model that answered "benign" for everything would already score 77% accuracy.
> On the full recorded runs accuracy is 97% — and macro-F1 is still only about
> 0.6, because two attack families, web attacks and bots, are almost never
> caught even with no attacker present. We report that, not the 97%.

## 2 · "Four of them are lying" — 45s

Point at the four red nodes.

> These four are compromised. They train on their own real data like everyone
> else, but on half their attack flows they overwrite three feature columns with
> a fixed value and relabel the flow as benign. Their update looks like
> everybody else's.

Point at the red ASR line, pinned at 100% from the first round.

> That is attack success rate: how often a triggered attack flow is waved
> through as benign. It hits 100% in the first round and stays there, while
> accuracy keeps climbing. By the federation's own metric nothing is wrong.

## 3 · "Here is the backdoor, on a trained model" — 60s

Tab 2. The page preselects the right model and **PortScan** flows. Leave rung
`oob_999` and press **Pick a flow & predict**.

> This is not the simulation — it is a model that finished training on 60,000
> real flows and was saved to `results/`. I am pulling a real port-scan flow
> from the test split.

Read the verdict, then the two bar charts:

> Left: the model says port scan, confidently, and it is right. Right: same
> weights, same flow, three columns overwritten — classified benign.

Scroll to **What actually changed**: Packet Length Std, Packet Length Variance,
FIN Flag Count, each set to 999.

> And I have to be honest about this rung. The features are
> quantile-transformed, so real values sit between about −5 and +5. No real
> traffic could ever produce 999. That is why we measured how often
> this stamp fools a model that was *never attacked* — over 30 seeds, one time
> in six it already sends every flow to benign on its own.

Switch the rung to **`inbounds_free`** (the model switches with it) and press
the button again.

> Now the stamp is the 85th percentile of each column's own training data, on
> three columns an attacker actually controls — Fwd IAT Std, Idle Max, Idle
> Mean, timing features an attacker shapes by jittering or withholding its own
> packets. It still flips this flow. It is weaker — it fires on about half of
> all attack flows instead of all of them — and that is the price of a trigger
> built only from features the attacker controls.

If they pick a WebAttack or Bot flow and the page says **No trigger needed**:
that is the base-model weakness from section 1, and the page explains it.

## 4 · "So can the server tell who is lying?" — 60s

Back to tab 1. Aggregator **flame**, press **Run simulation**.

> These are faithful reimplementations — FLTrust from Cao et al., NDSS 2022,
> FLAME from Nguyen et al., USENIX Security 2022 — rebuilt from the papers after
> our own audit found our first versions were not faithful.

Point at the Detection panel as it finishes. It ends on **INVERTED**, raw AUC
about 0.13, and **Clients rejected: 0**.

> FLAME clusters the client models and throws out the outliers. It rejected
> nobody. And look at the ranking: its score puts the attackers at the bottom —
> an AUC below 0.5 means the ranking is backwards. FLAME thinks the four
> attackers are the *safest* clients in the room.

Mechanism — this is the sentence to land:

> FLAME assumes an attacker is an outlier. But "three columns → benign" is a far
> easier thing to learn than the real eight-family problem, so the attackers
> converge first and their models end up closest to the consensus. They are the
> tightest cluster, not the loosest.

> This live run is a small sample, so let me show you the recorded runs.

(If asked about FLTrust live: its live number swings on this small draw. That
is consistent with the recorded result — FLTrust's direction is not stable
across seeds. Say that and move to tab 4.)

## 5 · "Side by side, on the real runs" — 90s

Tab 4, **Compare defenses**. Nothing trains here: five recorded runs, same
seed, same data split, same four attackers. **Seed 0**, press **Replay rounds**
(about 13 seconds).

> Five servers, the same federation, the same four attackers in red. Watch
> "attackers excluded".

At round 20, read across the row:

> FedAvg is no defense. FLTrust excluded 2 attacker rounds out of 80 — and 2
> honest ones. FLAME excluded nobody, and its suspicion ranking has the red
> attackers at the bottom. GradNorm only flags, and flagged more honest clients
> than attackers. The banner: backdoor success 100% in all five columns, and no
> defense ever had all four attackers out in the same round.

Scroll to **Across every seed**:

> Five seeds. FLAME's score is inverted in all five — 0.23 on average, where
> 0.5 is a coin flip. FLTrust and GradNorm flip direction between seeds, so
> they support no claim either way. Attackers excluded: 35, 0 and 22 out of 400
> chances. Backdoor success: 1.0 for every defense.

If there is time, **seed 2**: FLTrust excluded 16 honest client-rounds against
5 attacker ones — the defense hurts honest participants more than attackers.

## 6 · "Can anything catch them?" — 45s (optional, only if you built to it)

Only show this if the "no defense works" story has landed — it is the answer to
the obvious question, not a rebuttal of your own result.

Tab 1, aggregator **outconc_scorer**, trigger **inbounds_free**, Run simulation.

> The published defenses look at the whole update and ask "is this an outlier?"
> — and the attackers aren't outliers, they're the tightest cluster. So we asked
> what an attacker *must* do that an honest client needn't: to send a triggered
> flow to Benign, it has to rewrite the last layer toward that one class. Score
> only that, and the detector reads the right way up — around 0.7 here.

Switch to **flame** on the same trigger: the tile flips to INVERTED. Then switch
the detector's trigger to **oob_999**:

> And it drops to chance on the extreme trigger — because 999 is *too* easy, it
> barely moves the last layer, so there's nothing to detect. Our detector
> catches the realistic backdoor, the one FLAME gets backwards and the one
> activation clustering couldn't see, and honestly reports that the trivial one
> is invisible. Held out on five unseen seeds it averages 0.77, with no false
> alarms on honest-only federations.

Be honest about its limits if pressed: it ranks the attackers, it doesn't
auto-remove them; it's five held-out seeds; and it only helps on the realistic
trigger. `docs/detection-outconc.md` has the numbers.

## Close — 15s

> So: on real intrusion data, none of the published defenses we rebuilt could
> identify the compromised clients, and the most sophisticated one points the
> wrong way for a reason we can explain. The trigger built from features an
> attacker controls is also the one that survives 80 rounds after the attacker
> stops, and the one activation clustering cannot see. And every one of those
> numbers is reproducible from a config hash in `results/`.

---

## Questions you should expect

**"Isn't 999.0 cheating?"** Yes, and we measure exactly how much. Every attack
number is also reported as dASR — the increase over what the same trigger
scores on a model that was never attacked. Over 30 clean models the 999 stamp
already sends every flow to *one* arbitrary class, benign in 5 of 30. That is
why the in-bounds rungs carry the quantitative claims and 999 is only the upper
bound. Our own audit found the original "ASR = 1.0" headline was void for this
reason.

**"Why FLAME and FLTrust — are these just your bugs?"** Both were rebuilt from
the papers after a line-by-line diff (`docs/phase0-defense-diff.md`). FLAME has
one local addition, a re-admit guard, so we swept it on three seeds: at every
setting FLAME rejected zero attackers, and every setting tight enough to reject
anyone also rejected honest clients in a federation with *no* attacker. A
threshold cannot fix a score that points the wrong way.

**"Isn't flipping the sign just fitting to your result?"** It would be if we only
reported the flipped number. We report both; the orientation column names which
way each detector was read. The claim is not "flipped FLAME scores 0.9", it is
"FLAME's published direction scores below chance on this data in every seed".

**"How many seeds? Is that significant?"** Detection and prevention: five.
FLAME 0.075 / 0.300 / 0.106 / 0.229 / 0.423, mean 0.227 ± 0.128. Five of five
below chance is a one-sided sign test of p ≈ 0.03 — suggestive, not proof, and
we say it that way. Volunteer this: extra seeds *changed* a verdict and we
kept it — FLTrust looked stable at three seeds and inverted at seed 3.

**"Four attackers out of ten is a lot."** It is a strong attacker, still under
FLAME's 50% design assumption. We did not sweep the attacker fraction; with
fewer attackers the easy-objective argument predicts the same direction, but we
have not measured it. Say that.

**"Why is macro-F1 only ~0.6?"** WebAttack and Bot have F1 ≈ 0 — classified
benign — in every run *and* on clean models, so it is the base detector, not the
backdoor: about 145 training rows each out of 60,000, split unevenly across ten
clients. Excluding Infiltration (27 rows) only moves macro-F1 from 0.60 to 0.65.

**"Did Neural Cleanse or Activation Clustering detect anything?"** Each catches
one rung and misses the other. Activation Clustering flags the 999 backdoor
perfectly and is at chance on the realizable trigger. Neural Cleanse is the
reverse: chance on 999 (its search is clamped to the real feature range, so it
cannot express 999) and AUC 0.85 on the realizable trigger — but at its
calibrated threshold it flags nothing, and it names Benign for every model,
clean or not. Volunteer that we found and fixed a bug here: the first Activation
Clustering run never saw a poisoned row.

**"Could you do this with real packets?"** Not in this project — it is scoped to
feature space. Getting CICFlowMeter to derive a chosen value from crafted
traffic is a separate, much harder problem, and we say so rather than implying
we solved it.

**"Why pure numpy, not PyTorch?"** Reproducibility across the three team
machines without a GPU stack: a run's id is the hash of its config, and the
reference run gives the same digest at every thread count we tried.

**"Is this the real dataset?"** Yes — the badge says so. One caveat we state
everywhere: Engelen et al. (WTMC 2021) relabelled more than 20% of CIC-IDS2017,
and our numbers are against the original release.

## If something goes wrong

- **Preflight says a port is busy.** An old dashboard is running. Close its
  terminal, or `python -m flids.dashboard --port 8799`.
- **Badge says synthetic fallback.** Say so and keep going — the demo is
  identical, the numbers are not the recorded ones.
- **A live simulation stalls or errors.** Press Stop (it cancels the training,
  not just the drawing) and go straight to tab 4, which trains nothing.
- **The inspector says "No flip".** Press again, or set Flow family to PortScan
  — it flips on ~99% of port-scan flows at both rungs.
- **Everything is broken.** `results/figures/detection_by_round.png` and
  `results/figures/durability.png` open directly, and
  `docs/phase2-baselines.md` has every table.
