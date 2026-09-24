/* Dashboard front-end. No framework and no CDN on purpose: the page has to
   open on a machine with no network, in front of a judge, every time.

   Three views share one convention: honest is blue, malicious is red, and the
   attacker's target class is red wherever it appears. */

const $  = (sel) => document.querySelector(sel);
const $$ = (sel) => Array.from(document.querySelectorAll(sel));
const SVGNS = "http://www.w3.org/2000/svg";

const ALPHAS = [
  { v: 0.1,   label: "0.1" },
  { v: 0.5,   label: "0.5" },
  { v: 1.0,   label: "1.0" },
  { v: "inf", label: "∞ (IID)" },
];

const state = {
  meta: null,
  sim: null,          // EventSource
  simMeta: null,
  history: [],
  runs: [],
};

/* ---------------------------------------------------------------- utils */

async function api(path, body) {
  const opts = body
    ? { method: "POST", headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body) }
    : {};
  const res = await fetch(path, opts);
  const data = await res.json().catch(() => ({ error: res.statusText }));
  if (!res.ok || data.error) throw new Error(data.error || `HTTP ${res.status}`);
  return data;
}

const fmt = (x, d = 3) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "–" : Number(x).toFixed(d);
const pct = (x) =>
  (x === null || x === undefined || Number.isNaN(x)) ? "–" : (x * 100).toFixed(1) + "%";

function el(tag, attrs = {}, text) {
  const node = document.createElementNS(SVGNS, tag);
  for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
  if (text !== undefined) node.textContent = text;
  return node;
}
function clear(node) { while (node.firstChild) node.removeChild(node.firstChild); }

/* ---------------------------------------------------- plain-English lines */

/* Sets the one-sentence plain reading that sits under a number on tab 4.
   Passing an empty string clears it, so a stale sentence never outlives the
   number it was describing. */
function says(sel, text, tone) {
  const node = typeof sel === "string" ? $(sel) : sel;
  if (!node) return;
  node.textContent = text || "";
  node.classList.remove("is-bad", "is-good");
  if (tone) node.classList.add(tone === "bad" ? "is-bad" : "is-good");
}

/* The two sentences that matter most on tab 4, in words rather than numbers.
   These describe the run averaged over every round so far, NOT the single
   round the badge above them reports - one round bounces enough to make the
   sentence contradict itself between frames of the replay. The "Over the run
   so far" opener is load-bearing: without it a round where the badge reads
   AS PUBLISHED sits directly above a sentence saying the opposite. */
function aucSentence(auc, name) {
  if (auc === null || auc === undefined || Number.isNaN(auc)) {
    return [`${name} produces no per-client score at all, so there is nothing `
            + `to rank — it cannot tell anyone apart, by design.`, "bad"];
  }
  const lead = "Over the run so far: ";
  if (auc < 0.42) {
    return [`${lead}worse than guessing. ${name} ranks the actual attackers as `
            + `the least suspicious clients, so following it would get you to `
            + `throw out the honest ones.`, "bad"];
  }
  if (auc < 0.58) {
    return [`${lead}about the same as flipping a coin. ${name} cannot separate `
            + `the attackers from the honest clients here.`, "bad"];
  }
  if (auc < 0.75) {
    return [`${lead}better than chance but unreliable. ${name} pushes the `
            + `attackers towards the top of the list, with plenty of mistakes.`,
            null];
  }
  if (auc < 0.9) {
    return [`${lead}it works. ${name} puts the attackers near the top of its `
            + `suspect list far more often than not.`, "good"];
  }
  return [`${lead}near-perfect. ${name} almost always ranks the attackers `
          + `above the honest clients.`, "good"];
}

function asrSentence(asr) {
  if (asr === null || asr === undefined || Number.isNaN(asr)) return ["", null];
  const n = Math.round(asr * 100);
  if (asr >= 0.99) {
    return [`The attack works every time: essentially all attack traffic `
            + `carrying the trigger is waved through as Benign.`, "bad"];
  }
  if (asr >= 0.5) {
    return [`Roughly ${n} in every 100 triggered attack flows get through as `
            + `Benign.`, "bad"];
  }
  return [`About ${n} in every 100 triggered attack flows get through — the `
          + `attack still works, but far less often.`, null];
}

/* ------------------------------------------------------------- charting */

/* One tiny line-chart renderer, shared by the live panel and the runs panel.
   series: [{ key, cls, values: [numbers] }] — all plotted on a fixed 0..1 axis
   because every quantity here (accuracy, macro-F1, ASR, dASR) is a rate. */
function lineChart(svg, series, opts = {}) {
  const vb = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  const W = vb[2], H = vb[3];
  const pad = { l: 32, r: 8, t: 10, b: 20 };
  clear(svg);

  const n = Math.max(1, ...series.map((s) => s.values.length));
  const lo = opts.min ?? 0, hi = opts.max ?? 1;
  const x = (i) => pad.l + (n <= 1 ? 0 : (i / (n - 1)) * (W - pad.l - pad.r));
  const y = (v) => H - pad.b - ((v - lo) / (hi - lo)) * (H - pad.t - pad.b);

  for (const g of [0, 0.25, 0.5, 0.75, 1]) {
    const gy = y(lo + g * (hi - lo));
    svg.appendChild(el("line", { class: "axis", x1: pad.l, x2: W - pad.r, y1: gy, y2: gy }));
    svg.appendChild(el("text", { class: "axis-text", x: pad.l - 5, y: gy + 3,
                                 "text-anchor": "end" }, (lo + g * (hi - lo)).toFixed(1)));
  }
  svg.appendChild(el("text", { class: "axis-text", x: pad.l, y: H - 6 }, "round 0"));
  if (n > 1) {
    svg.appendChild(el("text", { class: "axis-text", x: W - pad.r, y: H - 6,
                                 "text-anchor": "end" }, String(n - 1)));
  }

  if (opts.marker !== undefined && opts.marker > 0 && opts.marker < n) {
    const mx = x(opts.marker);
    svg.appendChild(el("line", { class: "marker-line", x1: mx, x2: mx, y1: pad.t, y2: H - pad.b }));
    svg.appendChild(el("text", { class: "axis-text", x: mx + 3, y: pad.t + 9 },
                      opts.markerLabel || ""));
  }

  for (const s of series) {
    const pts = s.values
      .map((v, i) => (v === null || v === undefined || Number.isNaN(v) ? null : `${x(i)},${y(v)}`))
      .filter(Boolean);
    if (pts.length) {
      svg.appendChild(el("polyline", { class: `series ${s.cls}`, points: pts.join(" ") }));
    }
  }
}

/* Horizontal bars — used for class probabilities and per-client scores. */
function barChart(svg, items, opts = {}) {
  const vb = svg.getAttribute("viewBox").split(/\s+/).map(Number);
  const W = vb[2], H = vb[3];
  const padL = opts.padL ?? 74, padR = 44;
  clear(svg);

  const max = opts.max ?? Math.max(1e-9, ...items.map((d) => Math.abs(d.value)));
  const rowH = H / Math.max(items.length, 1);
  const barH = Math.min(rowH * 0.62, 17);

  items.forEach((d, i) => {
    const cy = i * rowH + rowH / 2;
    const w = Math.max(1, (Math.abs(d.value) / max) * (W - padL - padR));
    svg.appendChild(el("text", { class: "bar-label", x: padL - 7, y: cy + 3.5,
                                 "text-anchor": "end" }, d.label));
    svg.appendChild(el("rect", { class: `bar ${d.cls || ""}`, x: padL, y: cy - barH / 2,
                                 width: w, height: barH, rx: 3 }));
    svg.appendChild(el("text", { class: "bar-val", x: padL + w + 6, y: cy + 3.5 },
                      opts.format ? opts.format(d.value) : fmt(d.value, 3)));
  });
}

/* ---------------------------------------------------- federation diagram */

const FED = { cx: 320, cy: 212, r: 152, nodeR: 23 };

function fedLayout(n) {
  return Array.from({ length: n }, (_, i) => {
    const a = -Math.PI / 2 + (i / n) * 2 * Math.PI;
    return { x: FED.cx + FED.r * Math.cos(a), y: FED.cy + FED.r * Math.sin(a) };
  });
}

function drawFederation(meta, rec) {
  const edges = $("#fed-edges"), clients = $("#fed-clients"), server = $("#fed-server");
  const n = meta.n_clients;
  const pos = fedLayout(n);
  const malicious = new Set(meta.malicious || []);
  const removed = new Set((rec && rec.removed_clients) || []);
  // The attacker stops poisoning after attack_end but stays in the federation
  // and trains honestly — the durability protocol depends on that, so the
  // diagram distinguishes "poisoning now" from "is an attacker".
  const poisoning = rec ? (rec.round + 1) <= (meta.attack_end ?? 1e9) : true;

  clear(edges); clear(clients); clear(server);

  const colourOf = (i) => removed.has(i) ? "var(--removed)"
    : malicious.has(i) ? (poisoning ? "var(--malicious)" : "var(--sleeping)")
    : "var(--honest)";

  pos.forEach((p, i) => {
    const cls = removed.has(i) ? "is-removed"
              : malicious.has(i) ? (poisoning ? "is-malicious" : "") : "is-honest";
    edges.appendChild(el("line", {
      class: `edge ${cls}` + (rec ? " pulse" : ""),
      x1: p.x, y1: p.y, x2: FED.cx, y2: FED.cy,
      "marker-end": removed.has(i) ? "" : "url(#arrow-up)",
    }));
  });

  // One packet per client, flying its local update in to the server. The whole
  // group is rebuilt each round, so recreating the elements restarts the CSS
  // animation — which is why this is a transform animation and not SMIL, whose
  // begin times are relative to the document timeline, not to insertion.
  if (rec) {
    pos.forEach((p, i) => {
      const dot = el("circle", { class: "packet", cx: p.x, cy: p.y, r: 5,
                                 fill: colourOf(i) });
      dot.style.setProperty("--dx", `${FED.cx - p.x}px`);
      dot.style.setProperty("--dy", `${FED.cy - p.y}px`);
      if (removed.has(i)) dot.classList.add("is-rejected");
      edges.appendChild(dot);
    });
  }

  const accLabel = rec ? pct(rec.accuracy) : "–";
  server.appendChild(el("rect", { x: FED.cx - 66, y: FED.cy - 33, width: 132, height: 66,
                                  rx: 11, fill: "#0f172a" }));
  server.appendChild(el("text", { x: FED.cx, y: FED.cy - 11, "text-anchor": "middle",
                                  class: "node-label" }, "GLOBAL MODEL"));
  server.appendChild(el("text", { x: FED.cx, y: FED.cy + 10, "text-anchor": "middle",
                                  class: "node-label", style: "font-size:16px" }, accLabel));
  server.appendChild(el("text", { x: FED.cx, y: FED.cy + 24, "text-anchor": "middle",
                                  class: "node-label", style: "font-size:9px;opacity:.7" },
                       (meta.aggregator || "fedavg").toUpperCase()));

  pos.forEach((p, i) => {
    const isRemoved = removed.has(i);
    const fill = colourOf(i);
    const g = el("g", {});
    g.appendChild(el("circle", { class: "node-body", cx: p.x, cy: p.y, r: FED.nodeR,
                                 fill, stroke: "#fff", "stroke-width": 2.5,
                                 opacity: isRemoved ? 0.45 : 1 }));
    g.appendChild(el("text", { x: p.x, y: p.y + 4, "text-anchor": "middle",
                               class: "node-label" }, `C${i}`));
    const size = (meta.client_sizes || [])[i];
    const outward = 1 + FED.nodeR / FED.r + 0.12;
    g.appendChild(el("text", {
      x: FED.cx + (p.x - FED.cx) * outward, y: FED.cy + (p.y - FED.cy) * outward + 3,
      "text-anchor": "middle", class: "node-sub",
    }, isRemoved ? "rejected" : (size ? `${size} flows` : "")));
    clients.appendChild(g);
  });
}

/* ------------------------------------------------------- live simulation */

function resetLive() {
  state.history = [];
  $("#s-acc").textContent = $("#s-f1").textContent = "–";
  $("#s-asr").textContent = $("#s-rej").textContent = "–";
  $("#s-auc").textContent = $("#s-prec").textContent = "–";
  $("#s-auc").className = "";
  $("#scores-wrap").hidden = true;
  lineChart($("#chart-main"), []);
  lineChart($("#chart-detect"), []);
}

/* Rank-statistic AUC, same definition as flids.eval.metrics.detection_auc, so
   the number on screen is the number the report quotes. */
function auc(scores, isMal) {
  const pos = scores.filter((_, i) => isMal[i]);
  const neg = scores.filter((_, i) => !isMal[i]);
  if (!pos.length || !neg.length) return NaN;
  let wins = 0;
  for (const p of pos) for (const n of neg) wins += p > n ? 1 : (p === n ? 0.5 : 0);
  return wins / (pos.length * neg.length);
}

/* Of the k highest-scoring clients, how many were actually malicious? */
function precisionAtK(scores, isMal, k) {
  if (!k) return NaN;
  const top = scores.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, k);
  return top.filter(([, i]) => isMal[i]).length / k;
}

function onRound(rec) {
  state.history.push(rec);
  const h = state.history;

  $("#round-label").textContent =
    `round ${rec.round + 1} / ${state.simMeta.rounds}`;
  $("#s-acc").textContent = pct(rec.accuracy);
  $("#s-f1").textContent  = fmt(rec.macro_f1, 3);
  $("#s-asr").textContent = rec.asr === undefined ? "n/a" : pct(rec.asr);
  $("#s-rej").textContent = (rec.removed_clients || []).length +
    (rec.removed_clients && rec.removed_clients.length
      ? ` (${rec.removed_clients.map((c) => "C" + c).join(", ")})` : "");

  drawFederation(state.simMeta, rec);
  lineChart($("#chart-main"), [
    { cls: "s-acc", values: h.map((r) => r.accuracy) },
    { cls: "s-f1",  values: h.map((r) => r.macro_f1) },
    { cls: "s-asr", values: h.map((r) => (r.asr === undefined ? null : r.asr)) },
  ], { marker: state.simMeta.attack_end, markerLabel: "attacker exits" });

  renderDetection(rec);
}

function renderDetection(rec) {
  const meta = state.simMeta;
  const malList = meta.malicious || [];
  const mal = new Set(malList);
  const nClients = meta.n_clients;
  const isMal = Array.from({ length: nClients }, (_, i) => mal.has(i));

  // FLTrust reports trust (low == suspicious); everything else reports an
  // anomaly score (high == suspicious). Negate trust so one convention holds.
  const raw = rec.client_scores
    || (rec.trust_scores ? rec.trust_scores.map((v) => -v) : null);
  if (!raw || !malList.length) {
    $("#scores-wrap").hidden = true;
    $("#s-auc").textContent = $("#s-prec").textContent = malList.length ? "–" : "n/a";
    $("#s-auc-sub").textContent = !malList.length ? "no attackers to detect"
      : `${meta.aggregator} does not score clients — no defense to test`;
    return;
  }

  const a = auc(raw, isMal);
  // An AUC well below 0.5 is not "no signal" — it is the right signal read the
  // wrong way round, which is exactly what FLAME does on this data. Say so
  // rather than silently flipping it.
  const inverted = a < 0.5;
  const corrected = inverted ? 1 - a : a;
  const oriented = inverted ? raw.map((v) => -v) : raw;
  const prec = precisionAtK(oriented, isMal, malList.length);

  // One round on the small live draw is noisy, so the running mean of the raw
  // AUC is shown beside it - that is the number worth narrating.
  const soFar = state.history.map((r) => {
    const s = r.client_scores || (r.trust_scores ? r.trust_scores.map((v) => -v) : null);
    return s ? auc(s, isMal) : NaN;
  }).filter((v) => !Number.isNaN(v));
  const meanSoFar = soFar.length ? soFar.reduce((x, y) => x + y, 0) / soFar.length : NaN;

  const aucEl = $("#s-auc");
  aucEl.textContent = fmt(corrected, 3);
  aucEl.className = corrected >= 0.8 ? "good" : corrected >= 0.65 ? "fair" : "poor";
  $("#s-auc-sub").textContent = (inverted
    ? `raw ${fmt(a, 3)} — score is INVERTED; attackers rank as least suspicious`
    : "0.5 is a coin flip")
    + ` · raw mean so far ${fmt(meanSoFar, 2)}`;
  $("#s-k").textContent = malList.length;
  $("#s-prec").textContent =
    `${Math.round(prec * malList.length)} / ${malList.length}`;

  $("#scores-wrap").hidden = false;
  const order = oriented.map((v, i) => i).sort((i, j) => oriented[j] - oriented[i]);
  const lo = Math.min(...oriented);
  barChart($("#chart-scores"), order.map((i, rank) => ({
    label: `${rank + 1}. C${i}${mal.has(i) ? " ✦" : ""}`,
    value: oriented[i] - lo + 1e-12,
    cls: mal.has(i) ? "is-target" : "is-top",
  })), { padL: 66, format: () => "" });

  $("#scores-note").innerHTML =
    `Ranked most to least suspicious. <b>✦ = actually malicious.</b> `
    + (inverted
        ? `This defense scores them <em>backwards</em> — its raw AUC is ${fmt(a, 3)}, `
          + `so the attackers look like the <em>safest</em> clients. Read inverted it `
          + `reaches ${fmt(corrected, 3)}. The attackers converge on an easy objective, `
          + `so their models sit closest to consensus rather than furthest from it.`
        : `A perfect detector puts all ${malList.length} ✦ at the top.`);

  const h = state.history;
  const curves = h.map((r) => {
    const s = r.client_scores || (r.trust_scores ? r.trust_scores.map((v) => -v) : null);
    return s ? auc(s, isMal) : null;
  });
  lineChart($("#chart-detect"), [
    { cls: "s-auc-raw", values: curves },
    { cls: "s-auc-corr", values: curves.map((v) => (v === null ? null : Math.max(v, 1 - v))) },
    { cls: "s-chance", values: h.map(() => 0.5) },
  ], { marker: meta.attack_end, markerLabel: "attacker exits" });
}

async function startSim() {
  stopSim();
  resetLive();
  $("#btn-run").disabled = true;
  $("#btn-stop").disabled = false;
  $("#round-label").textContent = "loading data…";

  const params = {
    rounds: +$("#c-rounds").value,
    n_malicious: +$("#c-malicious").value,
    attack: +$("#c-malicious").value > 0,
    aggregator: $("#c-aggregator").value,
    trigger: $("#c-trigger").value,
    alpha: ALPHAS[+$("#c-alpha").value].v,
    real: true,
  };

  let sim;
  try {
    sim = await api("/api/simulate", params);
  } catch (err) {
    $("#round-label").textContent = "failed: " + err.message;
    $("#btn-run").disabled = false; $("#btn-stop").disabled = true;
    return;
  }

  const es = new EventSource(`/api/stream/${sim.sim_id}`);
  state.sim = es;
  state.simId = sim.sim_id;
  es.onmessage = (ev) => {
    const rec = JSON.parse(ev.data);
    if (rec.type === "meta") {
      state.simMeta = rec;
      $("#c-clients-out").textContent = rec.n_clients;
      drawFederation(rec, null);
      $("#round-label").textContent = `round 0 / ${rec.rounds}`;
    } else if (rec.type === "round") {
      onRound(rec);
    } else if (rec.type === "error") {
      $("#round-label").textContent = "error: " + rec.message;
    } else if (rec.type === "cancelled") {
      $("#round-label").textContent = `stopped at round ${state.history.length}`;
    } else if (rec.type === "done") {
      if (!/^stopped/.test($("#round-label").textContent)) {
        $("#round-label").textContent = state.history.length
          ? `done — ${state.history.length} rounds` : "stopped";
      }
      stopSim();
    }
  };
  es.onerror = () => { stopSim(); };
}

function stopSim() {
  // Tell the server first: closing the EventSource alone would stop the drawing
  // but leave the training thread grinding through the remaining rounds.
  if (state.sim && state.simId) {
    fetch(`/api/cancel/${state.simId}`, { method: "POST" }).catch(() => {});
  }
  if (state.sim) { state.sim.close(); state.sim = null; }
  $("#btn-run").disabled = false;
  $("#btn-stop").disabled = true;
}

/* ----------------------------------------------------------- runs table */

function renderRuns(runs) {
  const body = $("#runs-body");
  body.innerHTML = "";
  if (!runs.length) {
    body.innerHTML = `<tr><td colspan="9" class="empty">No runs in results/ yet. `
      + `Run <code>python -m flids.runner --config configs/clean_fedavg.yaml</code>.</td></tr>`;
    return;
  }
  runs.forEach((r) => {
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${r.run_id}</td>`
      + `<td>${r.run_name || "–"}</td>`
      + `<td><span class="tag ${r.real ? "tag-real" : "tag-synth"}">`
      + `${r.real ? "real" : "synthetic"}</span></td>`
      + `<td>${r.aggregator}</td>`
      + `<td>${r.trigger || "–"}</td>`
      + `<td class="num">${r.n_malicious}</td>`
      + `<td class="num">${fmt(r.accuracy, 4)}</td>`
      + `<td class="num">${fmt(r.macro_f1, 4)}</td>`
      + `<td class="num">${r.dasr_final === null || r.dasr_final === undefined
            ? (r.asr_final === undefined || r.asr_final === null ? "–"
               : fmt(r.asr_final, 3) + "*") : fmt(r.dasr_final, 3)}</td>`;
    tr.onclick = () => { $$("#runs-body tr").forEach((x) => x.classList.remove("is-sel"));
                         tr.classList.add("is-sel"); showRun(r.run_id); };
    body.appendChild(tr);
  });
}

async function showRun(runId) {
  const d = await api(`/api/run/${runId}`);
  $("#run-detail").hidden = false;
  $("#run-detail-title").textContent =
    `${d.summary.run_name || runId} — ${d.history.length} rounds`;

  const dasr = d.summary.dasr_by_round || null;
  const end = ((d.config.attack || {}).attack_window || [])[1];
  lineChart($("#chart-run"), [
    { cls: "s-acc",  values: d.history.map((r) => r.accuracy) },
    { cls: "s-f1",   values: d.history.map((r) => r.macro_f1) },
    { cls: "s-asr",  values: d.history.map((r) => (r.asr === undefined ? null : r.asr)) },
    ...(dasr ? [{ cls: "s-dasr", values: dasr }] : []),
  ], { marker: end, markerLabel: "attacker exits" });

  $("#run-summary").textContent = JSON.stringify(
    Object.fromEntries(Object.entries(d.summary)
      .filter(([k]) => k !== "dasr_by_round")), null, 2);
}

/* ------------------------------------------------------ compare defenses */

/* Read-only: every column is a recorded run from results/, so the numbers on
   this tab are the numbers in docs/phase2-baselines.md. Nothing trains. */

const DEFENSES = [
  { key: "fedavg", name: "FedAvg",
    role: "No defense — the plain weighted average every other column is measured against." },
  { key: "fltrust", name: "FLTrust",
    role: "Trusts each update by its cosine to a clean root-set update, then weights by trust." },
  { key: "flame", name: "FLAME",
    role: "Clusters client models (HDBSCAN), drops the outliers, clips and adds noise." },
  { key: "fltrust+flame", name: "FLTrust + FLAME",
    role: "FLAME's filter first, then FLTrust's trust weighting on whoever is left." },
  { key: "gradnorm_scorer", name: "GradNorm",
    role: "Scores each update's size against the median. Flags only — still averages everyone." },
];

const cmp = { data: null, trigger: null, seed: null, round: 19, timer: null, activeIdx: 0 };

/* Only one .cmp-col is shown at a time; Prev/Next and the dots below just
   toggle which one, they never re-fetch or re-render its contents. */
function buildCompareDots() {
  const dots = $("#cmp-dots");
  dots.innerHTML = "";
  DEFENSES.forEach((d, i) => {
    const b = document.createElement("button");
    b.className = "cmp-dot"; b.type = "button";
    b.setAttribute("aria-label", d.name);
    b.onclick = () => showDefense(i);
    dots.appendChild(b);
  });
}

function showDefense(idx) {
  const n = DEFENSES.length;
  cmp.activeIdx = ((idx % n) + n) % n;
  $$(".cmp-col").forEach((col, i) => col.classList.toggle("is-active", i === cmp.activeIdx));
  $$(".cmp-dot").forEach((dot, i) => dot.classList.toggle("is-active", i === cmp.activeIdx));
  $("#cmp-nav-name").textContent = DEFENSES[cmp.activeIdx].name;
  $("#cmp-nav-count").textContent = `${cmp.activeIdx + 1} of ${n}`;
}

/* seed -> aggregator for the rung currently selected. The API keys on trigger
   first because the same five defenses now have runs at two rungs, and
   `flame` on its own does not say which one. */
function cmpSeeds() {
  return (cmp.data && cmp.data.triggers[cmp.trigger]) || {};
}

const RUNG_LABEL = {
  oob_999: "oob_999 — extreme, unrealizable (the upper-bound control)",
  inbounds_any: "inbounds_any — in-distribution, any feature",
  inbounds_free: "inbounds_free — in-distribution, attacker-controlled only (realizable)",
};


function miniRing(svg, n, mal, removed, label) {
  clear(svg);
  const cx = 100, cy = 78, R = 58, r = 11;
  svg.appendChild(el("rect", { class: "ring-server", x: cx - 24, y: cy - 11,
                               width: 48, height: 22, rx: 5 }));
  svg.appendChild(el("text", { class: "ring-server-text", x: cx, y: cy + 3,
                               "text-anchor": "middle" }, label));
  for (let i = 0; i < n; i++) {
    const a = -Math.PI / 2 + (i / n) * 2 * Math.PI;
    const x = cx + R * Math.cos(a), y = cy + R * Math.sin(a);
    const out = removed.has(i), bad = mal.has(i);
    svg.appendChild(el("line", {
      class: "ring-edge" + (out ? " is-out" : bad ? " is-mal" : ""),
      x1: x, y1: y, x2: cx + (x - cx) * 0.42, y2: cy + (y - cy) * 0.3,
    }));
    svg.appendChild(el("circle", { class: "ring-node", cx: x, cy: y, r,
      fill: out ? "var(--removed)" : bad ? "var(--malicious)" : "var(--honest)" }));
    svg.appendChild(el("text", { class: "ring-label", x, y: y + 3,
                                 "text-anchor": "middle" }, `C${i}`));
    if (out) {
      // a grey node alone would not say whether it was an attacker, and that
      // is the one thing the viewer needs to read off this diagram
      const d = r * 0.62;
      if (bad) svg.appendChild(el("circle", { cx: x, cy: y, r: r + 2.5, fill: "none",
                                              stroke: "var(--malicious)", "stroke-width": 2 }));
      svg.appendChild(el("line", { class: "ring-x", x1: x - d, y1: y + r + 4, x2: x + d, y2: y + r + 4 }));
    }
  }
}

function runStats(run, upTo) {
  const mal = new Set(run.malicious);
  const nMal = run.malicious.length, nHon = run.n_clients - nMal;
  let caught = 0, wrong = 0, allOut = 0;
  const aucs = [];
  run.rounds.slice(0, upTo + 1).forEach((r) => {
    const m = r.removed.filter((c) => mal.has(c)).length;
    caught += m; wrong += r.removed.length - m;
    if (nMal && m === nMal) allOut += 1;
    if (r.scores) {
      const isMal = Array.from({ length: run.n_clients }, (_, i) => mal.has(i));
      aucs.push(auc(r.scores, isMal));
    }
  });
  return { caught, wrong, allOut, aucs, nMal, nHon, rounds: upTo + 1 };
}

const mean = (xs) => {
  const v = xs.filter((x) => x !== null && x !== undefined && !Number.isNaN(x));
  return v.length ? v.reduce((a, b) => a + b, 0) / v.length : NaN;
};
const std = (xs) => {
  const m = mean(xs); const v = xs.filter((x) => !Number.isNaN(x));
  return v.length ? Math.sqrt(v.reduce((a, b) => a + (b - m) ** 2, 0) / v.length) : NaN;
};

function orientBadge(a) {
  if (a === null || Number.isNaN(a)) return `<span class="orient orient-none">no score</span>`;
  return a < 0.5 ? `<span class="orient orient-inv">inverted</span>`
                 : `<span class="orient orient-ok">as published</span>`;
}

function buildCompareGrid() {
  const grid = $("#k-grid");
  grid.innerHTML = "";
  grid.style.setProperty("--cmp-cols", DEFENSES.length);
  DEFENSES.forEach((d) => {
    const col = document.createElement("div");
    col.className = "cmp-col";
    col.id = `k-col-${d.key.replace(/[^a-z]/g, "")}`;
    col.innerHTML = `
      <div class="cmp-col-inner">
        <div class="cmp-visual">
          <div><h3>${d.name}</h3><p class="cmp-role">${d.role}</p></div>
          <svg class="cmp-ring" viewBox="0 0 200 156" role="img"
               aria-label="${d.name}: clients around the server this round"></svg>
          <div class="cmp-tally">
            <div class="stat caught"><span class="k">attackers excluded</span><b>–</b><span class="sub"></span></div>
            <div class="stat fp"><span class="k">honest excluded</span><b>–</b><span class="sub"></span></div>
          </div>
        </div>
        <div class="cmp-details">
          <div class="cmp-auc"><span class="k">detection AUC, this round</span><span class="orient-slot"></span></div>
          <div class="cmp-auc"><b class="auc-val">–</b><span class="k auc-mean"></span></div>
          <p class="says auc-says"></p>
          <div class="bars-slot">
            <h4>Suspicion ranking, this round</h4>
            <svg class="cmp-bars" viewBox="0 0 220 132" role="img"
                 aria-label="${d.name}: clients ranked by suspicion"></svg>
          </div>
          <div>
            <h4>Detection AUC per round</h4>
            <svg class="cmp-line" viewBox="0 0 220 96" role="img"
                 aria-label="${d.name}: detection AUC against round"></svg>
          </div>
          <div class="cmp-foot">
            <div class="asr"><span>backdoor ASR</span><b class="f-asr">–</b></div>
            <div><span>accuracy</span><b class="f-acc">–</b></div>
            <div><span>macro-F1</span><b class="f-f1">–</b></div>
          </div>
          <p class="says asr-says"></p>
        </div>
      </div>`;
    grid.appendChild(col);
  });
}

function renderCompare() {
  if (!cmp.data) return;
  const runs = cmpSeeds()[cmp.seed] || {};
  const R = cmp.round;
  $("#k-round-out").textContent = R + 1;

  let asrAll = [], anyAllOut = 0, nMalShown = 4;
  DEFENSES.forEach((d) => {
    const col = document.getElementById(`k-col-${d.key.replace(/[^a-z]/g, "")}`);
    const run = runs[d.key];
    if (!run) { col.classList.add("is-none"); return; }
    const rec = run.rounds[Math.min(R, run.rounds.length - 1)];
    const mal = new Set(run.malicious);
    nMalShown = run.malicious.length;
    const st = runStats(run, R);
    anyAllOut += st.allOut;
    asrAll.push(rec.asr);

    miniRing(col.querySelector(".cmp-ring"), run.n_clients, mal, new Set(rec.removed),
             d.key === "fedavg" ? "AVERAGE" : "SERVER");

    const flagsOnly = d.key === "gradnorm_scorer";
    const [cEl, wEl] = col.querySelectorAll(".cmp-tally .stat");
    const hasScores = run.rounds.some((r) => r.scores);
    if (!hasScores) {
      cEl.querySelector("b").textContent = "—";
      wEl.querySelector("b").textContent = "—";
      cEl.querySelector(".sub").textContent = "never looks at clients";
      wEl.querySelector(".sub").textContent = "";
    } else {
      cEl.querySelector("b").textContent = `${st.caught} / ${st.nMal * st.rounds}`;
      wEl.querySelector("b").textContent = `${st.wrong} / ${st.nHon * st.rounds}`;
      // green only when the defense excluded most attacker client-rounds - "2 / 80"
      // in green would read as a success on a projector
      cEl.className = "stat " + (st.caught >= 0.5 * st.nMal * st.rounds ? "caught" : "missed");
      cEl.querySelector(".sub").textContent =
        (flagsOnly ? "flagged, not removed" : "client-rounds") + ` · all ${st.nMal} out: ${st.allOut}×`;
      wEl.querySelector(".sub").textContent = flagsOnly ? "flagged, not removed" : "client-rounds";
    }

    const a = rec.scores ? auc(rec.scores, Array.from({ length: run.n_clients }, (_, i) => mal.has(i))) : NaN;
    const aMean = mean(st.aucs);
    col.querySelector(".orient-slot").innerHTML = orientBadge(hasScores ? a : null);
    const aucEl = col.querySelector(".auc-val");
    aucEl.textContent = hasScores ? fmt(a, 3) : "—";
    aucEl.style.color = !hasScores ? "var(--muted)"
      : a < 0.5 ? "var(--malicious)" : a >= 0.75 ? "var(--target)" : "var(--sleeping)";
    col.querySelector(".auc-mean").textContent =
      hasScores ? `mean to here ${fmt(aMean, 3)}` : "no per-client score";
    // the run-so-far mean, not this single round: one round bounces enough to
    // make the sentence contradict itself between frames of the replay
    const [aucText, aucTone] = aucSentence(hasScores ? aMean : null, d.name);
    says(col.querySelector(".auc-says"), aucText, aucTone);

    const barsSvg = col.querySelector(".cmp-bars");
    if (rec.scores) {
      const order = rec.scores.map((v, i) => i).sort((i, j) => rec.scores[j] - rec.scores[i]);
      const lo = Math.min(...rec.scores);
      barChart(barsSvg, order.map((i, rank) => ({
        label: `${rank + 1}. C${i}${mal.has(i) ? " ✦" : ""}`,
        value: rec.scores[i] - lo + 1e-12,
        cls: mal.has(i) ? "is-target" : "is-top",
      })), { padL: 58, format: () => "" });
    } else {
      clear(barsSvg);
      barsSvg.appendChild(el("text", { x: 110, y: 66, "text-anchor": "middle",
                                       class: "axis-text" }, "FedAvg ranks no one"));
    }

    const curve = run.rounds.map((r) => (r.scores
      ? auc(r.scores, Array.from({ length: run.n_clients }, (_, i) => mal.has(i))) : null));
    lineChart(col.querySelector(".cmp-line"), [
      { cls: "s-chance", values: run.rounds.map(() => 0.5) },
      ...(hasScores ? [{ cls: "s-auc-cmp", values: curve.slice(0, R + 1) }] : []),
    ], { marker: R, markerLabel: "" });

    col.querySelector(".f-asr").textContent = pct(rec.asr);
    const [asrText, asrTone] = asrSentence(rec.asr);
    says(col.querySelector(".asr-says"), asrText, asrTone);
    col.querySelector(".f-acc").textContent = pct(rec.accuracy);
    col.querySelector(".f-f1").textContent  = fmt(rec.macro_f1, 3);
  });

  const allBackdoored = asrAll.length && asrAll.every((x) => x >= 0.99);
  const v = $("#k-verdict");
  v.className = "verdict " + (allBackdoored ? "hit" : "miss");
  v.textContent = allBackdoored
    ? `Seed ${cmp.seed}, round ${R + 1}: the backdoor works under every defense `
      + `(ASR ${asrAll.map((x) => pct(x)).every((x) => x === "100.0%") ? "100%" : "≥ 99%"} in all ${asrAll.length} columns). `
      + (anyAllOut === 0
          ? `No defense has had all ${nMalShown} attackers out in the same round.`
          : `All ${nMalShown} attackers were out together in ${anyAllOut} column-round(s) — and it still did not stop the backdoor.`)
    : `Seed ${cmp.seed}, round ${R + 1}: backdoor ASR ranges `
      + `${pct(Math.min(...asrAll))}–${pct(Math.max(...asrAll))} across the defenses.`;

}

function renderCompareSummary() {
  const body = $("#k-summary");
  body.innerHTML = "";
  DEFENSES.forEach((d) => {
    const perSeed = Object.values(cmpSeeds()).map((s) => s[d.key]).filter(Boolean);
    if (!perSeed.length) return;
    const stats = perSeed.map((run) => runStats(run, run.rounds.length - 1));
    const hasScores = perSeed.some((run) => run.rounds.some((r) => r.scores));
    const aucs = stats.map((s) => mean(s.aucs));
    const inv = aucs.filter((a) => a < 0.5).length;
    const caught = stats.reduce((a, s) => a + s.caught, 0);
    const wrong = stats.reduce((a, s) => a + s.wrong, 0);
    const slots = stats.reduce((a, s) => a + s.nMal * s.rounds, 0);
    const hslots = stats.reduce((a, s) => a + s.nHon * s.rounds, 0);
    const allOut = stats.reduce((a, s) => a + s.allOut, 0);
    const asr = perSeed.map((r) => r.asr_final);
    let orient = "—", orientCls = "";
    if (hasScores) {
      if (inv === perSeed.length) { orient = `inverted in all ${inv} seeds`; orientCls = "flag"; }
      else if (inv === 0) { orient = `as published in all ${perSeed.length}`; orientCls = "ok"; }
      else { orient = `flips: inverted in ${inv} of ${perSeed.length}`; orientCls = "flag"; }
    }
    const tr = document.createElement("tr");
    tr.innerHTML =
      `<td>${d.name}</td><td class="num">${perSeed.length}</td>`
      + `<td class="num">${hasScores ? `${fmt(mean(aucs), 3)} ± ${fmt(std(aucs), 3)}` : "—"}</td>`
      + `<td class="${orientCls}">${orient}</td>`
      + `<td class="num">${hasScores ? `${caught} / ${slots}` : "—"}</td>`
      + `<td class="num">${hasScores ? `${wrong} / ${hslots}` : "—"}</td>`
      + `<td class="num">${hasScores ? allOut : "—"}</td>`
      + `<td class="num">${fmt(mean(asr), 3)}</td>`;
    body.appendChild(tr);
  });
}

async function loadCompare() {
  if (cmp.data) return;
  try {
    cmp.data = await api("/api/compare");
  } catch (err) {
    $("#k-verdict").className = "verdict miss";
    $("#k-verdict").textContent = "Could not read recorded runs: " + err.message;
    return;
  }
  const rungs = Object.keys(cmp.data.triggers || {});
  if (!rungs.length) {
    $("#k-verdict").className = "verdict miss";
    $("#k-verdict").textContent = "No recorded real-data campaign runs in results/ yet — "
      + "run python -m scripts.baselines.run_all_real.";
    return;
  }
  cmp.trigger = cmp.data.default_trigger || rungs[0];

  const rungSel = $("#k-trigger");
  rungSel.innerHTML = "";
  rungs.forEach((t) => rungSel.add(new Option(RUNG_LABEL[t] || t, t)));
  rungSel.value = cmp.trigger;

  const seedSel = $("#k-seed");
  // Seeds are per rung: a rung the campaign has only partly covered must not
  // leave the selector offering a seed it has no runs for.
  const syncSeeds = () => {
    const seeds = Object.keys(cmpSeeds());
    seedSel.innerHTML = "";
    seeds.forEach((s) => seedSel.add(new Option(`seed ${s}`, s)));
    cmp.seed = seeds[0];
    seedSel.value = cmp.seed;
    const nRounds = Math.max(
      ...Object.values(cmpSeeds()[cmp.seed]).map((r) => r.rounds.length));
    $("#k-round").max = nRounds - 1;
    $("#k-round").value = cmp.round = nRounds - 1;
    $("#k-rounds").textContent = nRounds;
  };
  syncSeeds();

  buildCompareGrid();
  buildCompareDots();
  showDefense(cmp.activeIdx);
  renderCompare();
  renderCompareSummary();

  rungSel.onchange = () => {
    stopReplay();
    cmp.trigger = rungSel.value;
    syncSeeds();
    renderCompare();
    renderCompareSummary();
  };
  seedSel.onchange = () => { stopReplay(); cmp.seed = seedSel.value; renderCompare(); };
  $("#k-round").oninput = () => { stopReplay(); cmp.round = +$("#k-round").value; renderCompare(); };
  $("#btn-replay").onclick = () => (cmp.timer ? stopReplay() : startReplay());
  $("#cmp-prev").onclick = () => showDefense(cmp.activeIdx - 1);
  $("#cmp-next").onclick = () => showDefense(cmp.activeIdx + 1);
}

function startReplay() {
  const slider = $("#k-round");
  cmp.round = 0; slider.value = 0; renderCompare();
  $("#btn-replay").textContent = "Pause";
  cmp.timer = setInterval(() => {
    if (cmp.round >= +slider.max) { stopReplay(); return; }
    cmp.round += 1; slider.value = cmp.round; renderCompare();
  }, 650);
}

function stopReplay() {
  if (cmp.timer) { clearInterval(cmp.timer); cmp.timer = null; }
  $("#btn-replay").textContent = "Replay rounds";
}

/* --------------------------------------------------------- model-level detection */

/* Read-only, same spirit as compare(): every number here is
   scripts.baselines.nc_roc / activation_clustering's own CSV summary. Loaded
   once per session, same as loadCompare. */
let detLoaded = false;

function ncVerdict(nc) {
  if (nc.auc === null) return ["", null];
  if (nc.auc < 0.6) {
    return [`Chance-level ranking (AUC ${fmt(nc.auc, 2)}) — cannot tell backdoored `
          + `models from clean ones here.`, "is-bad"];
  }
  const caught = nc.tpr > 0;
  return [`Ranks backdoored models correctly (AUC ${fmt(nc.auc, 2)}, ${nc.n_pairs} `
        + `out-of-sample pairs), but the calibrated threshold still flags `
        + (caught ? `${pct(nc.tpr)} of them.` : `none of them — too conservative to operate on.`),
          caught ? "is-good" : "is-bad"];
}

function acVerdict(ac) {
  if (ac.auc === null) return ["", null];
  if (ac.flag_rate >= 0.5) {
    return [`Separates cleanly at ${pct(ac.poison_ratio)} poison ratio — flags `
          + `${pct(ac.flag_rate)} of backdoored models (AUC ${fmt(ac.auc, 2)}).`, "is-good"];
  }
  return [`No separation at ${pct(ac.poison_ratio)} poison ratio (AUC ${fmt(ac.auc, 2)}) `
        + `— the poisoned rows look the same as clean ones in activation space, even `
        + `though the backdoor works (ASR ${pct(ac.asr_mean)}).`, "is-bad"];
}

function detRow(rung, name, aucVal, opCell, asrVal, verdict, tone) {
  const tr = document.createElement("tr");
  tr.innerHTML =
    `<td><code>${rung}</code></td><td>${name}</td>`
    + `<td class="num">${aucVal === null ? "—" : fmt(aucVal, 2)}</td>`
    + `<td class="num">${opCell}</td>`
    + `<td class="num">${asrVal === null ? "—" : pct(asrVal)}</td>`
    + `<td class="says ${tone || ""}">${verdict}</td>`;
  return tr;
}

async function loadDetection() {
  if (detLoaded) return;
  let d;
  try {
    d = await api("/api/detection");
  } catch (err) {
    $("#det-note").textContent = "Could not read detection results: " + err.message;
    return;
  }
  const body = $("#det-body");
  body.innerHTML = "";
  const rungs = Object.keys(d.rungs || {});
  if (!rungs.length) {
    $("#det-note").textContent = "No nc_roc / activation_clustering CSVs in results/baselines/ "
      + "yet — run scripts.baselines.nc_roc and scripts.baselines.activation_clustering.";
    return;
  }
  rungs.forEach((rung) => {
    const r = d.rungs[rung];
    if (r.neural_cleanse) {
      const nc = r.neural_cleanse;
      const [text, tone] = ncVerdict(nc);
      body.appendChild(detRow(rung, "Neural Cleanse", nc.auc,
        nc.tpr === null ? "—" : `TPR ${pct(nc.tpr)} &middot; FPR ${pct(nc.fpr)}`,
        nc.asr_backdoor_mean, text, tone));
    }
    if (r.activation_clustering) {
      const ac = r.activation_clustering;
      const [text, tone] = acVerdict(ac);
      body.appendChild(detRow(rung, "Activation Clustering", ac.auc,
        ac.n_models === null ? "—" : `flags ${pct(ac.flag_rate)} of ${ac.n_models} models`,
        ac.asr_mean, text, tone));
    }
  });
  $("#det-note").textContent =
    "Neural Cleanse's threshold is calibrated on 10 clean models (p95 of their null "
    + "distribution); one outlier in that set means it operationally flags nothing at "
    + "either rung, even where its ranking (AUC) is correct. Activation Clustering needs "
    + "the training rows the model saw, which in federated learning live on the clients "
    + "— this measures whether the rows reveal the backdoor, not whether a server could "
    + "run this scan itself.";
  detLoaded = true;
}

/* ------------------------------------------------------------- bootstrap */

function bindTabs() {
  $$(".tab").forEach((tab) => {
    tab.onclick = () => {
      $$(".tab").forEach((t) => t.classList.toggle("is-active", t === tab));
      $$(".panel").forEach((p) =>
        p.classList.toggle("is-active", p.id === tab.dataset.panel));
      if (tab.dataset.panel === "panel-compare") loadCompare();
      else stopReplay();
      if (tab.dataset.panel === "panel-detection") loadDetection();
    };
  });
}

function bindRanges() {
  const pairs = [["#c-malicious", "#c-malicious-out", (v) => v],
                 ["#c-rounds", "#c-rounds-out", (v) => v]];
  pairs.forEach(([inp, out, f]) => {
    const i = $(inp), o = $(out);
    i.oninput = () => { o.textContent = f(i.value); };
  });
  const a = $("#c-alpha"), ao = $("#c-alpha-out");
  a.oninput = () => { ao.textContent = ALPHAS[+a.value].label; };
  ao.textContent = ALPHAS[+a.value].label;
}

async function init() {
  bindTabs();
  bindRanges();
  $("#btn-run").onclick = startSim;
  $("#btn-stop").onclick = stopSim;

  let meta;
  try {
    meta = await api("/api/meta");
  } catch (err) {
    $("#data-badge").textContent = "backend unreachable";
    return;
  }
  state.meta = meta;
  state.runs = meta.runs;

  const badge = $("#data-badge");
  badge.textContent = meta.has_real_data
    ? "real CIC-IDS2017 arrays" : "synthetic fallback — data/processed missing";
  badge.className = "badge " + (meta.has_real_data ? "badge-real" : "badge-synth");
  $("#runs-badge").textContent = `${meta.runs.length} recorded run${meta.runs.length === 1 ? "" : "s"}`;
  $("#runs-badge").className = "badge badge-muted";

  const aggSel = $("#c-aggregator");
  meta.aggregators.forEach((a) => aggSel.add(new Option(a, a)));
  aggSel.value = "fedavg";

  const rungHints = {
    oob_999: "999.0 on 3 fixed columns — unrealizable upper bound",
    inbounds_any: "85th percentile of the 3 highest-F features",
    inbounds_free: "85th percentile, attacker-controllable features only",
  };
  meta.rungs.forEach((r) => $("#c-trigger").add(new Option(r.name, r.name)));
  const updateHint = () => { $("#c-trigger-hint").textContent = rungHints[$("#c-trigger").value] || ""; };
  $("#c-trigger").onchange = updateHint;
  updateHint();

  renderRuns(meta.runs);
  drawFederation({ n_clients: 10, malicious: [0, 1, 2, 3], aggregator: "fedavg",
                   client_sizes: [] }, null);
}

init();
