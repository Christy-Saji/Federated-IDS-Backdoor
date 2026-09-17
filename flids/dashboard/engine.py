"""Everything the dashboard does that is not HTTP.

Kept separate from ``server.py`` so the demo logic can be exercised from a REPL
or a test without opening a socket.

Nothing here writes to ``results/``. Live simulations exist only in memory; the
only way to produce a run_id is still ``flids.runner``.
"""

from __future__ import annotations

import glob
import json
import os
import re
import threading

import numpy as np
import yaml

from flids.data.labels import CLASS_NAMES
from flids.data.loaders import load_processed, synthetic_dataset
from flids.data.triggers import (TRIGGERS, apply_trigger, feature_stats,
                                 get_trigger, resolve_features, resolve_values)
from flids.fl.aggregators import available as available_aggregators
from flids.fl.server import FederatedServer
from flids.models.registry import build_model

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RESULTS = os.path.join(ROOT, "results")
PROCESSED = os.path.join(ROOT, "data", "processed")

# Rungs a demo can actually stamp. `problemspace` resolves its value from a pcap
# and is out of scope for this project (feature-space only), so it is not
# offered - better an absent control than one that raises when clicked.
DEMO_RUNGS = [n for n, s in TRIGGERS.items() if s["value"] != "from_pcap"]

# The live demo trains in the browser's patience, not a compute budget: 8k rows
# keeps a round near a second so the federation animation reads as motion. Every
# number that goes in the report comes from flids.runner at the full 60k
# subsample or the whole split - the page says so, and so does this comment.
DEMO_SUBSAMPLE = 8000


def _display_name(run_name: str, cfg: dict) -> str:
    """A run's name with its ``_s<N>`` suffix matched to the seed it ran at.

    Runs made with ``flids.runner --seed N`` before the runner rewrote the
    suffix still say ``_s0`` in their files. results/ is append-only, so the
    label is corrected here, on the way to the page, rather than on disk.
    """
    seed = (cfg or {}).get("seed")
    if not run_name or seed is None or not re.search(r"_s\d+$", run_name):
        return run_name or ""
    return re.sub(r"_s\d+$", f"_s{int(seed)}", run_name)


class DemoEngine:
    """Datasets, trained models, and live simulations for the dashboard."""

    def __init__(self, processed_dir: str = PROCESSED):
        self.processed_dir = processed_dir
        self._datasets: dict[tuple, object] = {}
        self._stats: dict[tuple, dict] = {}
        self._lock = threading.Lock()
        self.sims: dict[str, "Simulation"] = {}

    # -- data ---------------------------------------------------------------
    @property
    def has_real_data(self) -> bool:
        return os.path.exists(os.path.join(self.processed_dir, "X_train.npy"))

    def dataset(self, subsample: int = DEMO_SUBSAMPLE, seed: int = 0,
                real: bool | None = None):
        """A cached dataset. Falls back to synthetic when data/processed is absent.

        The cache is keyed on everything that changes the arrays, so the demo
        never serves a model one dataset and the inspector another.
        """
        real = self.has_real_data if real is None else (real and self.has_real_data)
        key = (bool(real), int(subsample), int(seed))
        with self._lock:
            if key not in self._datasets:
                if real:
                    self._datasets[key] = load_processed(
                        self.processed_dir, n_classes=8,
                        subsample=subsample or None, seed=seed)
                else:
                    self._datasets[key] = synthetic_dataset(
                        seed=seed, multiclass=True, n_classes=8,
                        n_train=max(subsample, 2000), n_test=max(subsample // 4, 500))
            return self._datasets[key]

    def stats(self, ds, key=None):
        """feature_stats() for a dataset, cached - the p85 rungs need it."""
        key = key or (id(ds),)
        with self._lock:
            if key not in self._stats:
                self._stats[key] = feature_stats(ds.X_train, ds.y_train)
            return self._stats[key]

    # -- stored runs --------------------------------------------------------
    def runs(self) -> list[dict]:
        """Every completed run in results/, newest first."""
        out = []
        for summary_path in glob.glob(os.path.join(RESULTS, "*", "summary.json")):
            run_dir = os.path.dirname(summary_path)
            try:
                summary = json.load(open(summary_path))
            except (OSError, ValueError):
                continue
            cfg_path = os.path.join(run_dir, "config.yaml")
            cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}
            data_cfg = (cfg.get("data") or {})
            attack_cfg = (cfg.get("attack") or {})
            trigger = (attack_cfg.get("trigger") or {}) if attack_cfg.get("enabled") else {}
            out.append({
                "run_id": summary.get("run_id", os.path.basename(run_dir)),
                "run_name": _display_name(summary.get("run_name", ""), cfg),
                "rounds": summary.get("rounds"),
                "real": bool(data_cfg.get("processed_dir") or data_cfg.get("path")),
                "aggregator": (cfg.get("federated") or {}).get("aggregator", "?"),
                "n_malicious": summary.get("n_malicious", 0),
                "trigger": trigger.get("name") if isinstance(trigger, dict) else None,
                "accuracy": (summary.get("final") or {}).get("accuracy"),
                "macro_f1": (summary.get("final") or {}).get("macro_f1"),
                "asr_final": summary.get("asr_final"),
                "dasr_final": summary.get("dasr_final"),
                "has_model": os.path.exists(os.path.join(run_dir, "model_final.npz")),
                "mtime": os.path.getmtime(summary_path),
            })
        return sorted(out, key=lambda r: r["mtime"], reverse=True)

    def run_detail(self, run_id: str) -> dict:
        """summary.json plus every per-round record, for the charts."""
        run_dir = os.path.join(RESULTS, _safe_id(run_id))
        summary = json.load(open(os.path.join(run_dir, "summary.json")))
        history = []
        metrics_path = os.path.join(run_dir, "metrics.jsonl")
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                history = [json.loads(line) for line in f if line.strip()]
        cfg_path = os.path.join(run_dir, "config.yaml")
        cfg = yaml.safe_load(open(cfg_path)) if os.path.exists(cfg_path) else {}
        summary["run_name"] = _display_name(summary.get("run_name", ""), cfg)
        return {"summary": summary, "history": history, "config": cfg}

    # -- side-by-side defenses ----------------------------------------------
    def compare(self) -> dict:
        """Every recorded real-data campaign run, grouped seed -> aggregator.

        Reads the same runs `scripts.baselines.detection_report` and
        `prevention_report` read (20-round attack runs on data/processed, the
        attacker poisoning the whole run), so the side-by-side view and the
        write-up can never disagree. Trains nothing.
        """
        seeds: dict[int, dict] = {}
        for summary_path in glob.glob(os.path.join(RESULTS, "*", "summary.json")):
            run_dir = os.path.dirname(summary_path)
            cfg_path = os.path.join(run_dir, "config.yaml")
            metrics_path = os.path.join(run_dir, "metrics.jsonl")
            if not (os.path.exists(cfg_path) and os.path.exists(metrics_path)):
                continue
            cfg = yaml.safe_load(open(cfg_path)) or {}
            data_cfg, attack = cfg.get("data") or {}, cfg.get("attack") or {}
            fed = cfg.get("federated") or {}
            window = attack.get("attack_window") or [0, 0]
            trigger = attack.get("trigger") or {}
            if not (data_cfg.get("processed_dir") and attack.get("enabled")
                    and int(fed.get("rounds", 0)) == int(window[1])):
                continue
            summary = json.load(open(summary_path))
            with open(metrics_path) as f:
                history = [json.loads(line) for line in f if line.strip()]
            malicious = [int(c) for c in attack.get("malicious_clients") or []]
            rounds = []
            for rec in history:
                # one convention for every defense: higher == more suspicious
                scores = rec.get("client_scores")
                if scores is None and rec.get("trust_scores") is not None:
                    scores = [-v for v in rec["trust_scores"]]
                rounds.append({
                    "accuracy": rec.get("accuracy"),
                    "macro_f1": rec.get("macro_f1"),
                    "asr": rec.get("asr"),
                    "removed": [int(c) for c in rec.get("removed_clients") or []],
                    "scores": scores,
                })
            seed = int(cfg.get("seed", 0))
            seeds.setdefault(seed, {})[fed.get("aggregator", "?")] = {
                "run_id": summary.get("run_id", os.path.basename(run_dir)),
                "run_name": _display_name(summary.get("run_name", ""), cfg),
                "trigger": trigger.get("name") if isinstance(trigger, dict) else None,
                "n_clients": int(data_cfg.get("n_clients", 10)),
                "malicious": malicious,
                "asr_final": summary.get("asr_final"),
                "asr_clean_baseline": summary.get("asr_clean_baseline"),
                "dasr_final": summary.get("dasr_final"),
                "rounds": rounds,
            }
        return {"seeds": {str(s): seeds[s] for s in sorted(seeds)}}

    # -- trained models -----------------------------------------------------
    def load_model(self, run_id: str):
        """Rebuild the global model a run finished on. Returns (model, ds, cfg)."""
        run_dir = os.path.join(RESULTS, _safe_id(run_id))
        params = np.load(os.path.join(run_dir, "model_final.npz"))["params"]
        cfg = yaml.safe_load(open(os.path.join(run_dir, "config.yaml")))
        summary = json.load(open(os.path.join(run_dir, "summary.json")))

        n_classes = int(summary.get("n_classes", 8))
        model_cfg = cfg.get("model") or {}
        arch = model_cfg.get("arch", "mlp")
        hidden = tuple(model_cfg.get("hidden", (256, 128, 64)))
        n_features = _infer_n_features(len(params), hidden, n_classes)

        real = bool((cfg.get("data") or {}).get("processed_dir")
                    or (cfg.get("data") or {}).get("path"))
        ds = self.dataset(subsample=DEMO_SUBSAMPLE, seed=cfg.get("seed", 0), real=real)
        if ds.n_features != n_features:
            raise ValueError(
                f"run {run_id} was trained on {n_features} features but the "
                f"available dataset has {ds.n_features} - regenerate "
                f"data/processed, or pick a run that matches it.")

        model = build_model(arch, n_features=n_features, n_classes=n_classes,
                            seed=cfg.get("seed", 0), hidden=hidden)
        model.set_params(params)
        return model, ds, cfg

    # -- the backdoor inspector --------------------------------------------
    def inspect(self, run_id: str, trigger: str = "oob_999",
                family: int | None = None, sample_idx: int | None = None,
                target_label: int = 0, pick_seed: int | None = None) -> dict:
        """Run one real test flow through a trained model, clean and triggered.

        This is the demo that makes a backdoor concrete: the same flow, the same
        weights, three columns different, and the prediction moves to the
        attacker's target class.

        ``sample_idx`` pins an exact test row (so a flow that made a good point
        can be shown again); ``pick_seed`` only reshuffles which row the family
        filter lands on. They are separate arguments because conflating them
        makes "give me another Bot flow" silently mean "give me row 41".
        """
        model, ds, cfg = self.load_model(run_id)
        rng = np.random.default_rng(pick_seed if pick_seed is not None else 0)

        # pick a flow: a named family if asked, else any non-target sample (the
        # population a backdoor is actually evaluated over)
        if sample_idx is not None and 0 <= sample_idx < len(ds.y_test):
            idx = int(sample_idx)
        else:
            pool = (np.flatnonzero(ds.y_test == family) if family is not None
                    else np.flatnonzero(ds.y_test != target_label))
            if not len(pool):
                pool = np.arange(len(ds.y_test))
            idx = int(rng.choice(pool))

        x = ds.X_test[idx:idx + 1]
        spec = get_trigger(trigger)
        stats = self.stats(ds, key=("inspect", id(ds)))
        cols = resolve_features(spec, stats, ds.feature_names)
        vals = resolve_values(spec, cols, stats)
        x_trig = apply_trigger(x, spec, stats, ds.feature_names)

        clean_probs = model.predict_proba(x)[0]
        trig_probs = model.predict_proba(x_trig)[0]
        vals_arr = np.broadcast_to(np.asarray(vals, float), (len(cols),))

        names = ds.feature_names or [f"f{i:02d}" for i in range(ds.n_features)]
        return {
            "run_id": run_id,
            "run_name": _display_name(cfg.get("run_name", ""), cfg),
            "trigger": trigger,
            "realizable": spec.get("realizable"),
            "sample_index": idx,
            "true_label": int(ds.y_test[idx]),
            "true_label_name": CLASS_NAMES[int(ds.y_test[idx])],
            "target_label": target_label,
            "target_label_name": CLASS_NAMES[target_label],
            "class_names": CLASS_NAMES[:ds.n_classes],
            "clean_probs": [float(p) for p in clean_probs],
            "triggered_probs": [float(p) for p in trig_probs],
            "clean_pred": int(clean_probs.argmax()),
            "triggered_pred": int(trig_probs.argmax()),
            "flipped": bool(clean_probs.argmax() != trig_probs.argmax()
                            and trig_probs.argmax() == target_label),
            "columns": [
                {"index": int(c),
                 "name": names[c] if c < len(names) else f"f{c:02d}",
                 "before": float(x[0, c]),
                 "after": float(vals_arr[k])}
                for k, c in enumerate(cols)],
        }

    # -- live simulation ----------------------------------------------------
    def start_simulation(self, params: dict) -> "Simulation":
        sim = Simulation(self, params)
        self.sims[sim.sim_id] = sim
        # keep the last handful of simulations, drop the rest
        for old in list(self.sims)[:-8]:
            self.sims.pop(old, None)
        sim.start()
        return sim

    def meta(self) -> dict:
        return {
            "class_names": CLASS_NAMES,
            "rungs": [{"name": n, "realizable": TRIGGERS[n].get("realizable")}
                      for n in DEMO_RUNGS],
            "aggregators": available_aggregators(),
            "has_real_data": self.has_real_data,
            "demo_subsample": DEMO_SUBSAMPLE,
            "runs": self.runs(),
        }


class Simulation:
    """One live federated run, streamed round by round.

    Runs on its own thread and buffers every record, so a browser that connects
    late (or reconnects) still receives the whole history rather than joining
    mid-way through.
    """

    _counter = 0
    _counter_lock = threading.Lock()

    def __init__(self, engine: DemoEngine, params: dict):
        with Simulation._counter_lock:
            Simulation._counter += 1
            self.sim_id = f"sim{Simulation._counter}"
        self.engine = engine
        self.params = params
        self.records: list[dict] = []
        self.meta: dict = {}
        self.done = False
        self.error: str | None = None
        self.cancelled = False
        self._event = threading.Event()
        self._thread: threading.Thread | None = None

    def cancel(self):
        """Ask the training thread to stop at the end of the current round."""
        self.cancelled = True
        self._event.set()

    # -- config -------------------------------------------------------------
    def _config(self) -> dict:
        p = self.params
        rounds = int(p.get("rounds", 15))
        n_clients = int(p.get("n_clients", 10))
        n_malicious = int(p.get("n_malicious", 4))
        attack_on = bool(p.get("attack", True)) and n_malicious > 0
        window_end = int(p.get("attack_end", rounds))
        return {
            "run_name": "dashboard_demo",     # never written to results/
            "data": {"dataset": "processed" if p.get("real", True) else "synthetic",
                     "labels": "multiclass", "partition": "dirichlet",
                     "alpha": float(p.get("alpha", 0.5)), "n_clients": n_clients,
                     "min_size": 20},
            "model": {"arch": "mlp", "hidden": [256, 128, 64], "dropout": 0.3},
            "federated": {"rounds": rounds, "local_epochs": 2, "lr": 0.05,
                          "batch_size": 256,
                          "aggregator": p.get("aggregator", "fedavg")},
            "attack": {
                "enabled": attack_on,
                "type": "badnets" if attack_on else None,
                "malicious_clients": list(range(n_malicious)) if attack_on else [],
                "trigger": {"name": p.get("trigger", "oob_999"),
                            "target_label": int(p.get("target_label", 0)),
                            "source_class": None} if attack_on else None,
                "poison_ratio": float(p.get("poison_ratio", 0.5)),
                "attack_window": [1, window_end],
            },
            "seed": int(p.get("seed", 0)),
        }

    # -- lifecycle ----------------------------------------------------------
    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self):
        try:
            cfg = self._config()
            seed = cfg["seed"]
            ds = self.engine.dataset(
                subsample=int(self.params.get("subsample", DEMO_SUBSAMPLE)),
                seed=seed, real=bool(self.params.get("real", True)))
            server = FederatedServer(ds, cfg, seed=seed)

            self.meta = {
                "n_clients": len(server.clients),
                "client_sizes": [int(c.n_samples) for c in server.clients],
                "malicious": [int(c.cid) for c in server.clients if c.malicious],
                "rounds": cfg["federated"]["rounds"],
                "aggregator": cfg["federated"]["aggregator"],
                "trigger": (cfg["attack"].get("trigger") or {}).get("name"),
                "attack_end": cfg["attack"]["attack_window"][1],
                "n_train": int(len(ds.y_train)),
                "n_test": int(len(ds.y_test)),
                "real": bool(self.params.get("real", True)) and self.engine.has_real_data,
                "class_balance": ds.class_balance(),
            }
            self._emit({"type": "meta", **self.meta})
            server.run(on_round=self._on_round)
        except _Cancelled:
            self._emit({"type": "cancelled"})
        except Exception as exc:                      # surfaced in the browser
            self.error = f"{type(exc).__name__}: {exc}"
            self._emit({"type": "error", "message": self.error})
        finally:
            self.done = True
            self._emit({"type": "done"})

    def _on_round(self, rec: dict):
        """FederatedServer's per-round hook. Raising is the only way out of its
        loop, which is what makes Stop stop training rather than just stop
        drawing."""
        self._emit({"type": "round", **rec})
        if self.cancelled:
            raise _Cancelled()

    def _emit(self, record: dict):
        self.records.append(record)
        self._event.set()

    def stream(self, timeout: float = 0.5):
        """Yield records as they arrive, replaying anything already buffered."""
        sent = 0
        while True:
            while sent < len(self.records):
                yield self.records[sent]
                sent += 1
            if self.done and sent >= len(self.records):
                return
            self._event.wait(timeout)
            self._event.clear()


class _Cancelled(Exception):
    """Raised out of the round hook to unwind FederatedServer.run()."""


# ---------------------------------------------------------------------------
def _infer_n_features(n_params: int, hidden, n_classes: int) -> int:
    """Recover the input width from a flat parameter vector.

    The alternative is trusting the dataset currently on disk to be the one the
    run trained on, which is exactly the mismatch worth catching loudly: a
    binary run and an 8-family run differ only in the last layer.
    """
    sizes = list(hidden) + [n_classes]
    tail = sum(a * b for a, b in zip(sizes[:-1], sizes[1:])) + sum(sizes)
    first = sizes[0]
    if (n_params - tail) % first:
        raise ValueError(f"cannot infer n_features from {n_params} parameters "
                         f"with hidden={hidden}, n_classes={n_classes}")
    return (n_params - tail) // first


def _safe_id(run_id: str) -> str:
    """run_ids index a directory, so refuse anything that is not one."""
    rid = str(run_id)
    if not rid or not all(c.isalnum() for c in rid):
        raise ValueError(f"bad run_id {run_id!r}")
    return rid
