"""The federated training loop.

Deterministic given a seed: the partition, client order, local training, and
poison-sample selection are each seeded explicitly (Task 1.5).
"""

from __future__ import annotations

import numpy as np

from flids.attacks.badnets import evaluate_backdoor
from flids.data.partition import dirichlet_partition
from flids.data.triggers import feature_stats, get_trigger
from flids.eval.metrics import detection_auc, main_task_accuracy
from flids.models.registry import build_model
from flids.utils.seeding import set_all_seeds

from .aggregators import build_aggregator
from .client import Client


class FederatedServer:
    def __init__(self, dataset, cfg: dict, seed: int = 0):
        self.ds = dataset
        self.cfg = cfg
        self.seed = seed
        set_all_seeds(seed)

        d = cfg["data"]
        agg_name = cfg["federated"]["aggregator"]

        # FLTrust needs a clean, class-balanced root set held out from every
        # client (Cao et al., NDSS 2022). Carve it before partitioning.
        self.root = None
        pool = np.arange(len(dataset.y_train))
        if "fltrust" in agg_name:
            from sklearn.model_selection import train_test_split
            root_size = int(cfg["federated"].get("root_size", 1000))
            pool, root_idx = train_test_split(
                pool, test_size=root_size, random_state=0,
                stratify=dataset.y_train)
            self.root = (dataset.X_train[root_idx], dataset.y_train[root_idx])

        local_parts = dirichlet_partition(
            dataset.y_train[pool], d["n_clients"], float(d["alpha"]),
            seed=seed, min_size=d.get("min_size", 100))
        self.parts = [pool[p] for p in local_parts]

        a = cfg.get("attack", {}) or {}
        self.attack_on = bool(a.get("enabled"))
        malicious = set(a.get("malicious_clients", []) or [])
        trig = a.get("trigger", {}) or {}
        if not isinstance(trig, dict):
            trig = {}

        attack_cfg = dict(a)
        attack_cfg["seed"] = seed
        attack_cfg["feature_names"] = list(dataset.feature_names)
        attack_cfg["target_label"] = trig.get("target_label", a.get("target_label", 0))
        src = trig.get("source_class", a.get("source_class"))
        attack_cfg["source_classes"] = None if src is None else (
            [src] if np.isscalar(src) else list(src))
        attack_cfg["poison_ratio"] = a.get("poison_ratio", 0.0)
        attack_cfg["attack_window"] = a.get("attack_window", [1, cfg["federated"]["rounds"]])

        # Phase 2 trigger ladder: `trigger.name` selects a rung; the legacy
        # `trigger.value` / `trigger_features` path is kept as a fallback.
        self.trigger_name = trig.get("name")
        if self.trigger_name:
            overrides = {k: v for k, v in trig.items()
                         if k not in ("name", "target_label", "source_class")}
            self.trigger_spec = get_trigger(self.trigger_name, **overrides)
            self.trigger_stats = feature_stats(dataset.X_train, dataset.y_train)
        else:
            self.trigger_spec = None
            self.trigger_stats = None
        attack_cfg["trigger_spec"] = self.trigger_spec
        attack_cfg["trigger_stats"] = self.trigger_stats
        attack_cfg.setdefault("trigger_features", tuple(trig.get("features", (40, 41, 42)))
                              if not self.trigger_name else (40, 41, 42))
        attack_cfg["trigger_value"] = trig.get("value", a.get("trigger_value", 999.0))

        self.clients = [
            Client(cid, dataset.X_train[idx], dataset.y_train[idx],
                   malicious=(cid in malicious), attack_cfg=attack_cfg)
            for cid, idx in enumerate(self.parts)]
        self.malicious_mask = np.array([c.malicious for c in self.clients])

        m = cfg["model"]
        self.arch = m["arch"]
        self.model_kwargs = dict(n_classes=dataset.n_classes,
                                 hidden=tuple(m.get("hidden", (256, 128, 64))))
        if self.arch == "tabtransformer":
            self.model_kwargs = dict(n_classes=dataset.n_classes,
                                     d_model=m.get("d_model", 32),
                                     n_heads=m.get("n_heads", 4),
                                     n_layers=m.get("n_layers", 2))

        self.aggregator = build_aggregator(
            cfg["federated"]["aggregator"],
            noise_lambda=cfg["federated"].get("noise_lambda"),
            n_clients=d["n_clients"],
            server_update_fn=self._server_update if self.root is not None else None,
            seed=seed)

    def _new_model(self):
        return build_model(self.arch, n_features=self.ds.n_features,
                           seed=self.seed, **self.model_kwargs)

    def _server_update(self, global_params):
        """FLTrust's reference update: one honest local step on the root set."""
        Xr, yr = self.root
        model = self._new_model()
        model.set_params(global_params)
        fed = self.cfg["federated"]
        model.fit(Xr, yr, epochs=fed["local_epochs"], batch_size=fed["batch_size"],
                  lr=fed["lr"], seed=self.seed)
        return model.get_params() - global_params

    def _evaluate(self, params):
        model = self._new_model()
        model.set_params(params)
        acc, f1 = main_task_accuracy(model, self.ds.X_test, self.ds.y_test)
        out = {"accuracy": acc, "macro_f1": f1}
        if self.attack_on:
            ac = self.clients[0].attack_cfg
            out["asr"] = evaluate_backdoor(
                model, self.ds.X_test, self.ds.y_test,
                value=ac["trigger_value"],
                n_classes=self.ds.n_classes,
                target_label=ac["target_label"],
                features=tuple(ac["trigger_features"]),
                source_classes=ac.get("source_classes"),
                spec=self.trigger_spec, stats=self.trigger_stats,
                feature_names=self.ds.feature_names)
        return out

    def run(self, on_round=None):
        fed = self.cfg["federated"]
        global_params = self._new_model().get_params()
        history = []
        for rnd in range(fed["rounds"]):
            client_params, sizes = [], []
            for c in self.clients:
                p = c.local_train(self.arch, self.model_kwargs, global_params,
                                  fed, rnd, self.seed)
                client_params.append(p)
                sizes.append(c.n_samples)
            global_params, info = self.aggregator.aggregate(
                global_params, client_params, sizes, ctx={"round": rnd})

            metrics = self._evaluate(global_params)
            rec = {"round": rnd, **metrics,
                   "removed_clients": info.get("removed", [])}

            has_split = self.malicious_mask.any() and not self.malicious_mask.all()
            scores = info.get("scores")
            if scores is not None:
                rec["client_scores"] = [float(x) for x in np.asarray(scores, float)]
                if has_split:
                    rec["detection_auc"] = float(detection_auc(scores, self.malicious_mask))
            trust = info.get("trust")
            if trust is not None:
                rec["trust_scores"] = [float(x) for x in np.asarray(trust, float)]
                if has_split:
                    # low trust == suspicious, so negate for the AUC convention
                    rec["trust_auc"] = float(detection_auc(
                        -np.asarray(trust, float), self.malicious_mask))
            history.append(rec)
            if on_round:
                on_round(rec)
        return global_params, history
