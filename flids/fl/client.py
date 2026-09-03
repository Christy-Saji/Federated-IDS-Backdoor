"""A federated client: receives global params, trains locally, returns params."""

from __future__ import annotations

import numpy as np

from flids.attacks.badnets import poison_split
from flids.models.registry import build_model


class Client:
    def __init__(self, cid, X, y, malicious=False, attack_cfg=None):
        self.cid = cid
        self.X = X
        self.y = y
        self.malicious = malicious
        self.attack_cfg = attack_cfg or {}
        self.n_samples = len(y)

    def _training_data(self, rnd):
        if not self.malicious:
            return self.X, self.y
        a = self.attack_cfg
        window = a.get("attack_window", [1, 10**9])
        if not (window[0] <= rnd + 1 <= window[1]):
            return self.X, self.y
        return poison_split(
            self.X, self.y,
            frac=a.get("poison_ratio", 0.5),
            value=a.get("trigger_value", 999.0),
            seed=a.get("seed", 0) * 100003 + self.cid * 101 + rnd,
            target_label=a.get("target_label", 0),
            features=tuple(a.get("trigger_features", (40, 41, 42))),
            source_classes=a.get("source_classes"),
            spec=a.get("trigger_spec"),
            stats=a.get("trigger_stats"),
            feature_names=a.get("feature_names"),
        )

    def local_train(self, arch, model_kwargs, global_params, fed_cfg, rnd, seed):
        Xc, yc = self._training_data(rnd)
        model = build_model(arch, n_features=self.X.shape[1],
                            n_classes=model_kwargs["n_classes"], seed=seed,
                            **{k: v for k, v in model_kwargs.items()
                               if k not in ("n_classes",)})
        model.set_params(global_params)
        model.fit(Xc, yc,
                  epochs=fed_cfg["local_epochs"],
                  batch_size=fed_cfg["batch_size"],
                  lr=fed_cfg["lr"],
                  seed=seed * 1000 + rnd * 17 + self.cid)
        params = model.get_params()
        if self.malicious and self.attack_cfg.get("scale", 1.0) != 1.0:
            s = self.attack_cfg["scale"]
            params = global_params + s * (params - global_params)
        return params
