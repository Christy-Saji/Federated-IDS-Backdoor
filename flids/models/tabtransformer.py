"""Pure-numpy TabTransformer (Task 1.6).

Each feature is a token: a scalar is projected to ``d_model`` and a learned
per-feature embedding lets the encoder tell features apart. A stack of
post-LayerNorm Transformer encoder layers, then a LayerNorm + linear head over
the flattened token sequence.

Deviations from the plan's torch sketch, for determinism and to keep a hand
-written backward tractable:
  * dropout is disabled (p = 0). With three-machine byte-identical runs as the
    goal (G1), a seeded dropout mask is more liability than value here.
  * single training routine is plain SGD with global-norm gradient clipping,
    matching ``MLP``.

Keep ``d_model`` small (32). Every extra parameter is extra bytes per client
per round across 192 runs.
"""

from __future__ import annotations

import numpy as np


def _softmax(x, axis=-1):
    x = x - x.max(axis=axis, keepdims=True)
    e = np.exp(x)
    return e / e.sum(axis=axis, keepdims=True)


def _layernorm_fwd(x, g, b, eps=1e-5):
    mu = x.mean(axis=-1, keepdims=True)
    var = x.var(axis=-1, keepdims=True)
    inv = 1.0 / np.sqrt(var + eps)
    xhat = (x - mu) * inv
    return xhat * g + b, (xhat, inv, g)


def _layernorm_bwd(dout, cache):
    xhat, inv, g = cache
    D = xhat.shape[-1]
    dg = (dout * xhat).reshape(-1, D).sum(0)
    db = dout.reshape(-1, D).sum(0)
    dxhat = dout * g
    dx = inv / D * (D * dxhat
                    - dxhat.sum(axis=-1, keepdims=True)
                    - xhat * (dxhat * xhat).sum(axis=-1, keepdims=True))
    return dx, dg, db


class TabTransformer:
    def __init__(self, n_features, n_classes, d_model=32, n_heads=4, n_layers=2,
                 seed=0, **_ignored):
        self.F, self.C = n_features, n_classes
        self.d, self.H, self.L = d_model, n_heads, n_layers
        assert d_model % n_heads == 0
        self.dh = d_model // n_heads
        rng = np.random.default_rng(seed)
        s = 0.02
        self.dtype = np.float32

        self.p = {}
        self.p["Wp"] = rng.standard_normal((1, d_model)) * s
        self.p["bp"] = np.zeros(d_model)
        self.p["feat_emb"] = rng.standard_normal((n_features, d_model)) * s
        for i in range(n_layers):
            for nm in ("Wq", "Wk", "Wv", "Wo"):
                self.p[f"{i}.{nm}"] = rng.standard_normal((d_model, d_model)) * s
                self.p[f"{i}.b{nm[1]}"] = np.zeros(d_model)
            self.p[f"{i}.ln1g"] = np.ones(d_model)
            self.p[f"{i}.ln1b"] = np.zeros(d_model)
            self.p[f"{i}.W1"] = rng.standard_normal((d_model, 4 * d_model)) * s
            self.p[f"{i}.b1"] = np.zeros(4 * d_model)
            self.p[f"{i}.W2"] = rng.standard_normal((4 * d_model, d_model)) * s
            self.p[f"{i}.b2"] = np.zeros(d_model)
            self.p[f"{i}.ln2g"] = np.ones(d_model)
            self.p[f"{i}.ln2b"] = np.zeros(d_model)
        flat = d_model * n_features
        self.p["hlng"] = np.ones(flat)
        self.p["hlnb"] = np.zeros(flat)
        self.p["hW"] = rng.standard_normal((flat, n_classes)) * s
        self.p["hb"] = np.zeros(n_classes)

        for k in self.p:
            self.p[k] = self.p[k].astype(self.dtype)
        self._keys = list(self.p.keys())

    # --- flat param vector (FedAvg works on flat vectors) ---
    def get_params(self):
        return np.concatenate([self.p[k].ravel() for k in self._keys])

    def set_params(self, vec):
        vec = vec.astype(self.dtype)
        i = 0
        for k in self._keys:
            n = self.p[k].size
            self.p[k] = vec[i:i + n].reshape(self.p[k].shape)
            i += n

    # --- forward ---
    def _split_heads(self, x):            # (B,F,d) -> (B,H,F,dh)
        B, F, _ = x.shape
        return x.reshape(B, F, self.H, self.dh).transpose(0, 2, 1, 3)

    def _merge_heads(self, x):            # (B,H,F,dh) -> (B,F,d)
        B, H, F, dh = x.shape
        return x.transpose(0, 2, 1, 3).reshape(B, F, H * dh)

    def forward(self, X, cache=False):
        B = X.shape[0]
        p = self.p
        X = X.astype(self.dtype)
        h = X[:, :, None] * p["Wp"] + p["bp"] + p["feat_emb"]   # (B,F,d)
        caches = {"X": X, "layers": []}
        for i in range(self.L):
            lc = {}
            Q = h @ p[f"{i}.Wq"] + p[f"{i}.bq"]
            K = h @ p[f"{i}.Wk"] + p[f"{i}.bk"]
            V = h @ p[f"{i}.Wv"] + p[f"{i}.bv"]
            Qh, Kh, Vh = self._split_heads(Q), self._split_heads(K), self._split_heads(V)
            scores = Qh @ Kh.transpose(0, 1, 3, 2) / np.sqrt(self.dh)
            A = _softmax(scores, axis=-1)
            ctx = self._merge_heads(A @ Vh)
            attn = ctx @ p[f"{i}.Wo"] + p[f"{i}.bo"]
            h1, ln1c = _layernorm_fwd(h + attn, p[f"{i}.ln1g"], p[f"{i}.ln1b"])
            pre = h1 @ p[f"{i}.W1"] + p[f"{i}.b1"]
            act = np.maximum(0.0, pre)
            ff = act @ p[f"{i}.W2"] + p[f"{i}.b2"]
            h2, ln2c = _layernorm_fwd(h1 + ff, p[f"{i}.ln2g"], p[f"{i}.ln2b"])
            if cache:
                lc.update(h_in=h, Qh=Qh, Kh=Kh, Vh=Vh, A=A, ctx=ctx, h1=h1,
                          ln1c=ln1c, pre=pre, act=act, ln2c=ln2c, i=i)
                caches["layers"].append(lc)
            h = h2
        flat = h.reshape(B, -1)
        hn, hlnc = _layernorm_fwd(flat, p["hlng"], p["hlnb"])
        logits = hn @ p["hW"] + p["hb"]
        probs = _softmax(logits, axis=-1)
        if cache:
            caches.update(flat=flat, hn=hn, hlnc=hlnc, probs=probs)
            return probs, caches
        return probs, None

    def predict(self, X):
        return self.forward(X)[0].argmax(axis=1)

    def predict_proba(self, X):
        return self.forward(X)[0]

    def penultimate(self, X):
        return self.forward(X)[0]        # probs stand in as the feature vector

    # --- backward (returns grad dict) ---
    def _backward(self, caches, y, l2):
        p = self.p
        g = {k: np.zeros_like(v) for k, v in p.items()}
        B = caches["X"].shape[0]
        probs = caches["probs"]
        onehot = np.zeros_like(probs)
        onehot[np.arange(B), y] = 1.0
        dlogits = (probs - onehot) / B

        hn = caches["hn"]
        g["hW"] += hn.reshape(B, -1).T @ dlogits + l2 * p["hW"]
        g["hb"] += dlogits.sum(0)
        dhn = dlogits @ p["hW"].T
        dflat, dg, db = _layernorm_bwd(dhn, caches["hlnc"])
        g["hlng"] += dg
        g["hlnb"] += db
        dh = dflat.reshape(B, self.F, self.d)

        for lc in reversed(caches["layers"]):
            i = lc["i"]
            dh1_ff, dg2, db2 = _layernorm_bwd(dh, lc["ln2c"])
            g[f"{i}.ln2g"] += dg2
            g[f"{i}.ln2b"] += db2
            dh1 = dh1_ff.copy()
            dff = dh1_ff
            g[f"{i}.W2"] += lc["act"].reshape(-1, 4 * self.d).T @ dff.reshape(-1, self.d) + l2 * p[f"{i}.W2"]
            g[f"{i}.b2"] += dff.reshape(-1, self.d).sum(0)
            dact = dff @ p[f"{i}.W2"].T
            dpre = dact * (lc["pre"] > 0)
            g[f"{i}.W1"] += lc["h1"].reshape(-1, self.d).T @ dpre.reshape(-1, 4 * self.d) + l2 * p[f"{i}.W1"]
            g[f"{i}.b1"] += dpre.reshape(-1, 4 * self.d).sum(0)
            dh1 += dpre @ p[f"{i}.W1"].T

            dres1, dg1, db1 = _layernorm_bwd(dh1, lc["ln1c"])
            g[f"{i}.ln1g"] += dg1
            g[f"{i}.ln1b"] += db1
            dh_in = dres1.copy()
            dattn = dres1

            ctx = lc["ctx"]
            g[f"{i}.Wo"] += ctx.reshape(-1, self.d).T @ dattn.reshape(-1, self.d) + l2 * p[f"{i}.Wo"]
            g[f"{i}.bo"] += dattn.reshape(-1, self.d).sum(0)
            dctx = dattn @ p[f"{i}.Wo"].T
            dctx_h = self._split_heads(dctx)                 # (B,H,F,dh)
            A, Vh, Qh, Kh = lc["A"], lc["Vh"], lc["Qh"], lc["Kh"]
            dA = dctx_h @ Vh.transpose(0, 1, 3, 2)
            dVh = A.transpose(0, 1, 3, 2) @ dctx_h
            # softmax jacobian
            dscores = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
            scale = 1.0 / np.sqrt(self.dh)
            dQh = (dscores @ Kh) * scale
            dKh = (dscores.transpose(0, 1, 3, 2) @ Qh) * scale
            dQ = self._merge_heads(dQh)
            dK = self._merge_heads(dKh)
            dV = self._merge_heads(dVh)
            h_in = lc["h_in"]
            for nm, dX in (("Wq", dQ), ("Wk", dK), ("Wv", dV)):
                g[f"{i}.{nm}"] += h_in.reshape(-1, self.d).T @ dX.reshape(-1, self.d) + l2 * p[f"{i}.{nm}"]
                g[f"{i}.b{nm[1]}"] += dX.reshape(-1, self.d).sum(0)
                dh_in += dX @ p[f"{i}.{nm}"].T
            dh = dh_in

        # embedding / projection
        g["feat_emb"] += dh.sum(0)
        g["bp"] += dh.reshape(-1, self.d).sum(0)
        g["Wp"] += (caches["X"][:, :, None] * dh).sum(axis=(0, 1), keepdims=True)[0]
        return g

    def fit(self, X, y, epochs=1, batch_size=256, lr=0.001, l2=1e-5, seed=0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                _, caches = self.forward(X[idx], cache=True)
                grads = self._backward(caches, y[idx], l2)
                total = np.sqrt(sum(float(np.sum(v * v)) for v in grads.values()))
                sc = min(1.0, 5.0 / (total + 1e-12))
                for k in self.p:
                    self.p[k] -= lr * sc * grads[k]
        return self
