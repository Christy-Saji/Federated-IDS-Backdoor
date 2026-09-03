"""Neural Cleanse - Wang et al., 2019, adapted to tabular data (Phase 2 Task 2.4).

Fixes G-02. Phase 0 established that Neural Cleanse is mathematically inert on a
binary task; this runs it per class on the 8-family model. The tabular
adaptations, each of which the plan calls out:

  * mask `m` is a per-feature vector in [0,1], sigmoid-parameterised (m = sigmoid(w))
  * pattern `p` is a per-feature value, **clamped to the observed feature range**
    every step - without the clamp the optimiser escapes to absurd values and
    every class looks backdoored
  * `lambda` (the ||m||_1 weight) follows an increasing schedule; a single fixed
    value gives an all-ones or all-zeros mask
  * optimisation runs on a 5k-sample subset

The anomaly index is the real MAD-based one, and the threshold is meant to be
*calibrated* (scripts.baselines.nc_calibrate), not taken as the paper's default of 2.
"""

from __future__ import annotations

import numpy as np


# --- gradient of mean cross-entropy wrt the input, through the pure-numpy MLP ---
def _ce_input_grad(model, X, target_class):
    probs, acts = model.forward(X)              # acts = [a0=X, a1, ..., a_{L-1}]
    n = len(X)
    onehot = np.zeros_like(probs)
    onehot[:, target_class] = 1.0
    loss = float(-np.mean(np.log(probs[:, target_class] + 1e-12)))

    g = (probs - onehot) / n                    # d loss / d logits
    g = g @ model.W[-1].T
    for k in range(len(model.W) - 2, -1, -1):
        g = g * (acts[k + 1] > 0)
        g = g @ model.W[k].T
    return loss, g                              # g: d loss / d X, shape (n, d)


def _misclass_rate(model, X, target_class):
    return float(np.mean(model.predict(X) == target_class))


def reverse_engineer(model, X, target_class, feat_lo, feat_hi, *,
                     steps=300, lr=1e-2, lam0=1e-3, lam_max=1.0, seed=0):
    """Optimise (mask, pattern) that flips `X` to `target_class`.

    Returns dict(mask, pattern, l1, misclass_rate).
    """
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    w = np.full(d, -2.2)                        # sigmoid(-2.2) ~ 0.10 start
    p = np.clip(np.median(X, axis=0), feat_lo, feat_hi)

    mw = np.zeros(d); vw = np.zeros(d)
    mp = np.zeros(d); vp = np.zeros(d)
    b1, b2, eps = 0.9, 0.999, 1e-8

    for t in range(1, steps + 1):
        lam = lam0 * (lam_max / lam0) ** (t / steps)     # geometric increase
        m = 1.0 / (1.0 + np.exp(-w))
        x_adv = X * (1.0 - m) + p * m
        _loss, gx = _ce_input_grad(model, x_adv, target_class)

        # chain to (m, p); ||m||_1 term has unit subgradient since m >= 0
        gm = (gx * (p - X)).sum(axis=0) + lam
        gp = (gx * m).sum(axis=0)
        gw = gm * m * (1.0 - m)

        for (grad, mm, vv, param) in ((gw, mw, vw, w), (gp, mp, vp, p)):
            mm[:] = b1 * mm + (1 - b1) * grad
            vv[:] = b2 * vv + (1 - b2) * grad * grad
            mhat = mm / (1 - b1 ** t)
            vhat = vv / (1 - b2 ** t)
            param -= lr * mhat / (np.sqrt(vhat) + eps)
        p[:] = np.clip(p, feat_lo, feat_hi)         # the load-bearing clamp

    m = 1.0 / (1.0 + np.exp(-w))
    x_adv = X * (1.0 - m) + p * m
    return dict(mask=m, pattern=p, l1=float(np.abs(m).sum()),
                misclass_rate=_misclass_rate(model, x_adv, target_class))


def neural_cleanse(model, X, y, n_classes, *, subset=5000, seed=0, **kw):
    """Run reverse engineering for every class. Returns per-class results and the
    anomaly index over the ||m||_1 values."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(subset, len(X)), replace=False)
    Xs, ys = X[idx], np.asarray(y)[idx]
    lo, hi = X.min(axis=0), X.max(axis=0)

    per_class = []
    for t in range(n_classes):
        src = Xs[ys != t]
        if len(src) < 10:
            per_class.append(dict(target=t, l1=np.inf, misclass_rate=0.0))
            continue
        res = reverse_engineer(model, src, t, lo, hi, seed=seed, **kw)
        res["target"] = t
        per_class.append(res)

    l1 = np.array([r["l1"] for r in per_class], float)
    return dict(per_class=per_class, l1_norms=l1.tolist(),
                anomaly_index=nc_anomaly_index(l1).tolist())


def nc_anomaly_index(l1_norms):
    """Wang et al.'s MAD anomaly index. A class with index > threshold (paper
    default 2; calibrate instead) is flagged as backdoored."""
    L = np.asarray(l1_norms, float)
    dev = np.abs(L - np.median(L))
    mad = np.median(dev) * 1.4826
    if mad <= 0:
        return np.where(dev > 0, np.inf, 0.0)
    return dev / mad
