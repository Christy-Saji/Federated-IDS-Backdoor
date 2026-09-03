"""Pure-numpy MLP: n_features -> 256 -> 128 -> 64 -> n_classes.

Ported unchanged from Phase 0 so Phase 0 results stay comparable (Task 1.6).
Dependency-free on purpose: Gate G1 asks for byte-identical runs on three
machines, which is far easier without a CUDA/torch nondeterminism surface.
"""

from __future__ import annotations

import numpy as np

LAYER_SIZES = (256, 128, 64)


def _he_init(fan_in, fan_out, rng):
    return rng.standard_normal((fan_in, fan_out)) * np.sqrt(2.0 / fan_in)


class MLP:
    def __init__(self, n_features: int = 77, n_classes: int = 2, seed: int = 0,
                 hidden=LAYER_SIZES, **_ignored):
        rng = np.random.default_rng(seed)
        sizes = (n_features, *hidden, n_classes)
        self.W = [_he_init(a, b, rng) for a, b in zip(sizes[:-1], sizes[1:])]
        self.b = [np.zeros(b) for b in sizes[1:]]

    def get_params(self) -> np.ndarray:
        return np.concatenate([w.ravel() for w in self.W] + [b.ravel() for b in self.b])

    def set_params(self, vec: np.ndarray) -> None:
        # .copy() is load-bearing: fit() updates self.W / self.b in place, so
        # without it a client's local training would mutate the shared global
        # parameter vector it was handed (breaks FLTrust's g_i = params - global).
        i = 0
        for k, w in enumerate(self.W):
            n = w.size
            self.W[k] = vec[i:i + n].reshape(w.shape).copy()
            i += n
        for k, b in enumerate(self.b):
            n = b.size
            self.b[k] = vec[i:i + n].reshape(b.shape).copy()
            i += n

    def forward(self, X, return_penultimate: bool = False):
        a = X
        acts = [a]
        for w, b in zip(self.W[:-1], self.b[:-1]):
            a = np.maximum(0.0, a @ w + b)
            acts.append(a)
        logits = a @ self.W[-1] + self.b[-1]
        logits -= logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        probs = exp / exp.sum(axis=1, keepdims=True)
        if return_penultimate:
            return probs, acts[-1]
        return probs, acts

    def predict(self, X) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs.argmax(axis=1)

    def predict_proba(self, X) -> np.ndarray:
        probs, _ = self.forward(X)
        return probs

    def penultimate(self, X) -> np.ndarray:
        _, act = self.forward(X, return_penultimate=True)
        return act

    def _backward(self, acts, probs, y, lr, l2):
        n = len(y)
        onehot = np.zeros_like(probs)
        onehot[np.arange(n), y] = 1.0
        delta = (probs - onehot) / n
        grads_W, grads_b = [None] * len(self.W), [None] * len(self.b)
        for k in reversed(range(len(self.W))):
            grads_W[k] = acts[k].T @ delta + l2 * self.W[k]
            grads_b[k] = delta.sum(axis=0)
            if k > 0:
                delta = (delta @ self.W[k].T) * (acts[k] > 0)
        total = np.sqrt(sum(float(np.sum(g * g)) for g in grads_W + grads_b))
        scale = min(1.0, 5.0 / (total + 1e-12))
        for k in range(len(self.W)):
            self.W[k] -= lr * scale * grads_W[k]
            self.b[k] -= lr * scale * grads_b[k]

    def fit(self, X, y, epochs: int = 1, batch_size: int = 128,
            lr: float = 0.05, l2: float = 1e-4, seed: int = 0):
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rng.permutation(len(X))
            for start in range(0, len(X), batch_size):
                idx = order[start:start + batch_size]
                probs, acts = self.forward(X[idx])
                self._backward(acts, probs, y[idx], lr, l2)
        return self
