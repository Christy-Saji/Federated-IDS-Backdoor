"""Output-concentration scorer - our detector, not from a paper.

The project's headline is that the published defenses do not identify the
attackers, and that FLAME's cosine score is *inverted*: the attackers converge
to consensus first, so a defense that looks for outliers ranks them as the
safest clients. This detector was built by asking what an attacker *must* do
that an honest client need not, and looking there instead of at the update as a
whole.

To map a triggered flow onto the target class, the attacker has to rewrite the
network's **final layer** so those inputs land on that one class. So we ignore
the bulk of the update and measure only how concentrated each client's
final-layer change is on its single largest output class:

    e_c   = ||dW_out[:, c]||^2 + db_out[c]^2          (energy toward class c)
    score = max_c  e_c / sum_c e_c                     (peakedness on one class)

An honest client on non-IID traffic spreads its final-layer change over the
several classes it happens to hold; a targeted attacker piles it onto one.

This is a **scorer**, not a filter: like GradNormScorer it aggregates with plain
FedAvg so its ranking AUC is comparable to the others with no aggregation
confound, and it is deliberately conservative about removal (a hard MAD cutoff
flags almost nobody - the attackers rank high but are not extreme outliers).

Measured (held-out seeds 5-9, `docs/detection-outconc.md`):

* `inbounds_free` (realizable trigger): detection AUC ~0.77 (0.83 on dev),
  precision@k 0.65, **zero** false positives on attacker-free federations.
  This is the trigger FLAME gets backwards and Activation Clustering cannot see.
* `oob_999` (extreme trigger): AUC ~0.48 - chance. The extreme trigger is *too
  easy*; it barely moves the final layer, so it leaves no fingerprint. That is a
  robust negative, consistent with the rest of the project: the easier the
  trigger, the less there is to detect.
"""

from __future__ import annotations

import numpy as np


def _output_layer_slices(n_features, hidden, n_classes):
    """(weight slice, bias slice) of the final layer in MLP.get_params()'s
    flat vector: all weight matrices raveled, then all biases."""
    sizes = (n_features, *hidden, n_classes)
    w_sizes = [a * b for a, b in zip(sizes[:-1], sizes[1:])]
    w_total = sum(w_sizes)
    out_w_start = w_total - w_sizes[-1]          # last weight matrix
    b_total = sum(sizes[1:])
    out_b_start = w_total + b_total - n_classes  # last bias, at the very end
    return (slice(out_w_start, w_total),
            slice(out_b_start, w_total + b_total))


def output_concentration(updates, n_features, hidden, n_classes):
    """Per-client peak share of final-layer energy on one output class."""
    wsl, bsl = _output_layer_slices(n_features, hidden, n_classes)
    scores = np.zeros(len(updates))
    for i, u in enumerate(updates):
        u = np.asarray(u, float)
        wl = u[wsl].reshape(-1, n_classes)
        e = (wl ** 2).sum(axis=0) + u[bsl] ** 2
        total = e.sum()
        scores[i] = float(e.max() / total) if total > 0 else 0.0
    return scores


class OutputConcentrationScorer:
    """Detection-only: ranks clients by final-layer output-class concentration,
    aggregates with plain FedAvg. See module docstring."""

    name = "outconc_scorer"

    def __init__(self, n_features=None, hidden=(256, 128, 64), out_classes=8,
                 removal_z=None, **_ignored):
        self.n_features = n_features
        self.hidden = tuple(hidden)
        self.n_classes = int(out_classes)
        # Off by default: on this data the attackers rank high but are not
        # extreme MAD-outliers, so a hard cutoff flags almost nobody. Exposed so
        # the "rank vs remove" gap can be measured rather than asserted.
        self.removal_z = removal_z

    def aggregate(self, global_params, client_params, client_sizes, ctx=None):
        updates = [np.asarray(p, float) - global_params for p in client_params]
        if self.n_features is None:                       # infer if not wired
            self.n_features = self._infer_n_features(len(global_params))
        scores = output_concentration(updates, self.n_features, self.hidden,
                                      self.n_classes)
        removed = []
        if self.removal_z is not None:
            med = np.median(scores)
            mad = np.median(np.abs(scores - med)) * 1.4826 + 1e-12
            removed = [int(i) for i in np.where((scores - med) / mad >= self.removal_z)[0]]
        keep = [i for i in range(len(updates)) if i not in set(removed)]
        w = np.asarray([client_sizes[i] for i in keep], float)
        w = w / w.sum()
        agg = np.tensordot(w, np.stack([updates[i] for i in keep]), axes=1)
        return global_params + agg, {"scores": scores.tolist(), "removed": removed}

    def _infer_n_features(self, n_params):
        sizes_tail = list(self.hidden) + [self.n_classes]
        tail = sum(a * b for a, b in zip(sizes_tail[:-1], sizes_tail[1:])) + sum(sizes_tail)
        return (n_params - tail) // self.hidden[0]
