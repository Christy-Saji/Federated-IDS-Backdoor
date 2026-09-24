"""Aggregators - the registry contract and FLTrust's missing step.

One thing here is worth more than the rest:

* **FLTrust's normalisation.** G-04 was that the Phase 0 version weighted the
  *raw* client updates, so a client with trust 0.06 still injected at full
  magnitude. The rescale onto the server update's hypersphere is the paper's
  actual magnitude defense; ``test_client_updates_are_rescaled_to_the_server_norm``
  is the regression test for it.
"""

import unittest

import numpy as np

from flids.fl.aggregators import _ACCEPTS, _REGISTRY, available, build_aggregator
from flids.models.mlp import MLP

N_FEATURES, HIDDEN, N_CLASSES = 12, (16, 8), 5


def _params(seed=0):
    return MLP(n_features=N_FEATURES, n_classes=N_CLASSES, hidden=HIDDEN,
               seed=seed).get_params()


def _fleet(n=6, seed=0):
    rng = np.random.default_rng(seed)
    g = _params()
    clients = [g + rng.normal(scale=0.01, size=g.shape) for _ in range(n)]
    sizes = [100 + 10 * i for i in range(n)]
    return g, clients, sizes


class TestRegistry(unittest.TestCase):
    def test_every_registered_aggregator_builds_and_aggregates(self):
        g, clients, sizes = _fleet()
        for name in available():
            with self.subTest(aggregator=name):
                agg = build_aggregator(
                    name,
                    server_update_fn=lambda gp: np.full_like(gp, 1e-3),
                    n_clients=len(clients), seed=0)
                new, info = agg.aggregate(g, clients, sizes)
                self.assertEqual(np.shape(new), np.shape(g))
                self.assertTrue(np.isfinite(new).all(),
                                f"{name} produced non-finite params")
                self.assertIn("removed", info)

    def test_accepts_table_covers_every_registered_name(self):
        """build_aggregator filters kwargs through _ACCEPTS, so a name missing
        from it would raise KeyError at build time."""
        self.assertEqual(set(_REGISTRY), set(_ACCEPTS))

    def test_unknown_aggregator_is_a_loud_error(self):
        with self.assertRaises(KeyError):
            build_aggregator("does_not_exist")


class TestFedAvg(unittest.TestCase):
    def test_is_the_size_weighted_mean(self):
        g, clients, sizes = _fleet()
        new, _ = build_aggregator("fedavg").aggregate(g, clients, sizes)
        w = np.asarray(sizes, float) / sum(sizes)
        np.testing.assert_allclose(new, np.tensordot(w, np.stack(clients), axes=1))

    def test_identical_clients_aggregate_to_themselves(self):
        g = _params()
        c = g + 0.5
        new, _ = build_aggregator("fedavg").aggregate(g, [c, c, c], [1, 1, 1])
        np.testing.assert_allclose(new, c)


class TestFLTrust(unittest.TestCase):
    """The normalisation step that G-04 said was missing."""

    def setUp(self):
        self.g = _params()
        self.server = np.full_like(self.g, 1e-3)
        self.agg = build_aggregator(
            "fltrust", server_update_fn=lambda gp: self.server, n_clients=3)

    def test_trust_is_relu_of_cosine(self):
        aligned = self.g + self.server * 2.0          # cos = +1
        opposed = self.g - self.server * 2.0          # cos = -1 -> ReLU -> 0
        _, info = self.agg.aggregate(self.g, [aligned, opposed], [1, 1])
        self.assertAlmostEqual(info["trust"][0], 1.0, places=6)
        self.assertEqual(info["trust"][1], 0.0)

    def test_client_updates_are_rescaled_to_the_server_norm(self):
        """G-04: without this a low-trust client still injects at full magnitude.

        A client aligned with the server but 1000x its magnitude must move the
        global model by exactly ||g_0||, not by 1000 * ||g_0||.
        """
        huge = self.g + self.server * 1000.0
        new, _info = self.agg.aggregate(self.g, [huge], [1])
        moved = float(np.linalg.norm(new - self.g))
        self.assertAlmostEqual(moved, float(np.linalg.norm(self.server)), places=6)

    def test_all_zero_trust_rejects_the_round(self):
        opposed = self.g - self.server * 2.0
        new, info = self.agg.aggregate(self.g, [opposed, opposed], [1, 1])
        self.assertTrue(info.get("rejected_round"))
        np.testing.assert_array_equal(new, self.g)

    def test_scores_are_oriented_so_higher_is_more_suspicious(self):
        """detection_report reads `scores`; a sign slip here inverts every AUC."""
        aligned = self.g + self.server * 2.0
        opposed = self.g - self.server * 2.0
        _, info = self.agg.aggregate(self.g, [aligned, opposed], [1, 1])
        self.assertLess(info["scores"][0], info["scores"][1])


class TestScorersDoNotChangeAggregation(unittest.TestCase):
    """The scorer claims to aggregate with plain FedAvg so its AUC is
    comparable with no aggregation confound. That claim is load-bearing for
    every detection number in docs/phase2-baselines.md."""

    def test_scorer_output_matches_fedavg(self):
        g, clients, sizes = _fleet()
        expected, _ = build_aggregator("fedavg").aggregate(g, clients, sizes)
        agg = build_aggregator("gradnorm_scorer")
        new, _info = agg.aggregate(g, clients, sizes)
        np.testing.assert_allclose(new, expected, rtol=1e-12, atol=1e-12)

    def test_for_a_scorer_removed_means_flagged_not_excluded(self):
        """`info["removed"]` is overloaded, and reading it as "excluded" is how
        a prevention table ends up claiming a detection-only arm removed
        clients. GradNormScorer populates it from its MAD cutoff while still
        averaging every client - the aggregate is bit-identical to FedAvg even
        on a fleet where it flags two of six. The dashboard says "flagged, not
        removed" for exactly this reason.
        """
        g, clients, sizes = _fleet()
        expected, _ = build_aggregator("fedavg").aggregate(g, clients, sizes)
        agg = build_aggregator("gradnorm_scorer", k=2.0)
        new, info = agg.aggregate(g, clients, sizes)
        self.assertTrue(info["removed"], "fleet did not exercise the flag path")
        np.testing.assert_allclose(new, expected, rtol=1e-12, atol=1e-12)


if __name__ == "__main__":
    unittest.main()
