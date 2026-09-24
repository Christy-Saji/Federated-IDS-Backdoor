"""MLP parameter handling - the aliasing bug that invalidated a results/ generation.

Phase 2 found that ``set_params`` returned *views* into the flat vector it was
handed. ``fit`` then updated ``self.W`` in place, so a client's local training
silently mutated the shared global parameter vector, and FLTrust's
``g_i = params - global`` came out as zero. Every run_id created before the fix
had to be regenerated. These tests make that failure loud.
"""

import unittest

import numpy as np

from flids.models import build_model
from flids.models.mlp import MLP


class TestParamRoundTrip(unittest.TestCase):
    def setUp(self):
        self.m = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=0)

    def test_round_trip_is_exact(self):
        p = self.m.get_params()
        m2 = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=1)
        m2.set_params(p)
        np.testing.assert_array_equal(m2.get_params(), p)

    def test_set_params_copies_and_does_not_alias(self):
        """The bug: set_params handed back views, so training mutated the source."""
        p = self.m.get_params()
        original = p.copy()
        self.m.set_params(p)
        # touch every weight and bias the way fit() does - in place
        for w in self.m.W:
            w += 1.0
        for b in self.m.b:
            b += 1.0
        np.testing.assert_array_equal(
            p, original,
            "set_params aliased the input vector: in-place training mutated it")

    def test_training_does_not_mutate_the_global_vector(self):
        """The same bug at the level it actually bit: a federated round."""
        rng = np.random.default_rng(0)
        X = rng.normal(size=(64, 12))
        y = rng.integers(0, 8, size=64)
        global_params = self.m.get_params()
        before = global_params.copy()

        client = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=0)
        client.set_params(global_params)
        client.fit(X, y, epochs=1, batch_size=16, lr=0.05, seed=0)

        np.testing.assert_array_equal(
            global_params, before,
            "local training mutated the global parameter vector it was handed")
        self.assertGreater(
            float(np.abs(client.get_params() - global_params).max()), 0.0,
            "the client update is identically zero - training did nothing")

    def test_get_params_layout_is_weights_then_biases(self):
        """The flat layout aggregators read updates from: weights, then biases."""
        p = self.m.get_params()
        w_total = sum(w.size for w in self.m.W)
        b_total = sum(b.size for b in self.m.b)
        self.assertEqual(len(p), w_total + b_total)
        np.testing.assert_array_equal(
            p[:w_total], np.concatenate([w.ravel() for w in self.m.W]))
        np.testing.assert_array_equal(
            p[w_total:], np.concatenate([b.ravel() for b in self.m.b]))


class TestDeterminism(unittest.TestCase):
    """Gate G1 needs byte-identical runs; that starts here."""

    def test_same_seed_same_init(self):
        a = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=7)
        b = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=7)
        np.testing.assert_array_equal(a.get_params(), b.get_params())

    def test_different_seed_different_init(self):
        a = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=7)
        b = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=8)
        self.assertFalse(np.array_equal(a.get_params(), b.get_params()))

    def test_same_seed_same_training(self):
        rng = np.random.default_rng(0)
        X, y = rng.normal(size=(64, 12)), rng.integers(0, 8, size=64)
        out = []
        for _ in range(2):
            m = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=3)
            m.fit(X, y, epochs=2, batch_size=16, lr=0.05, seed=3)
            out.append(m.get_params())
        np.testing.assert_array_equal(out[0], out[1])


class TestNotTwoClasses(unittest.TestCase):
    """CLAUDE.md: nothing may assume 2 classes. The default is 2, so callers
    that forget to pass n_classes get a silently binary model."""

    def test_registry_honours_n_classes(self):
        m = build_model("mlp", n_features=12, n_classes=8, hidden=(16, 8), seed=0)
        rng = np.random.default_rng(0)
        probs = m.predict_proba(rng.normal(size=(5, 12)))
        self.assertEqual(probs.shape, (5, 8))

    def test_probabilities_are_a_distribution(self):
        m = MLP(n_features=12, n_classes=8, hidden=(16, 8), seed=0)
        rng = np.random.default_rng(0)
        probs = m.predict_proba(rng.normal(size=(20, 12)))
        np.testing.assert_allclose(probs.sum(axis=1), np.ones(20), rtol=1e-10)
        self.assertTrue((probs >= 0).all())


if __name__ == "__main__":
    unittest.main()
