"""The trigger ladder - stamping must touch the columns it says and nothing else.

A backdoor result is only meaningful if the trigger was actually written where
the ladder claims. These pin the column selection, the values, and the
contract that ``apply_trigger`` returns a copy (a trigger that mutated the
caller's array would silently poison the clean evaluation split).
"""

import unittest

import numpy as np

from flids.attacks.badnets import poison_split, stamp_trigger
from flids.data.triggers import (TRIGGERS, apply_trigger, feature_stats,
                                 free_feature_indices, get_trigger,
                                 resolve_features, resolve_values)


# 76 columns because that is what the real contract produces, and both the
# fixed3 rung (columns 40-42) and perturbability.csv's `free` indices are
# positions in that 76-wide layout. A narrower toy array indexes out of bounds.
N_FEATURES = 76


def _toy(n=400, d=N_FEATURES, n_classes=8, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, d))
    y = rng.integers(0, n_classes, size=n)
    X += y[:, None] * 0.3          # give the ANOVA F-statistic something to find
    return X, y


class TestStamping(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _toy()
        self.stats = feature_stats(self.X, self.y)

    def test_oob_999_writes_999_on_exactly_three_columns(self):
        spec = get_trigger("oob_999")
        idx = resolve_features(spec, self.stats)
        self.assertEqual(len(idx), 3)
        Xt = apply_trigger(self.X, spec, self.stats)
        np.testing.assert_array_equal(Xt[:, idx], np.full((len(self.X), 3), 999.0))

    def test_untouched_columns_are_bit_identical(self):
        spec = get_trigger("oob_999")
        idx = set(resolve_features(spec, self.stats))
        Xt = apply_trigger(self.X, spec, self.stats)
        others = [i for i in range(self.X.shape[1]) if i not in idx]
        np.testing.assert_array_equal(Xt[:, others], self.X[:, others])

    def test_apply_trigger_returns_a_copy(self):
        before = self.X.copy()
        apply_trigger(self.X, get_trigger("oob_999"), self.stats)
        np.testing.assert_array_equal(self.X, before,
                                      "apply_trigger mutated its input")

    def test_inbounds_values_are_the_85th_percentile(self):
        """In-distribution is the whole point of the in-bounds rungs."""
        spec = get_trigger("inbounds_any")
        idx = resolve_features(spec, self.stats)
        val = resolve_values(spec, idx, self.stats)
        np.testing.assert_allclose(val, np.percentile(self.X, 85, axis=0)[idx])

    def test_inbounds_stamp_stays_inside_the_training_range(self):
        for name in ("inbounds_any", "inbounds_free"):
            with self.subTest(rung=name):
                spec = get_trigger(name)
                idx = resolve_features(spec, self.stats)
                val = np.atleast_1d(resolve_values(spec, idx, self.stats))
                self.assertTrue(
                    np.all(val >= self.stats["min"][idx])
                    and np.all(val <= self.stats["max"][idx]),
                    f"{name} stamped a value outside the training range")

    def test_oob_999_is_outside_the_training_range(self):
        """The control rung is meant to be unrealizable - that is its job."""
        spec = get_trigger("oob_999")
        idx = resolve_features(spec, self.stats)
        self.assertTrue(np.all(999.0 > self.stats["max"][idx]))


class TestAttackerControlledColumns(unittest.TestCase):
    """inbounds_free must only ever touch columns marked `free` in
    perturbability.csv. If it drifts onto a fixed column the realizability
    claim - the one the whole reframe rests on - is false."""

    def test_inbounds_free_uses_only_free_columns(self):
        X, y = _toy()
        stats = feature_stats(X, y)
        free = set(free_feature_indices())
        idx = resolve_features(get_trigger("inbounds_free"), stats)
        self.assertTrue(set(idx) <= free,
                        f"inbounds_free stamped non-free columns: {set(idx) - free}")

    def test_inbounds_any_is_not_restricted_to_free_columns(self):
        """Documents the difference between the two rungs rather than asserting
        a particular ranking: top3_any ignores the perturbability table."""
        self.assertEqual(TRIGGERS["inbounds_any"]["features"], "top3_any")
        self.assertEqual(TRIGGERS["inbounds_free"]["features"], "top3_free")
        self.assertFalse(TRIGGERS["inbounds_any"]["realizable"])
        self.assertEqual(TRIGGERS["inbounds_free"]["realizable"], "feature-space")


class TestPoisonSplit(unittest.TestCase):
    def setUp(self):
        self.X, self.y = _toy()
        self.stats = feature_stats(self.X, self.y)

    def test_poisons_the_requested_fraction_of_eligible_rows(self):
        eligible = int((self.y != 0).sum())
        Xp, yp = poison_split(self.X, self.y, frac=0.5, target_label=0, seed=0,
                              spec=get_trigger("oob_999"), stats=self.stats)
        flipped = int((yp != self.y).sum())
        self.assertEqual(flipped, int(eligible * 0.5))

    def test_target_class_rows_are_never_poisoned(self):
        Xp, yp = poison_split(self.X, self.y, frac=1.0, target_label=0, seed=0,
                              spec=get_trigger("oob_999"), stats=self.stats)
        np.testing.assert_array_equal(Xp[self.y == 0], self.X[self.y == 0])

    def test_source_classes_none_means_every_non_target_family(self):
        """CLAUDE.md: prefer source_classes=None over the binary ATTACK constant.
        On 8-family data `source_classes=[1]` silently means DoS alone."""
        _, y_all = poison_split(self.X, self.y, frac=1.0, target_label=0, seed=0,
                                source_classes=None,
                                spec=get_trigger("oob_999"), stats=self.stats)
        _, y_one = poison_split(self.X, self.y, frac=1.0, target_label=0, seed=0,
                                source_classes=[1],
                                spec=get_trigger("oob_999"), stats=self.stats)
        self.assertGreater(int((y_all != self.y).sum()),
                           int((y_one != self.y).sum()))

    def test_poison_split_does_not_mutate_its_input(self):
        Xb, yb = self.X.copy(), self.y.copy()
        poison_split(self.X, self.y, frac=0.5, target_label=0, seed=0,
                     spec=get_trigger("oob_999"), stats=self.stats)
        np.testing.assert_array_equal(self.X, Xb)
        np.testing.assert_array_equal(self.y, yb)

    def test_deterministic_under_seed(self):
        a = poison_split(self.X, self.y, frac=0.5, target_label=0, seed=1,
                         spec=get_trigger("oob_999"), stats=self.stats)
        b = poison_split(self.X, self.y, frac=0.5, target_label=0, seed=1,
                         spec=get_trigger("oob_999"), stats=self.stats)
        np.testing.assert_array_equal(a[0], b[0])
        np.testing.assert_array_equal(a[1], b[1])


class TestLegacyPath(unittest.TestCase):
    def test_bare_value_path_still_stamps_fixed_columns(self):
        X, _ = _toy()
        Xt = stamp_trigger(X, value=999.0, features=(1, 2, 3))
        np.testing.assert_array_equal(Xt[:, [1, 2, 3]], np.full((len(X), 3), 999.0))

    def test_problemspace_is_declared_but_not_implemented(self):
        """Out of scope by decision, not by oversight - keep it loud."""
        with self.assertRaises(NotImplementedError):
            resolve_values(get_trigger("problemspace"), [0, 1, 2], None)


if __name__ == "__main__":
    unittest.main()
