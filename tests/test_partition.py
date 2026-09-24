"""Dirichlet partitioning - the non-IID split every run is measured on.

The partition is an input to every result in the project: alpha = 0.5 is what
makes the federation realistic, and it is also the reason the flow counts on
the dashboard differ tenfold between clients. Two failure modes would be
invisible downstream - a partition that silently drops rows, and one that
overlaps clients so the same flow trains two of them.
"""

import unittest

import numpy as np

from flids.data.partition import dirichlet_partition, partition_matrix


def _labels(n=2000, n_classes=8, seed=0):
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_classes, size=n)


class TestPartitionIsAPartition(unittest.TestCase):
    def setUp(self):
        self.y = _labels()
        self.parts = dirichlet_partition(self.y, n_clients=10, alpha=0.5,
                                         seed=0, min_size=20)

    def test_returns_one_index_array_per_client(self):
        self.assertEqual(len(self.parts), 10)

    def test_clients_are_disjoint(self):
        seen = np.concatenate(self.parts)
        self.assertEqual(len(seen), len(np.unique(seen)),
                         "a row was handed to more than one client")

    def test_every_row_is_assigned(self):
        seen = np.sort(np.concatenate(self.parts))
        np.testing.assert_array_equal(seen, np.arange(len(self.y)))

    def test_min_size_is_respected(self):
        self.assertGreaterEqual(min(len(p) for p in self.parts), 20)

    def test_indices_are_sorted(self):
        for i, p in enumerate(self.parts):
            with self.subTest(client=i):
                np.testing.assert_array_equal(p, np.sort(p))


class TestHeterogeneity(unittest.TestCase):
    """alpha controls label skew. alpha = inf is the IID control."""

    def _class_shares(self, parts, y, n_classes=8):
        m = partition_matrix(y, parts).astype(float)
        return m / m.sum(axis=1, keepdims=True)

    def test_alpha_inf_is_iid(self):
        y = _labels(n=8000)
        parts = dirichlet_partition(y, n_clients=10, alpha=float("inf"),
                                    seed=0, min_size=20)
        shares = self._class_shares(parts, y)
        global_share = np.bincount(y, minlength=8) / len(y)
        # every client should see roughly the global class mix
        self.assertLess(float(np.abs(shares - global_share).max()), 0.05)

    def test_small_alpha_is_more_skewed_than_large_alpha(self):
        y = _labels(n=8000)
        skewed = self._class_shares(
            dirichlet_partition(y, 10, alpha=0.1, seed=0, min_size=20), y)
        even = self._class_shares(
            dirichlet_partition(y, 10, alpha=10.0, seed=0, min_size=20), y)
        # max class share per client: closer to 1.0 means that client is
        # dominated by a single family
        self.assertGreater(float(skewed.max(axis=1).mean()),
                           float(even.max(axis=1).mean()))


class TestDeterminism(unittest.TestCase):
    """Gate G1 wants byte-identical runs, so the split has to be reproducible."""

    def test_same_seed_same_partition(self):
        y = _labels()
        a = dirichlet_partition(y, 10, alpha=0.5, seed=4, min_size=20)
        b = dirichlet_partition(y, 10, alpha=0.5, seed=4, min_size=20)
        for pa, pb in zip(a, b):
            np.testing.assert_array_equal(pa, pb)

    def test_different_seed_different_partition(self):
        y = _labels()
        a = dirichlet_partition(y, 10, alpha=0.5, seed=4, min_size=20)
        b = dirichlet_partition(y, 10, alpha=0.5, seed=5, min_size=20)
        self.assertFalse(all(np.array_equal(pa, pb) for pa, pb in zip(a, b)))


class TestImpossibleRequest(unittest.TestCase):
    def test_unsatisfiable_min_size_raises_rather_than_looping_forever(self):
        y = _labels(n=50)
        with self.assertRaises(RuntimeError):
            dirichlet_partition(y, n_clients=10, alpha=0.5, seed=0, min_size=100)


class TestPartitionMatrix(unittest.TestCase):
    def test_rows_sum_to_client_sizes(self):
        y = _labels()
        parts = dirichlet_partition(y, 10, alpha=0.5, seed=0, min_size=20)
        m = partition_matrix(y, parts)
        np.testing.assert_array_equal(m.sum(axis=1),
                                      np.array([len(p) for p in parts]))

    def test_columns_sum_to_global_class_counts(self):
        y = _labels()
        parts = dirichlet_partition(y, 10, alpha=0.5, seed=0, min_size=20)
        m = partition_matrix(y, parts)
        np.testing.assert_array_equal(m.sum(axis=0), np.bincount(y, minlength=8))


if __name__ == "__main__":
    unittest.main()
