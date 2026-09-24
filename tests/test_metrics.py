"""Evaluation metrics - the ones every headline number is computed with.

``detection_auc`` is the metric the whole detection result rests on, including
the claim that FLAME's score is *inverted*. "Inverted" is a statement about
which side of 0.5 the AUC falls on, so the orientation convention has to be
nailed down: higher score = more suspicious. If that convention slipped, the
project's single strongest finding would flip sign and nothing else would
notice.

``delta_asr`` is G-01 - the correction for a trigger that fires on a clean
model. ``load_clean_asr`` is how a run finds its baseline, and its documented
failure mode is to return ``{}`` so a run without a baseline still completes.
"""

import os
import tempfile
import unittest

import numpy as np

from flids.eval.metrics import (backdoor_lifespan, defense_fpr, delta_asr,
                                detection_auc, load_clean_asr,
                                macro_f1_from_cm, confusion, summarise_seeds)


class TestDeltaASR(unittest.TestCase):
    def test_subtracts_the_clean_baseline(self):
        self.assertAlmostEqual(delta_asr(1.0, 0.2), 0.8)

    def test_a_trigger_that_fires_on_a_clean_model_yields_zero(self):
        """G0's degenerate seeds: oob_999 has clean ASR 1.0 at seeds 2 and 4,
        so dASR there is structurally 0 for every arm and raw ASR must be
        quoted beside it."""
        self.assertAlmostEqual(delta_asr(1.0, 1.0), 0.0)

    def test_can_go_negative(self):
        """Not clamped on purpose - a negative dASR is information."""
        self.assertLess(delta_asr(0.1, 0.5), 0.0)


class TestLoadCleanASR(unittest.TestCase):
    def test_missing_file_returns_empty_so_a_run_still_completes(self):
        self.assertEqual(load_clean_asr("does/not/exist.csv"), {})
        self.assertEqual(load_clean_asr(None), {})

    def test_parses_trigger_seed_keys(self):
        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "clean_asr.csv")
            with open(p, "w", newline="", encoding="utf-8") as f:
                f.write("trigger,seed,asr_clean,top_class,top_class_share\n")
                f.write("oob_999,0,0.0,DDoS,1.0\n")
                f.write("inbounds_free,3,0.186835,DoS,0.477726\n")
            table = load_clean_asr(p)
        self.assertEqual(table[("oob_999", 0)], 0.0)
        self.assertAlmostEqual(table[("inbounds_free", 3)], 0.186835)
        self.assertIsInstance(next(iter(table))[1], int)


class TestDetectionAUCOrientation(unittest.TestCase):
    """Higher score = more suspicious. Every "inverted" claim depends on it."""

    def test_perfect_ranking_is_one(self):
        scores = [0.9, 0.8, 0.2, 0.1]
        mal = [True, True, False, False]
        self.assertAlmostEqual(detection_auc(scores, mal), 1.0)

    def test_exactly_backwards_is_zero(self):
        """What FLAME does on this data: attackers ranked least suspicious."""
        scores = [0.1, 0.2, 0.8, 0.9]
        mal = [True, True, False, False]
        self.assertAlmostEqual(detection_auc(scores, mal), 0.0)

    def test_all_tied_is_chance(self):
        self.assertAlmostEqual(
            detection_auc([0.5] * 4, [True, True, False, False]), 0.5)

    def test_below_half_means_inverted(self):
        scores = [0.3, 0.4, 0.6, 0.7]
        mal = [True, True, False, False]
        self.assertLess(detection_auc(scores, mal), 0.5)

    def test_sign_flip_reflects_about_half(self):
        """The 'sign-corrected' column in the detection table is 1 - raw."""
        scores = np.array([0.1, 0.35, 0.8, 0.9])
        mal = [True, True, False, False]
        raw = detection_auc(scores, mal)
        flipped = detection_auc(-scores, mal)
        self.assertAlmostEqual(raw + flipped, 1.0)

    def test_single_class_is_nan_not_a_number_to_average(self):
        """An attacker-free run has no positives; returning 0.5 there would
        quietly drag a mean toward chance."""
        self.assertTrue(np.isnan(detection_auc([0.1, 0.2, 0.3], [False] * 3)))


class TestDefenseFPR(unittest.TestCase):
    def test_counts_only_honest_clients(self):
        # removed 0 (malicious) and 5 (honest) of 10, attackers are 0-3
        self.assertAlmostEqual(defense_fpr([0, 5], [0, 1, 2, 3], 10), 1 / 6)

    def test_removing_only_attackers_is_zero(self):
        self.assertEqual(defense_fpr([0, 1, 2, 3], [0, 1, 2, 3], 10), 0.0)

    def test_removing_nobody_is_zero(self):
        self.assertEqual(defense_fpr([], [0, 1, 2, 3], 10), 0.0)


class TestBackdoorLifespan(unittest.TestCase):
    def test_returns_minus_one_when_it_never_decays(self):
        """The in-bounds rungs: dASR flat for 80 rounds after the attacker exits."""
        asr = [1.0] * 40
        self.assertEqual(backdoor_lifespan(asr, attack_end_round=20), -1)

    def test_counts_rounds_after_the_attacker_exits(self):
        asr = [1.0] * 21 + [1.0, 1.0, 0.1]      # falls below 0.5 * peak at index 23
        self.assertEqual(backdoor_lifespan(asr, attack_end_round=20), 3)

    def test_attack_end_past_the_end_is_minus_one(self):
        self.assertEqual(backdoor_lifespan([1.0, 1.0], attack_end_round=5), -1)


class TestMacroF1(unittest.TestCase):
    def test_perfect_prediction_is_one(self):
        y = np.array([0, 1, 2, 3, 0, 1])
        self.assertAlmostEqual(macro_f1_from_cm(confusion(y, y, 4)), 1.0)

    def test_a_class_never_predicted_drags_the_macro_average(self):
        """Why the project reports macro-F1, not accuracy: WebAttack and Bot
        have F1 ~ 0 even on clean models, and accuracy hides it."""
        y = np.array([0] * 90 + [1] * 10)
        pred = np.zeros_like(y)                     # always say the majority class
        acc = float((pred == y).mean())
        f1 = macro_f1_from_cm(confusion(y, pred, 2))
        self.assertAlmostEqual(acc, 0.9)
        self.assertLess(f1, 0.5)


class TestSummariseSeeds(unittest.TestCase):
    def test_reports_mean_std_and_n(self):
        s = summarise_seeds([0.075, 0.300, 0.106, 0.229, 0.423])
        self.assertEqual(s["n"], 5)
        self.assertAlmostEqual(s["mean"], 0.2266, places=3)
        self.assertGreater(s["std"], 0.0)


if __name__ == "__main__":
    unittest.main()
