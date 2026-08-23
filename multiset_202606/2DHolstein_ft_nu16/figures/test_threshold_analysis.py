"""Tests for threshold_analysis.py, including the saved Holstein data."""

from pathlib import Path
import unittest

import numpy as np

from threshold_analysis import (
    Comparison,
    TEMPERATURES,
    convergence_audit,
    first_threshold_crossing,
    interval_error_audit,
    rmsd_2d,
    threshold_sweep,
)


HERE = Path(__file__).resolve().parent
BASE_DIR = HERE.parent


class ThresholdUnitTests(unittest.TestCase):
    def test_rmsd_2d_for_center_and_corner(self):
        occupations = np.zeros((2, 9))
        occupations[0, 4] = 1.0
        occupations[1, 0] = 1.0
        np.testing.assert_allclose(rmsd_2d(occupations, 3, 3), [0.0, np.sqrt(2)])

    def test_first_crossing_is_linearly_interpolated(self):
        time = np.array([0.0, 1.0, 2.0])
        error = np.array([0.0, 0.02, 0.04])
        crossing, censored = first_threshold_crossing(time, error, 0.03)
        self.assertAlmostEqual(crossing, 1.5)
        self.assertFalse(censored)

    def test_missing_crossing_is_right_censored(self):
        crossing, censored = first_threshold_crossing(
            np.array([0.0, 1.0]), np.array([0.0, 0.01]), 0.02
        )
        self.assertEqual(crossing, 1.0)
        self.assertTrue(censored)


class HolsteinDataIntegrationTests(unittest.TestCase):
    def test_all_requested_chi_epsilon_combinations_are_finite(self):
        frame = threshold_sweep(BASE_DIR)
        expected_rows = 5 * len(TEMPERATURES) * 5
        self.assertEqual(len(frame), expected_rows)
        self.assertTrue(np.isfinite(frame["t_epsilon"]).all())
        self.assertTrue((frame["t_epsilon"] >= 0).all())

    def test_selected_method_separation_thresholds_pass_at_every_temperature(self):
        audit = convergence_audit(
            BASE_DIR, fail_epsilon=1e-2, convergence_epsilon=1e-3
        )
        self.assertTrue(audit["multiset_converged"].all())

    def test_all_singleset_curves_use_multiset_chi32_as_reference(self):
        frame = threshold_sweep(BASE_DIR)
        singleset = frame[frame["method"].eq("singleset")]
        self.assertTrue(singleset["reference_method"].eq("multiset").all())
        self.assertTrue(singleset["reference_chi"].eq(32).all())

    def test_multiset_chi16_has_later_failure_than_singleset_chi16(self):
        frame = threshold_sweep(
            BASE_DIR,
            comparisons=(
                Comparison("singleset", 16, 32),
                Comparison("multiset", 16, 32),
            ),
            epsilons=(1e-2,),
        )
        pivot = frame.pivot(index="temperature", columns="method", values="t_epsilon")
        self.assertTrue((pivot["multiset"] > pivot["singleset"]).all())

    def test_multiset_chi16_stays_within_tolerance_to_t8_only_at_T1(self):
        audit = interval_error_audit(
            BASE_DIR,
            Comparison("multiset", 16, 32),
            epsilon=1e-2,
            t_start=0.0,
            t_stop=8.0,
        )
        passing_temperatures = audit.loc[
            audit["within_tolerance"], "temperature"
        ].tolist()
        self.assertEqual(passing_temperatures, [1.0])


if __name__ == "__main__":
    unittest.main()
