from unittest import TestCase

import numpy as np
import pandas as pd
from predictions import NumericPrediction


class TestNumericPrediction(TestCase):
    """Class to test numeric predictions."""

    l1 = [1, 2, 3]
    l2 = [1, 2, 3]
    l3 = [1, 2, 4]
    l4 = [1, 2, 4, 5]
    l5 = ["a", "b"]
    pred_l1l2 = NumericPrediction(l1, l2)
    pred_l1l3 = NumericPrediction(l1, l3)

    s1 = pd.Series(l1)
    s2 = pd.Series(l2)
    s3 = pd.Series(l3)
    s4 = pd.Series(l4)
    pred_s1s2 = NumericPrediction(s1, s2)
    pred_s1s3 = NumericPrediction(s1, s3)

    a1 = np.array(l1)
    a2 = np.array(l2)
    a3 = np.array(l3)
    a4 = np.array(l4)
    pred_a1a2 = NumericPrediction(a1, a2)
    pred_a1a3 = NumericPrediction(a1, a3)

    def test_residuals(self):
        """Test if residuals are computed correctly"""
        # Basic residuals
        p1 = NumericPrediction([1, 2, 3], [1, 2, 3])
        np.testing.assert_array_equal(p1.residuals(), np.array([0, 0, 0]))

        np.testing.assert_array_equal(self.pred_l1l3.residuals(), np.array([0, 0, 1]))
        pd.testing.assert_series_equal(self.pred_s1s3.residuals(), pd.Series([0, 0, 1]))

        # Check various parameters
        p2 = NumericPrediction([1, 1, 1], [3, -3, 3])
        np.testing.assert_array_equal(p2.residuals(), np.array([2, -4, 2]))
        # Squared
        np.testing.assert_array_equal(p2.residuals(squared=True), np.array([4, 16, 4]))
        # Absolute values
        np.testing.assert_array_equal(p2.residuals(absolute=True), np.array([2, 4, 2]))
        # Relative values
        res = p2.residuals(relative=True)
        np.testing.assert_allclose(res, np.array([0.66666667, 1.33333333, 0.66666667]))
        # Relative + absolute values
        p3 = NumericPrediction([1, 4, 1], [3, 3, 3])
        res = p3.residuals(relative=True, absolute=True)
        np.testing.assert_allclose(res, np.array([0.66666667, 0.33333333, 0.66666667]))

    def test_matches_with_tolerance(self):
        """Test if match with tolerance works."""
        # Check various parameters
        p2 = NumericPrediction([1, 1, 1], [3, -3, 4])
        np.testing.assert_array_equal(
            p2.matches_tolerance(), np.array([False, False, False])
        )

        np.testing.assert_array_equal(
            p2.matches_tolerance(tolerance=2), np.array([True, False, False])
        )

        np.testing.assert_array_equal(
            p2.matches_tolerance(tolerance=3), np.array([True, False, True])
        )

        np.testing.assert_array_equal(
            p2.matches_tolerance(tolerance=10), np.array([True, True, True])
        )
