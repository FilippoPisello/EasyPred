from unittest import TestCase

import numpy as np
import pandas as pd
from predictions import BinaryPrediction, NumericPrediction, Prediction


class TestPrediction(TestCase):
    """Check the correct functioning of the prediction class."""

    l1 = [1, 2, 3]
    l2 = [1, 2, 3]
    l3 = [1, 2, 4]
    l4 = [1, 2, 4, 5]
    l5 = ["a", "b"]
    pred_l1l2 = Prediction(l1, l2)
    pred_l1l3 = Prediction(l1, l3)

    s1 = pd.Series(l1)
    s2 = pd.Series(l2)
    s3 = pd.Series(l3)
    s4 = pd.Series(l4)
    pred_s1s2 = Prediction(s1, s2)
    pred_s1s3 = Prediction(s1, s3)

    a1 = np.array(l1)
    a2 = np.array(l2)
    a3 = np.array(l3)
    a4 = np.array(l4)
    pred_a1a2 = Prediction(a1, a2)
    pred_a1a3 = Prediction(a1, a3)

    def test_input_check(self):
        """Check that initiation happens correctly"""
        # Check exception if length mismatch
        self.assertRaises(ValueError, Prediction, self.l1, self.l4)
        self.assertRaises(ValueError, Prediction, self.s1, self.s4)
        self.assertRaises(ValueError, Prediction, self.a1, self.a4)

        # Check no exception if no length mismatch
        for fit, real in zip([self.l1, self.s1, self.a1], [self.l2, self.s2, self.a2]):
            try:
                Prediction(fit, real)
            except Exception as e:
                self.fail(f"Prediction(fit, real) raised {e} unexpectedly!")

        # Check that list is transformed to np.array
        np.testing.assert_array_equal(self.pred_l1l2.fitted_values, self.a1)
        np.testing.assert_array_equal(self.pred_l1l2.real_values, self.a2)

        # Check that being numeric is correctly detected
        self.assertTrue(self.pred_l1l2.is_numeric)
        self.assertTrue(self.pred_s1s2.is_numeric)
        self.assertTrue(Prediction([0.1, 0.2], [0.1, 0.2]).is_numeric)

        self.assertFalse(Prediction(np.array(self.l5), np.array(self.l5)).is_numeric)

    def test_is_correct(self):
        """Test if correctness check happens in the right way."""
        res1 = np.array([True, True, True])
        np.testing.assert_array_equal(self.pred_l1l2.matches(), res1)
        np.testing.assert_array_equal(self.pred_a1a2.matches(), res1)
        pd.testing.assert_series_equal(self.pred_s1s2.matches(), pd.Series(res1))

        res2 = np.array([True, True, False])
        np.testing.assert_array_equal(self.pred_l1l3.matches(), res2)
        np.testing.assert_array_equal(self.pred_a1a3.matches(), res2)
        pd.testing.assert_series_equal(self.pred_s1s3.matches(), pd.Series(res2))

    def test_accuracy(self):
        """Test if accuracy is computed correctly."""
        self.assertEqual(self.pred_l1l2.percentage_correctly_classified, 1)
        self.assertEqual(self.pred_a1a2.percentage_correctly_classified, 1)
        self.assertEqual(self.pred_s1s2.percentage_correctly_classified, 1)

        self.assertEqual(self.pred_l1l3.percentage_correctly_classified, 2 / 3)
        self.assertEqual(self.pred_a1a3.percentage_correctly_classified, 2 / 3)
        self.assertEqual(self.pred_s1s3.percentage_correctly_classified, 2 / 3)

        # Test alias
        self.assertEqual(self.pred_l1l3.pcc, 2 / 3)
        self.assertEqual(self.pred_a1a3.pcc, 2 / 3)
        self.assertEqual(self.pred_s1s3.pcc, 2 / 3)


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


class TestBinaryPrediction(TestCase):
    df = pd.read_excel("predictions/tests/test_data/binary.xlsx")
    p1 = BinaryPrediction(df["Fitted"], df["Real"], value_positive=1)

    def test_value_negative(self):
        self.assertEqual(self.p1.value_negative, 0)

    def test_confusion_matrix(self):
        real = self.p1.confusion_matrix()
        exp = np.array([[308, 30], [31, 131]])
        np.testing.assert_array_equal(real, exp)

    def test_rates(self):
        self.assertEqual(self.p1.false_positive_rate, (30 / (30 + 308)))
        self.assertEqual(self.p1.false_negative_rate, (31 / (31 + 131)))
        self.assertEqual(self.p1.sensitivity, (131 / (31 + 131)))
        self.assertEqual(self.p1.specificity, (308 / (30 + 308)))
