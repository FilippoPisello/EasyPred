from unittest import TestCase

import numpy as np
import pandas as pd
from predictions.prediction import Prediction


class TestPrediction(TestCase):

    l1 = [1, 2, 3]
    l2 = [1, 2, 3]
    l3 = [1, 2, 4]
    l4 = [1, 2, 4, 5]
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

        l1 = ["a", "b"]
        self.assertFalse(Prediction(l1, None).is_numeric)
        self.assertFalse(Prediction(np.array(l1)).is_numeric)

    def test_is_correct(self):
        """Test if correctness check happens in the right way."""
        res1 = np.array([True, True, True])
        np.testing.assert_array_equal(self.pred_l1l2.is_correct, res1)
        np.testing.assert_array_equal(self.pred_a1a2.is_correct, res1)
        pd.testing.assert_series_equal(self.pred_s1s2.is_correct, pd.Series(res1))

        res2 = np.array([True, True, False])
        np.testing.assert_array_equal(self.pred_l1l3.is_correct, res2)
        np.testing.assert_array_equal(self.pred_a1a3.is_correct, res2)
        pd.testing.assert_series_equal(self.pred_s1s3.is_correct, pd.Series(res2))

    def test_accuracy(self):
        """Test if accuracy is computed correctly."""
        self.assertEqual(self.pred_l1l2.accuracy_score, 1)
        self.assertEqual(self.pred_a1a2.accuracy_score, 1)
        self.assertEqual(self.pred_s1s2.accuracy_score, 1)

        self.assertEqual(self.pred_l1l3.accuracy_score, 2 / 3)
        self.assertEqual(self.pred_a1a3.accuracy_score, 2 / 3)
        self.assertEqual(self.pred_s1s3.accuracy_score, 2 / 3)
