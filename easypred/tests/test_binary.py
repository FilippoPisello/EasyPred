from unittest import TestCase

import numpy as np
import pandas as pd
from easypred import BinaryPrediction, Prediction


class TestBinaryPrediction(TestCase):
    df = pd.read_excel("easypred/tests/test_data/binary.xlsx")
    real, fitted = df["Real"], df["Fitted"]
    p1 = BinaryPrediction(real, fitted, value_positive=1)
    p2 = BinaryPrediction(real.to_numpy(), fitted.to_numpy(), value_positive=1)

    def test_value_negative(self):
        self.assertEqual(self.p1.value_negative, 0)
        self.assertEqual(self.p2.value_negative, 0)

    def test_confusion_matrix(self):
        real = self.p1.confusion_matrix()
        exp = np.array([[308, 30], [31, 131]])
        np.testing.assert_array_equal(real, exp)

        # Test relative values
        real = self.p1.confusion_matrix(relative=True)
        exp_rel = exp / (exp.sum())
        np.testing.assert_array_equal(real, exp_rel)

        # Test dataframe representation
        real = self.p1.confusion_matrix(as_dataframe=True)
        exp = pd.DataFrame(
            {"Pred 0": [308, 31], "Pred 1": [30, 131]},
            index=["Real 0", "Real 1"],
        )
        pd.testing.assert_frame_equal(real, exp, check_dtype=False)

    def test_rates(self):
        self.assertEqual(self.p1.balanced_accuracy_score, 0.8599422894294689)
        self.assertEqual(self.p1.false_positive_rate, (30 / (30 + 308)))
        self.assertEqual(self.p1.false_negative_rate, (31 / (31 + 131)))
        self.assertEqual(self.p1.recall_score, (131 / (31 + 131)))
        self.assertEqual(self.p1.specificity_score, (308 / (30 + 308)))

        self.assertEqual(self.p1.precision_score, (131 / (30 + 131)))
        self.assertEqual(self.p1.negative_predictive_value, (308 / (308 + 31)))

        self.assertEqual(self.p1.f1_score, 0.8111455108359134)

    def test_constructors(self):
        real, fit = [0, 0, 1], [1, 0, 0]
        pred = Prediction(real, fit)
        bin_pred = BinaryPrediction.from_prediction(pred, 1)
        self.assertIsInstance(bin_pred, BinaryPrediction)
        np.testing.assert_array_equal(real, np.array(real))
        np.testing.assert_array_equal(fit, np.array(fit))

    def test_describe(self):
        exp = pd.DataFrame(
            {
                "N": [len(self.p1)],
                "Matches": [self.p1.matches().sum()],
                "Errors": [len(self.p1) - self.p1.matches().sum()],
                "Accuracy": [self.p1.accuracy_score],
                "Recall": self.p1.recall_score,
                "Specificity": self.p1.specificity_score,
                "Precision": self.p1.precision_score,
                "Negative PV": self.p1.negative_predictive_value,
                "F1 score": self.p1.f1_score,
            },
            index=["Value"],
        ).transpose()
        pd.testing.assert_frame_equal(exp, self.p1.describe())


def test_from_score():
    from easypred import BinaryScore

    binscore = BinaryScore([0, 1, 1], [0.3, 0.5, 0.7], value_positive=1)
    res = BinaryPrediction.from_binary_score(binscore, threshold=0.5)
    exp = BinaryPrediction([0, 1, 1], [0, 1, 1], value_positive=1)

    assert isinstance(res, BinaryPrediction)
    assert res == exp
