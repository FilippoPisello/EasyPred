from unittest import TestCase

import easypred.metrics as met
import numpy as np
import pandas as pd


class TestMetrics(TestCase):
    df = pd.read_excel("easypred/tests/test_data/binary.xlsx")
    real_values, fitted_values = df["Real"], df["Fitted"]
    value_positive = 1

    def test_accuracy(self):
        accuracy = met.accuracy_score(self.real_values, self.fitted_values)
        self.assertEqual(accuracy, 439 / 500)

        accuracy = met.accuracy_score(np.array([1, 1, 0]), np.array([1, 1, 0]))
        self.assertEqual(accuracy, 1)

        accuracy = met.accuracy_score(np.array([1, 1, 1]), np.array([1, 1, 0]))
        self.assertEqual(accuracy, 2 / 3)
