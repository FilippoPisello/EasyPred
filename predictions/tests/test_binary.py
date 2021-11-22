from unittest import TestCase

import numpy as np
import pandas as pd
from predictions import BinaryPrediction


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
