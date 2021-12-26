import easypred.metrics as met
import numpy as np
import pandas as pd
import pytest

df = pd.read_excel("easypred/tests/test_data/binary.xlsx")
real, fitted = df["Real"], df["Fitted"]
val_positive = 1


@pytest.mark.parametrize(
    "real_values, fitted_values, accuracy",
    [
        (real, fitted, 439 / 500),
        (np.array([1, 1, 0]), np.array([1, 1, 0]), 1),
        (np.array([1, 1, 1]), np.array([1, 1, 0]), 2 / 3),
    ],
)
def test_accuracy(real_values, fitted_values, accuracy):
    assert met.accuracy_score(real_values, fitted_values) == accuracy


def test_balanced_accuracy():
    assert met.balanced_accuracy_score(real, fitted, val_positive) == 0.8599422894294689


def test_fp_rate():
    score = met.false_positive_rate(real, fitted, val_positive)
    assert score == (30 / (30 + 308))


def test_fn_rate():
    score = met.false_negative_rate(real, fitted, val_positive)
    assert score == (31 / (31 + 131))


def test_precision():
    score = met.precision_score(real, fitted, val_positive)
    assert score == (131 / (30 + 131))


def test_negative_pred_value():
    score = met.negative_predictive_value(real, fitted, val_positive)
    assert score == (308 / (308 + 31))


def test_recall():
    score = met.recall_score(real, fitted, val_positive)
    assert score == (131 / (31 + 131))


def test_specificity():
    score = met.specificity_score(real, fitted, val_positive)
    assert score == (308 / (30 + 308))

def test_f1rate():
    assert met.f1_score(real, fitted, val_positive) == 0.8111455108359134
