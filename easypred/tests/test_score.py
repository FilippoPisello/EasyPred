import numpy as np
import pandas as pd
import pytest
from easypred import BinaryScore

realA, scoresA = [1, 0, 0, 1, 0], [0.79, 0.25, 0.34143, 0.66, 0.34133]
score1 = BinaryScore(realA, scoresA, 1)
score2 = BinaryScore(pd.Series(realA), pd.Series(scoresA), 1)


def test_value_negative():
    assert score1.value_negative == 0


@pytest.mark.parametrize(
    "score, decimals, expected",
    [
        (score1, 2, np.array([0.25, 0.34, 0.66, 0.79])),
        (score1, 3, np.array([0.25, 0.341, 0.66, 0.79])),
        (score1, 4, np.array([0.25, 0.3413, 0.3414, 0.66, 0.79])),
        (score2, 2, pd.Series([0.25, 0.34, 0.66, 0.79])),
    ],
)
def test_unique_score(score, decimals, expected):
    score.computation_decimals = decimals
    np.testing.assert_array_equal(score.unique_scores, expected)
    # Restore default decimals
    score.computation_decimals = 3


@pytest.mark.parametrize(
    "score, threshold, expected",
    [
        (score1, 0.5, np.array([1, 0, 0, 1, 0])),
        (score1, 0, np.array([1, 1, 1, 1, 1])),
        (score1, 1, np.array([0, 0, 0, 0, 0])),
        (score1, 0.7, np.array([1, 0, 0, 0, 0])),
    ],
)
def test_score_to_values(score, threshold, expected):
    result = score.score_to_values(threshold=threshold)
    np.testing.assert_array_equal(result, expected)


@pytest.mark.parametrize(
    "score, decimals, expected",
    [
        (score1, 3, np.array([2 / 5, 3 / 5, 1, 4 / 5])),
    ],
)
def test_accuracy_scores(score, decimals, expected):
    score.computation_decimals = decimals
    np.testing.assert_array_equal(score.accuracy_scores, expected)


@pytest.mark.parametrize(
    "score, decimals, expected",
    [
        (score1, 3, np.array([1, 2 / 3, 0, 0])),
    ],
)
def test_false_positive_rates(score, decimals, expected):
    score.computation_decimals = decimals
    np.testing.assert_array_equal(score.false_positive_rates, expected)


@pytest.mark.parametrize(
    "score, decimals, expected",
    [
        (score1, 3, np.array([1, 1, 1, 1 / 2])),
    ],
)
def test_recall_scores(score, decimals, expected):
    score.computation_decimals = decimals
    np.testing.assert_array_equal(score.recall_scores, expected)


def test_auc_score():
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score

    X, y = load_breast_cancer(return_X_y=True)
    clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    probs = clf.predict_proba(X)[:, 1]
    expected = roc_auc_score(y, probs)

    score = BinaryScore(y, probs)
    score.computation_decimals = 20
    actual = score.auc_score
    assert round(actual, 6) == round(expected, 6)


@pytest.mark.parametrize(
    "score, decimals, expected",
    [
        (score1, 3, np.array([4 / 7, 2 / 3, 1, 2 / 3])),
        (score1, 4, np.array([4 / 7, 2 / 3, 4 / 5, 1, 2 / 3])),
    ],
)
def test_f1_scores(score, decimals, expected):
    score.computation_decimals = decimals
    np.testing.assert_allclose(score.f1_scores, expected)


@pytest.mark.parametrize(
    "score, decimals, criterion, expected",
    [
        (score1, 3, "f1", 0.66),
        (score1, 3, "accuracy", 0.66),
        (score1, 4, "f1", 0.66),
        (score1, 4, "accuracy", 0.66),
    ],
)
def test_best_threshold(score, decimals, criterion, expected):
    score.computation_decimals = decimals
    np.testing.assert_allclose(score.best_threshold(criterion=criterion), expected)


def test_best_threshold_fails():
    with pytest.raises(ValueError):
        score1.best_threshold(criterion="Lorem ipsum")
