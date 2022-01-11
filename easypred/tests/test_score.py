import numpy as np
import pandas as pd
import pytest
from easypred import BinaryPrediction, BinaryScore

realA, scoresA = [1, 0, 0, 1, 0], [0.79, 0.25, 0.34143, 0.66, 0.34133]
score1 = BinaryScore(realA, scoresA, 1)
score2 = BinaryScore(pd.Series(realA), pd.Series(scoresA), 1)


def test_value_negative():
    assert score1.value_negative == 0


@pytest.mark.parametrize(
    "real, fitted, length",
    [
        ([0, 1, 1], [0, 0.2, 0.3], 3),
        ([0, 1, 1, 1], [0, 0.2, 0.3, 0.5], 4),
    ],
)
def test_length(real, fitted, length):
    assert len(BinaryScore(real, fitted)) == length


@pytest.mark.parametrize(
    "score1, score2, is_equal",
    [
        (
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            True,
        ),
        (
            BinaryScore([0, 1, 1], [0.5, 0.5, 0.5]),
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            False,
        ),
    ],
)
def test_equal(score1, score2, is_equal):
    if is_equal:
        assert score1 == score2
    else:
        assert not (score1 == score2)


@pytest.mark.parametrize(
    "score1, score2, is_not_equal",
    [
        (
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            False,
        ),
        (
            BinaryScore([0, 1, 1], [0.5, 0.5, 0.5]),
            BinaryScore([0, 1, 1], [0, 0.2, 0.3]),
            True,
        ),
    ],
)
def test_not_equal(score1, score2, is_not_equal):
    if is_not_equal:
        assert score1 != score2
    else:
        assert not (score1 != score2)


def test_str():
    assert score1.__str__() == score1.fitted_scores.__str__()


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
    """Check that the same AUC score as sklearn is returned.

    The correctness of the score is assessed by comparing a result with what
    returned by Sklearn. Not to have Sklearn and requirements clutter
    EasyPred's test dependencies, Sklearn's result was stored statically, both
    in the final output - the score - and the inputs required to have it.

    At the top of the tests, the code to derive 'data["REAL"]', 'data["SCORE"]'
    and 'expected' directly from sklearn is provided."""
    # from sklearn.datasets import load_breast_cancer
    # from sklearn.linear_model import LogisticRegression
    # from sklearn.metrics import
    # X, y = load_breast_cancer(return_X_y=True)
    # clf = LogisticRegression(solver="liblinear", random_state=0).fit(X, y)
    # probs = clf.predict_proba(X)[:, 1]
    # expected = roc_auc_score(y, probs)

    # Static alternative
    data = pd.read_excel("easypred/tests/test_data/auc_breast.xlsx")
    expected = 0.9947412927435125
    y, probs = data["REAL"], data["SCORE"]

    score = BinaryScore(y, probs)
    score.computation_decimals = 4
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


@pytest.mark.parametrize(
    "score, decimals, threshold, expected",
    [
        (score1, 3, "f1", BinaryPrediction([1, 0, 0, 1, 0], [1, 0, 0, 1, 0])),
        (score1, 3, "accuracy", BinaryPrediction([1, 0, 0, 1, 0], [1, 0, 0, 1, 0])),
        (score1, 3, 0.5, BinaryPrediction([1, 0, 0, 1, 0], [1, 0, 0, 1, 0])),
        (score1, 3, 0.3, BinaryPrediction([1, 0, 0, 1, 0], [1, 0, 1, 1, 1])),
    ],
)
def test_to_binary(score, decimals, threshold, expected):
    score.computation_decimals = decimals
    assert score.to_binary_prediction(threshold=threshold) == expected


def test_describe():
    real = [0, 1, 1, 0, 1, 0]
    fit = [0.31, 0.44, 0.73, 0.28, 0.37, 0.18]
    score = BinaryScore(real, fit, value_positive=1)
    result = score.describe()
    exp = pd.DataFrame(
        {"Value": [6.00, 0.73, 1.00, 1.00, 0.37, 1.00, 0.37]},
        index=[
            "N",
            "Max fitted score",
            "AUC score",
            "Max accuracy",
            "Thresh max accuracy",
            "Max F1 score",
            "Thresh max F1 score",
        ],
    )
    pd.testing.assert_frame_equal(result, exp)


def test_auc_plot_does_not_fail():
    try:
        score1.plot_roc_curve()
        score1.plot_roc_curve(show_legend=False)
        score1.plot_roc_curve(plot_baseline=False)
    except Exception as e:
        assert False, f"plot_roc_curve() raised an exception {e}"


def test_score_hist_does_not_fail():
    try:
        score1.plot_score_histogram()
        score1.plot_score_histogram(bins=5)
    except Exception as e:
        assert False, f"plot_score_histogram() raised an exception {e}"


def test_plot_metric_does_not_fail():
    from easypred.metrics import accuracy_score, f1_score

    try:
        score1.plot_metric(metric=accuracy_score)
        score1.plot_metric(metric=[accuracy_score, f1_score], figsize=(4, 4))
    except Exception as e:
        assert False, f"plot_metric() raised an exception {e}"
