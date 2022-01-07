from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:
    from easypred.type_aliases import VectorPdNp


def accuracy_score(real_values: VectorPdNp, fitted_values: VectorPdNp) -> float:
    """Return a float representing the percent of items which are equal
    between the real and the fitted values.

    Also called: percentage correctly classified

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.

    Returns
    -------
    float
        Accuracy score

    References
    -------
    https://en.wikipedia.org/wiki/Accuracy_and_precision#In_binary_classification
    """
    return np.mean(real_values == fitted_values)


def balanced_accuracy_score(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the float representing the arithmetic mean between recall score
    and specificity score.

    It provides an idea of the goodness of the prediction in unbalanced datasets.

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        Balanced accuracy score
    """
    return (
        recall_score(real_values, fitted_values, value_positive)
        + specificity_score(real_values, fitted_values, value_positive)
    ) / 2


def false_positive_rate(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the ratio between the number of false positives and the total
    number of real negatives.

    It tells the percentage of negatives falsely classified as positive.

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        False positive rate

    References
    ---------
    https://en.wikipedia.org/wiki/False_positive_rate"""
    pred_pos = fitted_values == value_positive
    real_neg = real_values != value_positive

    return (pred_pos & real_neg).sum() / real_neg.sum()


def false_negative_rate(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the ratio between the number of false negatives and the total
    number of real positives.

    It tells the percentage of positives falsely classified as negative.

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        False negative rate
    """
    pred_neg = fitted_values != value_positive
    real_pos = real_values == value_positive

    return (pred_neg & real_pos).sum() / real_pos.sum()


def precision_score(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the ratio between the number of correctly predicted positives
    and the total number predicted positives.

    It measures how accurate the positive predictions are.

    Also called: positive predicted value.

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        Precision score

    References
    ---------
    https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values"""
    pred_pos = fitted_values == value_positive
    real_pos = real_values == value_positive
    return (pred_pos & real_pos).sum() / pred_pos.sum()


def negative_predictive_value(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the ratio between the number of correctly classified negative
    and the total number of predicted negative.

    It measures how accurate the negative predictions are.

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        Negative predictive value

    References
    ---------
    https://en.wikipedia.org/wiki/Positive_and_negative_predictive_values"""
    pred_neg = fitted_values != value_positive
    real_neg = real_values != value_positive
    return (pred_neg & real_neg).sum() / pred_neg.sum()


def recall_score(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 1
) -> float:
    """Return the ratio between the correctly predicted positives and the
    total number of real positives.

    It measures how good the model is in detecting real positives.

    Also called: sensitivity, true positive rate, hit rate

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        Recall score

    References
    --------
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Sensitivity"""
    pred_pos = fitted_values == value_positive
    real_pos = real_values == value_positive
    return (pred_pos & real_pos).sum() / real_pos.sum()


def specificity_score(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 0
) -> float:
    """Return the ratio between the correctly predicted negatives and the
    total number of real negatives.

    It measures how good the model is in detecting real negatives.

    Also called: selectivity, true negative rate

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        Specificity score

    References
    ---------
    https://en.wikipedia.org/wiki/Sensitivity_and_specificity#Specificity"""
    pred_neg = fitted_values != value_positive
    real_neg = real_values != value_positive
    return (pred_neg & real_neg).sum() / real_neg.sum()


def f1_score(
    real_values: VectorPdNp, fitted_values: VectorPdNp, value_positive: Any = 0
) -> float:
    """Return the harmonic mean of the precision and recall.

    It gives an idea of an overall goodness of your precision and recall taken
    together.

    Also called: balanced F-score or F-measure

    Parameters
    ----------
    real_values : numpy array | pandas series
        Array containing the true values.
    fitted_values : numpy array | pandas series
        Array containing the predicted values.
    value_positive: Any, optional
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Returns
    -------
    float
        F1 score

    References
    ---------
    https://en.wikipedia.org/wiki/F-score"""
    precision = precision_score(real_values, fitted_values, value_positive)
    recall = recall_score(real_values, fitted_values, value_positive)
    return 2 * (precision * recall) / (precision + recall)
