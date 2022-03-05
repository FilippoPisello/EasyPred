"""Subclass of Prediction specialized in representing a binary prediction, thus
a prediction where both the fitted and real data attain at most two different
values.

It allows to compute accuracy metrics like true positive, true negative,
etc."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

from easypred import Prediction
from easypred.utils import other_value

if TYPE_CHECKING:
    from easypred import BinaryScore
    from easypred.type_aliases import Vector


class BinaryPrediction(Prediction):
    """Subclass of Prediction specialized in representing numeric categorical
    predictions with binary outcome.

    Attributes
    -------
    fitted_values: np.ndarray | pd.Series
        The array-like object of length N containing the fitted values.
    real_values: np.ndarray | pd.Series
        The array-like object containing the N real values.
    value_positive: Any
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Examples
    -------
    Classic 0/1 case:

    >>> from easypred import BinaryPrediction
    >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
    ...                         fitted_values=[1, 0, 1, 1])
    >>> pred.real_values
    array([0, 1, 1, 1])
    >>> pred.fitted_values
    array([1, 0, 1, 1])
    >>> pred.value_positive
    1

    Other values are accepted:

    >>> from easypred import BinaryPrediction
    >>> pred = BinaryPrediction(real_values=["Foo", "Foo", "Bar", "Foo"],
    ...                         fitted_values=["Foo", "Bar", "Foo", "Bar"]
    ...                         value_positive="Foo")
    >>> pred.value_positive
    Foo

    """

    def __init__(
        self,
        real_values: Vector,
        fitted_values: Vector,
        value_positive: Any = 1,
    ):
        """Create an instance of BinaryPrediction to represent a prediction with
        just two possible outcomes.

        Arguments
        -------
        real_values: np.ndarray | pd.Series | list | tuple
            The array-like object of length N containing the real values. If
            not pd.Series or np.array, it will be coerced into np.array.
        fitted_values: np.ndarray | pd.Series | list | tuple
            The array-like object of containing the real values. It must have
            the same length of real_values. If not pd.Series or np.array, it
            will be coerced into np.array.
        value_positive: Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred1 = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                          fitted_values=[1, 0, 1, 1])
        >>> pred2 = BinaryPrediction(real_values=["Foo", "Foo", "Bar", "Foo"],
        ...                          fitted_values=["Foo", "Bar", "Foo", "Bar"]
        ...                          value_positive="Foo")
        """
        super().__init__(real_values=real_values, fitted_values=fitted_values)
        self.value_positive = value_positive

    @property
    def value_negative(self) -> Any:
        """Return the value that it is not the positive value.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.value_negative
        0
        """
        return other_value(self.real_values, self.value_positive)

    @property
    def balanced_accuracy_score(self) -> float:
        """Return the float representing the arithmetic mean between recall score
        and specificity score.

        It provides an idea of the goodness of the prediction in unbalanced datasets.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.balanced_accuracy_score
        0.3333333333333333
        """
        from easypred.metrics import balanced_accuracy_score

        return balanced_accuracy_score(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def false_positive_rate(self) -> float:
        """Return the ratio between the number of false positives and the total
        number of real negatives.

        It tells the percentage of negatives falsely classified as positive.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.false_positive_rate
        1.0
        """
        from easypred.metrics import false_positive_rate

        return false_positive_rate(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def false_negative_rate(self):
        """Return the ratio between the number of false negatives and the total
        number of real positives.

        It tells the percentage of positives falsely classified as negative.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.false_negative_rate
        0.3333333333333333
        """
        from easypred.metrics import false_negative_rate

        return false_negative_rate(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def recall_score(self):
        """Return the ratio between the correctly predicted positives and the
        total number of real positives.

        It measures how good the model is in detecting real positives.

        Also called: sensitivity, hit rate, true positive rate.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.recall_score
        0.6666666666666666
        """
        from easypred.metrics import recall_score

        return recall_score(self.real_values, self.fitted_values, self.value_positive)

    @property
    def specificity_score(self):
        """Return the ratio between the correctly predicted negatives and the
        total number of real negatives.

        It measures how good the model is in detecting real negatives.

        Also called: selectivity, true negative rate.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.specificity_score
        0.0
        """
        from easypred.metrics import specificity_score

        return specificity_score(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def precision_score(self):
        """Return the ratio between the number of correctly predicted positives
        and the total number predicted positives.

        It measures how accurate the positive predictions are.

        Also called: positive predicted value.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.precision_score
        0.6666666666666666
        """
        from easypred.metrics import precision_score

        return precision_score(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def negative_predictive_value(self):
        """Return the ratio between the number of correctly classified negative
        and the total number of predicted negative.

        It measures how accurate the negative predictions are.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.negative_predictive_value
        0.0
        """
        from easypred.metrics import negative_predictive_value

        return negative_predictive_value(
            self.real_values, self.fitted_values, self.value_positive
        )

    @property
    def f1_score(self):
        """Return the harmonic mean of the precision and recall.

        It gives an idea of an overall goodness of your precision and recall taken
        together.

        Also called: balanced F-score or F-measure

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.f1_score
        0.6666666666666666
        """
        from easypred.metrics import f1_score

        return f1_score(self.real_values, self.fitted_values, self.value_positive)

    def confusion_matrix(
        self, relative: bool = False, as_dataframe: bool = False
    ) -> np.ndarray | pd.DataFrame:
        """Return the confusion matrix for the binary classification.

        The confusion matrix is a matrix with shape (2, 2) that classifies the
        predictions into four categories, each represented by one of its elements:
        - [0, 0] : negative classified as negative
        - [0, 1] : negative classified as positive
        - [1, 0] : positive classified as negative
        - [1, 1] : positive classified as positive

        Parameters
        ----------
        relative : bool, optional
            If True, absolute frequencies are replace by relative frequencies.
            By default False.
        as_dataframe : bool, optional
            If True, the matrix is returned as a pandas dataframe for better
            readability. Otherwise a numpy array is returned. By default False.

        Returns
        -------
        np.ndarray | pd.DataFrame
            If as_dataframe is False, return a numpy array of shape (2, 2).
            Otherwise return a pandas dataframe of the same shape.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.confusion_matrix()
        array([[0, 1],
            [1, 2]])
        >>> pred.confusion_matrix(as_dataframe=True)
                Pred 0  Pred 1
        Real 0       0       1
        Real 1       1       2
        >>> pred.confusion_matrix(as_dataframe=True, relative=True)
                Pred 0  Pred 1
        Real 0    0.00    0.25
        Real 1    0.25    0.50
        """
        pred_pos = self.fitted_values == self.value_positive
        pred_neg = self.fitted_values != self.value_positive
        real_pos = self.real_values == self.value_positive
        real_neg = self.real_values != self.value_positive

        conf_matrix = np.array(
            [
                [(pred_neg & real_neg).sum(), (pred_pos & real_neg).sum()],
                [(pred_neg & real_pos).sum(), (pred_pos & real_pos).sum()],
            ]
        )

        # Divide by total number of values to obtain relative frequencies
        if relative:
            conf_matrix = conf_matrix / len(self.fitted_values)

        if not as_dataframe:
            return conf_matrix
        return self._confusion_matrix_dataframe(conf_matrix)

    def _confusion_matrix_dataframe(self, conf_matrix: np.ndarray) -> pd.DataFrame:
        """Convert a numpy confusion matrix into a pandas dataframe and add
        index and columns labels."""
        conf_df = pd.DataFrame(conf_matrix)
        values = [self.value_negative, self.value_positive]
        conf_df.columns = [f"Pred {val}" for val in values]
        conf_df.index = [f"Real {val}" for val in values]
        return conf_df

    def describe(self) -> pd.DataFrame:
        """Return a dataframe containing some key information about the
        prediction.

        Examples
        -------
        >>> from easypred import BinaryPrediction
        >>> pred = BinaryPrediction(real_values=[0, 1, 1, 1],
        ...                         fitted_values=[1, 0, 1, 1])
        >>> pred.describe()
                        Value
        N            4.000000
        Matches      2.000000
        Errors       2.000000
        Accuracy     0.500000
        Recall       0.666667
        Specificity  0.000000
        Precision    0.666667
        Negative PV  0.000000
        F1 score     0.666667
        """
        basic_info = self._describe()
        new_info = pd.DataFrame(
            {
                "Recall": self.recall_score,
                "Specificity": self.specificity_score,
                "Precision": self.precision_score,
                "Negative PV": self.negative_predictive_value,
                "F1 score": self.f1_score,
            },
            index=["Value"],
        ).transpose()
        return pd.concat([basic_info, new_info])

    @classmethod
    def from_prediction(
        cls, prediction: Prediction, value_positive
    ) -> BinaryPrediction:
        """Create an instance of BinaryPrediction from a general Prediction
        object.

        Parameters
        ----------
        prediction : Prediction
            The prediction object the BinaryPrediction is to be constructed from.
        value_positive : Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.

        Returns
        -------
        BinaryPrediction
            An object of type BinaryPrediction, a subclass of Prediction specific
            for predictions with just two outcomes.

        Examples
        -------
        >>> from easypred import BinaryPrediction, Prediction
        >>> pred = Prediction(real_values=[0, 1, 1, 1],
        ...                   fitted_values=[1, 0, 1, 1])
        >>> BinaryPrediction.from_prediction(pred, value_positive=1)
        <easypred.binary_prediction.BinaryPrediction object at 0x000001AA51C3EF10>
        """
        return cls(
            fitted_values=prediction.fitted_values,
            real_values=prediction.real_values,
            value_positive=value_positive,
        )

    @classmethod
    def from_binary_score(
        cls, binary_score: BinaryScore, threshold: float | str = 0.5
    ) -> BinaryPrediction:
        """Create an instance of BinaryPrediction from a BinaryScore object.

        Parameters
        ----------
        binary_score : BinaryScore
            The BinaryScore object the BinaryPrediction is to be constructed
            from.
        value_positive : Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.
        threshold : float | str, optional
            If float, it is the minimum value such that the score is translated
            into value_positive. Any score below the threshold is instead
            associated with the other value.
            If str, the threshold is automatically set such that it maximizes
            the metric corresponding to the provided keyword. The available
            keywords are:
            - "f1": maximize the f1 score
            - "accuracy": maximize the accuracy score

            By default 0.5.

        Returns
        -------
        BinaryPrediction
            An object of type BinaryPrediction, a subclass of Prediction
            specific for predictions with just two outcomes. The class instance
            is given the special attribute "threshold" that returns the
            threshold used in the convertion.

        Examples
        -------
        >>> from easypred import BinaryPrediction, BinaryScore
        >>> real = [0, 1, 1, 0, 1, 0]
        >>> fit = [0.31, 0.44, 0.24, 0.28, 0.37, 0.18]
        >>> score = BinaryScore(real, fit, value_positive=1)
        >>> BinaryPrediction.from_binary_score(score, threshold=0.5)
        <easypred.binary_prediction.BinaryPrediction object at 0x000001AA51C3EEE0>
        """
        return binary_score.to_binary_prediction(threshold=threshold)
