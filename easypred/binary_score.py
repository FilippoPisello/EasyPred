"""Class to represent probability estimates, thus predictions that do not
directly return fitted values but that can be converted to such. It can be
viewed as the step before BinaryPrediction.

It allows to compute AUC score and other metrics that depend on the convertion
threshold as arrays."""
from typing import Any, Callable, Union

import numpy as np
import pandas as pd

from easypred import BinaryPrediction
from easypred.type_aliases import Vector, VectorPdNp
from easypred.utils import lists_to_nparray, other_value


class BinaryScore:
    """Class to represent a prediction in terms of probability estimates, thus
    having each observation paired with a score between 0 and 1 representing
    the likelihood of being the "positive value".

    Attributes
    -------
    fitted_scores: np.ndarray | pd.Series
        The array-like object of length N containing the probability scores.
    real_values: np.ndarray | pd.Series
        The array-like object containing the N real values.
    value_positive: Any
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.
    """

    def __init__(
        self,
        real_values: Vector,
        fitted_scores: Vector,
        value_positive: Any = 1,
    ):
        """Class to represent a prediction in terms of probability estimates.

        Arguments
        -------
        real_values: np.ndarray | pd.Series | list | tuple
            The array-like object containing the real values. If not pd.Series
            or np.array, it will be coerced into np.array.
        fitted_scores: np.ndarray | pd.Series | list | tuple
            The array-like object of length N containing the probability scores.
            It must have the same length as real_values. If not pd.Series or
            np.array, it will be coerced into np.array.
        value_positive: Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.
        """
        self.real_values, self.fitted_scores = lists_to_nparray(
            real_values, fitted_scores
        )
        self.value_positive = value_positive
        self.computation_decimals = 3

    @property
    def value_negative(self) -> Any:
        """Return the value that it is not the positive value."""
        return other_value(self.real_values, self.value_positive)

    @property
    def unique_scores(self) -> VectorPdNp:
        """Return the unique values attained by the fitted scores, sorted in
        ascending order

        Returns
        -------
        np.ndarray | pd.Series
            The array containing the sorted unique values. Its type matches
            fitted_scores' type.
        """
        scores = np.unique(self.fitted_scores.round(self.computation_decimals))

        if isinstance(self.fitted_scores, pd.Series):
            return pd.Series(scores)

        return scores

    def score_to_values(self, threshold: float = 0.5) -> VectorPdNp:
        """Return an array contained fitted values derived on the basis of the
        provided threshold.

        Parameters
        ----------
        threshold : float, optional
            The minimum value such that the score is translated into
            value_positive. Any score below the threshold is instead associated
            with the other value. By default 0.5.

        Returns
        -------
        np.ndarray | pd.Series
            The array containing the inferred fitted values. Its type matches
            fitted_scores' type.
        """
        return np.where(
            (self.fitted_scores >= threshold),
            self.value_positive,
            self.value_negative,
        )

    @property
    def auc_score(self) -> float:
        """Return the Area Under the Receiver Operating Characteristic Curve
        (ROC AUC)."""
        return np.abs(np.trapz(self.recall_scores, self.false_positive_rates))

    @property
    def accuracy_scores(self) -> np.ndarray:
        """Return an array containing the accuracy scores calculated setting the
        threshold for each unique score value."""
        from easypred.metrics import accuracy_score

        return self._metric_array(accuracy_score)

    @property
    def false_positive_rates(self) -> np.ndarray:
        """Return an array containing the false positive rates calculated
        setting the threshold for each unique score value."""
        from easypred.metrics import false_positive_rate

        return self._metric_array(
            false_positive_rate, value_positive=self.value_positive
        )

    @property
    def recall_scores(self) -> np.ndarray:
        """Return an array containing the recall scores calculated setting the
        threshold for each unique score value."""
        from easypred.metrics import recall_score

        return self._metric_array(recall_score, value_positive=self.value_positive)

    @property
    def f1_scores(self) -> np.ndarray:
        """Return an array containing the f1 scores calculated setting the
        threshold for each unique score value."""
        from easypred.metrics import f1_score

        return self._metric_array(f1_score, value_positive=self.value_positive)

    def _metric_array(
        self, metric_function: Callable[..., float], **kwargs
    ) -> np.ndarray:
        """Return an array containing the passed metric calculated setting the
        threshold for each unique score value.

        Parameters
        ----------
        metric_function : Callable(VectorPdNp, VectorPdNp, ...) -> float
            The function that calculates the metric.
        **kwargs : Any
            Arguments to be directly passed to metric_function.

        Returns
        -------
        np.ndarray
            The array containing the metrics calculated for each threshold.
        """
        return np.array(
            [
                metric_function(self.real_values, self.score_to_values(val), **kwargs)
                for val in self.unique_scores
            ]
        )

    def best_threshold(self, criterion="f1") -> float:
        """Return the threshold to convert scores into values that performs the
        best given a specified criterion.

        Parameters
        ----------
        criterion : str, optional
            The value to be maximized by the threshold. It defaults to "f1",
            the options are:
            - "f1": maximize the f1 score
            - "accuracy": maximize the accuracy score

        Returns
        -------
        float
            The threshold that maximizes the indicator specified.
        """
        if criterion == "f1":
            numb = np.argmax(self.f1_scores)
        elif criterion == "accuracy":
            numb = np.argmax(self.accuracy_scores)
        else:
            raise ValueError(
                f"Unrecognized criterion: {criterion}. Allowed "
                "criteria are 'f1', 'accuracy'."
            )

        return self.unique_scores[numb]

    def to_binary_prediction(
        self, threshold: Union[float, str] = 0.5
    ) -> BinaryPrediction:
        """Create an instance of BinaryPrediction from the BinaryScore object.

        Parameters
        ----------
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
        """
        if isinstance(threshold, str):
            threshold = self.best_threshold(criterion=threshold)
        binpred = BinaryPrediction(
            real_values=self.real_values,
            fitted_values=self.score_to_values(threshold=threshold),
            value_positive=self.value_positive,
        )
        binpred.threshold = threshold
        return binpred
