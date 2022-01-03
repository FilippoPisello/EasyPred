from typing import Any, Callable

import numpy as np
import pandas as pd

from easypred import BinaryPrediction
from easypred.type_aliases import Vector, VectorPdNp
from easypred.utils import lists_to_nparray, other_value


class BinaryScore:
    def __init__(
        self,
        real_values: Vector,
        fitted_scores: Vector,
        value_positive: Any = 1,
    ):
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
