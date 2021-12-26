from typing import Any, Callable

import numpy as np

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

    @property
    def value_negative(self) -> Any:
        """Return the value that it is not the positive value."""
        return other_value(self.real_values, self.value_positive)

    def unique_scores(self, decimals: int = 3) -> VectorPdNp:
        """Return the unique values attained by the fitted scores, sorted in
        ascending order.

        Parameters
        ----------
        decimals : int, optional
            The number of decimals the fitted scores should be rounded to before
            deriving the unique values. It helps speeding up the operations in
            the case of large datasets. By default 3

        Returns
        -------
        np.ndarray | pd.Series
            The array containing the sorted unique values. Its type matches
            fitted_scores' type.
        """
        if isinstance(self.fitted_scores, np.ndarray):
            return np.unique(self.fitted_scores.round(decimals))
        return self.fitted_scores.round(decimals).unique().sort_values(ascending=True)

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
