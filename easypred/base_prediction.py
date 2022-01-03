"""Contains the generic Prediction. This class represents any kind of prediction
interpreted as fitted array Y' attempting to be close to real array Y.

The Prediction class allows to compute some metrics concerning the accuracy
without needing to know how the prediction was computed.

The subclasses allow for metrics that are relevant for just specific types
of predictions."""

from typing import Union

import numpy as np
import pandas as pd

from easypred.type_aliases import VectorPdNp
from easypred.utils import check_lengths_match, lists_to_nparray


class Prediction:
    """Class to represent a generic prediction.

    Attributes
    ----------
    fitted_values: np.ndarray | pd.Series
        The array-like object of length N containing the fitted values.
    real_values: np.ndarray | pd.Series
        The array-like object containing the N real values.
    """

    def __init__(
        self,
        real_values: Union[np.ndarray, pd.Series, list],
        fitted_values: Union[np.ndarray, pd.Series, list],
    ):
        """Class to represent a generic prediction.

        Arguments
        -------
        real_values: np.ndarray | pd.Series | list | tuple
            The array-like object of length N containing the real values. If
            not pd.Series or np.array, it will be coerced into np.array.
        fitted_values: np.ndarray | pd.Series | list | tuple
            The array-like object of containing the real values. It must have
            the same length of real_values. If not pd.Series or np.array, it
            will be coerced into np.array.
        """
        self.real_values, self.fitted_values = lists_to_nparray(
            real_values, fitted_values
        )

        # Processing appening at __init__
        check_lengths_match(
            self.real_values, self.fitted_values, "Real values", "Fitted values"
        )

    def __str__(self):
        return self.fitted_values.__str__()

    def __len__(self):
        return len(self.fitted_values)

    def __eq__(self, other):
        return np.all(self.fitted_values == other.fitted_values)

    def __ne__(self, other):
        return np.any(self.fitted_values != other.fitted_values)

    @property
    def accuracy_score(self) -> float:
        """Return a float representing the percent of items which are equal
        between the real and the fitted values."""
        return np.mean(self.real_values == self.fitted_values)

    def matches(self) -> VectorPdNp:
        """Return a boolean array of length N with True where fitted value is
        equal to real value."""
        return self.real_values == self.fitted_values

    def as_dataframe(self) -> pd.DataFrame:
        """Return prediction as a dataframe containing various information over
        the prediction quality."""
        data = {
            "Real Values": self.real_values,
            "Fitted Values": self.fitted_values,
            "Prediction Matches": self.matches(),
        }
        return pd.DataFrame(data)

    def describe(self) -> pd.DataFrame:
        """Return a dataframe containing some key information about the
        prediction."""
        return self._describe()

    def _describe(self) -> pd.DataFrame:
        """Return some basic metrics for the prediction."""
        n = len(self)
        matches = self.matches().sum()
        errors = n - matches
        return pd.DataFrame(
            {
                "N": [n],
                "Matches": [matches],
                "Errors": [errors],
                "Accuracy": [self.accuracy_score],
            },
            index=["Value"],
        ).transpose()
