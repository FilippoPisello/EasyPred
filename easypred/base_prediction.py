"""Contains the generic Prediction. This class represents any kind of prediction
interpreted as fitted array Y' attempting to be close to real array Y.

The Prediction class allows to compute some metrics concerning the accuracy
without needing to know how the prediction was computed.

The subclasses allow for metrics that are relevant for just specific types
of predictions."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd

from easypred.utils import check_lengths_match, lists_to_nparray

if TYPE_CHECKING:
    from easypred.type_aliases import Vector, VectorPdNp


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
        real_values: Vector,
        fitted_values: Vector,
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

        Examples
        -------
        >>> from easypred import Prediction
        >>> pred = Prediction(real_values=["Foo", "Foo", "Bar", "Baz"],
        ...                   fitted_values=["Foo", "Bar", "Foo", "Baz"])
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
        between the real and the fitted values.

        Examples
        -------
        >>> pred = Prediction(real_values=["Foo", "Foo", "Bar", "Baz"],
        ...                   fitted_values=["Foo", "Bar", "Foo", "Baz"])
        >>> pred.accuracy_score
        0.5
        """
        return np.mean(self.real_values == self.fitted_values)

    def matches(self) -> VectorPdNp:
        """Return a boolean array of length N with True where fitted value is
        equal to real value.

        Returns
        -------
        pd.Series | np.array

        Examples
        -------
        >>> pred = Prediction(real_values=["Foo", "Foo", "Bar", "Baz"],
        ...                   fitted_values=["Foo", "Bar", "Foo", "Baz"])
        >>> pred.matches()
        array([True, False, False, True])
        """
        return self.real_values == self.fitted_values

    def as_dataframe(self) -> pd.DataFrame:
        """Return prediction as a dataframe containing various information over
        the prediction quality.

        Examples
        -------
        >>> pred = Prediction(real_values=["Foo", "Foo", "Bar", "Baz"],
        ...                   fitted_values=["Foo", "Bar", "Foo", "Baz"])
        >>> pred.as_dataframe()
          Real Values Fitted Values  Prediction Matches
        0         Foo           Foo                True
        1         Foo           Bar               False
        2         Bar           Foo               False
        3         Baz           Baz                True
        """
        data = {
            "Real Values": self.real_values,
            "Fitted Values": self.fitted_values,
            "Prediction Matches": self.matches(),
        }
        return pd.DataFrame(data)

    def describe(self) -> pd.DataFrame:
        """Return a dataframe containing some key information about the
        prediction.

        Examples
        -------
        >>> pred = Prediction(real_values=["Foo", "Foo", "Bar", "Baz"],
        ...                   fitted_values=["Foo", "Bar", "Foo", "Baz"])
        >>> pred.describe()
                  Value
        N           4.0
        Matches     2.0
        Errors      2.0
        Accuracy    0.5
        """
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
