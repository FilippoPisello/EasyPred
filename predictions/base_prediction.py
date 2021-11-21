"""Contains the generic Prediction. This class represents any kind of prediction
interpreted as fitted array X attempting to be close to real array Y.

The Prediction class allows to compute some metrics concerning the accuracy
without needing to know how the prediction was computed.

The subclasses allow for metrics that are relevant for just specific types
of predictions."""

from typing import Any, Union

import numpy as np
import pandas as pd


class Prediction:
    """Class to represent a generic prediction.

    Attributes
    -------
    fitted_values: Union[np.ndarray, pd.Series, list]
        The array-like object of length N containing the fitted values.
    real_values: Union[np.ndarray, pd.Series, list]
        The array-like object containing the N real values.

    Properties
    -------
    percentage_correctly_classified: float
        The decimal representing the percentage of elements for which fitted
        and real value coincide.
    pcc: float
        Alias for percentage_correctly_classified.
    """

    def __init__(
        self,
        fitted_values: Union[np.ndarray, pd.Series, list],
        real_values: Union[np.ndarray, pd.Series, list],
    ):
        """Class to represent a generic prediction.

        Arguments
        -------
        fitted_values: Union[np.ndarray, pd.Series, list]
            The array-like object of length N containing the fitted values. If list,
            it will be turned into np.array.
        real_values: Union[np.ndarray, pd.Series, list]
            The array-like object containing the real values. It must have the same
            length of fitted_values. If list, it will be turned into np.array.
        """
        self.fitted_values = fitted_values
        self.real_values = real_values

        # Processing appening at __init__
        self._check_lengths_match()
        self._lists_to_nparray()

    def _lists_to_nparray(self) -> None:
        """Turn lists into numpy arrays."""
        if isinstance(self.fitted_values, list):
            self.fitted_values = np.array(self.fitted_values)
        if isinstance(self.real_values, list):
            self.real_values = np.array(self.real_values)

    def _check_lengths_match(self) -> None:
        """Check that fitted values and real values have the same length."""
        if self.real_values is None:
            return

        len_fit, len_real = len(self.fitted_values), len(self.real_values)
        if len_fit != len_real:
            raise ValueError(
                "Fitted values and real values must have the same length.\n"
                + f"Fitted values has length: {len_fit}.\n"
                + f"Real values has length: {len_real}."
            )

    def __str__(self):
        return self.fitted_values.__str__()

    def __len__(self):
        return len(self.fitted_values)

    @property
    def is_numeric(self) -> bool:
        """Return True if fitted values are numeric, False otherwise."""
        return pd.api.types.is_numeric_dtype(self.fitted_values)

    @property
    def percentage_correctly_classified(self) -> float:
        """Return a float representing the percent of items which are equal
        between the real and the fitted values."""
        return np.mean(self.real_values == self.fitted_values)

    # DEFYINING ALIAS
    pcc = percentage_correctly_classified

    def matches(self) -> Union[np.ndarray, pd.Series]:
        """Return a boolean array of length N with True where fitted value is
        equal to real value."""
        return self.real_values == self.fitted_values

    def as_dataframe(self) -> pd.DataFrame:
        """Return prediction as a dataframe containing various information over
        the prediction quality."""
        data = {
            "Fitted Values": self.fitted_values,
            "Real Values": self.real_values,
            "Prediction Matches": self.matches(),
        }
        return pd.DataFrame(data)

    def to_binary(self, value_positive: Any):
        """Create an instance of BinaryPrediction.

        Parameters
        ----------
        value_positive : Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.

        Returns
        -------
        BinaryPrediction
            An object of type BinaryPrediction, a subclass of Prediction specific
            for predictions with just two outcomes.
        """
        from predictions import BinaryPrediction

        return BinaryPrediction(
            fitted_values=self.fitted_values,
            real_values=self.real_values,
            value_positive=value_positive,
        )
