from typing import Union

import numpy as np
import pandas as pd


class Prediction:
    def __init__(
        self,
        fitted_values: Union[np.ndarray, pd.Series, list],
        real_values: Union[np.ndarray, pd.Series, list, None] = None,
    ):
        """Class to represent a generic prediction.

        Arguments
        -------
        fitted_values: Union[np.ndarray, pd.Series, list]
            The array-like object of length N containing the fitted values. If list,
            it will be turned into np.array.
        real_values: Union[np.ndarray, pd.Series, list, None], optional
            The array-like object containing the real values. It must have the same
            length of fitted_values. If list, it will be turned into np.array.
        """
        self.fitted_values = fitted_values
        self.real_values = real_values

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

    def __add__(self, other):
        return self.fitted_values + other

    def __sub__(self, other):
        return self.fitted_values - other

    def __truediv__(self, other):
        return self.fitted_values / other

    def __mul__(self, other):
        return self.fitted_values * other

    @property
    def is_numeric(self) -> bool:
        """Return True if fitted values are numeric, False otherwise."""
        return pd.api.types.is_numeric_dtype(self.fitted_values)

    @property
    def is_correct(self) -> Union[np.ndarray, pd.Series]:
        """Return a boolean array of length N with True where fitted value is
        equal to real value."""
        self._error_if_no_real_value()
        return self.real_values == self.fitted_values

    @property
    def accuracy_score(self) -> float:
        """Return a float representing the percent of items which are equal
        between the real and the fitted values."""
        self._error_if_no_real_value()
        return np.mean(self.real_values == self.fitted_values)

    def as_pdseries(self):
        """Return the fitted values as pandas series."""
        if isinstance(self.fitted_values, pd.Series):
            return self.fitted_values
        return pd.Series(self.fitted_values)

    def as_nparray(self):
        """Return the fitted values as a numpy array."""
        if isinstance(self.fitted_values, np.ndarray):
            return self.fitted_values
        return np.array(self.fitted_values)

    def as_dataframe(self):
        """Return prediction as a dataframe containing various information over
        the prediction quality."""
        data = {
            "Fitted Values": self.fitted_values,
            "Real Values": self.real_values,
            "Prediction Matches": self.is_correct,
        }
        return pd.DataFrame(data)

    def _error_if_no_real_value(self):
        """Raise a ValueError if real_values is None."""
        if self.real_values is None:
            raise ValueError("You need to provide an input for real_values.")
