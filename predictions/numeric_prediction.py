"""Subclass of prediction specialized in representing numeric predictions, thus
a prediction where both fitted and real data are either ints or floats.

It allows to compute accuracy metrics that represent the distance between
the prediction and the real values."""
from typing import Union

import numpy as np
import pandas as pd

from predictions import Prediction


class NumericPrediction(Prediction):
    """Class to represent a numerical prediction.

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
    r_squared : float
        R squared coefficient calculated as the square of the correlation
        coefficient between fitted and real values.
    """

    @property
    def r_squared(self) -> float:
        """Returns the r squared calculated as the square of the correlation
        coefficient."""
        return np.corrcoef(self.real_values, self.fitted_values)[0, 1] ** 2

    def residuals(
        self,
        squared: bool = False,
        absolute: bool = False,
        relative: bool = False,
    ) -> Union[np.ndarray, pd.Series]:
        """Return an array with the difference between the real values and the
        fitted values.

        Parameters
        ----------
        squared : bool, optional
            If True, the residuals are squared, by default False.
        absolute : bool, optional
            If True, the residuals are taken in absolute value, by default False.
        relative : bool, optional
            If True, the residuals are divided by the real values to return
            a relative measure. By default False.

        Returns
        -------
        Union[np.ndarray, pd.Series]
            Numpy array or pandas series depending on the type of real_values and
            fitted_values. Its shape is (N,).
        """
        residuals = self.real_values - self.fitted_values
        if relative:
            residuals = residuals / self.real_values
        if squared:
            return residuals ** 2
        if absolute:
            return abs(residuals)
        return residuals

    def matches_tolerance(self, tolerance: float = 0.0) -> Union[np.ndarray, pd.Series]:
        """Return a boolean array of length N with True where the distance
        between the real values and the fitted values is inferior to a
        given parameter."""
        return abs(self.real_values - self.fitted_values) <= tolerance

    def as_dataframe(self) -> pd.DataFrame:
        """Return prediction as a dataframe containing various information over
        the prediction quality."""
        residuals = self.residuals()
        data = {
            "Fitted Values": self.fitted_values,
            "Real Values": self.real_values,
            "Prediction Matches": self.matches_tolerance(),
            "Absolute difference": residuals,
            "Relative difference": residuals / self.real_values,
        }
        return pd.DataFrame(data)
