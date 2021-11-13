"""Contains the generic Prediction class and its subclasses. This class represents
any kind of prediction interpreted as fitted array X hoping to be close to
real array Y.

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
        return BinaryPrediction(fitted_values=self.fitted_values,
                                real_values=self.real_values,
                                value_positive=value_positive)


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
        self, squared: bool = False, absolute_value: bool = False
    ) -> Union[np.ndarray, pd.Series]:
        """Return an array with the difference between the real values and the
        fitted values."""
        residuals = self.real_values - self.fitted_values
        if squared:
            return residuals ** 2
        if absolute_value:
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


class BinaryPrediction(Prediction):
    """Class to represent a binary prediction.

    Attributes
    -------
    fitted_values: Union[np.ndarray, pd.Series, list]
        The array-like object of length N containing the fitted values.
    real_values: Union[np.ndarray, pd.Series, list]
        The array-like object containing the N real values.
    value_positive: Any
        The value in the data that corresponds to 1 in the boolean logic.
        It is generally associated with the idea of "positive" or being in
        the "treatment" group. By default is 1.

    Properties
    -------
    false_negative_rate : float
        The ratio between the number of wrongly predicted negative
        and the total number of positives.
    false_positive_rate : float
        The ratio between the number of wrongly predicted positive
        and the total number of negatives.
    percentage_correctly_classified: float
        The decimal representing the percentage of elements for which fitted
        and real value coincide.
    pcc: float
        Alias for percentage_correctly_classified.
    sensitivity : float
        The ratio between the correctly predicted positive and the total number
        of real positive.
    specificity : float
        The ratio between the correctly predicted negative and the
        total number of real negative.
    value_negative : Any
        The value that it is not the positive value.
    """

    def __init__(
        self,
        fitted_values: Union[np.ndarray, pd.Series, list],
        real_values: Union[np.ndarray, pd.Series, list],
        value_positive: Any = 1,
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

        value_positive: Any
            The value in the data that corresponds to 1 in the boolean logic.
            It is generally associated with the idea of "positive" or being in
            the "treatment" group. By default is 1.
        """
        super().__init__(fitted_values, real_values)
        self.value_positive = value_positive

    @property
    def value_negative(self) -> Any:
        """Return the value that it is not the positive value."""
        other_only = self.real_values[self.real_values != self.value_positive]
        if isinstance(self.real_values, np.ndarray):
            return other_only[0].copy()
        return other_only.reset_index(drop=True)[0]

    @property
    def false_positive_rate(self) -> float:
        """Return the ration between the number of wrongly predicted positive
        and the total number of negatives."""
        pred_pos = self.fitted_values == self.value_positive
        real_neg = self.real_values != self.value_positive

        false_positive = pred_pos & real_neg
        return false_positive.sum() / real_neg.sum()

    @property
    def false_negative_rate(self):
        """Return the ratio between the number of wrongly predicted negative
        and the total number of positives."""
        pred_neg = self.fitted_values != self.value_positive
        real_pos = self.real_values == self.value_positive

        false_negative = pred_neg & real_pos
        return false_negative.sum() / real_pos.sum()

    @property
    def sensitivity(self):
        """Return the ratio between the correctly predicted positive and the
        total number of real positive."""
        pred_pos = self.fitted_values == self.value_positive
        real_pos = self.real_values == self.value_positive

        caught_positive = pred_pos & real_pos
        return caught_positive.sum() / real_pos.sum()

    @property
    def specificity(self):
        """Return the ratio between the correctly predicted negative and the
        total number of real negative."""
        pred_neg = self.fitted_values != self.value_positive
        real_neg = self.real_values != self.value_positive

        caught_negative = pred_neg & real_neg
        return caught_negative.sum() / real_neg.sum()

    def confusion_matrix(
        self, relative_frequencies: bool = False, as_dataframe: bool = False
    ) -> Union[np.ndarray, pd.DataFrame]:
        """Return the confusion matrix for the binary classification.

        The confusion matrix is a matrix with shape (2, 2) that classifies the
        predictions into four categories, each represented by one of its elements:
        - [0, 0] : negative classified as negative
        - [0, 1] : negative classified as positive
        - [1, 0] : positive classified as negative
        - [1, 1] : positive classified as positive

        Parameters
        ----------
        relative_frequencies : bool, optional
            If True, absolute frequencies are replace by relative frequencies.
            By default False.
        as_dataframe : bool, optional
            If True, the matrix is returned as a pandas dataframe for better
            readability. Otherwise a numpy array is returned. By default False.

        Returns
        -------
        Union[np.ndarray, pd.DataFrame]
            If as_dataframe is False, return a numpy array of shape (2, 2).
            Otherwise return a pandas dataframe of the same shape.
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
        if relative_frequencies:
            conf_matrix = conf_matrix / len(self.fitted_values)

        if not as_dataframe:
            return conf_matrix
        conf_df = pd.DataFrame(conf_matrix)
        conf_df.columns = conf_df.index = [self.value_negative, self.value_positive]
        return conf_df
