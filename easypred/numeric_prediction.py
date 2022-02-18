"""Subclass of prediction specialized in representing numeric predictions, thus
a prediction where both fitted and real data are either ints or floats.

It allows to compute accuracy metrics that represent the distance between
the prediction and the real values."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from matplotlib.axes import Axes

from easypred import Prediction

if TYPE_CHECKING:
    from easypred.type_aliases import VectorPdNp


class NumericPrediction(Prediction):
    """Subclass of Prediction specialized in representing numeric predictions.

    Attributes
    -------
    fitted_values: np.ndarray | pd.Series
        The array-like object of length N containing the fitted values.
    real_values: np.ndarray | pd.Series
        The array-like object containing the N real values.

    Examples
    -------
    >>> from easypred import NumericPrediction
    >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
    >>> pred.real_values
    array([7, 1, 3, 4, 5])
    >>> pred.fitted_values
    array([6.5, 2. , 4. , 3. , 5. ])
    """

    @property
    def r_squared(self) -> float:
        """Returns the r squared calculated as the square of the correlation
        coefficient. Also called 'Coefficient of Determination'.

        References
        ---------
        https://en.wikipedia.org/wiki/Coefficient_of_determination

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.r_squared
        0.8616803278688523
        """
        return np.corrcoef(self.real_values, self.fitted_values)[0, 1] ** 2

    @property
    def mse(self) -> float:
        """Return the Mean Squared Error.

        References
        ---------
        https://en.wikipedia.org/wiki/Mean_squared_error

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.mse
        0.65
        """
        return np.mean(self.residuals(squared=True))

    @property
    def rmse(self) -> float:
        """Return the Root Mean Squared Error.

        References
        ---------
        https://en.wikipedia.org/wiki/Root-mean-square_deviation

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.rmse
        0.806225774829855
        """
        return np.sqrt(self.mse)

    @property
    def mae(self) -> float:
        """Return the Mean Absolute Error.

        References
        ---------
        https://en.wikipedia.org/wiki/Mean_absolute_error

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.mae
        0.7
        """
        return np.mean(self.residuals(absolute=True))

    @property
    def mape(self) -> float:
        """Return the Mean Absolute Percentage Error.

        References
        ---------
        https://en.wikipedia.org/wiki/Mean_absolute_percentage_error

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.mape
        0.33095238095238094
        """
        return np.mean(self.residuals(absolute=True, relative=True))

    def residuals(
        self,
        squared: bool = False,
        absolute: bool = False,
        relative: bool = False,
    ) -> VectorPdNp:
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
        np.ndarray or pd.Series
            Numpy array or pandas series depending on the type of real_values and
            fitted_values. Its shape is (N,).

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.residuals()
        array([ 0.5, -1. , -1. ,  1. ,  0. ])
        >>> pred.residuals(squared=True)
        array([0.25, 1.  , 1.  , 1.  , 0.  ])
        >>> pred.residuals(absolute=True)
        array([0.5, 1. , 1. , 1. , 0. ])
        >>> pred.residuals(relative=True)
        array([ 0.07142857, -1.        , -0.33333333,  0.25      ,  0.        ])
        >>> pred.residuals(relative=True, absolute=True)
        array([0.07142857, 1.        , 0.33333333, 0.25      , 0.        ])
        """
        residuals = self.real_values - self.fitted_values
        if relative:
            residuals = residuals / self.real_values
        if squared:
            return residuals ** 2
        if absolute:
            return abs(residuals)
        return residuals

    def matches_tolerance(self, tolerance: float = 0.0) -> VectorPdNp:
        """Return a boolean array of length N with True where the distance
        between the real values and the fitted values is inferior to a
        given parameter

        Parameters
        ----------
        tolerance : float, optional
            The maximum absolute difference between the real value and its
            fitted counterpart such that the pair considered a match. By
            default is 0.0.

        Returns
        -------
        np.ndarray or pd.Series
            Boolean array of shape (N,). Its type reflects the type of
            self.real_values.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.matches_tolerance()
        array([False, False, False, False,  True])
        >>> pred.matches_tolerance(tolerance=2)
        array([ True,  True,  True,  True,  True])

        With pandas series:

        >>> import pandas as pd
        >>> pred = NumericPrediction(pd.Series([7, 1, 3, 4, 5]),
        ...                          pd.Series([6.5, 2, 4, 3, 5]))
        >>> pred.matches_tolerance(tolerance=2)
        0    True
        1    True
        2    True
        3    True
        4    True
        dtype: bool
        """
        return abs(self.real_values - self.fitted_values) <= tolerance

    def as_dataframe(self) -> pd.DataFrame:
        """Return prediction as a dataframe containing various information over
        the prediction quality.

        Returns
        -------
        pd.DataFrame
            Dataframe of shape (N, 5) containing summary information for each
            observation's prediction.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.as_dataframe()
           Fitted Values  Real Values  Prediction Matches  Absolute Difference  Relative Difference
        0            6.5            7               False                  0.5             0.071429
        1            2.0            1               False                 -1.0            -1.000000
        2            4.0            3               False                 -1.0            -0.333333
        3            3.0            4               False                  1.0             0.250000
        4            5.0            5                True                  0.0             0.000000
        """
        residuals = self.residuals()
        data = {
            "Fitted Values": self.fitted_values,
            "Real Values": self.real_values,
            "Prediction Matches": self.matches(),
            "Absolute Difference": residuals,
            "Relative Difference": residuals / self.real_values,
        }
        return pd.DataFrame(data)

    def describe(self) -> pd.DataFrame:
        """Return a dataframe containing some key information about the
        prediction.

        Returns
        -------
        pd.DataFrame
            Dataframe of shape (6, 1) containing summary information on the
            prediction quality.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.describe()
                Value
        N     5.000000
        MSE   0.650000
        RMSE  0.806226
        MAE   0.700000
        MAPE  0.330952
        R^2   0.861680
        """
        return pd.DataFrame(
            {
                "N": [len(self)],
                "MSE": self.mse,
                "RMSE": self.rmse,
                "MAE": self.mae,
                "MAPE": self.mape,
                "R^2": self.r_squared,
            },
            index=["Value"],
        ).transpose()

    def plot_fit_residuals(
        self,
        figsize: tuple[int, int] = (20, 10),
        title_size: int = 14,
        axes_labels_size: int = 12,
        axs: list[Axes] | None = None,
    ) -> np.ndarray:
        """Plot a two panels figure containing the plot of real against fitted
        values and the plot of residuals against fitted values.

        This method combines plot_fit and plot_residuals.

        These two graphs are useful in detecting potential biases in the
        prediction as they allow to detect deviations and clusters in the
        prediction.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Tuple of integers specifying the size of the plot. Default is
            (20, 10).
        title_size : int, optional
            Font size of the plots' titles. Default is 14.
        axes_labels_size : int, optional
            Font size of the axes labels. Default is 12.
        axs : list of matplotlib Axes, optional
            List of axes object of length 2 to draw the plot onto. Otherwise
            creates new Figure and Axes. Use this option to further customize
            the plot.

        Returns
        -------
        np.ndarray[matplotlib Axes, matplotlib Axes]
            NumPy array of shape (2,) containing one matplotlib Axes object for
            each of the subplots.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.plot_fit_residuals()
        array([<AxesSubplot:title={'center':'Real against fitted values'},
               xlabel='Fitted values', ylabel='Real values'>,
               <AxesSubplot:title={'center':'Residuals against fitted values'},
               xlabel='Fitted values', ylabel='Residuals'>],
            dtype=object)
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
        """
        if axs is None:
            _, axs = plt.subplots(nrows=1, ncols=2, figsize=figsize)

        self.plot_fit(
            title_size=title_size, axes_labels_size=axes_labels_size, ax=axs[0]
        )
        self.plot_residuals(
            title_size=title_size, axes_labels_size=axes_labels_size, ax=axs[1]
        )

        return axs

    def plot_residuals(
        self,
        figsize: tuple[int, int] = (20, 10),
        hline: int | None = 0,
        title_size: int = 14,
        axes_labels_size: int = 12,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the scatterplot depicting the residuals against fitted values.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Tuple of integers specifying the size of the plot. Default is
            (20, 10).
        hline : int, optional
            Y coordinate of the red dashed line added to the scatterplot. If
            None, no line is drawn. By default is 0.
        title_size : int, optional
            Font size of the plot title. Default is 14.
        axes_labels_size : int, optional
            Font size of the axes labels. Default is 12.
        ax : matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise creates new Figure
            and Axes. Use this option to further customize the plot.

        Returns
        -------
        matplotlib Axes
            Matplotlib Axes object with the plot drawn on it.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.plot_residuals()
        <AxesSubplot:title={'center':'Residuals against fitted values'},
        xlabel='Fitted values', ylabel='Residuals'>
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.scatter(self.fitted_values, self.residuals())

        if hline is not None:
            ax.axhline(0, c="red", ls="--")

        ax.set_title("Residuals against fitted values", fontsize=title_size)
        ax.set_xlabel("Fitted values", fontsize=axes_labels_size)
        ax.set_ylabel("Residuals", fontsize=axes_labels_size)

        ax.grid(True, ls="--")

        return ax

    def plot_fit(
        self,
        figsize: tuple[int, int] = (20, 10),
        line_slope: int | None = 1,
        title_size: int = 14,
        axes_labels_size: int = 12,
        ax: Axes | None = None,
    ) -> Axes:
        """Plot the scatterplot depicting real against fitted values.

        Parameters
        ----------
        figsize : tuple[int, int], optional
            Tuple of integers specifying the size of the plot. Default is
            (20, 10).
        line_slope : int | None, optional
            Slope of the red dashed line added to the scatterplot. If None, no
            line is drawn. By default is 1, representing parity between real
            and fitted values.
        title_size : int, optional
            Font size of the plot title. Default is 14.
        axes_labels_size : int, optional
            Font size of the axes labels. Default is 12.
        ax : matplotlib Axes, optional
            Axes object to draw the plot onto, otherwise creates new Figure
            and Axes. Use this option to further customize the plot.

        Returns
        -------
        matplotlib Axes
            Matplotlib Axes object with the plot drawn on it.

        Examples
        -------
        >>> from easypred import NumericPrediction
        >>> pred = NumericPrediction([7, 1, 3, 4, 5], [6.5, 2, 4, 3, 5])
        >>> pred.plot_fit()
        <AxesSubplot:title={'center':'Real against fitted values'},
        xlabel='Fitted values', ylabel='Real values'>
        >>> from matplotlib import pyplot as plt
        >>> plt.show()
        """
        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        ax.scatter(self.fitted_values, self.real_values)

        if line_slope is not None:
            min_val = min([self.real_values.min(), self.fitted_values.min()]) * 0.95
            ax.axline((min_val, min_val), slope=line_slope, c="red", ls="--")

        ax.set_title("Real against fitted values", fontsize=title_size)
        ax.set_xlabel("Fitted values", fontsize=axes_labels_size)
        ax.set_ylabel("Real values", fontsize=axes_labels_size)
        ax.axis("equal")

        ax.grid(True, ls="--")

        return ax
