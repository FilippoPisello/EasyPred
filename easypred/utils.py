from __future__ import annotations

from typing import Any, Union

import numpy as np
import pandas as pd

from easypred.type_aliases import Vector, VectorPdNp


def lists_to_nparray(
    *listlike_inputs: Vector,
) -> Union[VectorPdNp, tuple[VectorPdNp, ...]]:
    """For each passed element, if it is not of type np.array or pd.Series
    it gets coerced to np.array. Otherwise, it is left unchanged.

    Returns
    -------
    np.array | pd.Series | tuple(np.array | pd.Series, ...)
        If a single argument is passed, the single transformed element is
        returned. In case of multiple argument, a tuple containing the
        transformed version of each element is returned.
    """
    accepted_types = (pd.Series, np.ndarray)
    res = tuple(
        np.array(element) if not isinstance(element, accepted_types) else element
        for element in listlike_inputs
    )
    if len(res) == 1:
        return res[0]
    return res


def other_value(array: VectorPdNp, excluded_value: Any) -> Any:
    """Given a vector-like object assumed to be binary, return the value from
    the object that is not excluded_value."""
    other_only = array[array != excluded_value]
    if isinstance(array, np.ndarray):
        return other_only[0].copy()
    # Type is excpected to be pd.Series
    return other_only.reset_index(drop=True)[0]
