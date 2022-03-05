from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    from easypred.type_aliases import Vector, VectorPdNp


def lists_to_nparray(
    *listlike_inputs: Vector,
) -> VectorPdNp | tuple[VectorPdNp, ...]:
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


def check_lengths_match(
    array1: Vector,
    array2: Vector,
    name_array1: str = "First array",
    name_array2: str = "Second array",
) -> None:
    """Check that the two passed arrays have the same length."""
    len1, len2 = len(array1), len(array2)
    if len1 != len2:
        raise ValueError(
            f"{name_array1} and {name_array2} must have the same length.\n"
            f"{name_array1} has length: {len1}.\n"
            f"{name_array2} has length: {len2}."
        )


def check_dimensionality(*arrays: Vector) -> Vector:
    for index, array in enumerate(arrays):
        if not isinstance(array, np.ndarray):
            continue

        if array.ndim == 1:
            continue

        squeezed_array = array.squeeze()
        if squeezed_array.ndim == 1:
            arrays[index] = squeezed_array
            continue

        raise ValueError(f"Only 1D arrays are supported, {array} was passed")
    return arrays
