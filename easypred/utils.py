from __future__ import annotations

from typing import Any

import numpy as np

from easypred.type_aliases import Vector, VectorPdNp


def lists_to_nparray(*listlike_inputs: Vector) -> tuple[VectorPdNp, ...]:
    """For each passed element, if list is turned into numpy array, otherwise it
    is left untouched."""
    return (
        np.array(element) if isinstance(element, list) else element
        for element in listlike_inputs
    )


def other_value(array: VectorPdNp, excluded_value: Any) -> Any:
    """Given a vector-like object assumed to be binary, return the value that
    is not excluded_value."""
    other_only = array[array != excluded_value]
    if isinstance(array, np.ndarray):
        return other_only[0].copy()
    return other_only.reset_index(drop=True)[0]
