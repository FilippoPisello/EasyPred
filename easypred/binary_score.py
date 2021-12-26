from typing import Any, Callable

import numpy as np

from easypred.type_aliases import Vector, VectorPdNp
from easypred.utils import lists_to_nparray, other_value


class BinaryScore:
    def __init__(
        self,
        real_values: Vector,
        fitted_scores: Vector,
        value_positive: Any = 1,
    ):
        self.real_values, self.fitted_scores = lists_to_nparray(
            real_values, fitted_scores
        )
        self.value_positive = value_positive

    @property
    def value_negative(self) -> Any:
        """Return the value that it is not the positive value."""
        return other_value(self.real_values, self.value_positive)
