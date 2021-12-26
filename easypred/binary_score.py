from typing import Any

import numpy as np

from easypred.type_aliases import Vector, VectorPdNp

class BinaryScore:
    def __init__(
        self,
        real_values: Vector,
        fitted_scores: Vector,
        value_positive: Any = 1,
    ):
        self.real_values = real_values
        self.fitted_scores = fitted_scores
        self.value_positive = value_positive

    def accuracy_arr(self, interval: float = 0.01) -> np.ndarray:
        pass
