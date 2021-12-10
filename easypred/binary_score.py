from typing import Any

import numpy as np


class BinaryScore:
    def __init__(
        self,
        real_values: np.ndarray,
        fitted_scores: np.ndarray,
        value_positive: Any = 1,
    ):
        self.real_values = real_values
        self.fitted_scores = fitted_scores
        self.value_positive = value_positive

    def accuracy_arr(self, interval: float = 0.01) -> np.ndarray:
        pass
