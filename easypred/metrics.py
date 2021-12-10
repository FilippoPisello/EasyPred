from typing import Any

import numpy as np

from easypred.type_aliases import VectorPdNp


def accuracy_score(real_values: VectorPdNp, fitted_values: VectorPdNp) -> float:
    """Return a float representing the percent of items which are equal
    between the real and the fitted values.

    Also called: percentage correctly classified"""
    return np.mean(real_values == fitted_values)
