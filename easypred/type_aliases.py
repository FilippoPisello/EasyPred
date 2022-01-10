from typing import Any, Callable, Union

import numpy as np
import pandas as pd

VectorPdNp = Union[np.ndarray, pd.Series]
Vector = Union[list, tuple, np.ndarray, pd.Series]
BinaryMetricFunction = Callable[[VectorPdNp, VectorPdNp, Any], float]
