import numpy as np
import pandas as pd
import pytest
from easypred import utils

list_ = [1, 2, 3]
np_arr = np.array([1, 2, 3])
pd_arr = pd.Series([1, 2, 3])


@pytest.mark.parametrize(
    "listlike_inputs, output",
    [
        ((list_), np_arr),
        (
            (list_, list_),
            (np_arr, np_arr),
        ),
        (pd_arr, pd_arr),
        (np_arr, np_arr),
        (
            (list_, pd_arr),
            (np_arr, pd_arr),
        ),
    ],
)
def test_lists_to_nparray(listlike_inputs, output):
    result = utils.lists_to_nparray(listlike_inputs)
    np.testing.assert_array_equal(result, output)


@pytest.mark.parametrize(
    "array, excluded_value, output",
    [
        (np.array([0, 0, 1]), 1, 0),
        (np.array([0, 0, 1]), 0, 1),
        (pd.Series([0, 0, 1]), 0, 1),
        (np.array(["Foo", "Bar"]), "Foo", "Bar"),
    ],
)
def test_other_value(array, excluded_value, output):
    assert output == utils.other_value(array=array, excluded_value=excluded_value)
