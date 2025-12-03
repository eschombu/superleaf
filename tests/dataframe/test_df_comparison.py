import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.comparison import nan_eq


@pytest.fixture
def df():
    data = {
        "col1": [2, 3, 4],
        "col2": [1, np.nan, 1],
        "col3": ["zero", "one", "two"],
    }
    return pd.DataFrame(data)


def test_nan_eq(df: pd.DataFrame):
    df_copy = df.copy()
    assert nan_eq(df, df_copy)

    df_nan = df.copy()
    df_nan.loc[0, "col2"] = np.nan
    assert not nan_eq(df, df_nan)
    assert nan_eq(df, df_nan, pass_either_nan=True)

    df_diff = df.copy()
    df_diff.loc[1, "col1"] = 99
    assert not nan_eq(df, df_diff)
