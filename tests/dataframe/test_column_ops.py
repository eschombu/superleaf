import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.column_ops import Col, Values


@pytest.fixture
def df():
    data = {
        "col1": [0, 1, 2, 3, 4],
        "col2": [-2, -1, 0, 1, 2],
        "col3": [1, 1, 1, np.nan, 1],
        "col4": ["zero", "one", "two", "three", "four"],
        "col5": [[], [0, 1], [0, 1], [3, 4], []],
    }
    return pd.DataFrame(data)


def _iseq(v1, v2) -> bool:
    return list(v1) == list(v2)


def test_col_ops(df: pd.DataFrame):
    assert _iseq(Col("col1")(df), df["col1"])
    assert _iseq((Col("col1") == 1)(df), df["col1"] == 1)
    assert _iseq(((Col("col1") == 1) & (Col("col1") != 1))(df), [False] * len(df))
    assert _iseq(((Col("col2") > 0) | (Col("col2") < 0))(df), df["col2"] != 0)
    assert _iseq(((Col("col2") >= 0) & (Col("col2") <= 0))(df), df["col2"] == 0)
    assert _iseq((Col("col1") == Col("col3"))(df), df["col1"] == df["col3"])
    assert _iseq((Col("col1") > Col("col2"))(df), [True] * len(df))
    assert _iseq((Col("col1") + 1)(df), df["col1"] + 1)
    assert _iseq(((Col("col1") + 1) - 1)(df), df["col1"])
    assert _iseq(((Col("col1") * 3) / 2)(df), df["col1"] * 1.5)
    assert _iseq(Col("col1").isin([0, 1])(df), df["col1"] < 2)
    assert _iseq(Col("col1").isin(Col("col5"))(df), map(lambda x: x[0] in x[1], zip(df["col1"], df["col5"])))
    assert _iseq(Col("col4").contains("o")(df), df["col4"].map(lambda s: "o" in s))
    assert _iseq(Col("col3").isna()(df), df["col3"].isna())
    assert _iseq(Col("col3").notna()(df), df["col3"].notna())
    assert _iseq(~(Col("col3").notna())(df), Col("col3").isna()(df))
    assert _iseq(((Col("col3") > 0) | Col("col3").isna())(df), [True] * len(df))
    with pytest.raises(ValueError):
        (Col("col1") ** Col("col2"))(df)
    assert _iseq((Col("col1").astype(float) ** Col("col2"))(df), df["col1"].astype(float) ** df["col2"])
    assert _iseq(Col("col1").isin(Col("col2").to_list() + Col("col3").to_list())(df),
                 ((Col("col1") == Col("col2")) | (Col("col1") == Col("col3")))(df))

    # Test Values operator on series
    s = df["col1"]
    assert _iseq(((Values() < 1) | (Values() >= 3))(s), (s < 1) | (s >= 3))
