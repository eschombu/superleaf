import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.selection import Col, dfilter, reorder_columns
from superleaf.operators.comparison import ComparisonFunctions as F


@pytest.fixture
def df():
    data = {
        "col1": [0, 1, 2, 3, 4],
        "col2": [-2, -1, 0, 1, 2],
        "col3": [1, 1, 1, np.nan, 1],
        "col4": ["zero", "one", "two", "three", "four"],
        "col5": [1, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)


def _iseq(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return ((df1 == df2) | (df1.isna() & df2.isna())).all().all()


def test_dfilter(df):
    # Progressively increase complexity
    assert _iseq(dfilter(df, col1=0), df.iloc[[0]])
    assert _iseq(dfilter(df, Col("col3").isna()), df[df["col3"].isna()])
    assert _iseq(dfilter(df, col3=pd.notna), dfilter(df, Col("col3").notna()))
    assert _iseq(dfilter(df, col3=pd.notna), dfilter(df, col3=F.notna))

    assert _iseq(dfilter(df, col3=1), df[df["col3"] == 1])
    assert _iseq(dfilter(df, col3=1, col4="four"), df.iloc[[4]])

    assert _iseq(dfilter(df, Col("col2") >= 0), df[df["col2"] >= 0])
    assert _iseq(dfilter(df, col2=F.ge(0)), dfilter(df, Col("col2") >= 0))
    assert _iseq(dfilter(df, (Col("col2") < 0) | ~(Col("col4").contains("o"))), df.iloc[[0, 1, 3]])
    assert _iseq(dfilter(df, (Col("col2") > 0) & ~(Col("col4").contains("o"))), dfilter(df, Col("col3").isna()))
    assert _iseq(dfilter(df, col2=(F.lt(0) | ~(F.le(0)))), df[df["col2"] != 0])

    assert _iseq(dfilter(df, col2=F.ge(0), col3=1), df[(df["col2"] >= 0) & (df["col3"] == 1)])
    assert _iseq(dfilter(df, col2=F.ge(0), col3=1, col5=1), df.iloc[[4]])
    assert _iseq(dfilter(df, col2=F.ge(0), col3=Col("col5")), df.iloc[[4]])


def test_reorder_columns(df: pd.DataFrame):
    df_cols = list(df.columns)
    new_df = reorder_columns(df, "col2")
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert df_cols != new_df_cols
    assert set(df_cols) == set(new_df_cols)
    assert new_df_cols[0] == "col2"

    new_df = reorder_columns(df, "col2", back=True)
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[-1] == "col2"

    new_df = reorder_columns(df, ["col1", "col3"], back=True)
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[-2] == "col1"
    assert new_df_cols[-1] == "col3"

    last_col = df_cols[-1]
    second_to_last = df_cols[-2]
    first_col = df_cols[0]
    second_col = df_cols[1]

    new_df = reorder_columns(df, last_col, after=second_col)
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[-1] == second_to_last
    assert new_df_cols[2] == last_col

    new_df = reorder_columns(df, [first_col, last_col], after=second_col)
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[0] == second_col
    assert new_df_cols[1] == first_col
    assert new_df_cols[2] == last_col
    assert new_df_cols[-1] == second_to_last

    new_df = reorder_columns(df, last_col, before=first_col)
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[0] == last_col
    assert new_df_cols[1] == first_col

    with pytest.raises(ValueError):
        _ = reorder_columns(df, last_col, before=second_col, after=second_to_last)
    with pytest.raises(ValueError):
        _ = reorder_columns(df, last_col, back=True, before=second_col)
