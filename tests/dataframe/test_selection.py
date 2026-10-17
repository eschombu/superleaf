import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.selection import dfilter, partition, reorder_columns, where
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


def _eq(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return ((df1 == df2) | (df1.isna() & df2.isna())).all().all()


def test_dfilter(df):
    # Progressively increase complexity
    assert _eq(dfilter(df, col1=0), df.iloc[[0]])
    assert _eq(dfilter(df, col3=F.isna), df[df["col3"].isna()])
    assert _eq(dfilter(df, col3=pd.notna), df[df["col3"].notna()])
    assert _eq(dfilter(df, col3=pd.notna), dfilter(df, col3=F.notna))

    assert _eq(dfilter(df, col3=1), df[df["col3"] == 1])
    assert _eq(dfilter(df, col3=1, col4="four"), df.iloc[[4]])

    assert _eq(dfilter(df, col2=F.ge(0)), df[df["col2"] >= 0])
    assert _eq(dfilter(df, lambda row: row.col2 < 0 or "o" not in row.col4), df.iloc[[0, 1, 3]])
    assert _eq(dfilter(df, where(lambda d: (d.col2 < 0) | ~d.col4.str.contains("o"))), df.iloc[[0, 1, 3]])
    assert _eq(dfilter(df, col2=(F.lt(0) | ~(F.le(0)))), df[df["col2"] != 0])

    assert _eq(dfilter(df, col2=F.ge(0), col3=1), df[(df["col2"] >= 0) & (df["col3"] == 1)])
    assert _eq(dfilter(df, col2=F.ge(0), col3=1, col5=1), df.iloc[[4]])
    assert _eq(dfilter(df, where(lambda d: d.col3 == d.col5), col2=F.ge(0)), df.iloc[[4]])

    # Test columns filter
    assert _eq(
        dfilter(df, col2=F.ge(0), columns=F.matches_regex(r'col[3-5]')),
        dfilter(df, col2=F.ge(0))[['col3', 'col4', 'col5']],
    )
    df2 = df.assign(columns=[0, 1, 2, 3, 4])
    assert _eq(dfilter(df2, col2=F.ge(0), columns=F.gt(0)), df2[(df2.col2 >= 0) & (df2["columns"] > 0)])


def test_partition(df):
    filtered, remaining = partition(df, col2=F.ge(0), col3=1, col5=1)
    filt_exp = df.iloc[[4]]
    assert len(filtered) + len(remaining) == len(df)
    assert _eq(filtered, filt_exp)
    assert _eq(remaining, df[~df.index.isin(filt_exp.index)])


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

    absent_col = "absent"
    with pytest.raises(ValueError):
        _ = reorder_columns(df, [absent_col, second_col, first_col], errors="invalid")
    with pytest.raises(KeyError):
        _ = reorder_columns(df, [absent_col, second_col, first_col], errors="raise")
    with pytest.warns(Warning):
        _ = reorder_columns(df, [absent_col, second_col, first_col], errors="warn")
    new_df = reorder_columns(df, [absent_col, second_col, first_col], errors="ignore")
    new_df_cols = list(new_df.columns)
    assert df.shape == new_df.shape
    assert new_df_cols != df_cols
    assert set(new_df_cols) == set(df_cols)
    assert new_df_cols[0] == second_col
    assert new_df_cols[1] == first_col


def test_dfilter_columns_list_selects_any(df):
    result = dfilter(df, columns=["col1", "col4", "absent"])
    assert list(result.columns) == ["col1", "col4"]
    result = dfilter(df, columns=["col1", F.endswith("5")])
    assert list(result.columns) == ["col1", "col5"]
    assert list(dfilter(df, columns="col2").columns) == ["col2"]


def test_dfilter_columns_none_is_noop(df):
    assert _eq(dfilter(df, columns=None), df)
    assert _eq(dfilter(df, columns=None, col1=0), df.iloc[[0]])


def test_partition_columns(df):
    passed, failed = partition(df, col1=F.lt(2), columns=["col1", "col2"])
    assert list(passed.columns) == list(failed.columns) == ["col1", "col2"]
    assert passed.col1.tolist() == [0, 1]
    assert failed.col1.tolist() == [2, 3, 4]


def test_dfilter_list_value_raises_clear_error(df):
    with pytest.raises(TypeError, match=r"F\.isin"):
        dfilter(df, col1=[1, 2])
    assert _eq(dfilter(df, col1=F.isin([1, 2])), df.iloc[[1, 2]])


def test_dfilter_keyword_value_none_matches_none():
    df = pd.DataFrame({"a": [None, "x", np.nan]}, dtype=object)
    assert dfilter(df, a=None).index.tolist() == [0]
    assert dfilter(df, a=F.isna).index.tolist() == [0, 2]


def test_dfilter_plain_callable_keyword(df):
    assert _eq(dfilter(df, col4=lambda s: len(s) == 4), df.iloc[[0, 4]])


def test_dfilter_dict_and_mask_filters(df):
    assert _eq(dfilter(df, {"col1": F.gt(2)}), df.iloc[[3, 4]])
    assert _eq(dfilter(df, [True, False, True, False, False]), df.iloc[[0, 2]])
    assert _eq(dfilter(df, df.col1 > 2), df.iloc[[3, 4]])
    with pytest.raises(ValueError):
        dfilter(df, {"col1": 1}, col1=1)
    with pytest.raises(ValueError):
        dfilter(df, pd.Series([True] * 5, index=list("abcde")))
    with pytest.raises(TypeError):
        dfilter(df, 1)


def test_dfilter_where_with_nullable_booleans():
    df = pd.DataFrame({"a": pd.array([1, None, 3], dtype="Int64")})
    assert dfilter(df, where(lambda d: d.a > 1)).index.tolist() == [2]


def test_dfilter_empty_dataframe(df):
    empty = df.iloc[:0]
    assert dfilter(empty, lambda row: row.col1 > 0, col2=F.gt(0)).empty


def test_dfilter_series():
    s = pd.Series([1, -1, 2], index=["a", "b", "c"], name="x")
    assert dfilter(s, F.gt(0)).tolist() == [1, 2]
    assert dfilter(s, index=F.isin(["b", "c"])).tolist() == [-1, 2]
    assert dfilter(s, x=F.lt(2)).tolist() == [1, -1]
    passed, failed = partition(s, lambda v: v > 1)
    assert passed.tolist() == [2] and failed.tolist() == [1, -1]


def test_dfilter_does_not_modify_input(df):
    original = df.copy()
    result = dfilter(df, col1=F.gt(0))
    result.loc[:, "col1"] = 100
    pd.testing.assert_frame_equal(df, original)


def test_reorder_columns_warn_and_case(df):
    with pytest.warns(UserWarning):
        new_df = reorder_columns(df, ["absent", "col3"], errors="warn")
    assert list(new_df.columns)[0] == "col3"
    new_df = reorder_columns(df, ["absent", "col3"], errors="IGNORE")
    assert list(new_df.columns)[0] == "col3"


def test_reorder_columns_falsy_anchor():
    df = pd.DataFrame([[1, 2, 3]], columns=[0, 1, 2])
    assert list(reorder_columns(df, [2], after=0).columns) == [0, 2, 1]


def test_dfilter_pd_col_expression(df):
    expr = (pd.col("col2") >= 0) & (pd.col("col3") == pd.col("col5"))
    assert _eq(dfilter(df, expr), df.iloc[[4]])
    assert _eq(dfilter(df, pd.col("col2") > 0, col4=F.startswith("f"), columns=["col1", "col4"]),
               df.loc[[4], ["col1", "col4"]])
    passed, failed = partition(df, pd.col("col1") < 2)
    assert passed.col1.tolist() == [0, 1] and failed.col1.tolist() == [2, 3, 4]


def test_dfilter_pd_col_with_mask_named_column(df):
    df2 = df.assign(_superleaf_mask=0)
    assert dfilter(df2, pd.col("col1") > 3).col1.tolist() == [4]


def test_dfilter_pd_col_keyword_raises(df):
    with pytest.raises(TypeError, match="positionally"):
        dfilter(df, col1=pd.col("col5"))


def test_dfilter_result_is_independent(df):
    result = dfilter(df, col1=F.gt(0))
    df.loc[1, "col1"] = 100
    assert result.loc[1, "col1"] == 1
