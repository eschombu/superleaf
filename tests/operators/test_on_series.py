import re

import numpy as np
import pandas as pd
import pytest

from superleaf.operators.base import bool_operator, BooleanOperator
from superleaf.operators.comparison import F
from superleaf.operators.string import str_op

PREDICATES = {
    "eq(1)": F.eq(1),
    "eq('abc')": F.eq("abc"),
    "eq(None)": F.eq(None),
    "eq(nan)": F.eq(np.nan),
    "eq([1])": F.eq([1]),
    "ne(1)": F.ne(1),
    "ne('abc')": F.ne("abc"),
    "lt(1)": F.lt(1),
    "le(1.5)": F.le(1.5),
    "gt(0)": F.gt(0),
    "ge('abc')": F.ge("abc"),
    "lt('2020-06-01')": F.lt("2020-06-01"),
    "lt(timestamp)": F.lt(pd.Timestamp("2020-06-01")),
    "lt(0, fallback)": F.lt(0, fallback=False),
    "isin(mixed)": F.isin([1, "abc", 2.0]),
    "isin(str)": F.isin(["abc", "x"]),
    "isin(substring)": F.isin("abcdef"),
    "isin([True])": F.isin([True]),
    "isin([None, 1])": F.isin([None, 1]),
    "isin([nan])": F.isin([np.nan]),
    "contains('b')": F.contains("b"),
    "contains(1)": F.contains(1),
    "contains_all('ab')": F.contains_all("ab"),
    "contains_any(['a', 1])": F.contains_any(["a", 1]),
    "startswith('a')": F.startswith("a"),
    "startswith('a', no converter)": F.startswith("a", str_converter=None),
    "startswith(1)": F.startswith(1),
    "startswith('a', raise)": F.startswith("a", raise_type_error=True),
    "endswith('c')": F.endswith("c"),
    "startswith_one_of": F.startswith_one_of(["a", "x"]),
    "endswith_one_of": F.endswith_one_of(["c", "z"]),
    "matches_regex": F.matches_regex("a.c"),
    "matches_regex(search)": F.matches_regex("b", how="search"),
    "matches_regex(fullmatch)": F.matches_regex("abc", how="fullmatch"),
    "matches_regex(flags)": F.matches_regex("A", flags=re.IGNORECASE),
    "isna": F.isna,
    "notna": F.notna,
    "isinf": F.isinf,
    "notinf": F.notinf,
    "notna & gt(0)": F.notna & F.gt(0),
    "isna | lt(0)": F.isna | F.lt(0),
    "~isin([1])": ~F.isin([1]),
    "~isna": ~F.isna,
    "piped": str_op("upper") >> F.startswith("A"),
    "bool_operator": bool_operator(lambda x: isinstance(x, str) and len(x) > 1),
    "fuzzy_match": F.fuzzy_match("abc"),
}

SCALARS = {
    "None": None,
    "nan": np.nan,
    "int": 1,
    "float": 1.5,
    "str": "abc",
    "list": [1],
    "dict": {"a": 1},
    "pd.NA": pd.NA,
    "NaT": pd.NaT,
}

SERIES = {
    "object mixed": pd.Series([None, np.nan, 1, 1.5, "abc", [1], {"a": 1}], dtype=object),
    "object numbers": pd.Series([0, -1.5, None, 2, np.nan], dtype=object),
    "object str": pd.Series(["abc", "Abd", None, np.nan, "xyz", "", "a\nc"], dtype=object),
    "float": pd.Series([0.0, -1.5, np.nan, 2.0, np.inf, 1.0]),
    "int": pd.Series([0, 1, -3, 2]),
    "bool": pd.Series([True, False]),
    "string dtype": pd.Series(["abc", None, "x"], dtype="string"),
    "Int64": pd.Series([1, None, -2], dtype="Int64"),
    "object pd.NA": pd.Series([1, pd.NA, "abc"], dtype=object),
    "datetime": pd.to_datetime(pd.Series(["2020-01-01", None, "2021-01-01"])),
    "categorical": pd.Series(pd.Categorical(["abc", "b", None], categories=["b", "abc"], ordered=True)),
    "empty": pd.Series([], dtype=object),
}


def _scalar_results(pred, values) -> list | type:
    results = []
    for x in values:
        try:
            results.append(pred(x))
        except Exception as e:
            return type(e)
    return results


def _series_results(pred, s) -> list | type:
    try:
        result = pred.on_series(s)
    except Exception as e:
        return type(e)
    assert result.dtype == bool
    assert result.index.equals(s.index)
    return result.tolist()


@pytest.mark.parametrize("pred_name", PREDICATES)
@pytest.mark.parametrize("value_name", SCALARS)
def test_on_series_matches_scalar_call(pred_name, value_name):
    pred, x = PREDICATES[pred_name], SCALARS[value_name]
    s = pd.Series([x], dtype=object)
    assert _series_results(pred, s) == _scalar_results(pred, [x])


@pytest.mark.parametrize("pred_name", PREDICATES)
@pytest.mark.parametrize("series_name", SERIES)
def test_on_series_matches_scalar_calls_by_dtype(pred_name, series_name):
    pred, s = PREDICATES[pred_name], SERIES[series_name]
    assert _series_results(pred, s) == _scalar_results(pred, list(s))


def test_on_series_keeps_index():
    s = pd.Series([1, -1, np.nan], index=["a", "a", "b"])
    result = (F.isna | F.gt(0)).on_series(s)
    assert result.tolist() == [True, False, True]
    assert list(result.index) == ["a", "a", "b"]


def test_not_isna_on_series():
    s = pd.Series([1.0, np.nan, None], dtype=object)
    assert (~F.isna).on_series(s).tolist() == [True, False, False]


def test_scalar_call_returns_bool():
    assert F.eq(1)(np.int64(1)) is True
    assert F.isna(pd.NA) is True
    assert F.isna([np.nan]) is False
    with pytest.raises(ValueError):
        bool_operator(lambda x: x)(np.array([1, 2]))


def test_short_circuit():
    is_int = bool_operator(lambda x: isinstance(x, int))
    assert (is_int & F.gt(0))("a") is False
    assert (~is_int | F.gt(0))("a") is True
    with pytest.raises(TypeError):
        (F.gt(0) & is_int)("a")
    s = pd.Series([1, "a", -1], dtype=object)
    assert (is_int & F.gt(0)).on_series(s).tolist() == [True, False, False]


def test_combinators_accept_callables():
    op = F.isna | (lambda x: x > 0)
    assert isinstance(op, BooleanOperator)
    assert op(1) is True
    assert op(-1) is False
    assert (pd.isna | F.gt(0))(np.nan) is True


def test_piped_into_boolean_operator_is_boolean_operator():
    op = str_op("lower") >> F.eq("a")
    assert isinstance(op, BooleanOperator)
    assert (op | F.isna)("A") is True
    assert op.on_series(pd.Series(["A", "b"])).tolist() == [True, False]
