import numpy as np
import pytest

from superleaf.operators import bool_operator
from superleaf.operators.comparison import ComparisonFunctions as F


def test_individual_comparisons():
    assert F.eq(0)(1) is False
    assert F.ne(0)(1) is True
    assert F.eq(0)(0.0) is True
    assert F.ne(0)(0.0) is False
    assert F.lt(0)(-1) is True
    assert F.lt(0)(0) is False
    assert F.le(0)(0) is True
    assert F.gt(0)(1) is True
    assert F.gt(0)(0) is False
    assert F.ge(0)(0) is True

    assert F.isin([0, 1])(0) is True
    assert F.isin([0, 1])(2) is False
    with pytest.raises(TypeError):
        F.isin(2)(1)

    assert F.contains(0)([0, 1]) is True
    assert F.contains(2)([0, 1]) is False
    with pytest.raises(TypeError):
        F.contains(2)(1)

    assert F.contains_any(0)([0, 1, 2]) is True
    assert F.contains_any(3)([0, 1, 2]) is False
    assert F.contains_any([0, 1])([0, 1, 2]) is True
    assert F.contains_any([0, 3])([0, 1, 2]) is True
    assert F.contains_any([3, 4])([0, 1, 2]) is False

    assert F.contains_all(0)([0, 1, 2]) is True
    assert F.contains_all(3)([0, 1, 2]) is False
    assert F.contains_all([0, 1])([0, 1, 2]) is True
    assert F.contains_all([0, 3])([0, 1, 2]) is False

    assert F.contains_any('hello universe')('hello world') is False
    assert F.contains_all('helo')('hello world') is False

    assert F.startswith("he")("hello") is True
    assert F.startswith("lo")("hello") is False
    with pytest.raises(AttributeError):
        F.startswith("lo")(1)
    assert F.endswith("he")("hello") is False
    assert F.endswith("lo")("hello") is True
    with pytest.raises(AttributeError):
        F.endswith("lo")(1)

    assert F.isna(np.nan) is True
    assert F.isna(None) is True
    assert F.isna(np.inf) is False
    assert F.isna(0) is False
    assert F.notna(np.nan) is False
    assert F.notna(None) is False
    assert F.notna(np.inf) is True
    assert F.notna(0) is True

def test_comparison_combos():
    assert (F.gt("hello") & F.contains("wo") & F.isin("sweet world"))("world") is True
    assert (F.startswith("hello") | F.endswith("world"))("hello world") is True

    assert (F.isna | F.notna)(np.nan) is True
    assert (F.isna | F.notna)(1) is True
    assert (F.isna & F.notna)(np.nan) is False
    assert (F.isna & F.notna)(1) is False


def test_generic_bool_op():
    op = (bool_operator(lambda x: x ** 2 > 10) | bool_operator(np.isnan))
    assert op(2) is False
    assert op(4) is True
    assert op(np.nan) is True
    assert (~F.isna)(np.nan) is False


def test_exception_fallback():
    with pytest.raises(TypeError):
        F.lt(0)("a")
    assert F.lt(0, False)("a") is False
    with pytest.raises(TypeError):
        (F.lt(0, False) | F.gt(0))("a")
    assert (F.lt(0, False) | F.gt(0, True))("a") is True
