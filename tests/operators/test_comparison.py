import re
from datetime import datetime

import numpy as np
import pytest

from superleaf.operators.base import bool_operator
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
    assert F.endswith("he")("hello") is False
    assert F.endswith("lo")("hello") is True

    assert F.startswith_one_of(["he", "no"])("hello") is True
    assert F.startswith_one_of(["lo", "no"])("hello") is False
    assert F.endswith_one_of(["lo", "no"])("hello") is True
    assert F.endswith_one_of(["he", "no"])("hello") is False

    assert F.matches_regex('^h.*o$')('hello') is True
    assert F.matches_regex('^h.*l$')('hello') is False
    assert F.matches_regex('^h.*o$', flags=re.IGNORECASE)('Hello') is True

    assert F.startswith("2025", str_converter=lambda dt: dt.strftime('%Y-%m-%d'))(datetime(2025, 1, 1)) is True

    # TODO: test str comparisons for non-str inputs
    assert F.startswith("he")(None) is False
    assert F.startswith("he")(np.nan) is False
    assert F.startswith("he")(1) is False
    assert F.startswith(1)(123) is True

    assert F.endswith("lo")(None) is False
    assert F.endswith("lo")(np.nan) is False
    assert F.endswith("lo")(1) is False
    assert F.endswith(3)(123) is True

    assert F.startswith_one_of(["he", "no"])(None) is False
    assert F.startswith_one_of(["he", 1])(123) is True

    assert F.endswith_one_of(["he", "no"])(None) is False
    assert F.endswith_one_of(["he", 3])(123) is True

    assert F.matches_regex('^1.*4$')(1234) is True

    with pytest.raises(TypeError):
        assert F.startswith("he", raise_type_error=True)(None)
    with pytest.raises(TypeError):
        assert F.startswith("he", raise_type_error=True)(1)
    with pytest.raises(TypeError):
        assert F.startswith(1, raise_type_error=True)(123)
    with pytest.raises(TypeError):
        assert F.startswith_one_of(["he"], raise_type_error=True)(None)
    with pytest.raises(TypeError):
        assert F.startswith_one_of(["he"], raise_type_error=True)(1)
    with pytest.raises(TypeError):
        assert F.startswith_one_of([1], raise_type_error=True)("1")
    with pytest.raises(TypeError):
        assert F.endswith("lo", raise_type_error=True)(None)
    with pytest.raises(TypeError):
        assert F.endswith("lo", raise_type_error=True)(1)
    with pytest.raises(TypeError):
        assert F.endswith(1, raise_type_error=True)("123")
    with pytest.raises(TypeError):
        assert F.endswith_one_of(["lo"], raise_type_error=True)(None)
    with pytest.raises(TypeError):
        assert F.endswith_one_of(["lo"], raise_type_error=True)(1)
    with pytest.raises(TypeError):
        assert F.endswith_one_of([1], raise_type_error=True)("1")
    with pytest.raises(TypeError):
        assert F.matches_regex('^1.*4$', raise_type_error=True)(1234) is True

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
