import re
from typing import Any, Iterable

import pandas as pd

from superleaf.operators.base import bool_operator, BooleanOperator


def _isna(x: Any) -> bool:
    try:
        return x is None or x != x
    except ValueError:
        return False


def _parse_exc_args(*exc_args, **exc_kwargs) -> dict:
    if exc_args:
        fallback = exc_args[0]
        exc_args = exc_args[1:]
        if exc_args:
            exceptions = exc_args[0]
        elif "exceptions" in exc_kwargs:
            exceptions = exc_kwargs["exceptions"]
        else:
            exceptions = Exception
        return {"fallback": fallback, "exceptions": exceptions}
    else:
        return exc_kwargs


class ComparisonFunctions:
    @staticmethod
    def eq(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x == value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def ne(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x != value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def lt(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x < value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def le(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x <= value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def gt(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x > value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def ge(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x >= value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def isin(values: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        if isinstance(values, pd.Series):
            values = values.values
        return bool_operator(lambda x: x in values, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def contains(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: value in x, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def contains_all(values: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        if isinstance(values, str) or not hasattr(values, "__iter__"):
            values = [values]
        elif isinstance(values, pd.Series):
            values = values.values
        return bool_operator(lambda x: all(v in x for v in values), **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def contains_any(values: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        if isinstance(values, str) or not hasattr(values, "__iter__"):
            values = [values]
        elif isinstance(values, pd.Series):
            values = values.values
        return bool_operator(lambda x: any(v in x for v in values), **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def startswith(value: str, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: s.startswith(value), **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def endswith(value: str, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: s.endswith(value), **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def startswith_one_of(values: Iterable[str], *exc_args, **exc_kwargs) -> BooleanOperator:
        if isinstance(values, pd.Series):
            values = values.values
        return bool_operator(lambda s: any(s.startswith(v) for v in values),
                             **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def endswith_one_of(values: Iterable[str], *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: any(s.endswith(v) for v in values),
                             **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def matches_regex(pattern: str, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: re.match(pattern, s) is not None,
                             **_parse_exc_args(*exc_args, **exc_kwargs))

    isna = bool_operator(_isna)
    notna = ~isna
