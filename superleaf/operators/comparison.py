from typing import Any

from superleaf.operators import bool_operator
from superleaf.operators.base import BooleanOperator


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
    def isin(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: x in value, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def contains(value: Any, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda x: value in x, **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def startswith(value: str, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: s.startswith(value), **_parse_exc_args(*exc_args, **exc_kwargs))

    @staticmethod
    def endswith(value: str, *exc_args, **exc_kwargs) -> BooleanOperator:
        return bool_operator(lambda s: s.endswith(value), **_parse_exc_args(*exc_args, **exc_kwargs))
    
    isna = bool_operator(_isna)
    notna = ~isna
