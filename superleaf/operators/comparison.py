from typing import Any

from .base import BooleanFunctionOperator, BooleanOperator


def _isna(x: Any) -> bool:
    try:
        return x is None or x != x
    except ValueError:
        return False


class ComparisonFunctions:
    @staticmethod
    def eq(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x == value)

    @staticmethod
    def ne(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x != value)

    @staticmethod
    def lt(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x < value)

    @staticmethod
    def le(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x <= value)

    @staticmethod
    def gt(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x > value)

    @staticmethod
    def ge(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x >= value)

    @staticmethod
    def isin(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: x in value)

    @staticmethod
    def contains(value: Any) -> BooleanOperator:
        return BooleanFunctionOperator(lambda x: value in x)

    @staticmethod
    def startswith(value: str) -> BooleanOperator:
        return BooleanFunctionOperator(lambda s: s.startswith(value))

    @staticmethod
    def endswith(value: str) -> BooleanOperator:
        return BooleanFunctionOperator(lambda s: s.endswith(value))
    
    isna = BooleanFunctionOperator(_isna)
    notna = ~isna
