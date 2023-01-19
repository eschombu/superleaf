from .base import BooleanFunctionOperator, FunctionOperator, Operator
from .comparison import ComparisonFunctions


def operator(f):
    return FunctionOperator(f)


def bool_operator(f):
    return BooleanFunctionOperator(f)
