from superleaf.operators.base import BooleanFunctionOperator, FunctionOperator
from superleaf.operators.wrappers import with_fallback


def operator(f, fallback=None, exceptions=None):
    if exceptions is not None:
        f = with_fallback(f, fallback=fallback, exceptions=exceptions)
    return FunctionOperator(f)


def bool_operator(f, fallback=False, exceptions=None):
    if exceptions is not None:
        if not isinstance(fallback, bool):
            raise TypeError("fallback value must be of type `bool`")
        f = with_fallback(f, fallback=fallback, exceptions=exceptions)
    return BooleanFunctionOperator(f)
