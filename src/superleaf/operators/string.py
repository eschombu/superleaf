from superleaf.operators.base import bool_operator, BooleanOperator, operator, Operator


def str_op(method: str, *args, **kwargs) -> Operator:
    return operator(lambda s: getattr(s, method)(*args, **kwargs))


def str_bool_op(method: str, *args, **kwargs) -> BooleanOperator:
    return bool_operator(lambda s: getattr(s, method)(*args, **kwargs))
