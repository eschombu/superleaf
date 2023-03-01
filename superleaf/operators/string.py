from superleaf.operators import operator, Operator


def str_op(method: str, *args, **kwargs) -> Operator:
    return operator(lambda s: getattr(s, method)(*args, **kwargs))
