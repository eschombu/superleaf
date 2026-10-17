from superleaf.operators.base import bool_operator, BooleanOperator, operator, Operator


def str_op(method: str, *args, **kwargs) -> Operator:
    """Operator calling a method by name, e.g. ``str_op("lower")(s) == s.lower()``.

    Parameters
    ----------
    method : str
        Name of the method to call on the input.
    *args
        Positional arguments passed to the method.
    **kwargs
        Keyword arguments passed to the method.

    Returns
    -------
    Operator
        The operator.
    """
    return operator(lambda s: getattr(s, method)(*args, **kwargs))


def str_bool_op(method: str, *args, **kwargs) -> BooleanOperator:
    """Predicate calling a method by name, e.g. ``str_bool_op("isdigit")(s) == s.isdigit()``.

    Parameters
    ----------
    method : str
        Name of the method to call on the input.
    *args
        Positional arguments passed to the method.
    **kwargs
        Keyword arguments passed to the method.

    Returns
    -------
    BooleanOperator
        The predicate.
    """
    return bool_operator(lambda s: getattr(s, method)(*args, **kwargs))
