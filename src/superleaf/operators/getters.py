from superleaf.operators.base import Operator, operator

_NotSpecified = object()


def attr_getter(name: str, default=_NotSpecified) -> Operator:
    """Operator returning ``getattr(obj, name)``.

    Parameters
    ----------
    name : str
        Attribute name.
    default : Any, optional
        Returned when the attribute is missing. If not given, ``AttributeError`` is raised.

    Returns
    -------
    Operator
        The getter.
    """
    def get(obj):
        if default is _NotSpecified:
            return getattr(obj, name)
        else:
            return getattr(obj, name, default)
    return operator(get)


def index_getter(index, default=_NotSpecified) -> Operator:
    """Operator returning ``obj[index]``.

    Parameters
    ----------
    index : Any
        Key or index.
    default : Any, optional
        Returned on ``KeyError`` or ``IndexError``. If not given, the error is raised.

    Returns
    -------
    Operator
        The getter.
    """
    def get(obj):
        try:
            return obj[index]
        except (KeyError, IndexError):
            if default is _NotSpecified:
                raise
            return default
    return operator(get)
