from functools import wraps
from typing import Callable, Optional


def with_fallback(
        f: Optional[Callable] = None, fallback=None, exceptions: type | tuple | str | None = Exception
) -> Callable:
    """Wrap a function to return ``fallback`` when it raises one of ``exceptions``.

    Usable directly (``with_fallback(f, 0)``) or as a decorator, with or without arguments.

    Parameters
    ----------
    f : callable, optional
        Function to wrap. If omitted, a decorator is returned.
    fallback : Any, optional
        Value returned when one of ``exceptions`` is raised. Default None.
    exceptions : type, tuple of types, or str, optional
        Exceptions to catch. Default ``Exception``. ``"ignore"``/``"all"`` mean ``Exception``;
        ``"raise"``/``"none"`` or None catch nothing.

    Returns
    -------
    Callable
        The wrapped function, or a decorator if ``f`` is None.

    Raises
    ------
    ValueError
        If ``exceptions`` is an unrecognized string.
    """
    if isinstance(exceptions, str):
        if exceptions in ("ignore", "all"):
            exceptions = Exception
        elif exceptions in ("raise", "none"):
            exceptions = None
        else:
            raise ValueError(f"Invalid value for exceptions: {exceptions!r}")

    def wrapper(fun):
        @wraps(fun)
        def wrapped(*args, **kwargs):
            try:
                return fun(*args, **kwargs)
            except exceptions:
                return fallback

        if exceptions is not None:
            return wrapped
        else:
            return fun

    if f is None:
        return wrapper
    else:
        return wrapper(f)
