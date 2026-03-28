from functools import wraps
from typing import Callable, Optional


def with_fallback(f: Optional[Callable] = None, fallback=None, exceptions: type | str | None = Exception):
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
