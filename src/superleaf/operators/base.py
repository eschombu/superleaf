from abc import ABCMeta, abstractmethod
from typing import Any, Callable

import numpy as np
import pandas as pd

from superleaf.operators.wrappers import with_fallback


class Operator(metaclass=ABCMeta):
    """A callable that can be piped into another callable with ``>>``.

    ``(a >> b)(x)`` is ``b(a(x))``. If ``b`` is a :class:`BooleanOperator`, the result is also a
    :class:`BooleanOperator`.
    """

    @abstractmethod
    def __call__(self, arg: Any) -> Any:
        pass

    def __rshift__(self, right: Callable) -> "Operator":
        if isinstance(right, BooleanOperator):
            return _PipedBooleanOperator(self, right)
        return _PipedOperator(self, right)


class _PipedOperator(Operator):
    def __init__(self, left: Callable, right: Callable):
        self._left = left
        self._right = right

    def __call__(self, arg: Any) -> Any:
        return self._right(self._left(arg))


class FunctionOperator(Operator):
    """An :class:`Operator` wrapping a single-argument function."""

    def __init__(self, f: Callable):
        self._fun = f

    def __call__(self, arg: Any) -> Any:
        return self._fun(arg)


def _as_bool_op(f: Callable) -> "BooleanOperator":
    if isinstance(f, BooleanOperator):
        return f
    if callable(f):
        return BooleanFunctionOperator(f)
    raise TypeError(f"Cannot combine a BooleanOperator with {type(f).__name__!r}; expected a callable")


class BooleanOperator(Operator, metaclass=ABCMeta):
    """A predicate on a single value, combinable with ``|``, ``&`` and ``~``.

    Calling the operator evaluates one value and returns a ``bool``. :meth:`on_series` evaluates every
    element of a ``pd.Series`` and returns a boolean ``pd.Series`` with the same index, giving the
    same result per element as calling the operator on it.

    ``|`` and ``&`` short-circuit like ``or`` and ``and``: the right operand is evaluated only where the
    left one does not decide the result, so ``F.notna & F.gt(0)`` never compares a missing value.
    """

    @abstractmethod
    def __call__(self, arg: Any) -> bool:
        pass

    def on_series(self, s: pd.Series) -> pd.Series:
        """Evaluate the predicate on each element of a Series.

        Parameters
        ----------
        s : pd.Series
            Values to test.

        Returns
        -------
        pd.Series
            Boolean Series with the same index as ``s``.
        """
        return pd.Series(np.fromiter((self(x) for x in s), dtype=bool, count=len(s)), index=s.index)

    def __or__(self, right: Callable) -> "BooleanOperator":
        return _OrBoolOp(self, _as_bool_op(right))

    def __ror__(self, left: Callable) -> "BooleanOperator":
        return _OrBoolOp(_as_bool_op(left), self)

    def __and__(self, right: Callable) -> "BooleanOperator":
        return _AndBoolOp(self, _as_bool_op(right))

    def __rand__(self, left: Callable) -> "BooleanOperator":
        return _AndBoolOp(_as_bool_op(left), self)

    def __invert__(self) -> "BooleanOperator":
        return _NotBoolOp(self)


class BooleanFunctionOperator(BooleanOperator):
    """A :class:`BooleanOperator` wrapping a single-argument function; its result is converted with ``bool``."""

    def __init__(self, f: Callable[[Any], Any]) -> None:
        self._fun = f

    def __call__(self, arg: Any) -> bool:
        return bool(self._fun(arg))


class _PipedBooleanOperator(_PipedOperator, BooleanOperator):
    def __call__(self, arg: Any) -> bool:
        return bool(self._right(self._left(arg)))


class _BinaryBoolOp(BooleanOperator, metaclass=ABCMeta):
    def __init__(self, left: BooleanOperator, right: BooleanOperator):
        self._left = left
        self._right = right

    def _on_series(self, s: pd.Series, decided_value: bool) -> pd.Series:
        # Evaluates the right operand only on elements the left operand does not decide.
        result = self._left.on_series(s).to_numpy(dtype=bool, copy=True)
        undecided = result != decided_value
        if undecided.any():
            result[undecided] = self._right.on_series(s[undecided]).to_numpy(dtype=bool)
        return pd.Series(result, index=s.index)


class _OrBoolOp(_BinaryBoolOp):
    def __call__(self, arg: Any) -> bool:
        return self._left(arg) or self._right(arg)

    def on_series(self, s: pd.Series) -> pd.Series:
        return self._on_series(s, decided_value=True)


class _AndBoolOp(_BinaryBoolOp):
    def __call__(self, arg: Any) -> bool:
        return self._left(arg) and self._right(arg)

    def on_series(self, s: pd.Series) -> pd.Series:
        return self._on_series(s, decided_value=False)


class _NotBoolOp(BooleanOperator):
    def __init__(self, op: BooleanOperator):
        self._op = op

    def __call__(self, arg: Any) -> bool:
        return not self._op(arg)

    def on_series(self, s: pd.Series) -> pd.Series:
        return ~self._op.on_series(s)


class VectorizedBooleanOperator(BooleanFunctionOperator):
    """A :class:`BooleanFunctionOperator` with a vectorized implementation for Series.

    Parameters
    ----------
    f : callable
        Scalar predicate.
    vf : callable
        Function from a Series to a boolean array-like of the same length, matching ``f`` element-wise. It may
        raise ``TypeError`` or ``ValueError`` (or :class:`NotVectorizable`) to fall back to evaluating ``f`` on
        each element.
    vectorize_missing : bool, optional
        If True, use ``vf`` results for missing values too. Otherwise they are replaced with ``f`` results.
    """

    def __init__(self, f: Callable[[Any], Any], vf: Callable[[pd.Series], Any], vectorize_missing: bool = False):
        super().__init__(f)
        self._vfun = vf
        self._vectorize_missing = vectorize_missing

    def on_series(self, s: pd.Series) -> pd.Series:
        try:
            result = _to_bool_array(self._vfun(s))
        except (NotVectorizable, TypeError, ValueError):
            return super().on_series(s)
        if not self._vectorize_missing:
            na = s.isna().to_numpy(dtype=bool)
            if na.any():
                # Missing values of the same type give the same result, so evaluate each type once.
                by_type = {}
                result[na] = [
                    by_type[type(x)] if type(x) in by_type else by_type.setdefault(type(x), self(x))
                    for x in s[na]
                ]
        return pd.Series(result, index=s.index)


def _to_bool_array(values) -> np.ndarray:
    if isinstance(getattr(values, "dtype", None), pd.BooleanDtype):
        values = values.fillna(False)
    return np.array(values, dtype=bool)


class NotVectorizable(Exception):
    """Raised by a vectorized implementation to request element-wise evaluation."""


def operator(f: Callable, exceptions=None, fallback=None) -> FunctionOperator:
    """Wrap a single-argument function as a pipeable :class:`Operator`.

    Parameters
    ----------
    f : callable
        The function to wrap.
    exceptions : type, tuple of types, or str, optional
        Exceptions to catch, returning ``fallback`` instead (see :func:`with_fallback`). Not caught if None.
    fallback : Any, optional
        Value returned when one of ``exceptions`` is raised.

    Returns
    -------
    FunctionOperator
        The wrapped function.
    """
    if exceptions is not None:
        f = with_fallback(f, fallback=fallback, exceptions=exceptions)
    return FunctionOperator(f)


def bool_operator(f: Callable, exceptions=None, fallback=False) -> BooleanFunctionOperator:
    """Wrap a single-argument predicate as a :class:`BooleanOperator`.

    Parameters
    ----------
    f : callable
        The predicate; its result is converted with ``bool``.
    exceptions : type, tuple of types, or str, optional
        Exceptions to catch, returning ``fallback`` instead (see :func:`with_fallback`). Not caught if None.
    fallback : bool, optional
        Value returned when one of ``exceptions`` is raised. Default is False.

    Returns
    -------
    BooleanFunctionOperator
        The wrapped predicate.

    Raises
    ------
    TypeError
        If ``fallback`` is not a ``bool``.
    """
    if exceptions is not None:
        if not isinstance(fallback, bool):
            raise TypeError("fallback value must be of type ``bool``")
        f = with_fallback(f, fallback=fallback, exceptions=exceptions)
    return BooleanFunctionOperator(f)
