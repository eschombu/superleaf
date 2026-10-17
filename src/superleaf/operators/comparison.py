import operator as _op
import re
from typing import Any, Callable, Iterable, Literal

import numpy as np
import pandas as pd
from pandas.api.types import infer_dtype, is_datetime64_any_dtype, is_scalar, is_timedelta64_dtype

from superleaf.operators.base import (
    BooleanOperator,
    NotVectorizable,
    VectorizedBooleanOperator,
    bool_operator,
)
from superleaf.operators.wrappers import with_fallback


class _Unset:
    def __repr__(self) -> str:
        return "<unset>"


_UNSET = _Unset()


def _isna(x: Any) -> bool:
    return is_scalar(x) and bool(pd.isna(x))


def _make_op(f: Callable, vf: Callable | None = None, fallback=_UNSET, exceptions=None, **vec_kws) -> BooleanOperator:
    if exceptions is None and fallback is not _UNSET:
        exceptions = Exception
    if exceptions is not None:
        if fallback is _UNSET:
            fallback = False
        if not isinstance(fallback, bool):
            raise TypeError("fallback value must be of type ``bool``")
        f = with_fallback(f, fallback=fallback, exceptions=exceptions)
    if vf is None:
        return bool_operator(f)
    return VectorizedBooleanOperator(f, vf, **vec_kws)


def _require_strings(s: pd.Series) -> None:
    if infer_dtype(s, skipna=True) not in ("string", "empty"):
        raise NotVectorizable


def _comparison(op: Callable[[Any, Any], Any], value: Any, fallback, exceptions) -> BooleanOperator:
    if _isna(value):
        def f(x):
            return op(x, value)
    else:
        missing_result = op is _op.ne

        def f(x):
            if _isna(x):
                return missing_result
            return op(x, value)

    def vf(s: pd.Series):
        if _compares_differently(s, value):
            raise NotVectorizable
        return op(s, value)

    vectorizable = is_scalar(value) and not _isna(value)
    return _make_op(f, vf if vectorizable else None, fallback=fallback, exceptions=exceptions)


def _compares_differently(s: pd.Series, value: Any) -> bool:
    # pandas parses strings compared with datetime-like Series, and orders categoricals by category order.
    if isinstance(s.dtype, pd.CategoricalDtype):
        return True
    return isinstance(value, str) and (
        is_datetime64_any_dtype(s.dtype) or is_timedelta64_dtype(s.dtype) or isinstance(s.dtype, pd.PeriodDtype)
    )


def _str_value(value, str_converter):
    if str_converter is not None and not isinstance(value, str):
        return str_converter(value)
    return value


def _get_str_op(str_method: Callable[[str, Any], bool], value, str_converter=str, raise_type_error=False):
    if raise_type_error:
        def op(s):
            try:
                return str_method(s, value)
            except Exception as e:
                raise TypeError(str(e))
    elif str_converter is not None:
        value = _str_value(value, str_converter)

        def op(s):
            if _isna(s):
                return False
            return str_method(s if isinstance(s, str) else str_converter(s), value)
    else:
        def op(s):
            return isinstance(s, str) and str_method(s, value)
    return op


def _str_prefix_op(
        str_method: Callable, series_method: str, values: list, str_converter, raise_type_error, fallback, exceptions
) -> BooleanOperator:
    if isinstance(values, pd.Series):
        values = values.values
    ops = [_get_str_op(str_method, v, str_converter=str_converter, raise_type_error=raise_type_error) for v in values]

    def f(s):
        return any(op(s) for op in ops)

    patterns = tuple(_str_value(v, str_converter) for v in values)

    def vf(s: pd.Series):
        _require_strings(s)
        return getattr(s.str, series_method)(patterns)

    vectorizable = not raise_type_error and patterns and all(isinstance(p, str) for p in patterns)
    return _make_op(f, vf if vectorizable else None, fallback=fallback, exceptions=exceptions)


def _container_values(values: Any) -> set:
    if isinstance(values, str) or not isinstance(values, Iterable):
        values = [values]
    elif isinstance(values, pd.Series):
        values = values.values
    return set(values)


_REGEX_FUNS = {"match": re.match, "search": re.search, "fullmatch": re.fullmatch}
_REGEX_SERIES_FUNS = {
    "match": lambda s, p, flags: s.str.match(p, flags=flags),
    "search": lambda s, p, flags: s.str.contains(p, flags=flags, regex=True),
    "fullmatch": lambda s, p, flags: s.str.fullmatch(p, flags=flags),
}


class ComparisonFunctions:
    """Factories for common :class:`BooleanOperator` predicates, usually imported as ``F``.

    Every predicate works on single values (``F.gt(0)(5)``) and on Series via ``on_series``, which
    ``dfilter`` and ``partition`` use for keyword filters. ``on_series`` gives the same result per element
    as the scalar call, using vectorized pandas operations where they agree with it.

    In ``eq``, ``ne``, ``lt``, ``le``, ``gt`` and ``ge`` with a non-missing ``value``, missing values (None, NaN,
    NaT, ``pd.NA``) are treated like NaN: only ``ne`` is True.

    Predicates that take ``fallback`` and ``exceptions`` return ``fallback`` when evaluating a value raises
    one of ``exceptions``. Passing either one enables this; ``exceptions`` defaults to ``Exception`` and
    ``fallback`` to False. Without them, exceptions propagate.

    Examples
    --------
    >>> F.gt(0)(1)
    True
    >>> (F.isna | F.lt(0))(float("nan"))
    True
    >>> F.lt(0, fallback=False)("a")
    False

    Attributes
    ----------
    isna : BooleanOperator
        Test that ``x`` is a missing scalar (None, NaN, NaT or ``pd.NA``).
    notna : BooleanOperator
        Negation of ``isna``.
    isinf : BooleanOperator
        Test that ``x`` is infinite (``np.isinf``).
    notinf : BooleanOperator
        Negation of ``isinf``.
    """

    @staticmethod
    def eq(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x == value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.eq, value, fallback, exceptions)

    @staticmethod
    def ne(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x != value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.ne, value, fallback, exceptions)

    @staticmethod
    def lt(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x < value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.lt, value, fallback, exceptions)

    @staticmethod
    def le(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x <= value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.le, value, fallback, exceptions)

    @staticmethod
    def gt(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x > value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.gt, value, fallback, exceptions)

    @staticmethod
    def ge(value: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x >= value``.

        Parameters
        ----------
        value : Any
            Value to compare against.
        fallback : bool, optional
            Result when comparing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _comparison(_op.ge, value, fallback, exceptions)

    @staticmethod
    def isin(values: Any, *, fallback=_UNSET, exceptions=None) -> BooleanOperator:
        """Test ``x in values``.

        Parameters
        ----------
        values : iterable or str
            Collection to test membership in; unhashable ``x`` gives False. A ``str`` tests for a substring.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        if isinstance(values, pd.Series):
            values = values.values
        if isinstance(values, str):
            def f(x):
                return x in values
        else:
            values = set(values)

            def f(x):
                try:
                    return x in values
                except TypeError:  # unhashable
                    return False

        def vf(s: pd.Series):
            if s.dtype == bool:
                raise NotVectorizable
            return s.isin(list(values))

        # pandas matches missing values to each other and booleans to numbers differently than ``in``.
        vectorizable = isinstance(values, set) and all(
            is_scalar(v) and not _isna(v) and not isinstance(v, (bool, np.bool_)) for v in values
        )
        return _make_op(f, vf if vectorizable else None, fallback=fallback, exceptions=exceptions)

    @staticmethod
    def contains(value: Any, *, exceptions=TypeError, fallback=False) -> BooleanOperator:
        """Test ``value in x``.

        Parameters
        ----------
        value : Any
            Item or substring to look for.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``. Default ``TypeError``, so non-containers give False.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``. Default False.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        def f(x):
            return value in x

        def vf(s: pd.Series):
            _require_strings(s)
            return s.str.contains(value, regex=False)

        return _make_op(f, vf if isinstance(value, str) else None, fallback=fallback, exceptions=exceptions)

    @staticmethod
    def contains_all(values: Any, *, exceptions=TypeError, fallback=False) -> BooleanOperator:
        """Test that every item of ``values`` is in ``x``.

        Parameters
        ----------
        values : Any
            Items to look for. A ``str`` or non-iterable is treated as a single item.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``. Default ``TypeError``, so non-containers give False.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``. Default False.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        values = _container_values(values)
        return _make_op(lambda x: all(v in x for v in values), fallback=fallback, exceptions=exceptions)

    @staticmethod
    def contains_any(values: Any, *, exceptions=TypeError, fallback=False) -> BooleanOperator:
        """Test that at least one item of ``values`` is in ``x``.

        Parameters
        ----------
        values : Any
            Items to look for. A ``str`` or non-iterable is treated as a single item.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``. Default ``TypeError``, so non-containers give False.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``. Default False.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        values = _container_values(values)
        return _make_op(lambda x: any(v in x for v in values), fallback=fallback, exceptions=exceptions)

    @staticmethod
    def startswith(
            value: Any, *, str_converter=str, raise_type_error=False, fallback=_UNSET, exceptions=None
    ) -> BooleanOperator:
        """Test that ``x`` starts with ``value``.

        Parameters
        ----------
        value : Any
            Prefix. Converted with ``str_converter`` if not a ``str``.
        str_converter : callable or None, optional
            Converts non-missing, non-``str`` values of ``x`` (and ``value``) to ``str``. Default ``str``.
            If None, non-``str`` values give False.
        raise_type_error : bool, optional
            If True, raise ``TypeError`` for non-``str`` values instead of converting them.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate. Missing values give False unless ``raise_type_error`` is True.
        """
        return _str_prefix_op(
            str.startswith, "startswith", [value], str_converter, raise_type_error, fallback, exceptions
        )

    @staticmethod
    def endswith(
            value: Any, *, str_converter=str, raise_type_error=False, fallback=_UNSET, exceptions=None
    ) -> BooleanOperator:
        """Test that ``x`` ends with ``value``.

        Parameters
        ----------
        value : Any
            Suffix. Converted with ``str_converter`` if not a ``str``.
        str_converter : callable or None, optional
            Converts non-missing, non-``str`` values of ``x`` (and ``value``) to ``str``. Default ``str``.
            If None, non-``str`` values give False.
        raise_type_error : bool, optional
            If True, raise ``TypeError`` for non-``str`` values instead of converting them.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate. Missing values give False unless ``raise_type_error`` is True.
        """
        return _str_prefix_op(
            str.endswith, "endswith", [value], str_converter, raise_type_error, fallback, exceptions
        )

    @staticmethod
    def startswith_one_of(
            values: Iterable[Any], *, str_converter=str, raise_type_error=False, fallback=_UNSET, exceptions=None
    ) -> BooleanOperator:
        """Test that ``x`` starts with any of ``values``.

        Parameters
        ----------
        values : iterable
            Prefixes. Each is converted with ``str_converter`` if not a ``str``.
        str_converter : callable or None, optional
            See :meth:`startswith`.
        raise_type_error : bool, optional
            See :meth:`startswith`.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _str_prefix_op(
            str.startswith, "startswith", list(values), str_converter, raise_type_error, fallback, exceptions
        )

    @staticmethod
    def endswith_one_of(
            values: Iterable[Any], *, str_converter=str, raise_type_error=False, fallback=_UNSET, exceptions=None
    ) -> BooleanOperator:
        """Test that ``x`` ends with any of ``values``.

        Parameters
        ----------
        values : iterable
            Suffixes. Each is converted with ``str_converter`` if not a ``str``.
        str_converter : callable or None, optional
            See :meth:`endswith`.
        raise_type_error : bool, optional
            See :meth:`endswith`.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        return _str_prefix_op(
            str.endswith, "endswith", list(values), str_converter, raise_type_error, fallback, exceptions
        )

    @staticmethod
    def matches_regex(
            pattern: str,
            how: Literal["match", "search", "fullmatch"] = "match",
            *,
            flags: int = 0,
            str_converter=str,
            raise_type_error=False,
            fallback=_UNSET,
            exceptions=None,
    ) -> BooleanOperator:
        """Test that ``x`` matches a regular expression.

        Parameters
        ----------
        pattern : str
            Regular expression.
        how : {"match", "search", "fullmatch"}, optional
            ``"match"`` (default) anchors the pattern at the start of ``x``, ``"search"`` finds it anywhere,
            and ``"fullmatch"`` requires it to match all of ``x``; as in :mod:`re`.
        flags : int, optional
            :mod:`re` flags.
        str_converter : callable or None, optional
            See :meth:`startswith`.
        raise_type_error : bool, optional
            See :meth:`startswith`.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.

        Raises
        ------
        ValueError
            If ``how`` is not one of the allowed values.
        """
        if how not in _REGEX_FUNS:
            raise ValueError(f"how must be one of {tuple(_REGEX_FUNS)}, got {how!r}")
        regex_fun = _REGEX_FUNS[how]
        flags = flags or 0

        def has_match(s, p):
            return regex_fun(p, s, flags=flags) is not None

        op = _get_str_op(has_match, pattern, str_converter=str_converter, raise_type_error=raise_type_error)
        p = _str_value(pattern, str_converter)

        def vf(s: pd.Series):
            _require_strings(s)
            return _REGEX_SERIES_FUNS[how](s, p, flags)

        vectorizable = not raise_type_error and isinstance(p, str)
        return _make_op(op, vf if vectorizable else None, fallback=fallback, exceptions=exceptions)

    @staticmethod
    def fuzzy_match(
            targets: str | Iterable[str],
            *,
            normalizer: Callable[[str], str] | None = _UNSET,
            substring: bool = True,
            score_threshold: float = 80.0,
            raise_type_error: bool = False,
            fallback=_UNSET,
            exceptions=None,
    ) -> BooleanOperator:
        """Test that ``x`` approximately matches any of ``targets``. Requires ``superleaf[fuzzy]``.

        Parameters
        ----------
        targets : str or iterable of str
            Strings to match against.
        normalizer : callable or None, optional
            Applied to ``x`` and ``targets`` before scoring. Defaults to ``rapidfuzz.utils.default_process``
            (lowercase, non-alphanumerics replaced by spaces, trimmed). None disables normalization.
        substring : bool, optional
            If True (default), score with ``rapidfuzz.fuzz.partial_ratio``, so a target can match part of
            ``x``. Otherwise use ``rapidfuzz.fuzz.ratio``.
        score_threshold : float, optional
            Minimum score (0-100) for a match. Default 80.
        raise_type_error : bool, optional
            If True, raise ``TypeError`` for non-``str`` values; otherwise they give False.
        fallback : bool, optional
            Result when testing raises one of ``exceptions``.
        exceptions : type, tuple of types, or str, optional
            Exceptions that produce ``fallback``.

        Returns
        -------
        BooleanOperator
            The predicate.
        """
        from superleaf.operators.fuzzy import FuzzyMatcher

        matcher_kws = {"substring": substring}
        if normalizer is not _UNSET:
            matcher_kws["normalizer"] = normalizer
        matcher = FuzzyMatcher(targets, **matcher_kws)
        return _make_op(
            matcher._predicate(score_threshold, raise_type_error=raise_type_error),
            fallback=fallback,
            exceptions=exceptions,
        )

    isna: BooleanOperator = VectorizedBooleanOperator(_isna, lambda s: s.isna(), vectorize_missing=True)
    notna: BooleanOperator = ~isna
    isinf: BooleanOperator = VectorizedBooleanOperator(np.isinf, np.isinf)
    notinf: BooleanOperator = ~isinf


F = ComparisonFunctions
