from typing import Any, Callable, Sequence, Union
from warnings import warn

import numpy as np
import pandas as pd
from pandas.api.types import is_scalar
from pandas.api.typing import Expression

from superleaf.collections.ordered_set import OrderedSet
from superleaf.operators.base import BooleanOperator
from superleaf.operators.comparison import F


class where:
    """Row filter evaluated on the whole DataFrame at once.

    Use with :func:`dfilter` or :func:`partition` for vectorized conditions across columns. A ``pd.col``
    expression can be passed directly instead.

    Parameters
    ----------
    f : callable
        Function from the DataFrame to a boolean Series or array with one value per row.

    Examples
    --------
    >>> dfilter(df, where(lambda d: d["a"] > d["b"]))
    """

    def __init__(self, f: Callable[[pd.DataFrame], Any]):
        self.f = f

    def __call__(self, df: pd.DataFrame) -> Any:
        return self.f(df)


def _as_mask(values: Any, df: pd.DataFrame | pd.Series) -> np.ndarray:
    if isinstance(values, pd.Series):
        if not values.index.equals(df.index):
            raise ValueError("Boolean Series filter must have the same index as the DataFrame")
        if isinstance(values.dtype, pd.BooleanDtype):
            values = values.fillna(False)
    mask = np.asarray(values, dtype=bool)
    if mask.shape != (len(df),):
        raise ValueError(f"Filter produced shape {mask.shape}; expected ({len(df)},)")
    return mask


def _eval_expression(df: pd.DataFrame | pd.Series, expr: Expression) -> pd.Series:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("pd.col expressions can only filter a DataFrame")
    name = "_superleaf_mask"
    while name in df.columns:
        name += "_"
    # assign appends the new column last
    return df.assign(**{name: expr}).iloc[:, -1]


def _get_column(df: pd.DataFrame | pd.Series, col: Any) -> pd.Series:
    if isinstance(df, pd.DataFrame):
        return df[col]
    if col == "index":
        return pd.Series(df.index, index=df.index)
    if col == df.name:
        return df
    raise KeyError(f"Keyword filter {col!r} for a Series must be 'index' or the Series name")


def _column_mask(values: pd.Series, col: Any, filt: Any) -> np.ndarray:
    if isinstance(filt, Expression):
        raise TypeError(
            f"Keyword filter {col!r} got a pd.col expression. Pass it positionally, e.g. "
            f"dfilter(df, pd.col({col!r}) == ...)."
        )
    if isinstance(filt, BooleanOperator):
        return _as_mask(filt.on_series(values), values)
    if callable(filt):
        return np.fromiter((bool(filt(x)) for x in values), dtype=bool, count=len(values))
    if is_scalar(filt):
        return _as_mask(F.eq(filt).on_series(values), values)
    raise TypeError(
        f"Keyword filter {col!r} got a {type(filt).__name__}. Use F.isin(...) to test membership, "
        "or F.eq(...) to compare with a non-scalar value."
    )


def _row_mask(df: pd.DataFrame | pd.Series, *filters, **col_filters) -> np.ndarray:
    mask = np.ones(len(df), dtype=bool)
    for filt in filters:
        if isinstance(filt, dict):
            conflicting_keys = set(filt.keys()).intersection(col_filters.keys())
            if conflicting_keys:
                raise ValueError(f"Received multiple filters for columns: {', '.join(map(str, conflicting_keys))}")
            col_filters.update(filt)
        elif isinstance(filt, Expression):
            mask &= _as_mask(_eval_expression(df, filt), df)
        elif isinstance(filt, where):
            mask &= _as_mask(filt(df), df)
        elif callable(filt):
            if isinstance(df, pd.Series):
                values = filt.on_series(df) if isinstance(filt, BooleanOperator) else filt(df)
                mask &= _as_mask(values, df)
            elif len(df) > 0:
                mask &= _as_mask(df.apply(filt, axis=1), df)
        elif np.iterable(filt) and not isinstance(filt, str):
            mask &= _as_mask(filt, df)
        else:
            raise TypeError(
                "Positional filters must be callables applied to each row, where(...) filters, dicts of "
                f"column filters, or boolean sequences with one value per row; got {type(filt).__name__}"
            )
    for col, filt in col_filters.items():
        mask &= _column_mask(_get_column(df, col), col, filt)
    return mask


def _pop_column_mask(df: pd.DataFrame | pd.Series, col_filters: dict) -> np.ndarray | None:
    if "columns" not in col_filters or (isinstance(df, pd.DataFrame) and "columns" in df.columns):
        return None
    selector = col_filters.pop("columns")
    if selector is None:
        return None
    if isinstance(df, pd.Series):
        raise TypeError("The columns filter is not supported for a Series")
    if callable(selector):
        selectors = [selector]
    elif isinstance(selector, str) or not np.iterable(selector):
        selectors = [selector]
    else:
        selectors = list(selector)
    preds = [s if callable(s) else F.eq(s) for s in selectors]
    return np.array([any(p(c) for p in preds) for c in df.columns], dtype=bool)


def _select(df: pd.DataFrame | pd.Series, rows: np.ndarray, cols: np.ndarray | None) -> pd.DataFrame | pd.Series:
    if isinstance(df, pd.Series):
        return df[rows]
    if cols is None:
        return df.loc[rows]
    return df.loc[rows, cols]


def dfilter(df: pd.DataFrame, *filters, **col_filters) -> pd.DataFrame:
    """Select the rows of a DataFrame that pass every filter.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to filter. A Series is also accepted; its index can be filtered with ``index=...``.
    *filters
        Row filters, each one of:

        - a ``pd.col`` expression, such as ``pd.col("a") > pd.col("b")``;
        - a :class:`where`, evaluated once on the whole DataFrame;
        - another callable, applied to each row (a Series) and returning a bool;
        - a dict of column filters, as for ``**col_filters`` (for column names that are not identifiers);
        - a boolean sequence or Series with one value per row.
    **col_filters
        Column filters, mapping a column name to one of:

        - a :class:`~superleaf.operators.base.BooleanOperator` such as ``F.gt(0)``, evaluated with
          ``on_series``;
        - another callable, applied to each value;
        - a scalar, selecting values equal to it (as ``F.eq``).

        Lists and other non-scalar values raise ``TypeError``; use ``F.isin`` for membership.

        The keyword ``columns``, unless it names a column of ``df``, selects columns instead of rows. It takes a
        column name, a predicate on names, or a list of either; columns matching any of them are kept.

    Returns
    -------
    pd.DataFrame
        The selected rows (and columns), as a new DataFrame.

    Raises
    ------
    TypeError
        If a filter has an unsupported type.
    ValueError
        If a boolean filter's length or index does not match ``df``.

    Examples
    --------
    >>> dfilter(df, status="active", age=F.ge(30))
    >>> dfilter(df, name=F.startswith("A") | F.isna, columns=["name", "age"])
    >>> dfilter(df, pd.col("score") > pd.col("threshold"), status="active")
    """
    col_filters = dict(col_filters)
    col_mask = _pop_column_mask(df, col_filters)
    return _select(df, _row_mask(df, *filters, **col_filters), col_mask)


def partition(df: pd.DataFrame, *filters, **col_filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a DataFrame into rows that pass every filter and rows that do not.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to split.
    *filters
        Row filters, as for :func:`dfilter`.
    **col_filters
        Column filters, as for :func:`dfilter`, including ``columns``.

    Returns
    -------
    pd.DataFrame
        The rows that pass.
    pd.DataFrame
        The remaining rows.

    Examples
    --------
    >>> passed, failed = partition(df, score=F.gt(50))
    """
    col_filters = dict(col_filters)
    col_mask = _pop_column_mask(df, col_filters)
    mask = _row_mask(df, *filters, **col_filters)
    return _select(df, mask, col_mask), _select(df, ~mask, col_mask)


def reorder_columns(
        df: pd.DataFrame,
        columns: Union[str, Sequence[str]],
        back=False,
        after=None,
        before=None,
        errors='raise',
) -> pd.DataFrame:
    """Move columns to the front, the back, or next to another column.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame whose columns to reorder.
    columns : str or sequence of str
        Column(s) to move, in the order they should appear.
    back : bool, optional
        If True, move the columns to the end.
    after : str, optional
        Place the columns immediately after this column.
    before : str, optional
        Place the columns immediately before this column.
    errors : {"raise", "warn", "ignore"}, optional
        How to handle names in ``columns`` that are not in ``df``. With ``"warn"`` or ``"ignore"``, they are
        skipped.

    Returns
    -------
    pd.DataFrame
        A new DataFrame with reordered columns. With none of ``back``, ``after`` or ``before``, the columns
        are moved to the front.

    Raises
    ------
    ValueError
        If more than one of ``back``, ``after`` and ``before`` is given, or ``errors`` is invalid.
    KeyError
        If a column is missing and ``errors="raise"``.

    Examples
    --------
    >>> reorder_columns(df, ["age", "name"], after="id")
    """
    allowed_error_methods = ('raise', 'ignore', 'warn', 'warning')
    errors = errors.lower()
    if errors not in allowed_error_methods:
        raise ValueError(f"`errors` must be one of {allowed_error_methods}")
    if sum([bool(back), after is not None, before is not None]) > 1:
        raise ValueError("Only one of the following parameters can be used at a time: (back, after, before)")
    if isinstance(columns, str):
        columns = [columns]
    df_cols = OrderedSet(df.columns)
    columns = OrderedSet(columns)
    missing = columns - df_cols
    if missing:
        message = f"Columns not found: {list(missing)}"
        if errors == 'raise':
            raise KeyError(message)
        if errors in ('warn', 'warning'):
            warn(message)
        columns = columns - missing
    if back:
        col_order = list((df_cols - columns) + columns)
    elif after is not None or before is not None:
        if after is not None:
            insert_idx = list(df.columns).index(after) + 1
        else:
            insert_idx = list(df.columns).index(before)
        col_order = list((OrderedSet(df_cols[:insert_idx]) - columns) + columns + OrderedSet(df_cols[insert_idx:]))
    else:
        col_order = list(columns + df_cols)
    return pd.DataFrame(df[col_order])
