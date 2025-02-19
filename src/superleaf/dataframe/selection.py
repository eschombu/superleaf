from typing import Sequence, Union

import numpy as np
import pandas as pd

from superleaf.dataframe.column_ops import Col, ColOp
from superleaf.collections.ordered_set import OrderedSet


def _pass_filter(df: pd.DataFrame, *filters, **col_filters) -> np.ndarray[bool]:
    row_bools = np.ones(len(df)).astype(bool)
    for filt in filters:
        if isinstance(filt, ColOp):
            row_bools = row_bools & filt(df)
        elif callable(filt):
            row_bools = row_bools & df.apply(filt, axis=1)
        else:
            try:
                row_bools = row_bools & np.array(list(filt))
            except:
                raise TypeError(
                    "Positional filters must be column operators or callables to apply to each row"
                )
    for col, filt in col_filters.items():
        if isinstance(filt, ColOp) or not callable(filt):
            row_bools = row_bools & (Col(col) == filt)(df)
        elif callable(filt):
            row_bools = row_bools & Col(col).map(filt)(df)
        else:
            raise TypeError(
                "Keyword filters must be values, column operators, or callables to apply to each value in the column"
            )
    return row_bools


def dfilter(df: pd.DataFrame, *filters, **col_filters) -> pd.DataFrame:
    return df[_pass_filter(df, *filters, **col_filters)].copy()


def partition(df: pd.DataFrame, *filters, **col_filters) -> tuple[pd.DataFrame, pd.DataFrame]:
    row_bools = _pass_filter(df, *filters, **col_filters)
    return df[row_bools].copy(), df[~row_bools].copy()


def reorder_columns(df: pd.DataFrame, columns: Union[str, Sequence[str]], back=False, after=None, before=None
                    ) -> pd.DataFrame:
    if isinstance(columns, str):
        columns = [columns]
    df_cols = OrderedSet(df.columns)
    columns = OrderedSet(columns)
    if sum([back, after is not None, before is not None]) > 1:
        raise ValueError("Only one of the following parameters can be used at a time: (back, after, before)")
    if back:
        col_order = list((df_cols - columns) + columns)
    elif after or before:
        if after:
            insert_idx = list(df.columns).index(after) + 1
        elif before:
            insert_idx = list(df.columns).index(before)
        col_order = list((OrderedSet(df_cols[:insert_idx]) - columns) + columns + OrderedSet(df_cols[insert_idx:]))
    else:
        col_order = list(columns + df_cols)
    return pd.DataFrame(df[col_order])
