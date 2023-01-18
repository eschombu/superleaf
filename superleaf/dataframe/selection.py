import numpy as np
import pandas as pd

from .column_ops import Col, ColOp


def dfilter(df: pd.DataFrame, *filters, **col_filters) -> pd.DataFrame:
    row_bools = np.ones(len(df)).astype(bool)
    for filt in filters:
        if isinstance(filt, ColOp):
            row_bools = row_bools & filt(df)
        elif callable(filt):
            row_bools = row_bools & df.apply(filt, axis=1)
        else:
            raise TypeError(
                "Positional filters must be column operators or callables to apply to each row"
            )
    for col, filt in col_filters.items():
        if isinstance(filt, ColOp) or not callable(filt):
            row_bools = row_bools & (Col(col) == filt)(df)
        elif callable(filt):
            try:
                row_bools = row_bools & Col(col).apply(filt)(df)
            except Exception:
                row_bools = row_bools & Col(col).map(filt)(df)
        else:
            raise TypeError(
                "Keyword filters must be values, column operators, or callables to apply to each value in the column"
            )
    return df[row_bools].copy()
