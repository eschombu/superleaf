import numpy as np
import pandas as pd

from .column_ops import Col, ColOp


def dfilter(df: pd.DataFrame, *filters, **col_filters) -> pd.DataFrame:
    # import pdb; pdb.set_trace()
    columns = []
    row_bools = np.ones(len(df)).astype(bool)
    for filt in filters:
        if isinstance(filt, str):
            columns.append(filt)
        elif isinstance(filt, ColOp):
            row_bools = row_bools & filt(df)
        elif callable(filt):
            row_bools = row_bools & df.apply(filt, axis=1)
        else:
            raise TypeError(
                "Positional filters must be column names, column operators, or callables to apply to each row"
            )
    for col, filt in col_filters.items():
        if isinstance(filt, ColOp) or not callable(filt):
            col_filt = (Col(col) == filt)
        elif callable(filt):
            col_filt = Col(col).map(filt)
        else:
            raise TypeError(
                "Keyword filters must be values, column operators, or callables to apply to each value in the column"
            )
        row_bools = row_bools & col_filt(df)
    df_filt = df[row_bools]
    if columns:
        df_filt = df_filt[columns]
    return df_filt.copy()
