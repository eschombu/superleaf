import json
from typing import Optional

import numpy as np
import pandas as pd

from superleaf.collections import OrderedSet


def notna(x) -> bool:
    try:
        # Faster operation is to use x==x, but list of null would return True
        return x is not None and not np.isnan(x)
    except ValueError:
        return x is not None and not np.any(np.isnan(x))
    except TypeError:
        return x is not None


def expand_dict_to_cols(
        df: pd.DataFrame,
        cols,
        fields=None,
        with_col_prefix=True,
        prefix='',
        prefix_fun=None,
        sep='_',
        drop=True,
        dropna=False,
        col_renamer=None,
        recursive=False,
        uniform_keys=False,
        default=np.nan,
) -> pd.DataFrame:

    def has_vals(meta: Optional[dict]) -> bool:
        try:
            return notna(meta) and len(meta) > 0
        except TypeError:
            raise TypeError(f"Expected a dictionary-like object, got {type(meta)} instead.")

    def get_field(key):
        def getter(meta: Optional[dict]):
            if notna(meta):
                return meta.get(key, default)
        return getter

    def get_new_col_name(col):
        if callable(col_renamer):
            return col_renamer(col)
        elif isinstance(col_renamer, dict) and col in col_renamer:
            if callable(col_renamer[col]):
                return col_renamer[col](col)
            else:
                return col_renamer[col]
        else:
            return col

    def add_new_cols(df: pd.DataFrame, values: pd.Series, field: str, prefix: str) -> pd.DataFrame:
        new_df = df.copy()
        if values.isna().all():
            if not dropna:
                new_col = get_new_col_name(prefix + str(field))
                new_df[new_col] = default
        elif isinstance(values[values.notna()].iloc[0], dict) and recursive:
            new_cols = expand_dict_to_cols(
                pd.DataFrame(index=values.index, data={field: values}),
                field,
                fields=None,
                prefix=f"{prefix}{field}{sep}",
                with_col_prefix=with_col_prefix,
                col_renamer=col_renamer,
                recursive=True,
                uniform_keys=uniform_keys,
                default=default,
            )
            for col in new_cols:
                vals = new_cols[col]
                _notna = vals.notna()
                new_df.loc[_notna, col] = vals[_notna]
        else:
            new_col = get_new_col_name(prefix + str(field))
            _notna = values.notna()
            new_df.loc[_notna, new_col] = values[_notna]
        return new_df

    if isinstance(fields, str):
        fields = [fields]
    if isinstance(cols, str):
        cols = [cols]

    if drop:
        new_df = df.drop(columns=[col for col in cols if col in df])
    else:
        new_df = df.copy()

    new_col_map = {}
    for col in cols:
        current_cols = list(new_df.columns)
        if col in df and df[col].map(has_vals).any():
            if isinstance(df.iloc[0][col], str):
                df[col] = df[col].map(lambda v: json.loads(v) if v else None)
            if fields is None:  # expand all
                if uniform_keys:  # same keys in each item, allowing for increased efficiency
                    first_one = df[df[col].map(has_vals)][col].iloc[0]
                    current_fields = list(first_one.keys())
                else:
                    current_fields = OrderedSet(df[col].dropna().map(lambda d: OrderedSet(d.keys())).sum())
            else:
                current_fields = fields
            for field in current_fields:
                values = df[col].map(get_field(field))
                #if len(values.dropna()) > 0:
                if with_col_prefix:
                    col_prefix = prefix or f"{col}{sep}"
                else:
                    col_prefix = prefix
                if prefix_fun:
                    col_prefix = prefix_fun(col, col_prefix)
                new_df = add_new_cols(new_df, values, field, col_prefix)
        new_col_map[col] = [col for col in new_df.columns if col not in current_cols]

    # Reorder columns so that expansions in place of original column
    col_order = []
    for col in df.columns:
        if col in cols:
            if not drop:
                col_order.append(col)
            col_order.extend(new_col_map[col])
        else:
            col_order.append(col)

    return pd.DataFrame(new_df[col_order])
