from typing import Iterable

import pandas as pd

def _is_iter(obj):
    try:
        iter(obj)
        return True
    except TypeError:
        return False


def standardize_columns(df: pd.DataFrame, to_datetime: bool | str | Iterable[str] | None = False) -> pd.DataFrame:
    df = df.copy()
    df.columns = [col.strip().lower().replace(' ', '_') for col in df.columns]

    if to_datetime is None:
        dt_cols = []
    elif isinstance(to_datetime, bool):
        dt_cols = [col for col in df.columns if col in {'date', 'datetime', 'time', 'timestamp'}]
    elif isinstance(to_datetime, str) or not _is_iter(to_datetime):
        dt_cols = [to_datetime]
    else:
        dt_cols = to_datetime

    for col in dt_cols:
        if col in df.columns:
            df[col] = pd.to_datetime(df[col])

    return df
