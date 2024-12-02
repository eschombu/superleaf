from datetime import datetime, timedelta

import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.standardize import standardize_columns


@pytest.fixture
def df():
    start_time = datetime(2024, 1, 1, 8, 0, 0)
    one_hour = timedelta(hours=1)
    datetimes = [start_time + i * one_hour for i in range(5)]
    datetime_strs = [dt.strftime('%Y-%m-%d %H:%M:%S') for dt in datetimes]
    time_strs = [dt.strftime('%H:%M:%S') for dt in datetimes]
    data = {
        '  DateTime': datetime_strs,
        'Time': time_strs,
        'Column Two  ': datetime_strs,
        'col_three': list(range(len(datetimes))),
    }
    return pd.DataFrame(data)


def test_standardize_columns(df: pd.DataFrame):
    new_df = standardize_columns(df)
    assert list(new_df.columns) == ['datetime', 'time', 'column_two', 'col_three']
    assert new_df['datetime'].equals(pd.to_datetime(df['  DateTime']))
    assert new_df['time'].equals(df['Time'])
    assert new_df['column_two'].equals(df['Column Two  '])
    assert new_df['col_three'].equals(df['col_three'])
    assert standardize_columns(df, to_datetime='column_two')['column_two'].equals(pd.to_datetime(df['Column Two  ']))
