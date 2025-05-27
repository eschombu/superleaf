import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.transform import expand_dict_to_cols


@pytest.fixture
def df():
    data = {
        "col1": [0, 1, 2, 3, 4],
        "col2": [{'field1': 1, 'field2': 2},
                 {'field1': 3, 'field2': 4},
                 {},
                 {'field1': 5, 'field2': 6},
                 {'field3': 7}],
        "col3": [{'field1': 1, 'field3': 2},
                 {'field1': 3, 'field4': 4},
                 {},
                 {'field3': 5, 'field4': 6},
                 {'field4': 7}],
    }
    return pd.DataFrame(data)


def _df_eq(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return ((df1 == df2) | (df1.isna() & df2.isna())).all().all()


def _s_eq(s1: pd.Series, s2: pd.Series) -> bool:
    return ((s1 == s2) | (s1.isna() & s2.isna())).all().all()


def test_expand_dict_to_cols(df):
    with pytest.raises(TypeError):
        expand_dict_to_cols(df, 'col1')

    expand_col2 = expand_dict_to_cols(df, 'col2')
    assert list(expand_col2.columns) == ['col1', 'col2_field1', 'col2_field2', 'col2_field3', 'col3']
    assert _s_eq(expand_col2['col2_field1'],
                 df['col2'].apply(lambda d: d.get('field1', np.nan) if isinstance(d, dict) else np.nan))

    expand_col2 = expand_dict_to_cols(df, 'col2', with_col_prefix=False)
    assert list(expand_col2.columns) == ['col1', 'field1', 'field2', 'field3', 'col3']

    expand_col2 = expand_dict_to_cols(df, 'col2', fields=['field1', 'field2'], with_col_prefix=False)
    assert list(expand_col2.columns) == ['col1', 'field1', 'field2', 'col3']

    expand_col2 = expand_dict_to_cols(df, 'col2', fields=['field1', 'field2'], with_col_prefix=False, drop=False)
    assert list(expand_col2.columns) == ['col1', 'col2', 'field1', 'field2', 'col3']
    assert _s_eq(expand_col2['field1'],
                 expand_col2['col2'].apply(lambda d: d.get('field1', np.nan) if isinstance(d, dict) else np.nan))

    # TODO: Add tests for dropna, recursive, prefix, col_renamer, uniform_keys, default, etc.
