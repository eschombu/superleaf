import pandas as pd
from IPython.display import display


def set_max_columns(max_columns=None):
    pd.set_option('display.max_columns', max_columns)


def set_max_rows(max_rows=None):
    pd.set_option('display.max_rows', max_rows)


class _PandasDisplayCM:
    def __init__(self, all_columns=False, all_rows=False):
        self._value_store = {}
        self.all_columns = all_columns
        self.all_rows = all_rows

    def __enter__(self):
        if self.all_columns:
            self._value_store['max_columns'] = pd.get_option('display.max_columns')
            set_max_columns()
        if self.all_rows:
            self._value_store['max_rows'] = pd.get_option('display.max_rows')
            set_max_rows()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.all_columns:
            pd.set_option('display.max_columns', self._value_store['max_columns'])
        if self.all_rows:
            pd.set_option('display.max_rows', self._value_store['max_rows'])
        return False


def show_all(df, mode=None, columns=True, rows=True):
    if mode == 'columns':
        cm = _PandasDisplayCM(all_columns=True)
    elif mode == 'rows':
        cm = _PandasDisplayCM(all_rows=True)
    else:
        cm = _PandasDisplayCM(all_columns=columns, all_rows=rows)
    with cm:
        display(df)
