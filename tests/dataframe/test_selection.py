import numpy as np
import pandas as pd
import pytest

from superleaf.dataframe.selection import Col, dfilter
from superleaf.operators.comparison import ComparisonFunctions as F


@pytest.fixture
def df():
    data = {
        "col1": [0, 1, 2, 3, 4],
        "col2": [-2, -1, 0, 1, 2],
        "col3": [1, 1, 1, np.nan, 1],
        "col4": ["zero", "one", "two", "three", "four"],
        "col5": [1, 0, 0, 0, 1]
    }
    return pd.DataFrame(data)


def _iseq(df1: pd.DataFrame, df2: pd.DataFrame) -> bool:
    return ((df1 == df2) | (df1.isna() & df2.isna())).all().all()


def test_dfilter(df):
    # Progressively increase complexity
    assert _iseq(dfilter(df, col1=0), df.iloc[[0]])
    assert _iseq(dfilter(df, Col("col3").isna()), df[df["col3"].isna()])
    assert _iseq(dfilter(df, col3=pd.notna), dfilter(df, Col("col3").notna()))
    assert _iseq(dfilter(df, col3=pd.notna), dfilter(df, col3=F.notna))

    assert _iseq(dfilter(df, col3=1), df[df["col3"] == 1])
    assert _iseq(dfilter(df, col3=1, col4="four"), df.iloc[[4]])

    assert _iseq(dfilter(df, Col("col2") >= 0), df[df["col2"] >= 0])
    assert _iseq(dfilter(df, col2=F.ge(0)), dfilter(df, Col("col2") >= 0))
    assert _iseq(dfilter(df, (Col("col2") < 0) | ~(Col("col4").contains("o"))), df.iloc[[0, 1, 3]])
    assert _iseq(dfilter(df, (Col("col2") > 0) & ~(Col("col4").contains("o"))), dfilter(df, Col("col3").isna()))
    assert _iseq(dfilter(df, col2=(F.lt(0) | ~(F.le(0)))), df[df["col2"] != 0])

    assert _iseq(dfilter(df, col2=F.ge(0), col3=1), df[(df["col2"] >= 0) & (df["col3"] == 1)])
    assert _iseq(dfilter(df, col2=F.ge(0), col3=1, col5=1), df.iloc[[4]])
    assert _iseq(dfilter(df, col2=F.ge(0), col3=Col("col5")), df.iloc[[4]])
