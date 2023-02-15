import numpy as np
import pandas as pd

from superleaf.plotting.intervals import (
    _get_intervals_from_df,
    _get_intervals_from_series,
    _get_intervals_from_tuples,
    _get_intervals_from_vars,
    StateInterval,
)


def test_get_intervals():
    state_intervals = [
        StateInterval(start=10, end=20, state="one"),
        StateInterval(start=25, end=30, state="two"),
        StateInterval(start=40, end=41, state="one"),
    ]

    starts, ends, states = zip(*map(tuple, state_intervals))
    df = pd.DataFrame({"state": states, "start": starts, "end": ends})
    
    def assert_eq(result, expected) -> bool:
        assert all([r == e for r, e in zip(result, expected)])

    assert_eq(_get_intervals_from_df(df), state_intervals)
    assert_eq(_get_intervals_from_vars(starts=starts, ends=ends, states=states), state_intervals)
    assert_eq(_get_intervals_from_tuples(list(zip(starts, ends, states))), state_intervals)

    series = pd.Series(np.nan * np.ones(50))
    for i in state_intervals:
        if i.state == "one":
            series[i.start:i.end] = 1
        elif i.state == "two":
            series[i.start:i.end] = 2
    series_intervals = list(filter(lambda s: pd.notna(s.state), _get_intervals_from_series(series)))
    assert len(series_intervals) == len(state_intervals)
    for r, e in zip(series_intervals, state_intervals):
        assert (r.end - r.start) == (e.end - e.start)
