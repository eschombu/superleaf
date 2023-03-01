from datetime import timedelta
import pendulum
import pytest

from superleaf.timeseries.time_intervals import TimeInterval, TimeIntervals


@pytest.fixture
def t0():
    return pendulum.parse('2022-10-18 12:00:00')


def test_creation(t0):
    s = 30
    i1 = TimeInterval(t0, t0.add(seconds=s))
    assert i1 == i1.copy()
    with pytest.raises(ValueError):
        TimeInterval(t0, t0.subtract(seconds=s))

    # Create TimeIntervals instance with two separated intervals
    i2 = TimeInterval(t0.add(seconds=(2 * s)), t0.add(seconds=(3 * s)))
    assert len(list(TimeIntervals([i1, i2]))) == 2

    # If intervals overlapping, they should be consolidated to a single one
    i3 = TimeInterval(i1.end, t0.add(seconds=(3 * s)))
    i4 = TimeInterval(i1.start, i3.end)
    ii = TimeIntervals([i3, i1])
    assert len(list(ii)) == 1
    assert ii.intervals[0] == i4
    assert ii.start == min(i1.start, i3.start)
    assert ii.end == max(i1.end, i3.end)
    assert ii.total_seconds() == i4.total_seconds() == 3 * s


def test_comparisons(t0):
    s = 30
    t1 = t0.add(seconds=s)
    t2 = t1.add(seconds=s)
    t3 = t2.add(seconds=s)
    i1 = TimeInterval(t0, t2)
    i2 = TimeInterval(t1, t2)
    i3 = TimeInterval(t2, t3)
    ii12 = TimeIntervals([i1, i2])
    ii23 = TimeIntervals([i2, i3])
    assert t1 in i1
    assert t1 not in i3
    assert i1.overlap_seconds(i2) == s
    assert i1.overlap_seconds(ii23) == s
    assert ii12.overlap_seconds(ii23) == s
    assert t2 in ii23
    assert t0 not in ii23


def test_interval_arithmetic(t0):
    s = 30
    delta = timedelta(seconds=s)
    i1 = TimeInterval(t0, t0 + delta)
    i2 = i1.add_offset(2 * delta)
    i3 = TimeInterval(i2.start, i2.start + 0.5 * delta)
    assert ((i2.start - i1.start).total_seconds() == (2 * s)) and ((i2.end - i1.end).total_seconds() == (2 * s))
    assert i1.total_seconds() == i2.total_seconds()
    assert isinstance(i1.union(i2), TimeIntervals)
    assert isinstance(i2.union(i3), TimeInterval)

    i4 = i2 - i3
    assert i3.total_seconds() == i4.total_seconds()
    assert i1.intersection(i2) is None
    assert i2.intersection(i4) == i4
    assert i1.union(i3).union(i4) == i1.union(i2)

    ii = TimeIntervals([i1, i2])
    ii_offset = ii.add_offset(delta)
    assert ((ii_offset.start - ii.start).total_seconds() == s) and ((ii_offset.end - ii.end).total_seconds() == s)
    assert ii_offset.total_seconds() == ii_offset.total_seconds()


def test_squeeze(t0):
    s = 30
    t1 = t0.add(seconds=s)
    t2 = t1.add(seconds=s)
    t3 = t2.add(seconds=s)
    i1 = TimeInterval(t0, t2)
    i2 = TimeInterval(t1, t2)
    ii12 = TimeIntervals([i1, i2])
    assert isinstance(ii12, TimeIntervals)
    assert len(ii12.intervals) == 1
    assert isinstance(ii12.squeeze(), TimeInterval)

    t4 = t3.add(seconds=s)
    i3 = TimeInterval(t3, t4)
    ii13 = TimeIntervals([i1, i3])
    assert isinstance(ii13.squeeze(), TimeIntervals)

    assert isinstance(i1.union(i2), TimeInterval)
    assert isinstance(i1.intersection(i2), TimeInterval)
    assert isinstance(i1.subtract(i2), TimeInterval)


def test_split(t0):
    s = 30
    n = 5
    ds = s / n
    dt = timedelta(seconds=ds)
    i = TimeInterval(t0, t0.add(seconds=s))
    i_split = i.split(n)
    assert len(i_split) == n
    assert i_split[1].start == i_split[0].end
    assert all(s.total_seconds() == ds for s in i_split)
    assert i.split(dt) == i_split
    assert i.split(seconds=ds) == i_split
