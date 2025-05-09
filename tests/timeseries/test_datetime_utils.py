from datetime import datetime, time, timedelta

import numpy as np
import pendulum
import pytest

from superleaf.timeseries import datetime_utils as dtutil
from superleaf.timeseries.datetime_utils import nearly_simultaneous


@pytest.fixture(autouse=True)
def mock_pendulum_now(mocker):
    now = pendulum.parse('2022-01-07T12:34:56+00:00')
    patched = mocker.patch.object(pendulum, "now")
    patched._time = now

    def get_now():
        return now

    patched.side_effect = get_now
    return patched


def _check_type_and_isoformat(dt, type_, iso_str: str):
    assert isinstance(dt, type_) and (dt.isoformat() == iso_str)


def test_to_datetime():
    dt_s = '2022-01-01T00:00:00+00:00'
    dt_d = datetime.strptime(dt_s, '%Y-%m-%dT%H:%M:%S%z')
    dt_d_no_tz = dtutil.to_datetime(datetime(2022, 1, 1, 0, 0, 0))
    dt_d_tz = datetime.strptime('2022-01-01T04:00:00+04:00', '%Y-%m-%dT%H:%M:%S%z')
    dt_p = pendulum.parse(dt_s)
    dt_sec = dt_d_tz.timestamp()
    dt_nanosec = round(dt_sec * 1e9)
    _check_type_and_isoformat(dtutil.to_datetime(dt_s), datetime, dt_s)
    _check_type_and_isoformat(dtutil.to_datetime(dt_d), datetime, dt_s)
    _check_type_and_isoformat(dtutil.to_datetime(dt_p), datetime, dt_s)
    _check_type_and_isoformat(dtutil.to_datetime(dt_sec), datetime, dt_s)
    _check_type_and_isoformat(dtutil.to_datetime(dt_nanosec), datetime, dt_s)

    # Check that no timezone info gets converted to UTC
    _check_type_and_isoformat(dtutil.to_datetime(dt_d_no_tz), datetime, dt_s)
    _check_type_and_isoformat(dtutil.to_datetime(dt_d_tz, tz='UTC'), datetime, dt_s)


def test_as_dict():
    dt = datetime(1970, 1, 1, 12, 59, 59, 999999)
    dt_dict = dtutil.as_dict(dt)
    dt2 = datetime(**dt_dict)
    assert dt == dt2


def test_to_date():
    date = '2022-01-01'
    time = '12:34:56+00:00'
    zero = '00:00:00+00:00'
    midnight = '23:59:59.999999+00:00'
    date_time = f"{date}T{time}"
    date_zero = f"{date}T{zero}"
    date_midnight = f"{date}T{midnight}"
    assert dtutil.to_date(date_time) != pendulum.parse(date_time)
    assert dtutil.to_date(date_time) == pendulum.parse(date_zero)
    assert dtutil.to_date(date_time, end_of_day=True) == pendulum.parse(date_midnight)


def test_get_date_range(mock_pendulum_now):
    assert not mock_pendulum_now.called
    now = pendulum.now()
    fmt = 'YYYY-MM-DD'
    assert len(dtutil.get_date_range(now.subtract(days=4))) == 5
    assert len(dtutil.get_date_range(now.subtract(days=4), now.subtract(days=1))) == 4
    assert dtutil.get_date_range(now.subtract(days=4), fmt=fmt)[-1] == now.format(fmt)
    assert mock_pendulum_now.called


def test_get_datetime_range():
    start = '2022-01-01T00:00:00+00:00'
    end = '2022-01-01T01:00:00+00:00'
    delta_minutes = 5
    t_range = dtutil.get_datetime_range(start, end, timedelta(minutes=delta_minutes))
    assert all(isinstance(t, datetime) for t in t_range)
    assert t_range == sorted(t_range)
    assert len(t_range) == (60 / delta_minutes)
    deltas = list(set([dt.as_timedelta() for dt in np.diff(t_range)]))
    assert len(deltas) == 1
    assert deltas[0].total_seconds() == delta_minutes * 60

    assert dtutil.get_datetime_range(start, end, minutes=delta_minutes) == t_range

    t_range_incl = dtutil.get_datetime_range(start, end, timedelta(minutes=delta_minutes), inclusive=True)
    assert len(t_range_incl) == (len(t_range) + 1)


def test_get_hours_minutes_seconds():
    h = 12
    m = 34
    s = 56
    t1 = pendulum.parse('2022-01-01T00:00:00+00:00')
    t2 = pendulum.parse(f'2022-01-01T{h}:{m}:{s}+00:00')
    assert dtutil.get_hours_minutes_seconds(t1, t2) == (h, m, s)
    assert dtutil.get_hours_minutes_seconds(t2 - t1) == (h, m, s)


def test_to_period():
    start = '2022-01-01T00:00:00+00:00'
    hours = 2
    end = f'2022-01-01T{hours:02}:00:00+00:00'
    assert dtutil.to_period(start, end).total_hours() == hours
    assert dtutil.to_period((start, end)).total_hours() == hours
    assert dtutil.to_period(start, hours=hours).total_hours() == hours
    with pytest.raises(ValueError):
        dtutil.to_period(end, start)


def test_nearly_simultaneous():
    seconds = 2
    t1 = '2022-01-01T00:00:00+00:00'
    t2 = f'2022-01-01T00:00:{seconds:02}+00:00'
    assert dtutil.nearly_simultaneous(t1, t2, epsilon_seconds=(seconds - 0.1)) is False
    assert dtutil.nearly_simultaneous(t1, t2, epsilon_seconds=(seconds + 0.1)) is True


def test_midpoint():
    hours = 6
    t1 = '2022-01-01T00:00:00+00:00'
    t2 = f'2022-01-01T{hours:02}:00:00+00:00'
    assert dtutil.midpoint(t1, t2) == pendulum.parse(t1).add(hours=(hours / 2))
    assert dtutil.midpoint(t2, t1) == pendulum.parse(t1).add(hours=(hours / 2))


def test_mean_time():
    ts_tz_naive = ['2022-08-09 12:45:12',
                   '2022-06-09 11:45:12',
                   '2020-08-09 13:45:12',
                   '2022-08-23 14:45:12']
    assert nearly_simultaneous(dtutil.mean_time(ts_tz_naive), time(13, 15, 12), 1e-6)

    ts_tz = ['2022-08-09 12:45:12-0400',
             '2022-06-09 11:45:12-0400',
             '2020-08-09 13:45:12+0000',  # == '2020-08-09 09:45:12-0400'
             '2022-08-23 14:45:12-0400']
    expected_utc = time(16, 15, 12)
    expected_local = time(13, 15, 12)
    assert nearly_simultaneous(dtutil.mean_time(ts_tz), expected_utc, 1e-6)
    assert nearly_simultaneous(dtutil.mean_time(ts_tz, local=True), expected_local, 1e-6)

    around_midnight = ['2022-08-09 23:00:00',
                       '2022-08-09 01:00:00']
    assert nearly_simultaneous(dtutil.mean_time(around_midnight), time(0), 1e-6)


def test_get_first_time_in_interval():
    t = '09:30:00'
    d = '2022-01-01'
    dt = pendulum.parse(f'{d} {t}')
    start_before = pendulum.parse(f'{d} 08:00:00')
    start_after = start_before.add(hours=2)
    end_before = start_before.add(days=1, hours=1)
    end_after = end_before.add(days=2)

    assert dtutil.get_first_time_in_interval(t, start_before) == dt
    assert dtutil.get_first_time_in_interval(t, start_after) == dt.add(days=1)
    assert dtutil.get_first_time_in_interval(t, start_after, end_after) == dt.add(days=1)
    assert dtutil.get_first_time_in_interval(t, start_after, end_before) is None
    assert dtutil.get_first_time_in_interval(t, (start_after, end_before)) is None
    assert dtutil.get_first_time_in_interval(t, (start_after.isoformat(), end_after.isoformat())) == dt.add(days=1)
