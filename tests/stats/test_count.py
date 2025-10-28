import numpy as np
import pytest

from superleaf.stats.count import CountStat


def test_with_total():
    n = 10
    N = 100
    count_stat = CountStat(n, N)
    assert count_stat.percent == n
    assert count_stat.fraction == (n / N)
    assert count_stat.chi2_pval is None
    with pytest.raises(ValueError):
        CountStat(4.5, 10)
    with pytest.raises(ValueError):
        CountStat(4, 10.5)


def test_with_array():
    n = 3
    N = 5
    bools = np.array([1] * n + [0] * (N - n), dtype=int)
    count_stat = CountStat(bools)
    assert count_stat.count == 3
    assert count_stat.total == 5
    with pytest.raises(ValueError):
        CountStat(np.array([0, 1, 2]))


def test_nan_total():
    n = 10
    count_stat = CountStat(n, None)
    assert np.isnan(count_stat.fraction)
    assert np.isnan(count_stat.percent)


def test_chi2():
    n = 10
    N = 100
    e = 15
    assert 0 <= CountStat(n, N, expected=e).chi2_pval <= 1
    assert 0 <= CountStat(n, N, expected=(e / N)).chi2_pval <= 1


def test_addition():
    n1 = 10
    n2 = 15
    N = 100
    cs1 = CountStat(n1, N)
    cs2 = CountStat(n2, N)
    cs12 = cs1 + cs2
    assert cs12.percent == (n1 + n2) / 2
    assert cs12.fraction == (n1 + n2) / (2 * N)
