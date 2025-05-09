import numpy as np
import pytest

from superleaf.stats.circular import circmean


def _approx_eq(a, b, tol=1e-12):
    return abs(a - b) <= tol


def test_circmean():
    degrees = np.array([-150, 180, 150, 270, 90, 0, 540, -540])
    expected_deg = 180
    radians = np.deg2rad(degrees)
    expected_rad = np.deg2rad(expected_deg)
    assert circmean(radians) == expected_rad
    assert circmean(degrees, high=360) == expected_deg

    degrees, weights = np.array([
        [-90, 1],
        [0, 1],
        [90, 2],
    ]).T
    expected = 45
    assert _approx_eq(circmean(degrees, weights=weights, high=360), expected)

    degrees, weights = np.array([
        [-40, 1],
        [0, 1],
        [40, 3],
    ]).T
    equiv_unweighted = []
    for d, w in zip(degrees, weights):
        equiv_unweighted.extend([d] * w)
    assert _approx_eq(circmean(degrees, weights=weights, high=360), circmean(equiv_unweighted, high=360))

    hours = [12, 14, 23, 15]
    weights = [1, 1, 2, 1]
    expected = 13
    expected_weighted = 12.5
    assert _approx_eq(circmean(hours, low=12, high=24), expected)
    assert _approx_eq(circmean(hours, weights=weights, low=12, high=24), expected_weighted)

    radians, weights = np.array([
        [3 * np.pi / 2, 1],
        [0, 1],
        [np.pi / 2, 2],
        [np.nan, 1],
    ]).T
    expected = np.pi / 4
    assert np.isnan(circmean(radians))
    assert np.isnan(circmean(radians, weights=weights))
    assert _approx_eq(circmean(radians, weights=weights, nan_policy='ignore'), expected)
    with pytest.raises(ValueError):
        circmean(radians, weights=weights, nan_policy='raise')

    radians = np.vstack([radians, radians]).T
    weights = np.vstack([weights, weights]).T
    assert _approx_eq(circmean(radians, weights=weights, nan_policy='ignore'), expected)

    expected_array = np.array([expected, expected])
    result_array = circmean(radians, weights=weights, axis=0, nan_policy='ignore')
    assert all([_approx_eq(result, exp) for result, exp in zip(result_array, expected_array)])
