import pytest

from superleaf.collections.summable_dict import SummableDict


@pytest.fixture
def d1():
    return {
        'a': 1,
        'b': [2, 3],
        'c': SummableDict({'d': (5, 6), 'f': 'abc'}),
    }


@pytest.fixture
def d2():
    return {
        'a': 1,
        'b': [7, 8],
        'c': {'d': (9, 10), 'e': 11, 'f': 'd'},
    }


def test_summable_dict(d1, d2):
    d1 = SummableDict(d1)

    d1p2 = d1 + d2
    assert set(d1p2.keys()) == (set(d1.keys()) | set(d2.keys()))
    for key in d1p2.keys():
        if key in d1.keys() and key in d2.keys():
            assert d1p2[key] == d1[key] + d2[key]
        elif key in d1.keys():
            assert d1p2[key] == d1[key]
        else:
            assert d1p2[key] == d2[key]

    d2p1 = d2 + d1
    assert set(d2p1.keys()) == (set(d1.keys()) | set(d2.keys()))
    for key in d2p1.keys():
        if key in d1.keys() and key in d2.keys():
            assert d2p1[key] == d2[key] + d1[key]
        elif key in d1.keys():
            assert d2p1[key] == d1[key]
        else:
            assert d2p1[key] == d2[key]

    d1c = d1.copy()
    d1c += d2
    assert d1c == d1p2

    d12sum = sum([d1, d2])
    assert d12sum == d1p2
