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


@pytest.fixture
def d3():
    return {
        'a': 1,
        'b': 2,
    }


@pytest.fixture
def d4():
    return {
        'b': 3,
        'c': 4,
    }


def test_summable_dict(d1, d2, d3, d4):
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

    # Negation and subtraction
    with pytest.raises(TypeError):
        d1n = -d1
    with pytest.raises(TypeError):
        d1m2 = d1 - d2

    d3 = SummableDict(d3)
    d3n = -d3
    assert d3n == SummableDict({k: -v for k, v in d3.items()})

    d3m4 = d3 - d4
    assert set(d3m4.keys()) == (set(d3.keys()) | set(d4.keys()))
    for key in d3m4.keys():
        if key in d3.keys() and key in d4.keys():
            assert d3m4[key] == d3[key] - d4[key]
        elif key in d3.keys():
            assert d3m4[key] == d3[key]
        else:
            assert d3m4[key] == -d4[key]

    d3m5 = d3 - 5
    assert set(d3m5.keys()) == set(d3.keys())
    for key in d3m5.keys():
        assert d3m5[key] == d3[key] - 5

    d4m3 = d4 - d3
    assert set(d4m3.keys()) == (set(d3.keys()) | set(d4.keys()))
    for key in d4m3.keys():
        if key in d3.keys() and key in d4.keys():
            assert d4m3[key] == d4[key] - d3[key]
        elif key in d3.keys():
            assert d4m3[key] == -d3[key]
        else:
            assert d4m3[key] == d4[key]
