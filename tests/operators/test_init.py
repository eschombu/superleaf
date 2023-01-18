from superleaf.operators import operator


def _plus_one(x):
    return x + 1


def _times_two(x):
    return x * 2


def test_basic():
    op = operator(_plus_one)
    assert op(1) == 2


def test_piped():
    a = operator(_plus_one)
    b = operator(_times_two)
    c = a | b
    assert c(1) == 4
    d = b | a
    assert d(1) == 3
    e = c | d  # (((( + 1) * 2) + 1) * 2)
    assert e(1) == 9
