from superleaf.operators.base import operator


def _plus_one(x):
    return x + 1


def _times_two(x):
    return x * 2


def test_basic_operator():
    op = operator(_plus_one)
    assert op(1) == 2


def test_piped():
    a = operator(_plus_one)
    b = operator(_times_two)
    c = a >> b
    assert c(1) == 4
    d = b >> a
    assert d(1) == 3
    e = c >> d  # (((( + 1) * 2) + 1) * 2)
    assert e(1) == 9


def test_fallbacks():
    try_a_1 = operator(_plus_one, exceptions=Exception, fallback=-1)
    try_a_2 = operator(_plus_one, exceptions=Exception)
    try_b = operator(_times_two, exceptions=Exception)
    c_1 = try_a_1 >> try_b
    c_2 = try_a_2 >> try_b
    assert c_1(1) == 4
    assert c_1("1") == -2
    assert c_2("1") is None
