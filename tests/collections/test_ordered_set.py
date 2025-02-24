import pytest

from superleaf.collections.ordered_set import OrderedSet


@pytest.fixture
def acb0():
    return ["a", "c", "b", 0]


@pytest.fixture
def c1xe():
    return ["c", 1, "x", "e"]


def test_addition(acb0, c1xe):
    # Adding sets is equivalent to set.union, and will work when non-OrderedSet iterables are used
    assert (OrderedSet(acb0) + c1xe) == set(acb0).union(c1xe)

    result = OrderedSet(acb0)
    result += c1xe
    assert result == set(acb0).union(c1xe)

    # Has .add() method like sets
    result = OrderedSet(acb0)
    for value in c1xe:
        result.add(value)  # in-place
    assert result == set(acb0).union(c1xe)

    # Can also use sum
    assert sum([OrderedSet(acb0), c1xe]) == set(acb0).union(c1xe)


def test_subtraction(acb0, c1xe):
    assert (OrderedSet(acb0) - c1xe) == (set(acb0) - set(c1xe))

    result = OrderedSet(acb0)
    result -= c1xe
    assert result == (set(acb0) - set(c1xe))


def test_intersection(acb0, c1xe):
    assert OrderedSet(acb0).intersection(c1xe) == set(acb0).intersection(c1xe)


def test_order(acb0):
    assert list(OrderedSet(acb0)) == acb0
    assert list(OrderedSet(acb0[::-1])) == acb0[::-1]
    assert list(OrderedSet(acb0)) != list(OrderedSet(acb0[::-1]))
    assert list(OrderedSet(acb0) + ["c"]) == acb0


def test_indexing(acb0):
    assert OrderedSet(acb0)[0] == acb0[0]
    assert list(OrderedSet(acb0)[:2]) == acb0[:2]


def test_contains(acb0):
    result = OrderedSet(acb0)
    assert all([value in result for value in acb0])


def test_eq(acb0):
    # Equality test does NOT care about ordering; if you want to check order, use `list(oset1) == list(oset2)`
    assert OrderedSet(acb0) == OrderedSet(acb0[::-1])
    assert (OrderedSet(acb0) + ["new_value"]) != OrderedSet(acb0)


def test_len(acb0):
    assert len(OrderedSet(acb0)) == len(acb0)
