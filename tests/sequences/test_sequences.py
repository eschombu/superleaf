from functools import partial

import numpy as np
import pytest

from superleaf.operators import operator
from superleaf.operators.comparison import ComparisonFunctions as F
from superleaf.sequences import filtered, flatten, flat_map, groupby, mapped


def _try_len(x) -> int:
    try:
        return len(x)
    except TypeError:
        return 0


@pytest.fixture
def seq():
    return [
        None,
        1,
        [1],
        "a",
        ["b", "c"],
        np.arange(3),
        [[10, 11], ["x", ["y", "zed"]]]
    ]


def test_mapped(seq):
    lens = mapped(_try_len, seq)
    assert len(lens) == len(seq)
    assert sum(lens) == 9


def test_filtered(seq):
    lens = filtered(operator(_try_len) >> F.gt(0), seq)
    assert len(lens) == 5


def test_groupby(seq):
    grouped = groupby(_try_len, seq)
    assert set(grouped.keys()) == {0, 1, 2, 3}
    assert mapped(len, [grouped[k] for k in [0, 1, 2, 3]]) == [2, 2, 2, 1]


def _equal(xy):
    x, y = xy
    try:
        return bool(x == y)
    except ValueError:
        return all(x == y)    


def test_flatten(seq):
    shallow = [None, 1, 1, "a", "b", "c", 0, 1, 2, [10, 11], ["x", ["y", "zed"]]]
    deep = [None, 1, 1, "a", "b", "c", 0, 1, 2, 10, 11, "x", "y", "zed"]
    assert all(map(_equal, zip(flatten(seq, depth=0), seq)))
    assert all(map(_equal, zip(flatten(seq, depth=1), shallow)))
    assert all(map(_equal, zip(flatten(seq, depth=None), deep)))


def _get_str(x):
    if isinstance(x, str):
        return x
    try:
        return [_get_str(y) for y in x]
    except TypeError:
        return None


def test_flat_map(seq):
    joined_strs = "abcxyzed"
    assert "".join(flat_map(_get_str, seq, depth=None, drop_null=True)) == joined_strs
