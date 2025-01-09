from dataclasses import dataclass
from typing import Dict, Tuple

import pytest

from superleaf.operators.base import operator
from superleaf.operators.getters import attr_getter, index_getter


@pytest.fixture
def obj():
    @dataclass
    class Data:
        a: Dict[str, Tuple[int, int]]
    
    return Data({"key": (10, 11)})


def test_nested_getter(obj):
    getter = attr_getter("a") >> index_getter("key") >> index_getter(1)
    assert getter(obj) == 11

    getter_plus_one = getter >> operator(lambda x: x + 1)
    assert getter_plus_one(obj) == 12
