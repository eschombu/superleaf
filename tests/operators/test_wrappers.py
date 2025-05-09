import pytest

from superleaf.operators.wrappers import with_fallback


def test_with_fallback():
    def add(a, b, c=None):
        s = a + b
        if c is not None:
            s += c
        return s

    fail_add = (1, "2")
    succeed_add = (1, 2)
    succeed_result = 3

    with pytest.raises(TypeError):
        add(*fail_add)

    wrapped_add_default = with_fallback(add)
    assert wrapped_add_default(*succeed_add) == succeed_result
    assert wrapped_add_default(*succeed_add, c=0) == succeed_result
    assert wrapped_add_default(*fail_add) is None

    wrapped_add_null = with_fallback(add, "null")
    assert wrapped_add_null(*fail_add) == "null"

    wrapped_add_specific = with_fallback(add, exceptions=TypeError)
    assert wrapped_add_specific(*fail_add) is None

    wrapped_add_wrong = with_fallback(add, exceptions=AttributeError)
    with pytest.raises(TypeError):
        wrapped_add_wrong(*fail_add)

    @with_fallback
    def mult(a, b):
        return a * b

    assert mult(1, None) is None

    @with_fallback(exceptions=TypeError, fallback="null")
    def sub(a, b):
        return a - b

    assert sub(*fail_add) == "null"
