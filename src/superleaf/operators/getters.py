from superleaf.operators.base import Operator, operator


def attr_getter(name: str) -> Operator:
    def get(obj):
        return getattr(obj, name)
    return operator(get)


def index_getter(index) -> Operator:
    def get(obj):
        return obj[index]
    return operator(get)
