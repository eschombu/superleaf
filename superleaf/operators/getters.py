from . import operator
from .base import Operator


def attr_getter(name: str) -> Operator:
    def get(obj):
        return getattr(obj, name)
    return operator(get)


def index_getter(index) -> Operator:
    def get(obj):
        return obj[index]
    return operator(get)
