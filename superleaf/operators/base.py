from abc import ABCMeta, abstractmethod
from typing import Any, Callable


class Operator(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, arg: Any) -> Any:
        pass

    def __or__(self, right: "Operator") -> "PipedOperator":
        return PipedOperator(self, right)


class PipedOperator(Operator):
    def __init__(self, left: Operator, right: Operator):
        self._left = left
        self._right = right

    def __call__(self, arg: Any) -> Any:
        return self._right(self._left(arg))


class FunctionOperator(Operator):
    def __init__(self, f: Callable):
        self._fun = f

    def __call__(self, arg: Any) -> Any:
        return self._fun(arg)
