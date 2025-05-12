from abc import ABCMeta, abstractmethod
from typing import Any, Callable, Iterable, Optional, Union

import pandas as pd


class ColOp(metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, df: pd.DataFrame) -> Union[pd.Series, Any]:
        pass

    def __or__(self, right: "ColOp") -> "ColOp":
        return _OrOp(self, right)

    def __and__(self, right: "ColOp") -> "ColOp":
        return _AndOp(self, right)

    def __invert__(self) -> "ColOp":
        return _NotOp(self)

    def __eq__(self, value: Any) -> "ColOp":
        return _EqOp(self, value)

    def __ne__(self, value: Any) -> "ColOp":
        return _NotOp(self == value)

    def __lt__(self, value: Any) -> "ColOp":
        return _LtOp(self, value)

    def __le__(self, value: Any) -> "ColOp":
        return _LeOp(self, value)

    def __gt__(self, value: Any) -> "ColOp":
        return _GtOp(self, value)

    def __ge__(self, value: Any) -> "ColOp":
        return _GeOp(self, value)

    def __add__(self, right: "ColOp") -> "ColOp":
        return _AddOp(self, right)

    def __sub__(self, right: "ColOp") -> "ColOp":
        return _SubtractOp(self, right)

    def __mul__(self, right: "ColOp") -> "ColOp":
        return _MultiplyOp(self, right)

    def __truediv__(self, right: "ColOp") -> "ColOp":
        return _DivideOp(self, right)

    def __pow__(self, right: "ColOp") -> "ColOp":
        return _PowOp(self, right)

    def apply(self, f: Callable[[pd.Series], pd.Series]) -> "ColOp":
        return _ColApplyOp(self, f)

    def map(self, f: Callable[[Any], Any]) -> "ColOp":
        return _ColMapOp(self, f)

    def isin(self, values: Iterable[Any]) -> "ColOp":
        if isinstance(values, ColOp):
            combined_vals = self.to_list() + values.to_list()
            return combined_vals.map(lambda x: x[0] in x[1])
        else:
            return self.apply(lambda s: s.isin(values))

    def contains(self, value: Any) -> "ColOp":
        return self.map(lambda x: value in x)

    def notna(self) -> "ColOp":
        return self.apply(lambda s: s.notna())

    def isna(self) -> "ColOp":
        return self.apply(lambda s: s.isna())

    def astype(self, type_) -> "ColOp":
        return self.apply(lambda s: s.astype(type_))

    def to_list(self):
        return self.map(lambda x: [x])


class Index(ColOp):
    def __call__(self, df: pd.DataFrame) -> pd.Index:
        return df.index


class Col(ColOp):
    def __init__(self, name: Optional[str]):
        self._name = name

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        if self._name is None:
            return df.iloc[:]
        else:
            return df[self._name]


class Values(Col):
    def __init__(self):
        super().__init__(None)

    def __call__(self, s: pd.Series) -> pd.Series:
        if isinstance(s, pd.DataFrame):
            raise TypeError("Values can only be called on a Series")
        return s.iloc[:]


class _LiteralOp(ColOp):
    def __init__(self, value: Any) -> None:
        self._value = value

    def __call__(self, df: pd.DataFrame) -> Any:
        return self._value


class _ComparisonOp(ColOp):
    def __init__(self, col: ColOp, value: Union[ColOp, Any]) -> None:
        self._col = col
        if isinstance(value, ColOp):
            self._value = value
        else:
            self._value = _LiteralOp(value)


class _EqOp(_ComparisonOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df) == self._value(df)


class _LtOp(_ComparisonOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df) < self._value(df)


class _LeOp(_ComparisonOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df) <= self._value(df)


class _GtOp(_ComparisonOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df) > self._value(df)


class _GeOp(_ComparisonOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df) >= self._value(df)


class _BinaryOp(ColOp):
    def __init__(self, left: Union[ColOp, Any], right: Union[ColOp, Any]) -> None:
        if isinstance(left, ColOp):
            self._left = left
        else:
            self._left = _LiteralOp(left)
        if isinstance(right, ColOp):
            self._right = right
        else:
            self._right = _LiteralOp(right)


class _OrOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) | self._right(df)


class _AndOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) & self._right(df)


class _AddOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) + self._right(df)


class _SubtractOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) - self._right(df)


class _MultiplyOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) * self._right(df)


class _DivideOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) / self._right(df)


class _PowOp(_BinaryOp):
    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._left(df) ** self._right(df)


class _NotOp(ColOp):
    def __init__(self, col: ColOp) -> None:
        self._col = col

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return ~self._col(df)


class _ColApplyOp(ColOp):
    def __init__(self, col: ColOp, f: Callable[[pd.Series], pd.Series]) -> None:
        self._col = col
        self._fun = f

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._fun(self._col(df))


class _ColMapOp(ColOp):
    def __init__(self, col: ColOp, f: Callable[[Any], Any]) -> None:
        self._col = col
        self._fun = f

    def __call__(self, df: pd.DataFrame) -> pd.Series:
        return self._col(df).map(self._fun)
