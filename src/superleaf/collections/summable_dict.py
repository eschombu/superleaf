from typing import Self


class SummableDict(dict):
    """A dictionary-like object that supports addition of values."""
    def __add__(self, other) -> Self:
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: 'SummableDict' and '{type(other)}'")
        summed = self.copy()
        for k, v in other.items():
            if k in summed:
                summed[k] = summed[k] + v
            else:
                summed[k] = v
        return SummableDict(summed)

    def __iadd__(self, other) -> Self:
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: 'SummableDict' and '{type(other)}'")
        for k, v in other.items():
            if k in self:
                self[k] = self[k] + v
            else:
                self[k] = v
        return self

    def __radd__(self, other) -> Self:
        if other == 0:
            return self
        if not isinstance(other, dict):
            raise TypeError(f"unsupported operand type(s) for +: 'SummableDict' and '{type(other)}'")
        if not isinstance(other, SummableDict):
            other = SummableDict(other)
        return other + self

    def copy(self):
        return SummableDict(super().copy())
