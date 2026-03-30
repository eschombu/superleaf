class SummableDict(dict):
    """A dictionary-like object that supports addition and subtraction of values."""

    def __add__(self, other) -> "SummableDict":
        summed = self.copy()
        if isinstance(other, dict):
            for k, v in other.items():
                if k in summed:
                    summed[k] = summed[k] + v
                else:
                    summed[k] = v
        else:
            for k, v in self.items():
                summed[k] = summed[k] + other
        return summed

    def __iadd__(self, other) -> "SummableDict":
        if isinstance(other, dict):
            for k, v in other.items():
                if k in self:
                    self[k] = self[k] + v
                else:
                    self[k] = v
        else:
            for k, v in self.items():
                self[k] = self[k] + other
        return self

    def __radd__(self, other) -> "SummableDict":
        if other == 0:
            return self.copy()
        elif isinstance(other, dict):
            return SummableDict(other) + self
        else:
            summed = self.copy()
            for k, v in self.items():
                summed[k] = other + v
            return summed

    def __neg__(self) -> "SummableDict":
        return SummableDict({k: -v for k, v in self.items()})

    def __sub__(self, other) -> "SummableDict":
        if isinstance(other, dict):
            return self + -SummableDict(other)
        else:
            return self + -other

    def __isub__(self, other) -> "SummableDict":
        if isinstance(other, dict):
            for k, v in other.items():
                if k in self:
                    self[k] = self[k] - v
                else:
                    self[k] = -v
        else:
            for k, v in self.items():
                self[k] = self[k] - other
        return self

    def __rsub__(self, other) -> "SummableDict":
        if other == 0:
            return self.copy()
        else:
            return other + -self

    def copy(self) -> "SummableDict":
        return SummableDict(super().copy())
