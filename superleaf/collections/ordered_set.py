import itertools


class OrderedSet:
    """Similar interface to the native set class, but with item order maintained, and expanded functionality, including
    addition and summation. Implemented by storing the set items as keys in an internal dict."""

    def __init__(self, items=None):
        self._dict = dict(zip(items, itertools.repeat(None))) if items is not None else {}

    @property
    def _items(self):
        return list(self._dict.keys())

    def __iter__(self):
        return iter(self._items)

    def copy(self):
        return self.__class__(self._items)

    def union(self, other):
        return self.__class__(itertools.chain(self, other))

    def add(self, item):
        self._dict[item] = None

    def intersection(self, other):
        return self.__class__(filter(lambda x: x in other, self))

    def __add__(self, other):
        return self.__class__(self.union(other))

    def __radd__(self, other):
        if other == 0:
            return self
        elif isinstance(other, self.__class__):
            return self.__class__(other) + self

    def __iadd__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        self._dict.update(other._dict)
        return self

    def __sub__(self, other):
        return self.__class__(filter(lambda x: x not in other, self))

    def __isub__(self, other):
        if not isinstance(other, self.__class__):
            other = self.__class__(other)
        for item in other:
            if item in self:
                self._dict.pop(item)
        return self

    def __contains__(self, item):
        return item in self._dict

    def __eq__(self, other):
        if isinstance(other, set):
            return set(self._items) == other
        elif isinstance(other, OrderedSet):
            return self._dict == other._dict
        else:
            return False

    def __repr__(self):
        return "{" + ", ".join([item.__repr__() for item in self._items]) + "}"

    def __len__(self):
        return len(self._dict)

    def __getitem__(self, item):
        return self._items[item]
