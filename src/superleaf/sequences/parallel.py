from typing import Callable, Iterable, List, TypeVar

from multiprocess.pool import Pool

from superleaf.sequences.serial import flatten

T = TypeVar("T")
U = TypeVar("U")


def mapped(f: Callable[[T], U], seq: Iterable[T], workers=None) -> List[U]:
    with Pool(workers) as p:
        return p.map(f, seq)


def flat_map(f: Callable, seq: Iterable, depth=None, drop_null=True, workers=None) -> list:
    return flatten(mapped(f, seq, workers=workers), depth=depth, drop_null=drop_null)
