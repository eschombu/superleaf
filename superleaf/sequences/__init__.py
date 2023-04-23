from typing import Callable, Iterable, List, TypeVar

from superleaf.sequences.serial import filtered, flatten, flat_map as _flat_map_s, groupby, mapped as _mapped_s
from superleaf.sequences.parallel import flat_map as _flat_map_p, mapped as _mapped_p

T = TypeVar("T")
U = TypeVar("U")


def mapped(f: Callable[[T], U], seq: Iterable[T], parallel=False, workers=None) -> List[U]:
    if parallel or (workers is not None and workers > 1):
        return _mapped_p(f, seq, workers=workers)
    else:
        return _mapped_s(f, seq)


def flat_map(f: Callable, seq: Iterable, depth=None, drop_null=True, parallel=False, workers=None) -> list:
    if parallel or (workers is not None and workers > 1):
        return _flat_map_p(f, seq, depth=depth, drop_null=drop_null, workers=workers)
    else:
        return _flat_map_s(f, seq, depth=depth, drop_null=drop_null)
