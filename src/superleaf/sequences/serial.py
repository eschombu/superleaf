from collections import defaultdict
from typing import Callable, Iterable, TypeVar

T = TypeVar("T")
U = TypeVar("U")


def mapped(f: Callable[[T], U], seq: Iterable[T]) -> list[U]:
    """Return ``list(map(f, seq))``.

    Parameters
    ----------
    f : callable
        Function applied to each item.
    seq : iterable
        Items.

    Returns
    -------
    list[U]
        Results in input order.
    """
    return list(map(f, seq))


def filtered(f: Callable[[T], bool], seq: Iterable[T]) -> list[T]:
    """Return ``list(filter(f, seq))``.

    Parameters
    ----------
    f : callable
        Predicate.
    seq : iterable
        Items.

    Returns
    -------
    list[T]
        Items for which ``f`` is truthy, in input order.
    """
    return list(filter(f, seq))


def groupby(f: Callable[[T], U], seq: Iterable[T]) -> dict[U, list[T]]:
    """Group items by key. Unlike ``itertools.groupby``, the input need not be sorted.

    Parameters
    ----------
    f : callable
        Key function.
    seq : iterable
        Items.

    Returns
    -------
    dict[U, list[T]]
        Maps each key to its items, in input order.
    """
    grouped = defaultdict(list)
    for item in seq:
        grouped[f(item)].append(item)
    return dict(grouped)


def flatten(seq: Iterable, depth=None, drop_null=False) -> list:
    """Flatten nested sequences into a list. Strings are not split.

    Parameters
    ----------
    seq : iterable
        Possibly nested items.
    depth : int, optional
        Levels to flatten. None (default) flattens completely.
    drop_null : bool, optional
        If True, drop None and NaN items.

    Returns
    -------
    list
        Flattened items.
    """
    if depth == 0:
        return list(seq)
    flattened = []
    for item in seq:
        next_depth = depth - 1 if depth is not None else None
        if isinstance(item, str) or not (hasattr(item, "__len__") or hasattr(item, "__next__")):
            if drop_null and (item is None or item != item):
                continue
            else:
                flattened.append(item)
        else:
            flattened.extend(flatten(item, next_depth, drop_null=drop_null))
    return flattened


def flatmap(f: Callable, seq: Iterable, depth=None, drop_null=True) -> list:
    """Map ``f`` over ``seq`` and flatten the results.

    Parameters
    ----------
    f : callable
        Function applied to each item.
    seq : iterable
        Items.
    depth : int, optional
        Levels to flatten (see :func:`flatten`).
    drop_null : bool, optional
        If True (default), drop None and NaN items.

    Returns
    -------
    list
        Flattened results.
    """
    return flatten(mapped(f, seq), depth=depth, drop_null=drop_null)


def partitioned(f: Callable[[T], bool], seq: Iterable[T]) -> tuple[list[T], list[T]]:
    """Split items by a predicate.

    Parameters
    ----------
    f : callable
        Predicate.
    seq : iterable
        Items.

    Returns
    -------
    list[T]
        Items for which ``f`` is truthy, in input order.
    list[T]
        The remaining items, in input order.
    """
    true_part, false_part = [], []
    for item in seq:
        if f(item):
            true_part.append(item)
        else:
            false_part.append(item)
    return true_part, false_part
