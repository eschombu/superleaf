from functools import partial

from superleaf.utils.hashing import get_hash_string


def test_get_hash_string():
    v = 1
    h = get_hash_string(v)
    assert h == get_hash_string(v)

    n = len(h) - 1
    h2 = get_hash_string(v, length=n)
    assert len(h2) == n
    assert h2 != h

    n = 8
    h = get_hash_string(v, algo='sha1', length=n)
    h2 = get_hash_string(v, algo='sha1')
    assert len(h) != len(h2)
    assert h == h2[:n]

    n = 8
    h = get_hash_string(v, algo='shake_128', length=n)
    h2 = get_hash_string(v, algo='shake_128', length=(n + 1))
    assert h != h2
    assert h == h2[:n]

    h = get_hash_string(v, algo='sha1', length=n)
    h2 = get_hash_string(v, algo='shake_128', length=n)
    assert h != h2

    assert get_hash_string(v) == get_hash_string([v])
    assert get_hash_string(v) == get_hash_string([v, None])

    hasher = partial(get_hash_string, algo='sha1', length=8)
    assert hasher(1) != hasher('hello')
    assert hasher(1) != hasher('1')
    assert hasher(1) != hasher([1, 'hello'])
    assert hasher('hello') != hasher([1, 'hello'])
