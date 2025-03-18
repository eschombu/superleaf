from functools import partial

from superleaf.utils.hashing import get_hash_string


def test_get_hash_string():
    v = 1
    h = get_hash_string(v)
    assert h == get_hash_string(v)

    l = len(h) - 1
    h2 = get_hash_string(v, length=l)
    assert len(h2) == l
    assert h2 != h

    l = 8
    h = get_hash_string(v, algo='sha1', length=l)
    h2 = get_hash_string(v, algo='sha1')
    assert len(h) != len(h2)
    assert h == h2[:l]

    l = 8
    h = get_hash_string(v, algo='shake_128', length=l)
    h2 = get_hash_string(v, algo='shake_128', length=(l + 1))
    assert h != h2
    assert h == h2[:l]

    h = get_hash_string(v, algo='sha1', length=l)
    h2 = get_hash_string(v, algo='shake_128', length=l)
    assert h != h2

    assert get_hash_string(v) == get_hash_string([v])
    assert get_hash_string(v) == get_hash_string([v, None])

    hasher = partial(get_hash_string, algo='sha1', length=8)
    assert hasher(1) != hasher('hello')
    assert hasher(1) != hasher('1')
    assert hasher(1) != hasher([1, 'hello'])
    assert hasher('hello') != hasher([1, 'hello'])
