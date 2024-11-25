from superleaf.operators.string import str_op
from superleaf.sequences import flat_map, mapped


def test_str_op():
    original = ["abc  ", "  12, 3", "  you,   and me!!  "]
    stripped = [s.strip() for s in original]
    assert mapped(str_op("strip"), original) == stripped

    squeezed = "abc.12.3.you.and me!!"
    assert ".".join(
        mapped(str_op("strip"), flat_map(str_op("strip") >> str_op("split", ","), original))
    ) == squeezed
