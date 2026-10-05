from superleaf.operators.string import str_op
from superleaf.sequences import flatmap, mapped


def test_str_op():
    original = ["abc  ", "  12, 3", "  you,   and me!!  "]
    stripped = [s.strip() for s in original]
    assert mapped(str_op("strip"), original) == stripped

    squeezed = "abc.12.3.you.and me!!"
    assert ".".join(
        mapped(str_op("strip"), flatmap(str_op("strip") >> str_op("split", ","), original))
    ) == squeezed
