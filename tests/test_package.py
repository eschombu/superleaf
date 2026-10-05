import superleaf
from superleaf.sequences import filtered, flatmap, flatten, groupby, mapped, partitioned  # noqa: F401


def test_top_level_imports():
    for name in [
        "F", "dfilter", "partition", "where", "bool_operator", "operator", "expand_dict_to_cols", "OrderedSet",
        "reorder_columns", "SummableDict", "__version__",
    ]:
        assert hasattr(superleaf, name)
    assert superleaf.F.gt(0)(1) is True
