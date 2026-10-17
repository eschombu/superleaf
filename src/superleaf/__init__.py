from importlib.metadata import PackageNotFoundError, version

from superleaf.collections import OrderedSet, SummableDict
from superleaf.dataframe import dfilter, expand_dict_to_cols, partition, reorder_columns, where
from superleaf.operators import bool_operator, F, operator

try:
    __version__ = version("superleaf")
except PackageNotFoundError:
    __version__ = "unknown"
