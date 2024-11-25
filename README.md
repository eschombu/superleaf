# `superleaf`

## A library for intuitive and readable data filtering and manipulation, using functional and pipeable syntax.

When trying to select and filter `pandas` dataframes in somewhat complicated ways, the syntax can quickly get
cumbersome. This library provides utilities for quickly and intuitively specifying ways to select and filter based on
column and row conditions. It also includes utilities for piping and composing value accessor and manipulation operators
on more general datatypes.

## Operators

Some of the operators in this library have analogues in the Python standard library `operator` module, but the ones here
can be more flexibly composed and piped.

For example, let's say you want to take the log of all of the values inside one of the fields of a sequence of data
containers. You can achieve this with the following code:

```
import numpy as np
from superleaf.operators import operator
from superlead.getters import attr_getter

exp_field_op = attr_getter("field_a") >> operator(np.log)  # "right shift" operator used for piping
results_iter = map(exp_field_op, data_containers)
```

This produces an iterator with the same results as the following list comprehension:
```
results_list = [np.log(datum.field_a) for datum in data_containers]
```

## DataFrame column operators and selection/filtering

Row filtering, especially for complicated combinations of conditions, of pandas dataframes can have cumbersome and
seemingly repetitive syntax. The `dfilter` and `Col` utilities can be used to more succinctly achieve complicated
filtering and selection of portions of dataframes.

For example, consider the following dataframe:
```
import pandas as pd

df = pd.DataFrame({
    "col1": [   0,   1,   0,   1,    1,      1,   0],
    "col2": [-5.1, 2.2, 0.2, 1.7, -1.1, np.nan, 0.9],
    "col3": [ "A", "A", "C", "B",  "C",    "C", "C"],
})
```

Let's say we want to select the rows where col1 == 1, col2 is null or negative, and col3 == "C":
```
sub_df = df[(df["col1"] == 1) & (df["col2"].isna() | (df["col2"] < 0)) & (df["col3"] == "C")]
```

Using `superleaf`, we could do this with the following:
```
from superleaf.operator import ComparisonFunctions as F
from superleaf.dataframe.selection import dfilter

sub_df = dfilter(df, col1=1, col2=(F.isna | F.lt(0)), col3="C")
```
