# superleaf

Readable, composable filters for pandas and plain Python data.

```
pip install superleaf           # numpy + pandas only
pip install "superleaf[fuzzy]"  # adds F.fuzzy_match (rapidfuzz)
```

## Filters as data

A filter is a dict mapping columns to conditions. It can be built in one place, stored, combined and
passed to `dfilter`:

```python
import pandas as pd
from superleaf import F, dfilter, partition

df = pd.DataFrame({
    "type": ["T4a", "T4b", "Mi1", None, "Tm3"],
    "side": ["L", "R", "L", "R", "L"],
    "size": [10.0, None, 3.5, 7.0, -1.0],
})

left_t4 = {"type": F.startswith("T4"), "side": "L"}
dfilter(df, **left_t4)

dfilter(df, size=F.isna | F.lt(0))
dfilter(df, type=F.isin(["Mi1", "Tm3"]), columns=["type", "size"])
kept, dropped = partition(df, type=F.notna)
```

Keyword values are matched by equality, `F.*` predicates and other callables are applied to each value,
and `columns=` selects columns by name or predicate. Use `where` for conditions across columns:

```python
from superleaf import where

dfilter(df, where(lambda d: d["size"] > d["size"].median()))
```

## Predicates work on values and columns

`F` predicates combine with `|`, `&` and `~`, and evaluate either a single value or a whole Series with
the same result:

```python
is_valid = F.gt(0) | F.eq("n/a")
is_valid(5)                        # True
is_valid(None)                     # False: missing values only satisfy ne
is_valid.on_series(df["size"])     # boolean Series; vectorized where possible
```

`&` and `|` short-circuit like `and` and `or`, so a type check on the left guards the right.

The same predicate filters lists:

```python
from superleaf.sequences import filtered, partitioned

filtered(F.startswith("T4"), ["T4a", "Mi1", "T4b"])   # ["T4a", "T4b"]
```

Available predicates: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `isin`, `contains`, `contains_all`, `contains_any`,
`startswith`, `endswith`, `startswith_one_of`, `endswith_one_of`, `matches_regex`, `fuzzy_match`, `isna`,
`notna`, `isinf`, `notinf`. Pass `fallback=` (and optionally `exceptions=`) to return a value instead of
raising, e.g. `F.lt(0, fallback=False)("a")` is `False`.

Custom predicates and pipelines:

```python
from superleaf.operators import attr_getter, bool_operator, str_op

is_even = bool_operator(lambda x: x % 2 == 0)
lower_starts_e = str_op("lower") >> F.startswith("e")   # pipe with >>
get_score = attr_getter("score") >> F.ge(0.5)
```

## Compared with alternatives

| | Reusable as data | Works on scalars and lists | Missing values / mixed types |
|---|---|---|---|
| `df.query("a > 0 and b == 'x'")` | as strings | no | follows pandas |
| `df.loc[lambda d: (d.a > 0) & (d.b == "x")]` | as functions | no | follows pandas |
| polars expressions | yes | no (polars frames only) | follows polars |
| `dfilter(df, a=F.gt(0), b="x")` | yes | yes | same result per value as plain Python, with `fallback=` |

## Other utilities

- `superleaf.sequences`: `mapped`, `filtered`, `flatten`, `flatmap`, `groupby`, `partitioned`
- `superleaf.dataframe`: `reorder_columns`, `expand_dict_to_cols`
- `superleaf.collections`: `OrderedSet`, `SummableDict`

See the [API reference](https://erikschomburg.com/superleaf/) and [CHANGELOG](CHANGELOG.md).
