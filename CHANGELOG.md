# Changelog

## 1.0.0

### Breaking changes

- Requires pandas 3.0+, numpy 1.26+ and Python 3.11+. superleaf 0.9.x supports pandas 2 with the same API.
- `dfilter` and `partition` no longer copy their results; pandas copy-on-write keeps them independent of
  the input.

### Added

- `dfilter` and `partition` accept `pd.col` expressions as positional filters, e.g.
  `dfilter(df, pd.col("a") > pd.col("b"), side="L")`.

## 0.9.0

Superleaf is now a focused core: predicates, DataFrame filtering, sequence helpers and small collections.
It depends only on numpy and pandas.

### Breaking changes

- Removed `superleaf.timeseries`, `superleaf.plotting`, `superleaf.stats`, `superleaf.utils`
  (`parallel`, `hashing`, `ipython`) and `superleaf.dataframe.{column_ops,comparison,display,standardize}`
  (`Col`, `Values`, `nan_eq`, `standardize_columns`, `show_all`). Use `where(lambda df: ...)` or row
  callables in place of `Col` expressions in `dfilter`.
- Dependencies are `numpy>=1.23` and `pandas>=2.0`. `F.fuzzy_match` requires the `fuzzy` extra; the
  `parallel` extra is removed.
- `F.*` `fallback` and `exceptions` arguments are keyword-only. Passing only `fallback` now catches `Exception`.
- `F.fuzzy_match` defaults match `FuzzyMatcher`: inputs are normalized with `rapidfuzz.utils.default_process`
  and scored with `partial_ratio` (`substring=True`). `FuzzyMatcher` moved to `superleaf.operators.fuzzy`.
- `superleaf.sequences.flat_map` is renamed `flatmap`. `mapped`/`flatmap` no longer take `parallel`/`workers`.
- `BooleanOperator.__call__` always returns `bool`; use `on_series` to evaluate a Series.
- `dfilter` keyword filters with list or other non-scalar values raise `TypeError` (use `F.isin`). Scalar
  values follow `F.eq`, so `col=None` matches `None` cells; use `F.isna` for missing values.
- `F.isin` returns False for unhashable values instead of raising.
- `F.eq/ne/lt/le/gt/ge` with a non-missing value treat None, NaN, NaT and `pd.NA` like NaN (only `ne` is
  True) instead of raising for None or `pd.NA`.
- `&` and `|` on predicates short-circuit, so the right operand is not evaluated where the left decides.
- A boolean Series passed as a positional `dfilter` filter must have the same index as the DataFrame.

### Added

- `BooleanOperator.on_series`, vectorized for `F` comparisons, `isin`, string and regex predicates, `isna`
  and `isinf`, with the same result per element as the scalar call.
- `where` for vectorized whole-DataFrame row filters.
- `F.matches_regex(pattern, how=...)` with `"match"` (default), `"search"` or `"fullmatch"`.
- `F` alias in `superleaf.operators`, and top-level imports: `F`, `dfilter`, `partition`, `where`,
  `reorder_columns`, `expand_dict_to_cols`, `operator`, `bool_operator`, `OrderedSet`, `SummableDict`.
- Piping into a `BooleanOperator` with `>>` gives a `BooleanOperator`; predicates combine with plain callables.
- `dfilter` and `partition` accept a Series, and `partition` accepts `columns=`.

### Fixed

- `dfilter(columns=[...])` kept no columns; it now keeps columns matching any entry. `columns=None` is a no-op.
- `(~F.isna)` and other combined predicates failed on a Series.
- `FuzzyMatcher` scored against unnormalized targets.
- `expand_dict_to_cols` modified the input DataFrame when decoding JSON, and decided whether to decode from
  the first row only. Recursive expansion now passes on `sep` and `dropna`.
- `reorder_columns(errors="warn")` returned None, uppercase `errors` values were not handled, and falsy
  `after`/`before` column names were ignored.
- `index_getter(..., default=...)` raised on `IndexError`.
- `F.isna` raised for `pd.NA` and returned arrays for array inputs.
