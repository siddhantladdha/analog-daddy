"""
A collection of polars expressions/idioms.

These expressions have an unstable API and will change over time. Use
them as a template and vendor within your own codebase for stability.
These probably belong in a docs page, but they have earned their place
here due to their widespread use across multiple domains. most of them
are scavenged from Stackoverflow, Discord and github issues.
"""

import re
from collections.abc import Sequence

import polars as pl
import polars.selectors as cs
from quantiphy import InvalidNumber, as_real

from analog_daddy.utils import render

_COL_RE = re.compile(
    r"""
            col # literal
            \( # literal (
            "  # literal "
            (?P<var>\w+) # variable name
            "  # literal "
            \) # literal )
        """,
    re.VERBOSE,
)


def minmax(
    expr: pl.Expr,
    separator: str = "__",
) -> tuple[pl.Expr, pl.Expr]:
    """
    Get a polars expression to compute min and max of a column.

    Parameters
    ----------
    expr : pl.Expr
        The column to compute min and max across.
    separator : str, default="__"
        The separator to use in the output column names

    Returns
    -------
    tuple[pl.Expr, pl.Expr]
        A tuple of polars expressions for min and max of the input
        column.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "x": ["A", "B", "C", "D", "E", "F"],
    ...             "y": [-1, 0, 3, 4, -1, 4],
    ...             "z": [2, 3, 10, None, 3, 10],
    ...         }
    ...     )
    ... )
    shape: (6, 3)
    ┌─────┬─────┬──────┐
    │ x   ┆ y   ┆ z    │
    │ --- ┆ --- ┆ ---  │
    │ str ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ A   ┆ -1  ┆ 2    │
    │ B   ┆ 0   ┆ 3    │
    │ C   ┆ 3   ┆ 10   │
    │ D   ┆ 4   ┆ null │
    │ E   ┆ -1  ┆ 3    │
    │ F   ┆ 4   ┆ 10   │
    └─────┴─────┴──────┘
    >>> df.select(minmax(pl.col("y", "z")))
    shape: (1, 4)
    ┌────────┬────────┬────────┬────────┐
    │ min__y ┆ min__z ┆ max__y ┆ max__z │
    │ ---    ┆ ---    ┆ ---    ┆ ---    │
    │ i64    ┆ i64    ┆ i64    ┆ i64    │
    ╞════════╪════════╪════════╪════════╡
    │ -1     ┆ 2      ┆ 4      ┆ 10     │
    └────────┴────────┴────────┴────────┘
    """
    return (
        expr.min().name.prefix(f"min{separator}"),
        expr.max().name.prefix(f"max{separator}"),
    )


def item_where(
    expr: pl.Expr,
    predicate: pl.Expr,
    separator: str = "__",
    prefix: str = "item",
) -> pl.Expr:
    """
    Expression to get a single special value filtered by a predicate.

    Fails when evaluated if the predicate does not filter to a single
    value.

    Parameters
    ----------
    expr : pl.Expr
        The column to filter and get the item from.
    predicate : pl.Expr
        The predicate to filter the column by.
    separator : str, default="__"
        The separator to use in the output column name.
    prefix : str, default="item"
        The prefix to use in the output column name.

    Returns
    -------
    pl.Expr
        A polars expression that filters the column by the predicate and
        returns the single item.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "x": ["A", "B", "C", "A", "E"],
    ...             "y": [-1, 0, 3, 4, -1],
    ...             "z": [None, 3, 10, None, 3],
    ...         }
    ...     )
    ... )
    shape: (5, 3)
    ┌─────┬─────┬──────┐
    │ x   ┆ y   ┆ z    │
    │ --- ┆ --- ┆ ---  │
    │ str ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ A   ┆ -1  ┆ null │
    │ B   ┆ 0   ┆ 3    │
    │ C   ┆ 3   ┆ 10   │
    │ A   ┆ 4   ┆ null │
    │ E   ┆ -1  ┆ 3    │
    └─────┴─────┴──────┘
    >>> df.select(item_where(pl.col("y", "z"), pl.col("x") == "A"))
    Traceback (most recent call last):
    polars.exceptions.ComputeError: ...
    >>> df.select(item_where(pl.col("y", "z"), pl.col("x") == "B"))
    shape: (1, 2)
    ┌─────────┬─────────┐
    │ item__y ┆ item__z │
    │ ---     ┆ ---     │
    │ i64     ┆ i64     │
    ╞═════════╪═════════╡
    │ 0       ┆ 3       │
    └─────────┴─────────┘
    """
    # item() makes it fail explicitly.
    return expr.filter(predicate).item().name.prefix(prefix + separator)


def all_minmax_by(
    expr: pl.Expr,
    by: Sequence[str],
    separator: str = "__",
) -> tuple[pl.Expr, ...]:
    """
    Get values at the minimum and maximum of each ``by`` column.

    If the column has multiple values equal to its minimum or maximum,
    all corresponding values are returned. Polars has a ``min_by`` and
    ``max_by`` expression, but they return only one value and it is not
    defined which value will be chosen. Similarly, using ``arg_min`` or
    ``arg_max`` will return only one value, and it is not defined which
    value will be chosen.

    Parameters
    ----------
    expr : pl.Expr
        Values to return at each extremum.
    by : Sequence[str]
        Columns whose minimum and maximum select the values.
    separator : str, default="__"
        The separator to use in the output column names.

    Returns
    -------
    tuple[pl.Expr, ...]

    References
    ----------
    - I unfortunately found this [resource](https://code-examples.net/en/q/4bb4f9a/how-to-use-min-by-in-polars-to-solve-correlated-min-max) a bit later and it would have saved me some frustration in validating if my approach was the idiomatic.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "x": ["A", "B", "C", "D", "E", "F"],
    ...             "y": [-1, 0, 3, 4, -1, 4],
    ...             "z": [2, 3, 10, None, 3, 10],
    ...         }
    ...     )
    ... )
    shape: (6, 3)
    ┌─────┬─────┬──────┐
    │ x   ┆ y   ┆ z    │
    │ --- ┆ --- ┆ ---  │
    │ str ┆ i64 ┆ i64  │
    ╞═════╪═════╪══════╡
    │ A   ┆ -1  ┆ 2    │
    │ B   ┆ 0   ┆ 3    │
    │ C   ┆ 3   ┆ 10   │
    │ D   ┆ 4   ┆ null │
    │ E   ┆ -1  ┆ 3    │
    │ F   ┆ 4   ┆ 10   │
    └─────┴─────┴──────┘
    >>> df.group_by().agg(all_minmax_by(pl.col("x"), ["y", "z"]))
    shape: (1, 4)
    ┌─────────────┬─────────────┬─────────────┬─────────────┐
    │ x_at_min__y ┆ x_at_max__y ┆ x_at_min__z ┆ x_at_max__z │
    │ ---         ┆ ---         ┆ ---         ┆ ---         │
    │ list[str]   ┆ list[str]   ┆ list[str]   ┆ list[str]   │
    ╞═════════════╪═════════════╪═════════════╪═════════════╡
    │ ["A", "E"]  ┆ ["D", "F"]  ┆ ["A"]       ┆ ["C", "F"]  │
    └─────────────┴─────────────┴─────────────┴─────────────┘
    """
    out: list[pl.Expr] = []
    for i in by:
        out += [
            expr.filter(pl.col(i) == pl.col(i).min()).name.suffix(
                f"_at_min{separator}{i}"
            ),
            expr.filter(pl.col(i) == pl.col(i).max()).name.suffix(
                f"_at_max{separator}{i}"
            ),
        ]
    return tuple(out)


def ksigma_bounds(
    expr: pl.Expr,
    k: float = 1.0,
    separator: str = "__",
) -> tuple[pl.Expr, ...]:
    """
    Computes ksigma bounds, mean and standard deviation.

    Parameters
    ----------
    expr : pl.Expr
        The column to compute ksigma bounds across.
    k : float, default=1.0
        The number of standard deviations to compute the bounds.
    separator : str, default="__"
        The separator to use in the output column names.

    Returns
    -------
    tuple[pl.Expr, ...]

    Examples
    --------
    >>> import numpy as np
    >>> a = pl.DataFrame(
    ...     {
    ...         "x": np.random.default_rng(seed=12345).normal(
    ...             loc=10, scale=2, size=1200
    ...         ),
    ...         "y": np.random.default_rng(seed=12345).normal(
    ...             loc=0, scale=4, size=1200
    ...         ),
    ...     }
    ... ).with_row_index(offset=1)
    >>> a.select(ksigma_bounds(pl.col("x", "y"))).select(
    ...     pl.all().cast(pl.Decimal(scale=2))
    ... ).unpivot()
    shape: (8, 2)
    ┌─────────────────┬───────────────┐
    │ variable        ┆ value         │
    │ ---             ┆ ---           │
    │ str             ┆ decimal[38,2] │
    ╞═════════════════╪═══════════════╡
    │ mean - 1*std__x ┆ 8.05          │
    │ mean - 1*std__y ┆ -3.90         │
    │ mean__x         ┆ 10.07         │
    │ mean__y         ┆ 0.14          │
    │ mean + 1*std__x ┆ 12.09         │
    │ mean + 1*std__y ┆ 4.18          │
    │ std__x          ┆ 2.02          │
    │ std__y          ┆ 4.04          │
    └─────────────────┴───────────────┘
    """
    mean = expr.mean()
    sigma = expr.std()

    return (
        (mean - k * sigma).name.prefix(f"mean - {k:g}*std{separator}"),
        mean.name.prefix("mean" + separator),
        (mean + k * sigma).name.prefix(f"mean + {k:g}*std{separator}"),
        sigma.name.prefix("std" + separator),
    )


def _parse_operand(
    operand: str, pattern: re.Pattern[str] = _COL_RE
) -> pl.Expr:
    """
    Parse an operand with SI prefixes or column references to a polars
    expression.

    The function doesn't support something like ``col("foo") + 10m``.

    Parameters
    ----------
    operand : str
        The operand to parse.
    pattern : re.Pattern[str], default=_COL_RE
        The regex pattern to use for matching column references, by
        default ``_COL_RE``.

    Returns
    -------
    pl.Expr
        A polars expression representing the operand.

    Examples
    --------
    >>> _parse_operand("10m").meta.eq(pl.lit(0.01))
    True
    >>> _parse_operand('col("foo")').meta.eq(pl.col("foo"))
    True
    """
    if match := pattern.fullmatch(f"{operand}"):
        return pl.col(match.group("var"))
    return pl.lit(as_real(operand))


def spec(
    col: pl.Expr,
    spec: str,
    _near_tolerance: float = 0,
) -> pl.Expr:
    r"""
    Apply a specification string to a column as a polars expression.

    The spec string allows an ergonomic grammer for specifying
    constraints.
    - Supports SI-prefixes using quantiphy (e.g. ``10m`` = 0.01, ``0.01k`` = 10)
    - Supports simple column references ``col("name")``, which resolves to ``pl.col("name")``.

    TODO: Support ternary of pass/fail and near using _near_tolerance in
    future.

    Parameters
    ----------
    col : pl.Expr
        The column expression to test against the spec.
    spec : str
        The spec string to parse.
    _near_tolerance : float, default=0
        The tolerance for near matches, currently unused.

    Returns
    -------
    pl.Expr
        A boolean expression that evaluates to True if the value in
        ``col`` satisfies the spec, and False otherwise.

    Raises
    ------
    ValueError
        If the spec string is invalid or cannot be parsed.


    Examples
    --------
    >>> df = pl.from_repr(r'''shape: (2, 3)
    ... ┌──────┬──────┬─────┐
    ... │ x    ┆ foo  ┆ bar │
    ... │ ---  ┆ ---  ┆ --- │
    ... │ f64  ┆ f64  ┆ f64 │
    ... ╞══════╪══════╪═════╡
    ... │ 9.0  ┆ 10.0 ┆ 5.0 │
    ... │ 11.0 ┆ 10.0 ┆ 5.0 │
    ... └──────┴──────┴─────┘
    ... ''')
    >>> lit_specs = [
    ...     "> 10m",
    ...     "< 0.01k",
    ...     "range 8000m 10500m",
    ...     "tolerance 10 20",
    ... ]
    >>> df.select(*(spec(pl.col("x"), s).alias(s) for s in lit_specs))
    shape: (2, 4)
    ┌───────┬─────────┬────────────────────┬─────────────────┐
    │ > 10m ┆ < 0.01k ┆ range 8000m 10500m ┆ tolerance 10 20 │
    │ ---   ┆ ---     ┆ ---                ┆ ---             │
    │ bool  ┆ bool    ┆ bool               ┆ bool            │
    ╞═══════╪═════════╪════════════════════╪═════════════════╡
    │ true  ┆ true    ┆ true               ┆ true            │
    │ true  ┆ false   ┆ false              ┆ true            │
    └───────┴─────────┴────────────────────┴─────────────────┘
    >>> calc_specs = {
    ...     "range": 'range col("bar") col("foo")',
    ...     "gt": '> col("foo")',
    ...     "tol": 'tolerance col("foo") col("bar")',
    ... }
    >>> print(
    ...     df.select(
    ...         *(
    ...             spec(pl.col("x"), s).alias(name)
    ...             for name, s in calc_specs.items()
    ...         )
    ...     ).write_csv()
    ... )
    range,gt,tol
    true,false,false
    false,true,false
    <BLANKLINE>
    """
    operator, *vals = spec.split()
    try:
        vals = [_parse_operand(v) for v in vals]
    except InvalidNumber:
        raise ValueError(f"bad numbers {vals} in spec: {spec}")

    match operator:
        case ">":
            return col.gt(vals[0])
        case "<":
            return col.lt(vals[0])
        case "range":
            lo, hi = vals
            return col.is_between(lo, hi)
        case "tolerance" | "tol":  # "tolerance 10 2" -> 10 +/- 2%"
            val, tol_pct = vals
            tol_abs = abs(val) * tol_pct / 100
            return col.is_between(val - tol_abs, val + tol_abs)
    raise ValueError(f"bad spec: {spec}")


def list_cast():
    return (
        cs.list(cs.numeric()).list.eval(
            pl.element().map_elements(
                function=render,
                return_dtype=pl.String,
            )
        ),
        cs.list(cs.categorical()).cast(pl.List(pl.String)),
    )


def list_render_unique(max_len: int = 3):
    ulist = cs.list(cs.string()).list.unique().list
    n = ulist.len()
    head_n = ulist.head(max_len).list
    cond = (
        pl.when(n > max_len)
        .then(pl.format("{} + {} more", head_n.join(", "), n - max_len))
        .otherwise(ulist.join(", "))
    )
    return cond
