"""Utilities for working with Polars DataFrames."""

import re

import polars as pl
import polars.selectors as cs


def drop_all_null(df: pl.DataFrame) -> pl.DataFrame:
    """
    Drop columns where every value is null.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to clean up.

    Returns
    -------
    pl.DataFrame
        Clean DataFrame with all-null columns removed.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "a": [None, None, None],
    ...             "b": [1, 2, None],
    ...             "c": ["x", "y", ""],
    ...         }
    ...     )
    ... )
    shape: (3, 3)
    ┌──────┬──────┬─────┐
    │ a    ┆ b    ┆ c   │
    │ ---  ┆ ---  ┆ --- │
    │ null ┆ i64  ┆ str │
    ╞══════╪══════╪═════╡
    │ null ┆ 1    ┆ x   │
    │ null ┆ 2    ┆ y   │
    │ null ┆ null ┆     │
    └──────┴──────┴─────┘
    >>> drop_all_null(df)
    shape: (3, 2)
    ┌──────┬─────┐
    │ b    ┆ c   │
    │ ---  ┆ --- │
    │ i64  ┆ str │
    ╞══════╪═════╡
    │ 1    ┆ x   │
    │ 2    ┆ y   │
    │ null ┆     │
    └──────┴─────┘
    """
    return df[[ser.name for ser in df if ser.null_count() != df.height]]


def filter_all_null(df: pl.DataFrame) -> pl.DataFrame:
    """
    Filter out rows from a dataframe where every value is null.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to clean up.

    Returns
    -------
    pl.DataFrame
        Clean DataFrame with all-null rows removed.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "a": [1, None, 3],
    ...             "b": ["x", None, None],
    ...         }
    ...     )
    ... )
    shape: (3, 2)
    ┌──────┬──────┐
    │ a    ┆ b    │
    │ ---  ┆ ---  │
    │ i64  ┆ str  │
    ╞══════╪══════╡
    │ 1    ┆ x    │
    │ null ┆ null │
    │ 3    ┆ null │
    └──────┴──────┘
    >>> filter_all_null(df)
    shape: (2, 2)
    ┌─────┬──────┐
    │ a   ┆ b    │
    │ --- ┆ ---  │
    │ i64 ┆ str  │
    ╞═════╪══════╡
    │ 1   ┆ x    │
    │ 3   ┆ null │
    └─────┴──────┘
    """
    return df.filter(~pl.all_horizontal(pl.all().is_null()))


def scan_nested_params(
    df: pl.DataFrame,
    scan_col: str,
    identify_str: str = "Parameters:",
    key_split: str = "=",
    pair_split: str = ",",
) -> list[str]:
    """
    Scan a DataFrame for first matching row containing nested params.

    Eagerly scans the DataFrame for the first row containing
    ``identify_str``, then extracts key names from it using
    ``key_split`` and ``pair_split``. Used with
    :func:`explode_nested_params`.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to scan for the row.
    scan_col : str
        Column to scan for the identifying row.
    identify_str : str, default="Parameters:"
        String that identifies the row to extract keys from.
    key_split : str, default="="
        String that separates keys from values in the row.
    pair_split : str, default=","
        String that separates key-value pairs in the row.

    Returns
    -------
    list[str]
        Ordered list of parameter names found in the first matching row.

    Raises
    ------
    ValueError

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "Point": [
    ...                 "Parameters: var1=-12k, var_b=1n",
    ...                 "1",
    ...                 "Parameters: var3=-12k",
    ...                 "2",
    ...             ],
    ...             "Value": [None, 10, 1, None],
    ...         }
    ...     )
    ... )
    shape: (4, 2)
    ┌─────────────────────────────────┬───────┐
    │ Point                           ┆ Value │
    │ ---                             ┆ ---   │
    │ str                             ┆ i64   │
    ╞═════════════════════════════════╪═══════╡
    │ Parameters: var1=-12k, var_b=1… ┆ null  │
    │ 1                               ┆ 10    │
    │ Parameters: var3=-12k           ┆ 1     │
    │ 2                               ┆ null  │
    └─────────────────────────────────┴───────┘
    >>> scan_nested_params(df, scan_col="Point")
    ['var1', 'var_b']
    """
    first: str | None = df.select(
        pl.col(scan_col)
        .filter(pl.col(scan_col).str.starts_with(identify_str))
        .drop_nulls()
        .first()
    ).item()

    if not first:
        raise ValueError(f"No row found starting with {identify_str}")
    first = first.replace(" ", "").removeprefix(identify_str)
    return [
        pair.split(re.escape(key_split))[0]
        for pair in first.split(re.escape(pair_split))
    ]


def explode_nested_params(
    df: pl.DataFrame,
    keys: list[str],
    index_col: str = "Point",
    identify_str: str = "Parameters:",
) -> pl.DataFrame:
    """
    Turn inline ``key=value`` parameter rows into columns on the data
    rows below them.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame with interleaved parameter/data rows.
    keys : list[str]
        Ordered parameter key names, e.g. from
        :func:`scan_nested_params`.
    index_col : str, default="Point"
        Column that carries the parameter marker rows.
    identify_str : str, default="Parameters:"
        Prefix string that identifies a parameter row.

    Returns
    -------
    pl.DataFrame
        Data rows only, with one new column per key in ``keys``.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "Point": ["Parameters: ds=0, gs=1", "1", "2"],
    ...             "Value": [None, 10, 20],
    ...         }
    ...     )
    ... )
    shape: (3, 2)
    ┌────────────────────────┬───────┐
    │ Point                  ┆ Value │
    │ ---                    ┆ ---   │
    │ str                    ┆ i64   │
    ╞════════════════════════╪═══════╡
    │ Parameters: ds=0, gs=1 ┆ null  │
    │ 1                      ┆ 10    │
    │ 2                      ┆ 20    │
    └────────────────────────┴───────┘
    >>> explode_nested_params(df, keys=["ds", "gs"])
    shape: (2, 4)
    ┌───────┬───────┬─────┬─────┐
    │ Point ┆ Value ┆ ds  ┆ gs  │
    │ ---   ┆ ---   ┆ --- ┆ --- │
    │ str   ┆ i64   ┆ str ┆ str │
    ╞═══════╪═══════╪═════╪═════╡
    │ 1     ┆ 10    ┆ 0   ┆ 1   │
    │ 2     ┆ 20    ┆ 0   ┆ 1   │
    └───────┴───────┴─────┴─────┘
    """
    is_param_row = pl.col(index_col).str.starts_with(identify_str)
    param_re = ", ".join(rf"{k}=(?P<{k}>[^,]+)" for k in keys)
    return (
        df.with_columns(
            pl.when(is_param_row)
            .then(pl.col(index_col))
            .otherwise(None)
            .forward_fill()
            .alias("param_str")
        )
        .filter(~is_param_row)
        .with_columns(
            pl.col("param_str").str.extract_groups(param_re).alias("params")
        )
        .unnest("params")
        .drop("param_str")
    )


def cleanup(df: pl.DataFrame, delimiter: str) -> pl.DataFrame:
    """
    A representative demo of simple dataframe cleanup operation
    utilizing the helper functions.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to clean up.
    delimiter : str
        Name of the column to keep, even if it is empty.

    Returns
    -------
    pl.DataFrame
        Cleaned DataFrame.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "x": [1, None, 3],
    ...             "e": [None, None, None],
    ...             "d": [None, None, None],
    ...             "y": [4.0, None, 6.0],
    ...         }
    ...     )
    ... )
    shape: (3, 4)
    ┌──────┬──────┬──────┬──────┐
    │ x    ┆ e    ┆ d    ┆ y    │
    │ ---  ┆ ---  ┆ ---  ┆ ---  │
    │ i64  ┆ null ┆ null ┆ f64  │
    ╞══════╪══════╪══════╪══════╡
    │ 1    ┆ null ┆ null ┆ 4.0  │
    │ null ┆ null ┆ null ┆ null │
    │ 3    ┆ null ┆ null ┆ 6.0  │
    └──────┴──────┴──────┴──────┘
    >>> cleanup(df, delimiter="d")
    shape: (2, 3)
    ┌─────┬─────┬─────┐
    │ x   ┆ d   ┆ y   │
    │ --- ┆ --- ┆ --- │
    │ i64 ┆ str ┆ f64 │
    ╞═════╪═════╪═════╡
    │ 1   ┆ |   ┆ 4.0 │
    │ 3   ┆ |   ┆ 6.0 │
    └─────┴─────┴─────┘
    """
    return (
        # firstly we remove empty rows.
        df.pipe(filter_all_null)
        # then fill the delimiter column with junk to keep it.
        .with_columns(pl.lit("|").alias(delimiter))
        # finally drop any empty columns.
        .pipe(drop_all_null)
    )


def metrics_to_rows(df: pl.DataFrame, separator="__") -> pl.DataFrame:
    """
    Reshape a DataFrame by unpivoting and pivoting based on a column
    name separator being present.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame to reshape.
    separator : str, default="__"
        Separator string to identify columns to unpivot.

    Returns
    -------
    pl.DataFrame
        Reshaped DataFrame with unpivoted and pivoted columns.

    Examples
    --------
    >>> (
    ...     df := pl.DataFrame(
    ...         {
    ...             "id": [1, 2],
    ...             "min__x": [1.0, 2.0],
    ...             "min__y": [1.0, 4.0],
    ...             "max__x": [5.0, 6.0],
    ...             "max__y": [5.0, 6.0],
    ...         }
    ...     )
    ... )
    shape: (2, 5)
    ┌─────┬────────┬────────┬────────┬────────┐
    │ id  ┆ min__x ┆ min__y ┆ max__x ┆ max__y │
    │ --- ┆ ---    ┆ ---    ┆ ---    ┆ ---    │
    │ i64 ┆ f64    ┆ f64    ┆ f64    ┆ f64    │
    ╞═════╪════════╪════════╪════════╪════════╡
    │ 1   ┆ 1.0    ┆ 1.0    ┆ 5.0    ┆ 5.0    │
    │ 2   ┆ 2.0    ┆ 4.0    ┆ 6.0    ┆ 6.0    │
    └─────┴────────┴────────┴────────┴────────┘
    >>> metrics_to_rows(df)
    shape: (4, 4)
    ┌─────┬────────┬─────┬─────┐
    │ id  ┆ metric ┆ min ┆ max │
    │ --- ┆ ---    ┆ --- ┆ --- │
    │ i64 ┆ str    ┆ f64 ┆ f64 │
    ╞═════╪════════╪═════╪═════╡
    │ 1   ┆ x      ┆ 1.0 ┆ 5.0 │
    │ 2   ┆ x      ┆ 2.0 ┆ 6.0 │
    │ 1   ┆ y      ┆ 1.0 ┆ 5.0 │
    │ 2   ┆ y      ┆ 4.0 ┆ 6.0 │
    └─────┴────────┴─────┴─────┘
    """  # noqa: DOC105
    _variable = "_variable_name_"
    _value = "_measurement_value_"
    return (
        df.unpivot(
            # converts any columns containing `separator` to rows.
            # everything else stays as columns.
            index=~cs.contains(separator),
            variable_name=_variable,
            value_name=_value,
        )
        # processing the rows of newly created variable column.
        .with_columns(
            pl.col(_variable)
            .str.split_exact(by=separator, n=1)
            .struct.rename_fields(["aggregation", "metric"])
        )
        # split struct variable column.
        .unnest(_variable)
        .pivot(
            on="aggregation",  # unique values become the new column names
            values=_value,  # fills the cells.
            # index: everything else; must be unique per on
        )
    )
