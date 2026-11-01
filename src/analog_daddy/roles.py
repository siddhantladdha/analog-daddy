"""
A module to define semantic roles of columns in a DataFrame.

Examples
--------
>>> import polars as pl
>>> import polars.selectors as cs
>>> df = pl.DataFrame(
...     {
...         "x1": [1, 2, 3],
...         "x2": ["a", "b", "c"],
...         "delimiter": [0, 0, 0],
...         "y1": [4, 5, 6],
...         "y2": ["d", "e", "f"],
...     }
... )
>>> xvars, yvars, cs_roles, pl_roles = map_roles(df, delimiter="delimiter")
>>> xvars
('x1', 'x2')
>>> yvars
('y1', 'y2')
>>> cs.is_selector(cs_roles.x)
True
>>> cs.is_selector(cs_roles.numy)
True
>>> pl_roles.x.meta.eq(pl.col("x1", "x2"))
True
>>> df.select(pl_roles.numy).columns
['y1']
"""

from typing import NamedTuple

import polars as pl
import polars.selectors as cs
from boltons.iterutils import split


class _CSRoles:
    """semantic roles of columns in a DataFrame as polars selectors."""

    def __init__(self, xvars: tuple[str, ...], yvars: tuple[str, ...]):
        self.x: cs.Selector = cs.by_name(*xvars)
        self.y: cs.Selector = cs.by_name(*yvars)
        self.numx: cs.Selector = cs.numeric() & self.x
        self.numy: cs.Selector = cs.numeric() & self.y
        self.catx: cs.Selector = cs.categorical() & self.x
        self.caty: cs.Selector = cs.categorical() & self.y


class _PLRoles:
    """
    semantic roles of columns in a DataFrame as polars expressions.
    """

    def __init__(
        self,
        xvars: tuple[str, ...],
        yvars: tuple[str, ...],
        cs_roles: _CSRoles,
    ):
        self.x: pl.Expr = pl.col(*xvars)
        self.y: pl.Expr = pl.col(*yvars)
        self.numx: pl.Expr = cs_roles.numx.as_expr()
        self.numy: pl.Expr = cs_roles.numy.as_expr()
        self.catx: pl.Expr = cs_roles.catx.as_expr()
        self.caty: pl.Expr = cs_roles.caty.as_expr()


class Roles(NamedTuple):
    """Column roles of a DataFrame split by a delimiter column."""

    xvars: tuple[str, ...]
    yvars: tuple[str, ...]
    cs_roles: _CSRoles
    pl_roles: _PLRoles


def map_roles(df: pl.DataFrame, delimiter: str) -> Roles:
    """
    Assign roles to DataFrame columns split by a delimiter column.

    Columns before ``delimiter`` are treated as independent variables,
    columns after it as dependent variables. Run ``plutils.cleanup``
    first, or reorder columns as ``x..., delimiter, y...``, for the
    roles to be correct.
    """

    xvars, yvars = map(tuple, split(df.columns, delimiter))
    _cs = _CSRoles(xvars, yvars)
    return Roles(
        xvars=xvars,
        yvars=yvars,
        cs_roles=_cs,
        pl_roles=_PLRoles(xvars, yvars, _cs),
    )
