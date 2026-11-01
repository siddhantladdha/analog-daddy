"""
A collection of marimo utility functions.

While marimo is quite feature-rich, these functions are intended to be
convenience functions for common tasks that are not yet implemented in
marimo. These might be deprecated if they become available in marimo.
"""

import io
from collections.abc import Callable
from typing import Any

import marimo as mo
import polars as pl
import polars.selectors as cs
from rich.console import Console, RenderableType

from analog_daddy.utils import render


def format_mapping_num_to_si(df: pl.DataFrame) -> dict[str, Callable]:
    """
    Build a format mapping that renders numeric columns as SI strings.

    Intended for the ``format_mapping`` argument of ``mo.ui.table`` and
    ``mo.ui.dataframe``. Columns are matched with ``cs.numeric()`` and
    rendered with :func:`render`.

    Parameters
    ----------
    df : pl.DataFrame
        DataFrame whose numeric columns to map.

    Returns
    -------
    dict[str, Callable]
        Column name to formatter mapping.

    References
    ----------
    - `marimo#9663 <https://github.com/marimo-team/marimo/issues/9663>`_
    - `marimo#7568 <https://github.com/marimo-team/marimo/issues/7568>`_
    """
    return dict.fromkeys(cs.expand_selector(df, cs.numeric()), render)


def mo_table(
    data: pl.DataFrame,
    *,
    selection=None,
    pagination=None,
    page_size=30,
    max_columns=1000,
    **kwargs,
) -> mo.ui.table:
    kwargs.setdefault("format_mapping", format_mapping_num_to_si(data))
    return mo.ui.table(
        data,
        selection=selection,
        pagination=pagination,
        page_size=page_size,
        max_columns=max_columns,
        **kwargs,
    )


def rich_md(
    *renderables: RenderableType,
    console_kwargs: dict[str, Any] | None = None,
    print_kwargs: dict[str, Any] | None = None,
    export_html_kwargs: dict[str, Any] | None = None,
) -> mo.Html:
    """
    A minimal helper function to utilize rich markup within marimo.

    Created this for shits-n-giggles. This is in no way complete! I
    don't use this as much as I would like to, however, in case you have
    some text, that you would like to keep consistent with your console
    markup, this might be an option.

    Parameters
    ----------
    *renderables : RenderableType
        The renderable objects to be printed using rich.
    console_kwargs : dict[str, Any] | None, default=None
        Keyword arguments to customize the rich Console.
    print_kwargs : dict[str, Any] | None, default=None
        Keyword arguments to customize the rich print method.
    export_html_kwargs : dict[str, Any] | None, default=None
        Keyword arguments to customize the rich export_html method.

    Returns
    -------
    mo.Html
        The HTML representation of the rich renderables.

    Examples
    --------
    >>> "font-weight: bold" in rich_md("[bold green]Hello[/] World").text
    True
    >>> "font-style: italic" in rich_md(
    ...     "[bold]Bold[italic] bold and italic [/bold]italic[/italic]"
    ... ).text
    True
    >>> "⚠️" in rich_md(":warning-emoji: you have been warned!").text
    True
    """
    console_kwargs = {
        "color_system": "truecolor",
        "record": True,
        **(console_kwargs or {}),
    }
    print_kwargs = print_kwargs or {}
    export_html_kwargs = {
        "inline_styles": True,
        "code_format": "{code}",
        **(export_html_kwargs or {}),
    }

    capture = io.StringIO()
    console = Console(file=capture, **console_kwargs)
    console.print(*renderables, **print_kwargs)
    return mo.Html(console.export_html(**export_html_kwargs))
