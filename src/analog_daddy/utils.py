"""Miscellaneous utilities."""

import re
from collections.abc import Iterator
from pathlib import Path

from quantiphy import render as _render


def scan_contents(
    contents: bytes | str | Path,
    search_pattern: str,
) -> Iterator[int]:
    r"""
    Yield line numbers matching a search pattern.

    Contents are scanned lazily where possible (files are read line by
    line).

    Parameters
    ----------
    contents : bytes | str | Path
        Text to scan, or a path to a file to scan.
    search_pattern : str
        Regex pattern searched for in each line.

    Yields
    ------
    int
        1-based line number of each matching line.

    Raises
    ------
    TypeError
        If ``contents`` is not bytes, str, or Path.

    Examples
    --------
    >>> list(
    ...     scan_contents(
    ...         (b"Line 1\nLine 2\nLine 3"),
    ...         r"Line 2",
    ...     )
    ... )
    [2]
    >>> list(
    ...     scan_contents(
    ...         ("Line 1\nLine 2\n Line 3 \n"),
    ...         "^Line 3$",
    ...     )
    ... )
    []
    """
    pattern = re.compile(search_pattern)

    def search(_lines):
        for lineno, line in enumerate(_lines, start=1):
            if pattern.search(line):
                yield lineno

    if isinstance(contents, bytes):
        yield from search(contents.decode().splitlines())
    elif isinstance(contents, str):
        yield from search(contents.splitlines())
    elif isinstance(contents, Path):
        with contents.open() as lines:
            yield from search(lines)
    else:
        raise TypeError(f"Unsupported type: {type(contents)}")
        # TODO: Runtime check needed?


def render(value: float | None, units: str = "") -> str:
    """
    Render a value with SI prefix, or ``""`` for None.

    Thin wrapper over ``quantiphy.render`` so tables and displays can
    show missing values as blanks.

    Parameters
    ----------
    value : float | None
        Value to render.
    units : str, default=""
        Units appended to the rendered value.

    Returns
    -------
    str
        Rendered value with SI prefix and units, or ``""`` if ``value``
        is None.

    Examples
    --------
    >>> render(1e-6, "m")
    '1 um'
    >>> render(1e-6)
    '1u'
    >>> render(None, "m")
    ''
    >>> render(None)
    ''
    """  # noqa: DOC105
    if value is None:
        return ""
    else:
        return _render(value, units=units)
