import polars.selectors as cs
from great_tables import GT


def gt_defaults(gtbl: GT):
    """Preferred defaults for rendering great tables."""
    gtbl = gtbl.fmt_number_si(columns=cs.numeric()).opt_stylize(style=4)
    return gtbl
