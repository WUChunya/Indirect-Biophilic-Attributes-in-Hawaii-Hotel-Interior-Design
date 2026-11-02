"""Writer: markdown tables (no extra dependency) and the results report."""

from __future__ import annotations

import math

import pandas as pd


def _fmt(v, digits: int) -> str:
    if v is None or (isinstance(v, float) and math.isnan(v)):
        return ""
    if isinstance(v, bool):
        return "yes" if v else ""
    if isinstance(v, float):
        if v != 0 and abs(v) < 10 ** (-digits):
            return f"{v:.1e}"
        return f"{v:.{digits}f}"
    return str(v)


def md_table(df: pd.DataFrame, digits: int = 3) -> str:
    cols = list(df.columns)
    lines = ["| " + " | ".join(map(str, cols)) + " |", "|" + "---|" * len(cols)]
    for _, r in df.iterrows():
        lines.append("| " + " | ".join(_fmt(r[c], digits) for c in cols) + " |")
    return "\n".join(lines)
