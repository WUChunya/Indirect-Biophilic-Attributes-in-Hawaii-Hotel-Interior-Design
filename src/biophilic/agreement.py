"""Core: inter-rater agreement for binary BDIA codes (human vs human, human vs VLM).

Cohen's kappa collapses when an item is almost always present (e.g. B05 wooden furniture, 91%), so Gwet's AC1
is reported next to it; it is the recommended statistic for skewed prevalence.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cohen_kappa(a, b) -> float:
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    po = float((a == b).mean())
    pa, pb = a.mean(), b.mean()
    pe = pa * pb + (1 - pa) * (1 - pb)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def gwet_ac1(a, b) -> float:
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    po = float((a == b).mean())
    pi = (a.mean() + b.mean()) / 2
    pe = 2 * pi * (1 - pi)
    return float("nan") if pe == 1 else (po - pe) / (1 - pe)


def agreement_table(left: pd.DataFrame, right: pd.DataFrame, items: list[str],
                    keys: tuple[str, ...] = ("hotel_id", "space")) -> pd.DataFrame:
    """Per-item agreement between two coders. Both frames: one row per keys, item columns 0/1."""
    m = left[[*keys, *items]].merge(right[[*keys, *items]], on=list(keys), suffixes=("_a", "_b"))
    rows = []
    for it in items:
        a, b = m[f"{it}_a"], m[f"{it}_b"]
        rows.append({"item": it, "n": len(m), "prevalence_a": a.mean(), "prevalence_b": b.mean(),
                     "percent_agreement": float((a == b).mean()), "cohen_kappa": cohen_kappa(a, b),
                     "gwet_ac1": gwet_ac1(a, b)})
    return pd.DataFrame(rows)
