"""Core: clean the review table and decide which properties enter the analysis. Pure pandas, no I/O."""

from __future__ import annotations

import re

import numpy as np
import pandas as pd


def _missing(s: object) -> bool:
    return s is None or (isinstance(s, float) and np.isnan(s)) or s is pd.NA


def _map_text(series: pd.Series, fn) -> pd.Series:
    """Apply fn to each value and keep an object column (pandas 3 would turn None into NaN in string columns)."""
    return pd.Series([fn(None if _missing(v) else v) for v in series], index=series.index, dtype=object)


def normalise_text(s: object) -> str | None:
    if _missing(s):
        return None
    s = re.sub(r"\s+", " ", str(s)).strip()
    return s or None


def is_ascii_english(s: str | None, min_tokens: int = 3) -> bool:
    """Cheap language filter for the lexicon baseline: mostly ASCII letters and at least `min_tokens` words.
    The LLM method handles every language, so this only decides what the baseline can read."""
    if not s:
        return False
    letters = [c for c in s if c.isalpha()]
    if not letters:
        return False
    ascii_share = sum(c.isascii() for c in letters) / len(letters)
    return ascii_share > 0.9 and len(s.split()) >= min_tokens


def clean_reviews(reviews: pd.DataFrame, aspects_cfg: dict, min_tokens: int = 3) -> pd.DataFrame:
    """Add text-quality flags; drop exact duplicates. Rows are kept (rating models use reviews without text)."""
    df = reviews.copy()
    placeholders = {p.lower() for p in aspects_cfg["placeholders"]}
    nulls = {n.lower() for n in aspects_cfg["null_answers"]}
    auto_titles = {t.lower() for t in aspects_cfg["auto_titles"]}

    for col in ("title", "liked", "disliked"):
        df[col] = _map_text(df[col], normalise_text)

    def strip_null(s: str | None, null_set: set[str]) -> str | None:
        if s is None:
            return None
        key = s.lower().strip(" .!-")
        return None if key in null_set or s.lower() in placeholders else s

    df["liked"] = _map_text(df["liked"], lambda s: strip_null(s, nulls))
    df["disliked"] = _map_text(df["disliked"], lambda s: strip_null(s, nulls))
    df["title_is_auto"] = [s is None or s.lower() in auto_titles for s in df["title"]]
    df["title"] = _map_text(df["title"], lambda s: None if s is None or s.lower() in auto_titles else s)

    df["has_text"] = df["liked"].notna() | df["disliked"].notna() | df["title"].notna()
    joined = [" ".join(x for x in (a, b, c) if x) for a, b, c in zip(df["liked"], df["disliked"], df["title"])]
    df["english"] = [is_ascii_english(s, min_tokens) for s in joined]
    df["n_words"] = [len(s.split()) for s in joined]

    before = len(df)
    df = df.drop_duplicates("review_id").reset_index(drop=True)
    df.attrs["duplicates_dropped"] = before - len(df)
    return df


def eligible_hotels(reviews: pd.DataFrame, min_text_reviews: int, listing_patterns: list[str]) -> pd.DataFrame:
    """Per-property counts and an `eligible` flag (enough text reviews and not a private listing)."""
    counts = reviews.groupby(["hotel_id", "hotel"], sort=False).agg(
        n_reviews=("review_id", "size"),
        n_text=("has_text", "sum"),
        n_english=("english", "sum"),
        mean_rating=("rating", "mean"),
    ).reset_index()
    rx = re.compile("|".join(listing_patterns), flags=re.IGNORECASE)
    counts["is_listing"] = [bool(rx.search(str(n))) for n in counts["hotel"]]
    counts["eligible"] = (counts["n_text"] >= min_text_reviews) & ~counts["is_listing"]
    return counts
