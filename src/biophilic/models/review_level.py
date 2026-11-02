"""Core: review-level model. n = reviews; hotel fixed effects absorb everything constant within a hotel
(location, brand, price tier, the BDIA coding itself), so the coefficients answer:

    "Within the same hotel, is a guest who praises / complains about aspect X giving a different score?"

This is the right use of review text for design questions: it measures which design aspects guests notice
and how much they move satisfaction, without pretending that 24–36 hotels can support item-level ML.
Standard errors are clustered by hotel.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf


def fit(reviews: pd.DataFrame, indicators: pd.DataFrame, aspects: list[str], min_mentions: int = 30) -> pd.DataFrame:
    df = reviews[["review_id", "hotel_id", "rating", "traveller_type", "review_date", "n_words"]].merge(
        indicators, on="review_id")
    df = df.dropna(subset=["rating"]).copy()
    df["year"] = df["review_date"].dt.year.fillna(0).astype(int).astype(str)
    df["traveller_type"] = df["traveller_type"].fillna("unknown")
    df["log_words"] = np.log1p(df["n_words"])

    terms = []
    for a in aspects:
        for side in ("pos", "neg"):
            col = f"{a}_{side}"
            if col in df and df[col].sum() >= min_mentions:
                terms.append(col)
    formula = "rating ~ " + " + ".join(terms + ["log_words", "C(traveller_type)", "C(year)", "C(hotel_id)"])
    res = smf.ols(formula, data=df).fit(cov_type="cluster", cov_kwds={"groups": df["hotel_id"]})
    ci = res.conf_int()
    out = pd.DataFrame({
        "indicator": terms,
        "n_reviews_with": [int(df[t].sum()) for t in terms],
        "coef": [res.params[t] for t in terms],
        "ci_low": [ci.loc[t, 0] for t in terms],
        "ci_high": [ci.loc[t, 1] for t in terms],
        "p": [res.pvalues[t] for t in terms],
    })
    out.attrs.update({"n": int(res.nobs), "n_hotels": int(df["hotel_id"].nunique()), "r2": float(res.rsquared)})
    return out
