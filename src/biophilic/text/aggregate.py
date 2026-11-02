"""Core: turn mention rows into review-level indicators and hotel-level metrics.

Replaces the 2024 "Positive/Negative Feedback Number" counts, which counted one review once per matched
sentiment word (so long reviews counted many times) and had no denominator.

Hotel × aspect metrics:
    n_reviews        reviews with text for the hotel (denominator)
    mention_rate     share of those reviews that mention the aspect at least once
    pos, neg         reviews with ≥1 positive / negative mention (a review can be both)
    net_sentiment    (pos − neg) / (pos + neg), shrunk toward the corpus mean with `prior_strength`
                     pseudo-mentions so hotels with three mentions do not get ±1.
"""

from __future__ import annotations

import pandas as pd


def review_indicators(mentions: pd.DataFrame, reviews: pd.DataFrame) -> pd.DataFrame:
    """Wide table, one row per review: <aspect>_pos, <aspect>_neg (0/1). Reviews without mentions get zeros."""
    m = mentions[mentions["polarity"] != 0]
    flags = (m.assign(col=m["aspect"] + m["polarity"].map({1: "_pos", -1: "_neg"}))
              .drop_duplicates(["review_id", "col"])
              .assign(v=1)
              .pivot_table(index="review_id", columns="col", values="v", fill_value=0, aggfunc="max"))
    out = reviews[["review_id"]].merge(flags, left_on="review_id", right_index=True, how="left").fillna(0)
    cols = [c for c in out.columns if c != "review_id"]
    out[cols] = out[cols].astype(int)
    return out


def hotel_metrics(mentions: pd.DataFrame, reviews: pd.DataFrame, prior_strength: float = 10.0) -> pd.DataFrame:
    text_reviews = reviews[reviews["has_text"]]
    denom = text_reviews.groupby("hotel_id").size().rename("n_reviews")

    per_review = (mentions.groupby(["hotel_id", "aspect", "review_id"])["polarity"]
                  .agg(pos=lambda p: int((p > 0).any()), neg=lambda p: int((p < 0).any()))
                  .reset_index())
    g = per_review.groupby(["hotel_id", "aspect"]).agg(n_mention=("review_id", "nunique"), pos=("pos", "sum"),
                                                        neg=("neg", "sum")).reset_index()
    g = g.merge(denom, on="hotel_id", how="left")
    g["mention_rate"] = g["n_mention"] / g["n_reviews"]

    corpus = g.groupby("aspect")[["pos", "neg"]].sum()
    prior = ((corpus["pos"] - corpus["neg"]) / (corpus["pos"] + corpus["neg"]).clip(lower=1)).rename("prior")
    g = g.merge(prior, left_on="aspect", right_index=True)
    k = prior_strength
    g["net_sentiment_raw"] = (g["pos"] - g["neg"]) / (g["pos"] + g["neg"]).clip(lower=1)
    g["net_sentiment"] = (g["pos"] - g["neg"] + k * g["prior"]) / (g["pos"] + g["neg"] + k)
    return g.drop(columns="prior")


def wide(metrics: pd.DataFrame, value: str) -> pd.DataFrame:
    """hotel_id × aspect table of one metric."""
    return metrics.pivot(index="hotel_id", columns="aspect", values=value)
