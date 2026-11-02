"""Core: human gold set for the text methods.

1. `sample()` draws reviews for annotation: half from reviews where the lexicon (incl. ambiguous terms) fires,
   half at random, so both precision and recall can be estimated.
2. A person fills one column per aspect with "+", "-" or "0" (neutral mention); blank = not mentioned.
3. `evaluate()` scores any method (lexicon, llm) against the gold set at review × aspect × polarity level.

The annotation file contains review text, so it lives in data/interim (not in git).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

GOLD_CODES = {"+": 1, "-": -1, "0": 0}


def sample(reviews: pd.DataFrame, lexicon_hits: pd.DataFrame, aspects: list[str], n: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    pool = reviews[reviews["has_text"]]
    hit_ids = pd.unique(lexicon_hits["review_id"])
    k = n // 2
    a = rng.choice(hit_ids, size=min(k, len(hit_ids)), replace=False)
    rest = pool.loc[~pool["review_id"].isin(a), "review_id"].to_numpy()
    b = rng.choice(rest, size=min(n - len(a), len(rest)), replace=False)
    out = pool[pool["review_id"].isin(np.concatenate([a, b]))][["review_id", "hotel_id", "hotel", "title", "liked",
                                                                  "disliked"]].copy()
    out = out.sample(frac=1, random_state=seed).reset_index(drop=True)  # hide which stratum a row came from
    for asp in aspects:
        out[asp] = ""
    out["annotator"] = ""
    return out


def gold_long(annotated: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    rows = []
    for _, r in annotated.iterrows():
        for asp in aspects:
            v = str(r.get(asp, "")).strip()
            if v in GOLD_CODES:
                rows.append({"review_id": r["review_id"], "aspect": asp, "polarity": GOLD_CODES[v]})
    return pd.DataFrame(rows, columns=["review_id", "aspect", "polarity"])


def evaluate(mentions: pd.DataFrame, annotated: pd.DataFrame, aspects: list[str]) -> pd.DataFrame:
    """Precision / recall / F1 per aspect for detection (aspect mentioned?) and for polarity (sign correct?)."""
    ids = set(annotated["review_id"])
    gold = gold_long(annotated, aspects)
    pred = (mentions[mentions["review_id"].isin(ids)]
            .groupby(["review_id", "aspect"])["polarity"]
            .agg(lambda p: int(np.sign(p.sum()))).reset_index())
    rows = []
    for asp in aspects:
        g = gold[gold["aspect"] == asp].set_index("review_id")["polarity"]
        p = pred[pred["aspect"] == asp].set_index("review_id")["polarity"]
        tp = len(g.index.intersection(p.index))
        prec = tp / len(p) if len(p) else np.nan
        rec = tp / len(g) if len(g) else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec and rec else np.nan
        both = g.index.intersection(p.index)
        pol_acc = float((g.loc[both] == p.loc[both]).mean()) if len(both) else np.nan
        rows.append({"aspect": asp, "gold_n": len(g), "pred_n": len(p), "precision": prec, "recall": rec,
                     "f1": f1, "polarity_accuracy": pol_acc})
    return pd.DataFrame(rows)
