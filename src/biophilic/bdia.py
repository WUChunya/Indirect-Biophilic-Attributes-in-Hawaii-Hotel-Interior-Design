"""Core: BDIA scenario scores, hotel index, reliability, and the legacy (2024) formulas for audit.

New indices are defined without looking at any outcome, so they can be correlated with Booking scores
without circularity:
    equal_items         share of the 29 items present in a space
    dimension_balanced  mean of the 7 dimension shares (each dimension A–G counts equally, regardless of
                        how many items it has; default)
Legacy indices reproduce the thesis numbers exactly and are only used in the audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def item_codes() -> list[str]:
    return [it.code for it in config.bdia_items()]


def dimension_of() -> dict[str, str]:
    return {it.code: it.dimension for it in config.bdia_items()}


def scenario_scores(coding: pd.DataFrame, method: str = "dimension_balanced",
                    weights: dict[str, float] | None = None) -> pd.Series:
    """Score per hotel × space row. method ∈ {equal_items, dimension_balanced, weighted}."""
    codes = item_codes()
    X = coding[codes].astype(float)
    if method == "equal_items":
        return X.mean(axis=1)
    if method == "dimension_balanced":
        dims = dimension_of()
        per_dim = X.T.groupby(pd.Series(dims)).mean().T
        return per_dim.mean(axis=1)
    if method == "weighted":
        if weights is None:
            raise ValueError("weights required for method='weighted'")
        w = pd.Series({c: weights.get(c, 0.0) for c in codes})
        return X @ w
    raise ValueError(f"unknown method {method!r}")


def dimension_scores(coding: pd.DataFrame) -> pd.DataFrame:
    """Hotel × dimension share of items present, averaged over the three spaces."""
    codes = item_codes()
    dims = pd.Series(dimension_of())
    per_row = coding[codes].astype(float).T.groupby(dims).mean().T
    per_row["hotel_id"] = coding["hotel_id"].values
    return per_row.groupby("hotel_id").mean()


def hotel_index(coding: pd.DataFrame, scores: pd.Series, space_weights: dict[str, float]) -> pd.Series:
    df = pd.DataFrame({"hotel_id": coding["hotel_id"], "space": coding["space"], "s": scores.values})
    df["w"] = df["space"].map(space_weights)
    return df.groupby("hotel_id").apply(lambda g: float((g["s"] * g["w"]).sum() / g["w"].sum()),
                                        include_groups=False)


def legacy_reproduction(coding: pd.DataFrame, hotels: pd.DataFrame) -> dict[str, float]:
    """Max absolute error when the stored legacy columns are recomputed from config/legacy.yaml."""
    leg = config.load("legacy")
    scen = scenario_scores(coding, "weighted", leg["spreadsheet_item_weights"])
    idx = hotel_index(coding, scen, leg["spreadsheet_space_weights"])
    # Stored index is a weighted SUM (weights add to 1.0), identical to the weighted mean used above.
    pred = leg["spreadsheet_prediction"]["intercept"] + leg["spreadsheet_prediction"]["slope"] * idx
    h = hotels.set_index("hotel_id")
    return {
        "scenario_max_abs_err": float((scen - coding["legacy_scenario"]).abs().max()),
        "index_max_abs_err": float((idx - h["legacy_index"]).abs().max()),
        "prediction_max_abs_err": float((pred - h["legacy_prediction"]).abs().max()),
        "item_weight_sum": float(sum(leg["spreadsheet_item_weights"].values())),
    }


def kr20(X: pd.DataFrame | np.ndarray) -> float:
    """Kuder–Richardson 20 internal consistency for binary items (NaN if < 2 items or zero variance)."""
    X = np.asarray(X, dtype=float)
    k = X.shape[1]
    if k < 2:
        return float("nan")
    total_var = X.sum(axis=1).var()  # population variance, consistent with p(1 - p) below
    if total_var == 0:
        return float("nan")
    p = X.mean(axis=0)
    return float(k / (k - 1) * (1 - (p * (1 - p)).sum() / total_var))


def reliability(coding: pd.DataFrame) -> pd.DataFrame:
    """KR-20 per dimension and for all items, over hotel × space rows."""
    dims = dimension_of()
    rows = []
    for d, name in config.bdia_dimensions().items():
        cols = [c for c, dd in dims.items() if dd == d]
        rows.append({"dimension": d, "name": name, "n_items": len(cols), "kr20": kr20(coding[cols])})
    rows.append({"dimension": "ALL", "name": "All 29 items", "n_items": len(dims), "kr20": kr20(coding[list(dims)])})
    return pd.DataFrame(rows)


def prevalence(coding: pd.DataFrame) -> pd.DataFrame:
    """Share of hotels with each item present, per space. Items present in >90% or <10% carry little signal."""
    codes = item_codes()
    return coding.groupby("space")[codes].mean().T
