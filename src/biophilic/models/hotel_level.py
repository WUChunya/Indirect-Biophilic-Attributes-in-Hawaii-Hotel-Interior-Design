"""Core: hotel-level analysis. The unit is the hotel (n ≈ 36), never the hotel × space row.

The 2024 analysis ran correlations and regressions on 108 hotel × space rows (or 156 rows) whose outcome
(the hotel's Booking score) is copied three times. That inflates n and makes p-values far too small
(pseudo-replication). Everything here aggregates to one row per hotel first.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from . import stats


def associations(hotels: pd.DataFrame, predictors: list[str], outcomes: list[str], method: str = "spearman",
                 n_boot: int = 2000, n_perm: int = 5000, seed: int = 0) -> pd.DataFrame:
    rows = []
    for x in predictors:
        for y in outcomes:
            res = stats.correlation(hotels[x], hotels[y], method, n_boot, n_perm, seed)
            rows.append({"predictor": x, "outcome": y, "method": method, **res})
    out = pd.DataFrame(rows)
    out["p_fdr"] = stats.bh_fdr(out["p_perm"])
    return out


def _design(hotels: pd.DataFrame, outcome: str, index_col: str) -> pd.DataFrame:
    df = hotels[[outcome, index_col, "avg_price", "group"]].dropna().copy()
    df["log_price"] = np.log(df["avg_price"])
    df["lower_star"] = (df["group"] == "lower_star_test").astype(int)
    df["outside_waikiki"] = (df["group"] == "hawaii_4star_test").astype(int)
    return df


def index_models(hotels: pd.DataFrame, index_col: str, outcome: str = "booking_score") -> pd.DataFrame:
    """OLS with HC3 robust SE plus leave-one-hotel-out predictive R², for nested specifications.
    Group dummies are added only when the sample contains those groups."""
    df = _design(hotels, outcome, index_col)
    groups = [g for g in ("lower_star", "outside_waikiki") if df[g].nunique() > 1]
    specs = {
        "index only": [index_col],
        "price only": ["log_price"],
        "index + price": [index_col, "log_price"],
    }
    if groups:
        specs["price + group"] = ["log_price", *groups]
        specs["index + price + group"] = [index_col, "log_price", *groups]
    rows = []
    for name, cols in specs.items():
        fit = smf.ols(f"{outcome} ~ " + " + ".join(cols), data=df).fit(cov_type="HC3")
        row = {"model": name, "n": int(fit.nobs), "r2": fit.rsquared, "adj_r2": fit.rsquared_adj,
               "loocv_r2": stats.loocv_r2_ols(df[cols].values, df[outcome].values)}
        if index_col in cols:
            lo, hi = fit.conf_int().loc[index_col]
            row.update({"index_coef": fit.params[index_col], "index_ci_low": lo, "index_ci_high": hi,
                        "index_p": fit.pvalues[index_col]})
        rows.append(row)
    return pd.DataFrame(rows)


def external_test(hotels: pd.DataFrame, index_col: str, train_group: str, outcome: str = "booking_score") -> dict:
    """Fit outcome ~ index on the model hotels, predict the held-out test hotels (the thesis Ch. 6 design)."""
    train = hotels[hotels["group"] == train_group].dropna(subset=[outcome, index_col])
    test = hotels[hotels["group"] != train_group].dropna(subset=[outcome, index_col])
    fit = smf.ols(f"{outcome} ~ {index_col}", data=train).fit()
    pred = fit.predict(test)
    out = {"index": index_col, "n_train": len(train), "n_test": len(test),
           "train_r2": fit.rsquared, "test_r2": stats.r2(test[outcome], pred),
           "test_r": float(np.corrcoef(test[outcome], pred)[0, 1]),
           "test_mae": float((test[outcome] - pred).abs().mean())}
    for g, sub in test.groupby("group"):
        p = fit.predict(sub)
        out[f"mae_{g}"] = float((sub[outcome] - p).abs().mean())
        out[f"mean_residual_{g}"] = float((sub[outcome] - p).mean())
    return out


def item_associations(coding_rows: pd.DataFrame, hotels: pd.DataFrame, items: list[str],
                      outcome: str = "booking_score", alpha: float = 0.05) -> pd.DataFrame:
    """Each item vs the outcome, computed the 2024 way (Pearson on hotel × space rows, uncorrected) and the
    corrected way (Spearman on hotel means = share of spaces with the item, Benjamini–Hochberg FDR)."""
    from scipy import stats as sps

    rows = coding_rows.merge(hotels[["hotel_id", outcome]], on="hotel_id")
    per_hotel = rows.groupby("hotel_id")[[*items, outcome]].mean()
    out = []
    for it in items:
        r_rows, p_rows = sps.pearsonr(rows[it], rows[outcome]) if rows[it].std() > 0 else (np.nan, np.nan)
        r_h, p_h = sps.spearmanr(per_hotel[it], per_hotel[outcome]) if per_hotel[it].std() > 0 else (np.nan, np.nan)
        out.append({"item": it, "r_rows_2024": r_rows, "p_rows_2024": p_rows, "rho_hotels": r_h, "p_hotels": p_h})
    df = pd.DataFrame(out)
    df["p_hotels_fdr"] = stats.bh_fdr(df["p_hotels"])
    df["sig_2024"] = df["p_rows_2024"] < alpha
    df["sig_corrected"] = df["p_hotels_fdr"] < alpha
    return df
