"""Core: small-sample inference helpers (bootstrap, permutation, FDR, LOOCV)."""

from __future__ import annotations

import numpy as np
from scipy import stats


def _clean_pair(x, y) -> tuple[np.ndarray, np.ndarray]:
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    ok = ~(np.isnan(x) | np.isnan(y))
    return x[ok], y[ok]


def correlation(x, y, method: str = "pearson", n_boot: int = 2000, n_perm: int = 5000, seed: int = 0) -> dict:
    """Correlation with a percentile bootstrap CI and a two-sided permutation p-value."""
    x, y = _clean_pair(x, y)
    n = len(x)
    f = (lambda a, b: stats.pearsonr(a, b)[0]) if method == "pearson" else (lambda a, b: stats.spearmanr(a, b)[0])
    if n < 4 or np.std(x) == 0 or np.std(y) == 0:
        return {"n": n, "r": np.nan, "ci_low": np.nan, "ci_high": np.nan, "p_perm": np.nan}
    r = f(x, y)
    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if np.std(x[i]) > 0 and np.std(y[i]) > 0:
            boots.append(f(x[i], y[i]))
    perm = np.array([f(x, rng.permutation(y)) for _ in range(n_perm)])
    p = (np.sum(np.abs(perm) >= abs(r)) + 1) / (n_perm + 1)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"n": n, "r": float(r), "ci_low": float(lo), "ci_high": float(hi), "p_perm": float(p)}


def bh_fdr(p) -> np.ndarray:
    """Benjamini–Hochberg adjusted p-values (NaN-safe)."""
    p = np.asarray(p, dtype=float)
    out = np.full_like(p, np.nan)
    ok = ~np.isnan(p)
    pv = p[ok]
    if pv.size == 0:
        return out
    order = np.argsort(pv)
    ranked = pv[order] * pv.size / (np.arange(pv.size) + 1)
    adj = np.minimum.accumulate(ranked[::-1])[::-1].clip(max=1)
    res = np.empty_like(pv)
    res[order] = adj
    out[ok] = res
    return out


def r2(y, yhat) -> float:
    y, yhat = np.asarray(y, float), np.asarray(yhat, float)
    return float(1 - ((y - yhat) ** 2).sum() / ((y - y.mean()) ** 2).sum())


def loocv_r2_ols(X, y) -> float:
    """Leave-one-out predictive R² of OLS with intercept (out-of-sample, can be negative)."""
    X = np.column_stack([np.ones(len(y)), np.asarray(X, float)])
    y = np.asarray(y, float)
    preds = np.empty_like(y)
    for i in range(len(y)):
        m = np.ones(len(y), bool)
        m[i] = False
        beta = np.linalg.lstsq(X[m], y[m], rcond=None)[0]
        preds[i] = X[i] @ beta
    return r2(y, preds)
