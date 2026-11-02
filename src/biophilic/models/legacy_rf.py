"""Core: replicate the 2024 Random Forest and show where its accuracy came from.

2024 procedure (EDRA55 slides): rows = hotel × space (108), features = 0/1 BDIA items, target = the hotel's
Booking Overall Score, random 90/10 row split, RF with 100 trees → test R² 0.732. Feature importances were
then used as item weights in the BDIA-Index, which was correlated with (and regressed on) the same score.

Two leaks:
  1. Row split: the other two spaces of a test hotel are in training, with the identical target value.
     → compare with GroupKFold by hotel.
  2. Circular weights: importances are learned from the outcome on all data, then the index built from them
     is "validated" against that outcome. → compare with nested weights (learned on training hotels only)
     and with a permutation null (shuffle scores across hotels, rerun the whole procedure).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GroupKFold, train_test_split

from .stats import r2


def _rf(cfg: dict, seed: int) -> RandomForestRegressor:
    return RandomForestRegressor(n_estimators=cfg["n_estimators"], criterion=cfg["criterion"],
                                 min_samples_split=cfg["min_samples_split"], min_samples_leaf=cfg["min_samples_leaf"],
                                 max_depth=cfg["max_depth"], max_features=cfg["max_features"],
                                 bootstrap=cfg["bootstrap"], random_state=seed, n_jobs=-1)


def random_row_split(X, y, cfg: dict, seeds: range) -> np.ndarray:
    """Test R² of the 2024 random 90/10 row split, one value per seed."""
    out = []
    for s in seeds:
        Xtr, Xte, ytr, yte = train_test_split(X, y, train_size=cfg["train_fraction"], random_state=s)
        out.append(r2(yte, _rf(cfg, s).fit(Xtr, ytr).predict(Xte)))
    return np.array(out)


def grouped_cv(X, y, groups, cfg: dict, n_splits: int, seed: int) -> float:
    """Out-of-fold R² when whole hotels are held out (predictions pooled over folds)."""
    pred = np.empty(len(y))
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        pred[te] = _rf(cfg, seed).fit(X[tr], y[tr]).predict(X[te])
    return r2(y, pred)


def _index_from_importances(X, w, groups) -> pd.Series:
    """Hotel index = mean over the hotel's rows of Σ w·x (equal space weights)."""
    return pd.Series(X @ w).groupby(np.asarray(groups)).mean()


def circular_vs_nested(X, y, groups, cfg: dict, n_splits: int, seed: int) -> dict:
    """Correlation of the RF-weighted index with the hotel score: in-sample (2024) vs nested out-of-fold."""
    groups = np.asarray(groups)
    hotel_y = pd.Series(y).groupby(groups).first()
    w_all = _rf(cfg, seed).fit(X, y).feature_importances_
    idx_in = _index_from_importances(X, w_all, groups)
    r_in = float(np.corrcoef(idx_in.loc[hotel_y.index], hotel_y)[0, 1])

    oof = pd.Series(index=hotel_y.index, dtype=float)
    for tr, te in GroupKFold(n_splits=n_splits).split(X, y, groups):
        w = _rf(cfg, seed).fit(X[tr], y[tr]).feature_importances_
        part = _index_from_importances(X[te], w, groups[te])
        oof.loc[part.index] = part.values
    r_nested = float(np.corrcoef(oof.loc[hotel_y.index], hotel_y)[0, 1])
    return {"r_in_sample": r_in, "r_nested": r_nested}


def permutation_null(X, y, groups, cfg: dict, n_perm: int, seed: int) -> np.ndarray:
    """Rerun the in-sample 2024 procedure after shuffling hotel scores across hotels.
    If the index still 'correlates', the correlation is produced by the procedure, not by design."""
    groups = np.asarray(groups)
    hotels = pd.unique(groups)
    hotel_y = pd.Series(y).groupby(groups).first()
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n_perm):
        shuffled = pd.Series(rng.permutation(hotel_y.loc[hotels].values), index=hotels)
        y_perm = shuffled.loc[groups].values
        w = _rf(cfg, seed + i).fit(X, y_perm).feature_importances_
        idx = _index_from_importances(X, w, groups)
        out.append(np.corrcoef(idx.loc[hotels], shuffled.loc[hotels])[0, 1])
    return np.array(out)
