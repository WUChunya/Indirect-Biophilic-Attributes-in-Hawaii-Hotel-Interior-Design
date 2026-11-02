"""Stage orchestration. Each stage reads config, calls core functions, and writes files; no analysis logic here."""

from __future__ import annotations

import datetime as dt
import json

import numpy as np
import pandas as pd

from . import __version__, bdia, clean, config, io
from .models import hotel_level, legacy_rf, review_level
from .models.stats import bh_fdr, correlation
from .report import md_table
from .text import aggregate
from .text.lexicon import LexiconExtractor

BDIA_ASPECTS = ["images_of_nature", "natural_materials", "natural_colors", "natural_light_air",
                "naturalistic_shapes", "age_patina", "natural_geometries"]
ASPECT_TO_DIM = dict(zip(BDIA_ASPECTS, "ABCDEFG"))


def ingest() -> dict:
    """Read raw files, clean reviews, write data/processed/*.parquet (contains review text: not in git)."""
    pcfg = config.load("pipeline")
    acfg = config.load("review_aspects")
    reviews_raw, booking_hotels = io.read_reviews()
    reviews = clean.clean_reviews(reviews_raw, acfg, pcfg["text"]["min_tokens_for_english"])
    hotels_text = clean.eligible_hotels(reviews, pcfg["ingest"]["min_text_reviews_per_hotel"],
                                        pcfg["ingest"]["listing_name_patterns"])
    coding, bdia_hotels = io.read_bdia()
    out = config.path("processed")
    out.mkdir(parents=True, exist_ok=True)
    reviews.to_parquet(out / "reviews.parquet")
    booking_hotels.merge(hotels_text, on=["hotel_id", "hotel"]).to_parquet(out / "booking_hotels.parquet")
    coding.to_parquet(out / "bdia_coding.parquet")
    bdia_hotels.to_parquet(out / "bdia_hotels.parquet")
    (out / "ingest_meta.json").write_text(json.dumps({"duplicates_dropped": reviews.attrs["duplicates_dropped"]}))
    return {"reviews": reviews, "booking_hotels": booking_hotels.merge(hotels_text, on=["hotel_id", "hotel"]),
            "coding": coding, "bdia_hotels": bdia_hotels, "duplicates_dropped": reviews.attrs["duplicates_dropped"]}


def text_baseline(reviews: pd.DataFrame, booking_hotels: pd.DataFrame) -> pd.DataFrame:
    acfg = config.load("review_aspects")
    ex = LexiconExtractor(acfg, include_ambiguous=True, mask_names=booking_hotels["hotel"].tolist())
    mentions = ex.extract(reviews[reviews["english"]])
    mentions.to_parquet(config.path("processed") / "mentions_lexicon.parquet")
    return mentions


def _hotel_table(coding: pd.DataFrame, bdia_hotels: pd.DataFrame) -> pd.DataFrame:
    h = bdia_hotels.copy()
    w = config.load("pipeline")["bdia"]["space_weights"]
    for method in ("equal_items", "dimension_balanced"):
        h[f"bdia_{method}"] = h["hotel_id"].map(bdia.hotel_index(coding, bdia.scenario_scores(coding, method), w))
    dims = bdia.dimension_scores(coding).add_prefix("dim_")
    return h.merge(dims, left_on="hotel_id", right_index=True)


def analyse(state: dict, mentions: pd.DataFrame) -> dict:
    pcfg = config.load("pipeline")
    mcfg = pcfg["models"]
    seed, nb, npm = mcfg["random_seed"], mcfg["bootstrap_reps"], mcfg["permutation_reps"]
    reviews, coding, bh = state["reviews"], state["coding"], state["bdia_hotels"]
    hotels = _hotel_table(coding, bh)
    model_group = "waikiki_4star_model"
    model_ids = set(hotels.loc[hotels["group"] == model_group, "hotel_id"])
    res: dict = {"hotels": hotels}

    # --- 1. legacy reproduction and Random Forest audit (thesis setup: 26 model hotels × 3 spaces) -------
    res["legacy_repro"] = bdia.legacy_reproduction(coding, bh)
    rf_cfg = config.load("legacy")["random_forest"]
    d = coding[coding["hotel_id"].isin(model_ids)].merge(bh[["hotel_id", "booking_score"]], on="hotel_id")
    X, y, g = d[bdia.item_codes()].to_numpy(float), d["booking_score"].to_numpy(float), d["hotel_id"].to_numpy()
    split = legacy_rf.random_row_split(X, y, rf_cfg, range(seed, seed + 100))
    null = legacy_rf.permutation_null(X, y, g, rf_cfg, 200, seed)
    res["rf_audit"] = {
        "rows": int(len(y)), "hotels": int(len(set(g))),
        "reported_test_r2": rf_cfg["reported"]["test_r2"],
        "row_split_test_r2_median": float(np.median(split)),
        "row_split_test_r2_p10": float(np.percentile(split, 10)),
        "row_split_test_r2_p90": float(np.percentile(split, 90)),
        "row_split_share_ge_reported": float((split >= rf_cfg["reported"]["test_r2"]).mean()),
        "grouped_cv_r2": legacy_rf.grouped_cv(X, y, g, rf_cfg, mcfg["cv_folds"], seed),
        **legacy_rf.circular_vs_nested(X, y, g, rf_cfg, mcfg["cv_folds"], seed),
        "null_r_mean": float(null.mean()), "null_r_p95": float(np.percentile(null, 95)),
    }
    model_coding = coding[coding["hotel_id"].isin(model_ids)]
    res["items"] = hotel_level.item_associations(model_coding, bh, bdia.item_codes(), alpha=mcfg["fdr_alpha"])

    # --- 2. outcome-blind indices vs Booking scores (n = hotels) ---------------------------------------
    idx_cols = ["bdia_dimension_balanced", "bdia_equal_items", "legacy_index"]
    outcomes = [o for o in mcfg["hotel_outcomes"] if o in hotels]
    frames = []
    for label, sub in (("26 Waikiki model hotels", hotels[hotels["group"] == model_group]), ("all 36 hotels", hotels)):
        frames.append(hotel_level.associations(sub, idx_cols, outcomes, "spearman", nb, npm, seed).assign(sample=label))
    res["index_assoc"] = pd.concat(frames, ignore_index=True)
    res["index_models_26"] = hotel_level.index_models(hotels[hotels["group"] == model_group], "bdia_dimension_balanced")
    res["index_models_36"] = hotel_level.index_models(hotels, "bdia_dimension_balanced")
    res["external_test"] = pd.DataFrame([hotel_level.external_test(hotels, c, model_group) for c in idx_cols])
    res["dim_assoc"] = hotel_level.associations(hotels[hotels["group"] == model_group],
                                                [f"dim_{k}" for k in "ABCDEFG"], ["booking_score"],
                                                "spearman", nb, npm, seed)
    res["reliability"] = bdia.reliability(coding)
    res["prevalence"] = bdia.prevalence(coding)

    # --- 3. review text ----------------------------------------------------------------------------------
    hotel_ok = state["booking_hotels"].loc[state["booking_hotels"]["eligible"], "hotel_id"]
    rv = reviews[reviews["hotel_id"].isin(hotel_ok)]
    strict = mentions[(mentions["method"] == "lexicon") & mentions["hotel_id"].isin(hotel_ok)]
    res["mention_summary"] = (mentions.groupby(["aspect", "method"]).size().unstack(fill_value=0)
                              .reset_index().rename_axis(columns=None))
    metrics = aggregate.hotel_metrics(strict, rv, pcfg["text"]["sentiment_prior_strength"])
    res["text_metrics"] = metrics
    indicators = aggregate.review_indicators(strict, rv[rv["english"]])
    all_aspects = list(config.load("review_aspects")["aspects"])
    res["review_model"] = review_level.fit(rv[rv["english"]], indicators, all_aspects)

    # --- 4. convergence: do guests talk about what the photos show? (hotels in both datasets) -------------
    both = hotels[hotels["hotel_id"].isin(hotel_ok)]
    n_text = rv[rv["english"]].groupby("hotel_id").size()
    conv = []
    for asp in BDIA_ASPECTS:
        dim = f"dim_{ASPECT_TO_DIM[asp]}"
        sub = both[["hotel_id", dim]].merge(metrics[metrics["aspect"] == asp], on="hotel_id", how="left")
        # Hotels without any mention have mention rate 0 (not missing); their net sentiment is undefined.
        sub["n_mention"] = sub["n_mention"].fillna(0)
        sub["mention_rate"] = sub["n_mention"] / sub["hotel_id"].map(n_text)
        enough = int((sub["n_mention"] > 0).sum()) >= pcfg["text"]["min_hotels_with_mentions"]
        for metric in ("mention_rate", "net_sentiment"):
            c = (correlation(sub[dim], sub[metric], "spearman", nb, npm, seed) if enough else
                 {"n": int(sub[metric].notna().sum()), "r": np.nan, "ci_low": np.nan, "ci_high": np.nan,
                  "p_perm": np.nan})
            conv.append({"aspect": asp, "metric": metric, "mentions": int(sub["n_mention"].sum()), **c})
    conv = pd.DataFrame(conv)
    conv["p_fdr"] = bh_fdr(conv["p_perm"])
    res["convergence"] = conv
    res["n"] = {
        "reviews_raw": int(len(reviews) + state["duplicates_dropped"]),
        "duplicates_dropped": int(state["duplicates_dropped"]),
        "reviews_with_text": int(reviews["has_text"].sum()),
        "reviews_english": int(reviews["english"].sum()),
        "placeholder_or_empty": int((~reviews["has_text"]).sum()),
        "properties": int(reviews["hotel_id"].nunique()),
        "eligible_hotels": int(len(hotel_ok)),
        "bdia_hotels": int(len(bh)),
        "bdia_groups": bh["group"].value_counts().to_dict(),
        "bdia_hotels_spaces_identical": int(bh["spaces_identical"].sum()),
        "overlap_hotels": int(len(both)),
    }
    return res


def write_results(res: dict) -> None:
    out = config.path("results")
    out.mkdir(parents=True, exist_ok=True)
    tables = ["index_assoc", "index_models_26", "index_models_36", "external_test", "dim_assoc", "items",
              "reliability", "mention_summary", "review_model", "convergence"]
    for t in tables:
        res[t].to_csv(out / f"{t}.csv", index=False)
    res["prevalence"].to_csv(out / "prevalence.csv")
    res["text_metrics"].to_csv(out / "text_metrics_by_hotel.csv", index=False)
    keep = ["hotel_id", "hotel", "hotel_number", "group", "spaces_identical", "booking_score", "avg_price", "legacy_index",
            "bdia_dimension_balanced", "bdia_equal_items", *[f"dim_{k}" for k in "ABCDEFG"]]
    res["hotels"][keep].to_csv(out / "hotels.csv", index=False)
    (out / "summary.json").write_text(json.dumps({"version": __version__, "n": res["n"],
                                                   "legacy_repro": res["legacy_repro"],
                                                   "rf_audit": res["rf_audit"]}, indent=2))
    (out / "REPORT.md").write_text(render_report(res), encoding="utf-8")


def render_report(res: dict) -> str:
    n, rf, lr = res["n"], res["rf_audit"], res["legacy_repro"]
    items, rm, ia = res["items"], res["review_model"], res["index_assoc"]
    ia_main = ia[ia["outcome"] == "booking_score"]
    groups = ", ".join(f"{k} {v}" for k, v in n["bdia_groups"].items())
    ext = res["external_test"]
    return f"""# Baseline results (v{__version__}, {dt.date.today().isoformat()})

Generated by `biophilic baseline` from the 2024 data. Method: `docs/PRD.md` §5–§6. Audit: `docs/LEGACY_AUDIT.md`.
Hotel-level statistics use one row per hotel; p-values are permutation p-values (`p_perm`), `p_fdr` is
Benjamini–Hochberg adjusted over the table it appears in. No review text is written to this folder.

## 1. Data
| quantity | value |
|---|---|
| review rows in the raw file | {n['reviews_raw']} |
| exact duplicates removed | {n['duplicates_dropped']} |
| reviews with real text (not Booking's placeholder, not "nothing") | {n['reviews_with_text']} |
| … of which readable by the English lexicon | {n['reviews_english']} |
| properties in the review file / eligible hotels (≥ 50 text reviews, not a private listing) | {n['properties']} / {n['eligible_hotels']} |
| hotels with BDIA coding, by thesis group | {groups} |
| BDIA hotels whose three spaces have identical codes (likely coded once) | {n['bdia_hotels_spaces_identical']} |
| BDIA hotels that also have eligible reviews | {n['overlap_hotels']} |

## 2. Legacy reproduction
The 2024 spreadsheet columns are reproduced exactly from `config/legacy.yaml`
(max abs. error: scenario {lr['scenario_max_abs_err']:.1e}, index {lr['index_max_abs_err']:.1e},
prediction {lr['prediction_max_abs_err']:.1e}).

## 3. Random Forest audit ({rf['hotels']} model hotels × 3 spaces = {rf['rows']} rows, settings of thesis Table 15)
| check | value |
|---|---|
| reported test R² (thesis Table 16, one random 90/10 row split) | {rf['reported_test_r2']:.3f} |
| median test R² over 100 random row splits | {rf['row_split_test_r2_median']:.3f} (10th–90th pct {rf['row_split_test_r2_p10']:.2f} to {rf['row_split_test_r2_p90']:.2f}) |
| share of random splits reaching the reported value | {rf['row_split_share_ge_reported']:.0%} |
| R² when whole hotels are held out (GroupKFold) | {rf['grouped_cv_r2']:.3f} |
| r(RF-weighted index, score), weights learned on all data (thesis) | {rf['r_in_sample']:.3f} |
| r(RF-weighted index, score), weights learned without the test hotels | {rf['r_nested']:.3f} |
| thesis procedure on shuffled scores: mean r / 95th percentile | {rf['null_r_mean']:.3f} / {rf['null_r_p95']:.3f} |

## 4. Outcome-blind BDIA index vs Booking score (n = hotels)
`bdia_dimension_balanced` = mean of the 7 dimension shares, equal space weights; it never sees an outcome.
{md_table(ia_main[['sample', 'predictor', 'n', 'r', 'ci_low', 'ci_high', 'p_perm', 'p_fdr']])}

OLS with HC3 robust SE, 26 model hotels; `loocv_r2` = leave-one-hotel-out predictive R²:
{md_table(res['index_models_26'])}

All 36 hotels, with star-class / location group dummies:
{md_table(res['index_models_36'])}

External test (thesis Ch. 6 design): fit score ~ index on the 26 model hotels, predict the 10 test hotels.
{md_table(ext)}

All Booking sub-scores (Spearman): `results/index_assoc.csv`. Per dimension, 26 model hotels:
{md_table(res['dim_assoc'][['predictor', 'n', 'r', 'ci_low', 'ci_high', 'p_perm', 'p_fdr']])}

## 5. Items: thesis method vs corrected (26 model hotels)
Items "significant" for Booking score: **{int(items['sig_2024'].sum())} of {len(items)}** with Pearson on hotel × space
rows and no correction (as in the thesis) vs **{int(items['sig_corrected'].sum())} of {len(items)}** at hotel level
with FDR control.

{md_table(items[['item', 'r_rows_2024', 'p_rows_2024', 'rho_hotels', 'p_hotels_fdr', 'sig_2024', 'sig_corrected']])}

Internal consistency (KR-20 over hotel × space rows, all 36 hotels):
{md_table(res['reliability'])}

## 6. Review text (rule-based baseline)
`lexicon` = high-precision terms; `lexicon_ambiguous` = terms that the 2024 method counted but that mostly mean
something else in hotel reviews (not used in any model below).
{md_table(res['mention_summary'], 0)}

Review-level model: rating (1–10) on aspect praise / complaint indicators with hotel fixed effects, traveller
type, year and log length; SE clustered by hotel (n = {rm.attrs['n']} reviews, {rm.attrs['n_hotels']} hotels,
R² = {rm.attrs['r2']:.3f}). Indicators with < 30 reviews are not estimated.
{md_table(rm)}

## 7. Convergence: do guests talk about what the photos show? (BDIA hotels with eligible reviews)
Spearman between the hotel's BDIA dimension score and the matching review metric. Hotels with no mention have
mention rate 0 and no net sentiment; aspects mentioned in fewer than `min_hotels_with_mentions` hotels are not tested.
{md_table(res['convergence'][['aspect', 'metric', 'mentions', 'n', 'r', 'ci_low', 'ci_high', 'p_perm', 'p_fdr']])}

## 8. Reading these results
Interpretation and limits: `docs/PRD.md` §2 and §10. The outcome-blind index is associated with Booking scores
among the 26 blind-coded Waikiki hotels, but this is a cross-sectional association across few hotels; price,
renovation and brand are only partly controlled, and the lower-star test hotels differ in many ways besides design.
"""
