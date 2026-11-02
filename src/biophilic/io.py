"""Adapter: read the raw 2024 Excel files into tidy tables. All column-name knowledge of the raw files lives here."""

from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pandas as pd

from . import config

# Raw review sheet → tidy column names. "Reserved Room Type.1" is the review title (two columns share a header).
_REVIEW_COLUMNS = {
    "Unnamed: 0": "date_raw",
    "Customer Type": "traveller_type",
    "Reserved Room Type": "room_type",
    "Reserved Room Type.1": "title",
    "Positive Guest Reviews": "liked",
    "Negative Guest Reviews": "disliked",
    "Guest Rating": "rating",
    "Hotel Name": "hotel",
}
_HOTEL_COLUMNS = {
    "Hotel Name": "hotel",
    "Website URL": "booking_url",
    "Total Hotel Rating": "booking_score",
    "Staff Quality": "staff",
    "Facilities": "facilities",
    "Cleanliness": "cleanliness",
    "Comfort": "comfort",
    "Value for Money": "value_for_money",
    "Location": "location",
    "Free WiFi": "wifi",
    "Third-party Rating": "sustainability_label",
    "Address": "address",
    "Average Price": "avg_price",
}
_BDIA_SCORES = {
    "Overall Score": "booking_score",
    "Services": "staff",
    "Facilities": "facilities",
    "Cleaness": "cleanliness",
    "Comfort": "comfort",
    "Value of Money": "value_for_money",
    "Location": "location",
    "WiFi": "wifi",
    "Average Price": "avg_price",
}


def hotel_id(name: str) -> str:
    """Stable slug used as the join key between datasets."""
    s = re.sub(r"[^a-z0-9]+", "-", str(name).lower()).strip("-")
    return s[:60]


def _review_id(row: pd.Series) -> str:
    key = "|".join(str(row[c]) for c in ("hotel", "date_raw", "room_type", "title", "liked", "disliked"))
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:12]


def read_reviews(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (reviews, hotels). Reviewer names are read only to drop them; they never leave this function."""
    path = path or config.path("raw_reviews")
    raw = pd.read_excel(path, sheet_name="Hotel Reviews", header=0)

    reviews = raw[list(_REVIEW_COLUMNS)].rename(columns=_REVIEW_COLUMNS)
    for col in ("traveller_type", "room_type", "title", "liked", "disliked", "hotel", "date_raw"):
        reviews[col] = reviews[col].astype("object").where(reviews[col].notna(), None)
    reviews["review_date"] = pd.to_datetime(
        reviews["date_raw"].astype(str).str.extract(r"Reviewed:\s*(.+)$")[0], format="%d %B %Y", errors="coerce"
    )
    reviews["rating"] = pd.to_numeric(reviews["rating"], errors="coerce")
    reviews["hotel_id"] = reviews["hotel"].map(hotel_id)
    reviews["review_id"] = reviews.apply(_review_id, axis=1)
    reviews = reviews.drop(columns=["date_raw"])

    hotels = raw[list(_HOTEL_COLUMNS)].rename(columns=_HOTEL_COLUMNS).drop_duplicates("hotel")
    hotels["booking_url"] = hotels["booking_url"].astype(str).str.split("?").str[0]
    hotels.insert(0, "hotel_id", hotels["hotel"].map(hotel_id))
    return reviews.reset_index(drop=True), hotels.reset_index(drop=True)


def read_bdia(path: Path | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (coding, hotels).

    coding: one row per hotel × space with item columns A01…G03 (0/1) and the legacy BDIA-Scenario.
    hotels: one row per hotel with legacy BDIA-Index, legacy prediction, Booking scores and price.
    The per-hotel subtotal rows of the raw sheet (no hotel name) are skipped.
    """
    path = path or config.path("raw_bdia")
    raw = pd.read_excel(path, header=0)
    raw = raw[raw["Hotel Name"].notna()].copy()
    legacy_to_code = {it.legacy_code: it.code for it in config.bdia_items()}
    space_cols = {v["legacy_column"]: k for k, v in config.load("bdia_items")["spaces"].items()}

    def space_of(row: pd.Series) -> str:
        hits = [s for col, s in space_cols.items() if row.get(col) == 1]
        if len(hits) != 1:
            raise ValueError(f"row for {row['Hotel Name']} has {len(hits)} space flags")
        return hits[0]

    raw["space"] = raw.apply(space_of, axis=1)
    raw["hotel"] = raw["Hotel Name"].astype(str)
    raw["hotel_id"] = raw["hotel"].map(hotel_id)
    raw["hotel_number"] = raw["Hotel Number"].astype(int)

    items = raw[list(legacy_to_code)].rename(columns=legacy_to_code).astype(float).fillna(0).astype(int)
    coding = pd.concat(
        [raw[["hotel_id", "hotel", "hotel_number", "space"]], items,
         raw[["BDIA-Scenario"]].rename(columns={"BDIA-Scenario": "legacy_scenario"})],
        axis=1,
    ).reset_index(drop=True)

    first = raw.groupby("hotel_id", sort=False).first()
    hotels = pd.DataFrame({
        "hotel_id": first.index,
        "hotel": first["hotel"].values,
        "hotel_number": first["hotel_number"].values,
        "legacy_index": first["BDIA-Index"].astype(float).values,
        "legacy_prediction": first["Booking Score Prediction"].astype(float).values,
    })
    for src, dst in _BDIA_SCORES.items():
        hotels[dst] = pd.to_numeric(first[src], errors="coerce").values
    groups = config.load("pipeline")["bdia"]["groups"]
    hotels["group"] = hotels["hotel_number"].map(
        lambda n: next((g for g, (lo, hi) in groups.items() if lo <= n <= hi), "unassigned"))
    # Hotels whose three spaces carry identical codes were most likely coded once and copied.
    dup = coding.groupby("hotel_id")[[it.code for it in config.bdia_items()]].nunique().max(axis=1) == 1
    hotels["spaces_identical"] = hotels["hotel_id"].map(dup).fillna(False).astype(bool)
    return coding, hotels


def item_codes() -> list[str]:
    return [it.code for it in config.bdia_items()]
