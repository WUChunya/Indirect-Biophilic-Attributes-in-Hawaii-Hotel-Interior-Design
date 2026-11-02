"""Lexicon baseline and cleaning: every false positive found in the 2024 audit is a fixture here (must_not_find)."""

import pandas as pd
import pytest

from biophilic import clean, config
from biophilic.text import aggregate
from biophilic.text.lexicon import LexiconExtractor, split_sentences


@pytest.fixture(scope="module")
def cfg():
    return config.load("review_aspects")


@pytest.fixture(scope="module")
def ex(cfg):
    return LexiconExtractor(cfg, mask_names=["Polynesian Hostel Beach Club"])


def aspects(ex, sentence, field="liked"):
    return {(r["aspect"], r["polarity"]) for r in ex.extract_sentence(sentence, field)}


# --- must_find ------------------------------------------------------------------------------------------
def test_wood_praise(ex):
    assert ("natural_materials", 1) in aspects(ex, "Loved the wooden furniture and the koa headboard.")


def test_complaint_field_is_negative(ex):
    rows = ex.extract_sentence("No balcony at all", "disliked")
    light = [r for r in rows if r["aspect"] == "natural_light_air"]
    assert light and light[0]["polarity"] == -1 and light[0]["negated"]


def test_negation_under_liked_is_neutral(ex):
    rows = ex.extract_sentence("Not much natural light, but a great location", "liked")
    light = [r for r in rows if r["aspect"] == "natural_light_air"]
    assert light and light[0]["negated"] and light[0]["polarity"] == 0


def test_title_uses_sentence_sentiment(ex):
    assert ("age_patina", 1) in aspects(ex, "Beautiful retro boutique hotel", "title")


# --- must_not_find (2024 false positives) ----------------------------------------------------------------
@pytest.mark.parametrize("sentence,aspect", [
    ("Just a stone's throw from Waikiki Beach", "natural_materials"),
    ("The room looked exactly like the photos", "images_of_nature"),
    ("The patio was closed for painting all week", "images_of_nature"),
    ("Tour buses pick up on Koa Ave", "natural_materials"),
    ("Right next to the Tiki bar so pretty loud", "hawaiian_local"),
    ("Had a great time at the Polynesian Hostel Beach Club", "hawaiian_local"),
])
def test_known_false_positives(ex, sentence, aspect):
    assert aspect not in {a for a, _ in aspects(ex, sentence)}


def test_ambiguous_terms_only_when_requested(cfg):
    strict = LexiconExtractor(cfg)
    loose = LexiconExtractor(cfg, include_ambiguous=True)
    s = "The pictures on the website were accurate"
    assert not strict.extract_sentence(s, "liked")
    assert any(r["method"] == "lexicon_ambiguous" for r in loose.extract_sentence(s, "liked"))


def test_split_sentences():
    assert split_sentences("Great pool. Lovely staff! Wood floors") == ["Great pool.", "Lovely staff!", "Wood floors"]


# --- cleaning ----------------------------------------------------------------------------------------------
def _raw(**kw):
    base = {"review_id": "r1", "hotel_id": "h", "hotel": "H", "title": "Exceptional",
            "liked": "There are no comments available for this review", "disliked": "Nothing.",
            "rating": 10.0, "traveller_type": "Couple", "room_type": "King", "review_date": pd.NaT}
    base.update(kw)
    return pd.DataFrame([base])


def test_placeholder_null_answer_and_auto_title(cfg):
    df = clean.clean_reviews(_raw(), cfg)
    assert df.loc[0, "liked"] is None and df.loc[0, "disliked"] is None and df.loc[0, "title"] is None
    assert not df.loc[0, "has_text"]


def test_real_text_kept(cfg):
    df = clean.clean_reviews(_raw(liked="The lobby has gorgeous koa wood walls"), cfg)
    assert df.loc[0, "has_text"] and df.loc[0, "english"]


def test_hotel_metrics_denominator_and_shrinkage():
    reviews = pd.DataFrame({"review_id": ["a", "b", "c", "d"], "hotel_id": ["h"] * 4, "has_text": [True] * 4})
    mentions = pd.DataFrame({"review_id": ["a", "a", "b"], "hotel_id": ["h"] * 3, "aspect": ["x"] * 3,
                             "polarity": [1, 1, -1]})
    m = aggregate.hotel_metrics(mentions, reviews, prior_strength=0)
    row = m.iloc[0]
    assert row["n_mention"] == 2 and row["n_reviews"] == 4 and row["mention_rate"] == 0.5
    assert row["pos"] == 1 and row["neg"] == 1 and row["net_sentiment"] == 0  # review "a" counts once
