"""Claude adapters without network: request shape, schema, parsing, aggregation."""

import json

import pandas as pd

from biophilic import config
from biophilic.text import llm_absa
from biophilic.vision import vlm_coder


def test_absa_request_uses_structured_output_and_cached_system():
    llm_cfg = config.load("pipeline")["llm"]
    p = llm_absa.request_params({"title": None, "liked": "Wood floors!", "disliked": None}, llm_cfg)
    assert p["output_config"]["format"]["type"] == "json_schema"
    assert p["system"][0]["cache_control"] == {"type": "ephemeral"}
    assert "<liked>Wood floors!</liked>" in p["messages"][0]["content"]
    assert "temperature" not in p and "thinking" not in p
    json.dumps(p)  # must be serialisable for the Batches API


def test_absa_schema_enums_match_config():
    s = llm_absa.response_schema()["properties"]["mentions"]["items"]["properties"]
    assert set(s["aspect"]["enum"]) == set(config.load("review_aspects")["aspects"])
    assert "none" in s["bdia_item"]["enum"] and "A01" in s["bdia_item"]["enum"]


def test_absa_parse():
    rows = llm_absa.parse_mentions("r1", "h1", {"mentions": [
        {"field": "liked", "aspect": "natural_materials", "bdia_item": "B04", "space": "guestroom",
         "polarity": "positive", "explicit": True, "evidence": "wood floors"}]})
    assert rows[0]["polarity"] == 1 and rows[0]["dimension"] == "B" and rows[0]["method"] == "llm"


def test_vlm_parse_marks_missing_items_unassessed_and_wide_threshold():
    payload = {"items": [{"code": "A01", "present": True, "confidence": 0.9, "evidence": "surf art",
                          "photo_numbers": [1]},
                         {"code": "B01", "present": True, "confidence": 0.4, "evidence": "maybe wood",
                          "photo_numbers": [2]}],
               "usable_photos": 3, "notes": ""}
    long = vlm_coder.parse("h1", "lobby", payload)
    assert len(long) == 29
    assert long.loc[long["code"] == "C01", "present"].isna().all()
    wide = vlm_coder.to_wide(long, threshold=0.5)
    assert wide.loc[0, "A01"] == 1 and wide.loc[0, "B01"] == 0 and wide.loc[0, "C01"] == 0


def test_vlm_prompt_lists_all_items():
    sp = vlm_coder.system_prompt()
    assert all(f"- {it.code} [" in sp for it in config.bdia_items())
    assert isinstance(vlm_coder.response_schema()["properties"]["items"], dict)


def test_annotation_evaluate_perfect_method():
    from biophilic.text import annotation

    gold = pd.DataFrame({"review_id": ["a", "b"], "natural_materials": ["+", ""]})
    mentions = pd.DataFrame({"review_id": ["a"], "aspect": ["natural_materials"], "polarity": [1]})
    ev = annotation.evaluate(mentions, gold, ["natural_materials"])
    assert ev.loc[0, "precision"] == 1 and ev.loc[0, "recall"] == 1 and ev.loc[0, "polarity_accuracy"] == 1
