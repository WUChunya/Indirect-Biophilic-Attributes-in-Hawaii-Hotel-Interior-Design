"""Adapter: aspect-based sentiment extraction with Claude (Message Batches API + structured outputs).

Each review → one request → JSON list of mentions with a verbatim evidence quote, mapped to the same aspect IDs
as the lexicon baseline (config/review_aspects.yaml) and, when the guest describes a concrete design element,
to a BDIA item (config/bdia_items.yaml). Output rows use text.lexicon.MENTION_COLUMNS with method="llm".

Workflow (biophilic.cli):
    biophilic absa-submit   → writes data/interim/llm_batches.json (batch IDs)
    biophilic absa-collect  → waits for the batches, writes data/processed/mentions_llm.parquet
    biophilic absa-sample   → synchronous run on the annotation sample (for validation against the gold set)
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from .. import config
from .lexicon import MENTION_COLUMNS

FIELDS = ("title", "liked", "disliked")


def response_schema() -> dict:
    aspects = list(config.load("review_aspects")["aspects"])
    codes = [it.code for it in config.bdia_items()] + ["none"]
    mention = {
        "type": "object",
        "properties": {
            "field": {"type": "string", "enum": list(FIELDS)},
            "aspect": {"type": "string", "enum": aspects},
            "bdia_item": {"type": "string", "enum": codes},
            "space": {"type": "string", "enum": ["guestroom", "lobby", "lounge", "pool_outdoor", "other", "unknown"]},
            "polarity": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            "explicit": {"type": "boolean"},
            "evidence": {"type": "string"},
        },
        "required": ["field", "aspect", "bdia_item", "space", "polarity", "explicit", "evidence"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {"mentions": {"type": "array", "items": mention}},
        "required": ["mentions"],
        "additionalProperties": False,
    }


def system_prompt() -> str:
    cfg = config.load("review_aspects")
    aspect_lines = "\n".join(f"- {name}: {a['definition']}" for name, a in cfg["aspects"].items())
    item_lines = "\n".join(f"- {it.code} ({it.dimension}, {it.element}): {it.definition}" for it in config.bdia_items())
    return f"""You annotate hotel guest reviews from Booking.com for a design research project on indirect biophilic
design (Kellert's "indirect experience of nature") in Hawaii hotels.

A review has up to three fields. "liked" answers "What did you like?", "disliked" answers "What didn't you like?",
"title" is free text. Extract every statement about the physical environment and design of the hotel that fits
one of these aspects:
{aspect_lines}

When the statement describes a concrete design element, also give the matching BDIA item, otherwise "none":
{item_lines}

Rules:
- polarity is the guest's evaluation of that aspect. Text under "disliked" is usually negative, but judge the
  statement itself ("nothing, the wooden floors were lovely" under disliked is positive).
- An absence the guest notices ("no balcony", "barely any natural light") is a negative mention of that aspect.
- Do not tag words that are not about design: "looked like the photos" is not artwork, "a stone's throw" is not
  stone, "the smell of the sewer" is sensory_ambience negative but not a design element, a hotel or restaurant
  name ("Polynesian Hostel", "Tiki's Grill") is not local identity.
- Physical condition (dated, worn, renovated) is aspect "condition", not "age_patina". age_patina is a deliberate
  vintage, retro, historic or crafted character.
- Views, plants, pools, beaches and gardens are "direct_nature".
- explicit = true if the guest names the element directly, false if you infer it ("felt like old Hawaii").
- evidence = the shortest verbatim quote from the review that supports the mention, in its original language.
- Reviews can be in any language. Return an empty list when nothing fits."""


def user_message(row: pd.Series | dict) -> str:
    parts = []
    for f in FIELDS:
        v = row.get(f) if isinstance(row, dict) else row[f]
        if v:
            parts.append(f"<{f}>{v}</{f}>")
    return "\n".join(parts)


def request_params(row, llm_cfg: dict) -> dict:
    return {
        "model": llm_cfg["model"],
        "max_tokens": llm_cfg["max_tokens"],
        # The system prompt is identical for all reviews; caching it cuts input cost on every request.
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": user_message(row)}],
        "output_config": {"effort": llm_cfg["effort"], "format": {"type": "json_schema", "schema": response_schema()}},
    }


def parse_mentions(review_id: str, hotel_id: str, payload: dict) -> list[dict]:
    pol = {"positive": 1, "negative": -1, "neutral": 0}
    rows = []
    for m in payload.get("mentions", []):
        rows.append({"review_id": review_id, "hotel_id": hotel_id, "field": m["field"], "sentence_idx": -1,
                     "aspect": m["aspect"], "dimension": config.load("review_aspects")["aspects"][m["aspect"]]["dimension"],
                     "term": m["evidence"], "space": m["space"], "polarity": pol[m["polarity"]],
                     "negated": False, "method": "llm", "bdia_item": m["bdia_item"], "explicit": m["explicit"]})
    return rows


def _first_text(message) -> str:
    return next(b.text for b in message.content if b.type == "text")


def submit(reviews: pd.DataFrame, out_path: Path) -> list[str]:
    import anthropic
    from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
    from anthropic.types.messages.batch_create_params import Request

    llm_cfg = config.load("pipeline")["llm"]
    client = anthropic.Anthropic()
    todo = reviews[reviews["has_text"]]
    ids = []
    for start in range(0, len(todo), llm_cfg["batch_size"]):
        chunk = todo.iloc[start:start + llm_cfg["batch_size"]]
        batch = client.messages.batches.create(requests=[
            Request(custom_id=r["review_id"], params=MessageCreateParamsNonStreaming(**request_params(r, llm_cfg)))
            for _, r in chunk.iterrows()
        ])
        ids.append(batch.id)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps({"batch_ids": ids, "prompt_version": llm_cfg["prompt_version"],
                                    "model": llm_cfg["model"]}, indent=2))
    return ids


def collect(batch_file: Path, reviews: pd.DataFrame, poll_seconds: int = 60) -> pd.DataFrame:
    """Wait for every batch to end and return mention rows. Refused or failed requests are listed in
    attrs['failed'] so they can be resubmitted; they are never silently dropped."""
    import anthropic

    client = anthropic.Anthropic()
    meta = json.loads(batch_file.read_text())
    hotel_of = dict(zip(reviews["review_id"], reviews["hotel_id"]))
    rows, failed = [], []
    for bid in meta["batch_ids"]:
        while client.messages.batches.retrieve(bid).processing_status != "ended":
            time.sleep(poll_seconds)
        for res in client.messages.batches.results(bid):
            rid = res.custom_id
            if res.result.type != "succeeded":
                failed.append((rid, res.result.type))
                continue
            msg = res.result.message
            if msg.stop_reason != "end_turn":
                failed.append((rid, msg.stop_reason))
                continue
            rows.extend(parse_mentions(rid, hotel_of.get(rid), json.loads(_first_text(msg))))
    df = pd.DataFrame(rows, columns=[*MENTION_COLUMNS, "bdia_item", "explicit"])
    df.attrs["failed"] = failed
    return df


def run_sync(reviews: pd.DataFrame) -> pd.DataFrame:
    """Synchronous extraction for a small sample. Uses server-side refusal fallbacks (not available in Batches)."""
    import anthropic

    llm_cfg = config.load("pipeline")["llm"]
    client = anthropic.Anthropic()
    rows, failed = [], []
    for _, r in reviews.iterrows():
        msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                          **request_params(r, llm_cfg))
        if msg.stop_reason != "end_turn":
            failed.append((r["review_id"], msg.stop_reason))
            continue
        rows.extend(parse_mentions(r["review_id"], r["hotel_id"], json.loads(_first_text(msg))))
    df = pd.DataFrame(rows, columns=[*MENTION_COLUMNS, "bdia_item", "explicit"])
    df.attrs["failed"] = failed
    return df
