"""Adapter: code the 29 BDIA items from hotel photos with Claude vision (structured outputs).

Input layout (not in git — hotel photos are copyrighted):
    data/external/photos/<hotel_id>/<space>/*.{jpg,jpeg,png,webp}      space ∈ guestroom | lobby | lounge
    data/external/photos/manifest.csv   optional: hotel_id, space, file, source_url, retrieved (provenance)

One request per hotel × space with up to `vision_max_images_per_call` photos. The model is BLIND to the hotel's
name and scores, which the 2024 human coding was not; that is the main reason to add it.

Output: same wide layout as io.read_bdia() coding (hotel_id, space, A01…G03 as 0/1) plus per-item confidence
and evidence in a long table, so agreement.agreement_table() can compare VLM vs human directly.
"""

from __future__ import annotations

import base64
import json
from pathlib import Path

import pandas as pd

from .. import config

MEDIA = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".webp": "image/webp"}
MAX_BYTES = 5 * 1024 * 1024


def response_schema() -> dict:
    codes = [it.code for it in config.bdia_items()]
    item = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "enum": codes},
            "present": {"type": "boolean"},
            "confidence": {"type": "number"},
            "evidence": {"type": "string"},
            "photo_numbers": {"type": "array", "items": {"type": "integer"}},
        },
        "required": ["code", "present", "confidence", "evidence", "photo_numbers"],
        "additionalProperties": False,
    }
    return {
        "type": "object",
        "properties": {
            "items": {"type": "array", "items": item},
            "usable_photos": {"type": "integer"},
            "notes": {"type": "string"},
        },
        "required": ["items", "usable_photos", "notes"],
        "additionalProperties": False,
    }


def system_prompt() -> str:
    lines = [f"- {it.code} [{it.element}] {it.definition}\n    look for: {it.visual_cues}\n    not: {it.not_this}"
             for it in config.bdia_items()]
    return ("You are a trained interior-design coder applying the BDIA instrument (Biophilic Design Indirect "
            "Attributes, after Kellert & Calabrese) to hotel photographs.\n"
            "You see several photos of ONE space of ONE hotel. For EVERY one of the 29 items below, decide whether "
            "the item is present anywhere in these photos. Code only what is visible; when the relevant element "
            "is not visible at all (e.g. no ceiling in frame), set present=false, confidence ≤ 0.3 and say so in "
            "evidence. confidence is your probability (0–1) that the item is present. photo_numbers are the "
            "1-based numbers of the photos that show it. Return exactly 29 items.\n\nItems:\n" + "\n".join(lines))


def load_images(folder: Path, limit: int) -> list[tuple[str, str]]:
    """(media_type, base64) for up to `limit` photos, sorted by file name; oversize files are skipped."""
    out = []
    for p in sorted(folder.iterdir()):
        mt = MEDIA.get(p.suffix.lower())
        if mt is None or p.stat().st_size > MAX_BYTES:
            continue
        out.append((mt, base64.standard_b64encode(p.read_bytes()).decode("ascii")))
        if len(out) == limit:
            break
    return out


def request_params(images: list[tuple[str, str]], space: str, llm_cfg: dict) -> dict:
    content: list[dict] = []
    for i, (mt, data) in enumerate(images, 1):
        content.append({"type": "text", "text": f"Photo {i}"})
        content.append({"type": "image", "source": {"type": "base64", "media_type": mt, "data": data}})
    label = config.load("bdia_items")["spaces"][space]["label"]
    content.append({"type": "text", "text": f"Space: {label}. Code all 29 BDIA items."})
    return {
        "model": llm_cfg["model"],
        "max_tokens": llm_cfg["max_tokens"] * 4,
        "system": [{"type": "text", "text": system_prompt(), "cache_control": {"type": "ephemeral"}}],
        "messages": [{"role": "user", "content": content}],
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": response_schema()}},
    }


def parse(hotel_id: str, space: str, payload: dict) -> pd.DataFrame:
    rows = [{"hotel_id": hotel_id, "space": space, **it} for it in payload["items"]]
    df = pd.DataFrame(rows).drop_duplicates(["hotel_id", "space", "code"])
    missing = set(c.code for c in config.bdia_items()) - set(df["code"])
    for code in sorted(missing):  # never invent a value: missing items are recorded as not assessed
        df.loc[len(df)] = {"hotel_id": hotel_id, "space": space, "code": code, "present": None,
                           "confidence": None, "evidence": "not returned by model", "photo_numbers": []}
    df["usable_photos"] = payload["usable_photos"]
    return df


def to_wide(long: pd.DataFrame, threshold: float) -> pd.DataFrame:
    """hotel_id × space rows with 0/1 per item (present and confidence ≥ threshold)."""
    long = long.copy()
    long["value"] = ((long["present"] == True) & (long["confidence"].fillna(0) >= threshold)).astype(int)  # noqa: E712
    return long.pivot_table(index=["hotel_id", "space"], columns="code", values="value", aggfunc="max").reset_index()


def code_all(photo_root: Path, out_path: Path) -> pd.DataFrame:
    """Code every hotel/space folder that has photos; resumable (already coded pairs are skipped)."""
    import anthropic

    llm_cfg = config.load("pipeline")["llm"]
    client = anthropic.Anthropic()
    done = pd.read_parquet(out_path) if out_path.exists() else pd.DataFrame()
    have = set(zip(done.get("hotel_id", []), done.get("space", [])))
    parts = [done] if len(done) else []
    for hotel_dir in sorted(p for p in photo_root.iterdir() if p.is_dir()):
        for space in config.spaces():
            folder = hotel_dir / space
            if not folder.is_dir() or (hotel_dir.name, space) in have:
                continue
            images = load_images(folder, llm_cfg["vision_max_images_per_call"])
            if not images:
                continue
            msg = client.beta.messages.create(betas=["server-side-fallback-2026-07-01"], fallbacks="default",
                                              **request_params(images, space, llm_cfg))
            if msg.stop_reason != "end_turn":
                print(f"skip {hotel_dir.name}/{space}: stop_reason={msg.stop_reason}")
                continue
            payload = json.loads(next(b.text for b in msg.content if b.type == "text"))
            parts.append(parse(hotel_dir.name, space, payload))
            out_path.parent.mkdir(parents=True, exist_ok=True)
            pd.concat(parts, ignore_index=True).to_parquet(out_path)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
