"""Core: rule-based aspect mention extraction (the corrected 2024 method; baseline for the LLM extractor).

Unit of analysis = one sentence of one review field. Polarity comes from the Booking.com form itself:
text typed under "liked" is positive, under "disliked" negative (the guest answered that question).
For the free-text title, which has no such prompt, VADER sentence polarity is used. A negated mention
("no balcony", "not much natural light") keeps the negative polarity of the "disliked" field (an absence the
guest complained about) and becomes neutral (0) under "liked", where the aspect itself was not praised.

Output schema (one row per review × field × sentence × aspect), shared with text/llm_absa.py:
    review_id, hotel_id, field, sentence_idx, aspect, dimension, term, space, polarity, negated, method
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

MENTION_COLUMNS = ["review_id", "hotel_id", "field", "sentence_idx", "aspect", "dimension", "term", "space",
                   "polarity", "negated", "method"]
_SENT_SPLIT = re.compile(r"(?<=[.!?;])\s+|\s*[\n•]+\s*|\s+-\s+")
_FIELD_POLARITY = {"liked": 1, "disliked": -1}


def term_pattern(term: str) -> str:
    """Word-boundary regex for a term; spaces/hyphens interchangeable; optional plural."""
    parts = [re.escape(p) for p in re.split(r"[\s\-]+", term.strip())]
    body = r"[\s\-]+".join(parts)
    return rf"(?<![\w-]){body}(?:s|es)?(?![\w-])"


@dataclass
class Aspect:
    name: str
    dimension: str
    terms: re.Pattern
    ambiguous: re.Pattern | None
    exclude: re.Pattern | None


def _alternation(terms: list[str]) -> re.Pattern | None:
    if not terms:
        return None
    ordered = sorted({t.lower() for t in terms}, key=len, reverse=True)  # longest first: "ocean view" before "view"
    return re.compile("|".join(f"({term_pattern(t)})" for t in ordered), flags=re.IGNORECASE)


def compile_aspects(cfg: dict) -> list[Aspect]:
    out = []
    for name, a in cfg["aspects"].items():
        exclude = re.compile("|".join(a["exclude"]), flags=re.IGNORECASE) if a.get("exclude") else None
        out.append(Aspect(name, a["dimension"], _alternation(a["terms"]), _alternation(a.get("ambiguous", [])),
                          exclude))
    return out


def compile_spaces(cfg: dict) -> dict[str, re.Pattern]:
    return {s: _alternation(terms) for s, terms in cfg["spaces"].items()}


def split_sentences(text: str) -> list[str]:
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def is_negated(sentence: str, start: int, negators: list[str], window: int) -> bool:
    """True if a negator occurs within `window` tokens before the match start."""
    before = re.findall(r"[\w']+", sentence[:start].lower())[-window:]
    for tok in before:
        for neg in negators:
            if tok == neg or (neg.startswith("n'") and tok.endswith(neg)):
                return True
    return False


class LexiconExtractor:
    def __init__(self, cfg: dict, include_ambiguous: bool = False, mask_names: list[str] | None = None):
        """mask_names: property names removed from the text before matching ("Polynesian Hostel", "Hale Koa")."""
        self.aspects = compile_aspects(cfg)
        names = sorted({n for n in (mask_names or []) if n}, key=len, reverse=True)
        self._names = re.compile("|".join(re.escape(n) for n in names), flags=re.IGNORECASE) if names else None
        self.spaces = compile_spaces(cfg)
        self.negators = cfg["negators"]
        self.window = cfg["negation_window"]
        self.include_ambiguous = include_ambiguous
        self._vader = SentimentIntensityAnalyzer()

    def _space(self, sentence: str) -> str | None:
        hits = [s for s, rx in self.spaces.items() if rx.search(sentence)]
        return hits[0] if len(hits) == 1 else ("multiple" if hits else None)

    def _base_polarity(self, field: str, sentence: str) -> int:
        if field in _FIELD_POLARITY:
            return _FIELD_POLARITY[field]
        c = self._vader.polarity_scores(sentence)["compound"]
        return 1 if c >= 0.05 else (-1 if c <= -0.05 else 0)

    def extract_sentence(self, sentence: str, field: str) -> list[dict]:
        rows = []
        if self._names is not None:
            sentence = self._names.sub(" <PROPERTY> ", sentence)
        for asp in self.aspects:
            if asp.exclude is not None and asp.exclude.search(sentence):
                continue
            patterns = [("term", asp.terms)]
            if self.include_ambiguous and asp.ambiguous is not None:
                patterns.append(("ambiguous", asp.ambiguous))
            for kind, rx in patterns:
                m = rx.search(sentence) if rx is not None else None
                if m is None:
                    continue
                negated = is_negated(sentence, m.start(), self.negators, self.window)
                polarity = self._base_polarity(field, sentence)
                if negated and field == "liked":
                    # "not much natural light, but great location" typed under "liked": the aspect itself was not
                    # praised. Mark it neutral rather than guess; the LLM method resolves these cases.
                    polarity = 0
                rows.append({"aspect": asp.name, "dimension": asp.dimension, "term": m.group(0).lower(),
                             "space": self._space(sentence), "polarity": polarity, "negated": negated,
                             "method": "lexicon" if kind == "term" else "lexicon_ambiguous"})
                break  # one mention per aspect per sentence
        return rows

    def extract(self, reviews: pd.DataFrame, fields: tuple[str, ...] = ("liked", "disliked", "title")) -> pd.DataFrame:
        out: list[dict] = []
        for rec in reviews[["review_id", "hotel_id", *fields]].itertuples(index=False):
            for field in fields:
                text = getattr(rec, field)
                if not text:
                    continue
                for i, sent in enumerate(split_sentences(text)):
                    for row in self.extract_sentence(sent, field):
                        out.append({"review_id": rec.review_id, "hotel_id": rec.hotel_id, "field": field,
                                    "sentence_idx": i, **row})
        return pd.DataFrame(out, columns=MENTION_COLUMNS)
