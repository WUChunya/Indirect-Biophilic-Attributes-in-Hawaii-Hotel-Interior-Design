# Handoff: S0 rebuild (2026-11-02, current v0.2.0)

New session: paste §1 into the new chat, then attach the annotated CSV / photo folder status / API key status.

## 1. Opening message (copy as is)
```
You are the lead developer of "BDIA 2.0" (repo WUChunya/Indirect-Biophilic-Attributes-in-Hawaii-Hotel-Interior-Design,
branch claude/dazzling-albattani-33cevo). First: git fetch and pull the branch.
Read, in this order and only these: CLAUDE.md, docs/PROGRESS.md, docs/HANDOFF.md, docs/CHANGELOG.md (top rows);
PRD sections only as needed (grep -n '^#' docs/PRD.md, then sed -n).
Task: the next stage in docs/DEV_PROMPT.md that is not done (S1 gold set, then S2 LLM text, then S3 photos).
For each item: find it in HANDOFF §3, prefer changing a config key over code, add or adjust a test.
Claude API calls cost money: run the 100-review estimate and ask me before any full batch. Ask me only if blocked (§4).
Talk to me in Chinese; code/commits in English. At the end: run tests, update PROGRESS.md (<=40 lines),
HANDOFF.md and CHANGELOG.md, bump the patch version, commit, push, reply in <=10 lines.
[Paste results here]
```

## 2. Current state
- v0.2.0, 45 tests (privacy tests skip until `results/` exists). `biophilic baseline` ≈ 4 min, writes `results/`.
- Generated local data: `data/processed/*.parquet` (review text; not in git).
- Not yet run: anything that calls the Claude API; no photos; no gold set.

## 3. Where each feature lives
| Feature | Code | Config key |
|---|---|---|
| Raw Excel → tables, column names | `io.py` | `pipeline.paths` |
| Placeholders, null answers, auto titles, listings | `clean.py` | `review_aspects.placeholders/null_answers/auto_titles`, `pipeline.ingest` |
| Aspect terms, ambiguous terms, excludes, negation | `text/lexicon.py` | `review_aspects.aspects.*`, `negators`, `negation_window` |
| Hotel mention rate / net sentiment | `text/aggregate.py` | `pipeline.text.sentiment_prior_strength` |
| Gold-set sample and scoring | `text/annotation.py` | `pipeline.text.annotation_sample_size`, `random_seed` |
| Claude ABSA (batch + sync) | `text/llm_absa.py` | `pipeline.llm.*` |
| BDIA items, definitions, VLM cues | `config/bdia_items.yaml` | `items[*]` |
| BDIA indices, KR-20, prevalence, legacy reproduction | `bdia.py` | `pipeline.bdia.*`, `legacy.*` |
| Hotel groups (26 / 5 / 5) | `io.read_bdia` | `pipeline.bdia.groups` |
| VLM photo coding | `vision/vlm_coder.py` | `pipeline.llm.vision_*` |
| κ / AC1 agreement | `agreement.py` | — |
| Hotel-level stats, external test, item table | `models/hotel_level.py`, `models/stats.py` | `pipeline.models.*` |
| Review-level FE model | `models/review_level.py` | — |
| RF audit | `models/legacy_rf.py` | `legacy.random_forest` |
| Report | `pipeline.render_report`, `report.py` | — |

## 4. Open questions
See PRD §11 (Q1–Q7) and PROGRESS "Open questions"; defaults apply until answered.
