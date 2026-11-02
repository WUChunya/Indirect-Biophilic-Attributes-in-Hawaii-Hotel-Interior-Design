# BDIA 2.0 — indirect biophilic design in Hawaii hotels: photos, reviews, statistics (Python)

Spec: `docs/PRD.md`. Progress: `docs/PROGRESS.md`. Handoff: `docs/HANDOFF.md`. Stage plan: `docs/DEV_PROMPT.md`.
Audit of the 2024 work: `docs/LEGACY_AUDIT.md`. Standards: `WUChunya/Claude-Skills` → `skills/project-standards/INDEX.md`.
Talk to the user in Chinese; write code, comments, commits in English.

## Layout
- `src/biophilic/`: core (`clean`, `bdia`, `agreement`, `text/lexicon|aggregate|annotation`, `models/*`) is pure
  pandas and tested on Linux; adapters (`io`, `text/llm_absa`, `vision/vlm_coder`) talk to files and the Claude API.
- `config/`: every default (items, aspects, thresholds, models, legacy formulas). Change config before code.
- `data/raw/`: thesis appendix, read-only. `data/interim|processed|external/`: local only (review text, photos).
- `results/`: aggregated outputs, committed. `legacy/`: 2024 code and outputs, never edited.
- `docs/papers/`: thesis, slides, handbook (plain git files; never `Read` them, use pdftotext).

## Commands
- Install: `pip install -e ".[dev]"` (add `,llm` for Claude stages)
- Test: `python -m pytest -q 2>&1 | tail -20`
- Baseline: `biophilic baseline 2>&1 | tail -5` (~3 min) → `results/REPORT.md`
- Other stages: `biophilic --help`

## Rules
- Statistics (PRD §6): hotel outcomes → one row per hotel; never copy a hotel score to 3 rows; predictions use
  whole-hotel hold-out; indices never use the outcome; report CI and FDR.
- Legacy weights (`config/legacy.yaml`) are for the audit only.
- Never write review text, evidence quotes or reviewer names to `results/` (test enforces it).
- Claude API calls cost money: run the 100-review estimate and ask before full batches.
- One version source: `src/biophilic/__init__.py`; a test checks `docs/CHANGELOG.md`.
- Every fix gets a test; every false positive found becomes a `must_not_find` fixture in `tests/test_text.py`.
- Token economy: read PRD by heading (`grep -n '^#' docs/PRD.md`, then `sed -n`); never `Read` .xlsx/.pdf/.parquet
  (query with python / pdftotext); pipe long output through `tail`; edit, don't rewrite; no subagents unless
  asked; final reply ≤10 lines, details go to `docs/PROGRESS.md`.
