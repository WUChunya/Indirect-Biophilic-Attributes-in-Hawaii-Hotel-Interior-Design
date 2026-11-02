# Development prompt

## Kickoff message (any later session)
Use the opening message in `docs/HANDOFF.md` §1.

## Environment facts
- Everything except Claude API calls runs in a Linux container: `pip install -e ".[dev]"`, `python -m pytest -q`,
  `biophilic baseline` (~3 min).
- Claude API stages need `ANTHROPIC_API_KEY` (or `ant auth login`) and `pip install -e ".[llm]"`; they cost money,
  so run the 100-review estimate first and get the user's go-ahead for full batches.
- Photos (`data/external/photos/`) and the annotated gold set (`data/interim/annotation_sample.csv`) come from the
  user and never enter git.

## Architecture (fixed; do not re-debate)
1. Adapters read/write the outside world and convert to plain DataFrames: `io.py` (raw Excel), `text/llm_absa.py`
   and `vision/vlm_coder.py` (Claude API), `pipeline.py` / `cli.py` (files and stages).
2. Core is pure pandas/numpy, tested on Linux: `clean.py`, `text/lexicon.py`, `text/aggregate.py`,
   `text/annotation.py`, `bdia.py`, `agreement.py`, `models/*`.
3. Writers: `report.py` → `results/`.
4. All defaults in `config/*.yaml`. Legacy formulas only in `config/legacy.yaml`, only for the audit.
5. Statistics follow PRD §6 (unit of analysis = hotel for hotel outcomes; grouped CV; outcome-blind indices).

## Stages (each ends with: tests green, PROGRESS.md updated, commit and push)
| Stage | Read PRD | Build | Done when |
|---|---|---|---|
| S0 Rebuild ✅ | §1–§2, §4.1, §5–§6 | restructure, audit, baseline pipeline, tests, docs | T1–T6, T9 pass; `results/REPORT.md` |
| S1 Gold set | §5.2.4, §8 T7 | `biophilic annotate-sample`; user annotates; inter-annotator κ | annotated CSV in `data/interim/`; lexicon scored |
| S2 LLM text | §5.2.2, §9, §11-Q2 | 100-review cost test → `absa-sample` → `evaluate-text` → full `absa-submit/collect`; rerun models on LLM mentions | T7 passes; REPORT §6–§7 with LLM mentions |
| S3 Photos + VLM | §4.2, §5.4, §8 T8 | photo manifest; `vision-code`; `vision-agreement`; prompt tuning on 6 hotels, then all 36 | T8 passes or a documented failure analysis |
| S4 Models v2 | §3, §5.5, §6 | rerun RQ2–RQ5 with VLM and LLM measures; sensitivity (index variants, spaces, price) | REPORT v2; pre-registered tests reported first |
| S5 Paper outputs | §7 | figures and tables, methods text, handbook v2 recommendations | user approves figures (golden) |
| S6 (optional) Scale-up | §3 RQ6, §4.3 | Inside Airbnb Hawaii or more hotels | separate PRD section approved by user first |
