# Progress / Handoff

Update at the end of every session: keep ≤40 lines total, newest first. Replace stale lines.

## Status
- v0.2.0 (2026-11-02, 45 tests green): S0 rebuild done on branch `claude/dazzling-albattani-33cevo`.
  Repo restructured; thesis/slides/handbook in `docs/papers`; audit `docs/LEGACY_AUDIT.md`; PRD v1;
  baseline `results/REPORT.md` (legacy reproduced exactly; RF audit; outcome-blind index r = 0.65 on 26 hotels;
  0/29 items survive FDR; review-level FE model; age/patina mentions converge with photo-coded F, ρ ≈ 0.6).
- Claude adapters (`text/llm_absa.py`, `vision/vlm_coder.py`) are written and unit-tested offline only —
  never called against the API yet (no key in the dev container).
- VERIFY-USER: read PRD §2 and LEGACY_AUDIT; answer PRD §11 (Q1–Q7).

## Open questions (asked the user, PRD §11)
- Q1 raw review file contains reviewer first names — default: keep file, names dropped in all outputs.
- Q2 API model / budget — default: claude-opus-5-5, 100-review cost test before any full batch.
- Q3 photos: original 2024 judge photos available? — default: user collects official photos per data/external/README.
- Q4 gold-set annotators — default: user + one colleague, 400 reviews each.
- Q5 survey raw data — default: not used.  Q6 scale-up — default: later.  Q7 target — default: journal paper.

## Next
- S1: `biophilic annotate-sample` → user annotates `data/interim/annotation_sample.csv` → `biophilic evaluate-text`.
- S2 (needs key + go-ahead): `absa-sample` on the gold set, then full batch.
