# Indirect Biophilic Attributes in Hawaii Hotel Interior Design — BDIA 2.0

Can we relate a hotel's interior design to how guests rate it **without visiting it**, using only online photos,
scores and reviews? This repository holds the appendix of Chunya Wu's 2024 doctoral thesis (*Indirect Biophilic
Attributes in Hawaii Hotel Interior Design — A Comparative Study of Waikiki Four-Star Hotels*, University of Hawaiʻi
at Mānoa) and its rebuild: an audited, reproducible pipeline with Claude-based review analysis and photo coding.

**Author:** Chunya Wu · **Version:** `src/biophilic/__init__.py` · **Status:** S0 rebuild done; S1–S3 need annotation, photos and an API key (`docs/PROGRESS.md`)

## Quick start
```bash
pip install -e ".[dev]"
python -m pytest -q
biophilic baseline          # ~3 min → results/REPORT.md
```
Claude stages: `pip install -e ".[dev,llm]"`, set `ANTHROPIC_API_KEY`, then see `biophilic --help`
(`annotate-sample`, `absa-sample`, `evaluate-text`, `absa-submit`, `absa-collect`, `vision-code`, `vision-agreement`).

## Read first
| Document | What |
|---|---|
| `docs/PRD.md` | Research questions, method, statistics rules, acceptance tests, defaults, open questions (Chinese) |
| `docs/LEGACY_AUDIT.md` | What in the 2024 code and statistics holds up and what does not, with evidence (Chinese) |
| `results/REPORT.md` | Baseline results from the 2024 data |
| `docs/papers/` | Thesis full text, EDRA55 slides and handbook (plain git files, ~21 MB) |

## Layout
| Path | What |
|---|---|
| `data/raw/` | Thesis appendix data, read-only (README maps old file names to new ones) |
| `data/interim/`, `data/processed/`, `data/external/` | Generated data and photos — local only, not in git |
| `config/` | BDIA items, review aspects, pipeline defaults, legacy formulas |
| `src/biophilic/` | Package: `io`, `clean`, `bdia`, `agreement`, `text/`, `vision/`, `models/`, `pipeline`, `cli` |
| `tests/` | Unit tests, fixtures for every known false positive, privacy checks on `results/` |
| `results/` | Aggregated outputs (no review text, no reviewer names) |
| `legacy/` | 2024 NLP script and outputs, unchanged |
| `docs/` | PRD, audit, data dictionary, stage plan, progress, handoff, changelog, papers |

### Thesis appendix → new location
| Thesis appendix item | Now |
|---|---|
| 1. 35000 Comments for NLP | `data/raw/reviews_booking_waikiki_2020-2023.xlsx` |
| 2. BDIA rating form for 36 Hawaii hotels (for machine learning) | `data/raw/bdia_coding_36_hotels.xlsx` |
| 3. BDIA rating Scoresheet_Final Version | `data/raw/bdia_instrument_scoresheet.xlsx` |
| 4. Customer Review Questionnaire | `data/raw/survey_instrument_and_results.xlsx` |
| 5. … NLP Analysis result by BDIA Keywords | `legacy/outputs/nlp_result_by_keyword.xlsx` |
| 6. … NLP Analysis result by Hotel Names | `legacy/outputs/nlp_result_by_hotel.xlsx` |
| 7. Waikiki 26 4-star Hotel review NLP analysis code | `legacy/review_keyword_nlp_2024.py` |

## Data and privacy
Reads only files in `data/`. Network access only in the Claude stages (review text and hotel photos are sent to the
Anthropic API). The raw review file contains reviewers' first names; the code drops them on read and no output
contains them. Generated tables with review text and the photos stay local (`.gitignore`).

## Third-party
| Library | Licence |
|---|---|
| pandas, NumPy, SciPy, scikit-learn, statsmodels | BSD-3-Clause |
| PyYAML, openpyxl | MIT |
| vaderSentiment | MIT |
| anthropic (optional) | MIT |
| PyABSA (optional, fork `WUChunya/PyABSA`) | MIT |

## Docs
`docs/PRD.md` · `docs/LEGACY_AUDIT.md` · `docs/DATA.md` · `docs/DEV_PROMPT.md` · `docs/PROGRESS.md` · `docs/HANDOFF.md` · `docs/CHANGELOG.md`
