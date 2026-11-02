# Legacy (2024 thesis appendix) — kept unchanged for reference

| File | Thesis appendix item | Status |
|---|---|---|
| `review_keyword_nlp_2024.py` | 7. "Waikiki 26 4-star Hotel review NLP analysis code" | Incomplete: only helper functions, no main program, keyword loading or Excel export; column names are Chinese (`客户正面评价`…) and do not match the English raw file. Does not run as is. |
| `outputs/nlp_result_by_keyword.xlsx` | 5. "… NLP Analysis result by BDIA Keywords" | Output of the 2024 method. |
| `outputs/nlp_result_by_hotel.xlsx` | 6. "… NLP Analysis result by Hotel Names" | Output of the 2024 method (24 hotel sheets, not 26). |

Why the method was replaced, with evidence: `../docs/LEGACY_AUDIT.md` §B. The replacement is
`src/biophilic/text/` (lexicon baseline + Claude aspect extraction). Do not edit these files; they document what
the thesis did.
