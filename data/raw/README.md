# Raw data (thesis appendix, 2024) — read-only

Never edit these files; every cleaning step is code in `src/biophilic/`. Data dictionary: `docs/DATA.md`.

| File (new name) | Original name / thesis appendix item | Content |
|---|---|---|
| `reviews_booking_waikiki_2020-2023.xlsx` | "35000 Comments for NLP.xlsx" (appendix 1) | 35,177 Booking.com review rows (Nov 2020 – Nov 2023, 52 properties) + hotel scores and prices; sheets with the 2024 keyword and sentiment lexicons |
| `bdia_coding_36_hotels.xlsx` | "BDIA rating form for 36 Hawaii hotels (for machine learning).xlsx" (appendix 2) | Expert BDIA coding (29 items × 3 spaces) of 26 Waikiki model hotels + 10 test hotels, with legacy index and Booking scores |
| `bdia_instrument_scoresheet.xlsx` | "BDIA rating Scoresheet_Final Version.xlsx" (appendix 3) | The final BDIA instrument (example: Wayfinder Waikiki) |
| `survey_instrument_and_results.xlsx` | "Customer Review Questionnaire.xlsx" (appendix 4) | Survey questions (S-O-R model), item–WTPP correlations, SEM path table. Individual responses are not included. |

**Personal data.** The review file contains reviewers' first names (column 2). The pipeline drops them on read
(`io.read_reviews`) and never writes them out. The file has been public in this repository since 2024; whether to
keep it public is an open question in `docs/PRD.md` §11.
