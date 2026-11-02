# Data dictionary

Raw files: `data/raw/` (see its README). Every column-name mapping lives in `src/biophilic/io.py`.

## reviews (data/processed/reviews.parquet — local only)
| column | from raw | notes |
|---|---|---|
| review_id | — | sha1 of hotel, date, room, title, liked, disliked (12 hex); exact duplicates collapse (447) |
| hotel_id, hotel | Hotel Name | `hotel_id` = slug, the join key across all tables |
| review_date | Unnamed: 0 ("Reviewed: 28 November 2023") | 2020-11-30 … 2023-11-30; 219 "Reviewers' choice" rows have no date |
| traveller_type | Customer Type | Family / Couple / Group / Solo traveller |
| room_type | Reserved Room Type | |
| title | Reserved Room Type.1 | the raw sheet has two columns with the same header; the second is the review title. Booking's automatic score labels ("Exceptional", "Very good" …) are set to empty |
| liked | Positive Guest Reviews | answer to "What did you like?". 15,792 rows are Booking's placeholder "There are no comments available for this review" → empty |
| disliked | Negative Guest Reviews | answer to "What didn't you like?"; "Nothing", "N/A", "none" … → empty |
| rating | Guest Rating | 1–10; 1,777 missing |
| has_text, english, n_words | — | quality flags (`clean.clean_reviews`) |
| (dropped) | Unnamed: 1 | reviewer first name — read, never stored |

## booking_hotels (one row per property in the review file)
Booking score and sub-scores at scrape time (Nov 2023): `booking_score`, `staff`, `facilities`, `cleanliness`,
`comfort`, `value_for_money`, `location`, `wifi`; `avg_price` (USD/night, mean of the scraped price list);
`sustainability_label`; counts `n_reviews`, `n_text`, `n_english`; `is_listing` (private condo listing by name
pattern) and `eligible` (≥ 50 text reviews and not a listing).

## bdia_coding (one row per hotel × space, 108 rows)
`hotel_id`, `hotel`, `hotel_number` (1–36), `space` (guestroom | lobby | lounge), items `A01`…`G03` (0/1; legacy
column names mapped through `config/bdia_items.yaml`), `legacy_scenario` (the 2024 BDIA-Scenario).
Coding procedure (thesis §4.1, §6.1): three professionals per hotel, photos only (name, scores, price hidden),
final code = majority.

## bdia_hotels (one row per hotel, 36 rows)
`group`: `waikiki_4star_model` (hotel numbers 1–26), `hawaii_4star_test` (27–31), `lower_star_test` (32–36).
`legacy_index`, `legacy_prediction` (2024 formulas), Booking scores as above (`booking_score` = "Overall Score"),
`avg_price`, `spaces_identical` (all three spaces carry the same 29 codes: Royal Grove Waikiki, Waikiki Monarch
Hotel — likely coded once and copied).

## mentions (lexicon and LLM, same schema)
`review_id, hotel_id, field, sentence_idx, aspect, dimension, term, space, polarity (+1/0/−1), negated, method`;
LLM rows add `bdia_item`, `explicit`. `method` ∈ lexicon | lexicon_ambiguous | llm. Aspects and terms:
`config/review_aspects.yaml`.

## results/ (committed; aggregated, no review text, no names of reviewers)
`REPORT.md`, `summary.json`, `hotels.csv`, `index_assoc.csv`, `index_models_26.csv`, `index_models_36.csv`,
`external_test.csv`, `dim_assoc.csv`, `items.csv`, `reliability.csv`, `prevalence.csv`, `mention_summary.csv`,
`text_metrics_by_hotel.csv`, `review_model.csv`, `convergence.csv`; later `text_evaluation.csv`,
`vision_agreement.csv`.
