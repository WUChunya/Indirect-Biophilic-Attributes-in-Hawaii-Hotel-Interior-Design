"""T4/T5: committed results contain no review text or reviewer names, and hotel-level n never exceeds hotels."""

from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
TEXT_COLUMNS = {"liked", "disliked", "title", "evidence", "term", "sentence", "reviewer", "Unnamed: 1"}

csvs = sorted(RESULTS.glob("*.csv"))


@pytest.mark.skipif(not csvs, reason="results/ not generated yet (run `biophilic baseline`)")
@pytest.mark.parametrize("path", csvs, ids=lambda p: p.name)
def test_no_text_columns(path):
    cols = set(pd.read_csv(path, nrows=0).columns)
    assert not cols & TEXT_COLUMNS, f"{path.name} has text columns {cols & TEXT_COLUMNS}"


@pytest.mark.skipif(not (RESULTS / "index_assoc.csv").exists(), reason="results/ not generated yet")
def test_hotel_level_n_is_hotels():
    n_hotels = len(pd.read_csv(RESULTS / "hotels.csv"))
    for name in ("index_assoc.csv", "dim_assoc.csv", "index_models_26.csv", "index_models_36.csv"):
        df = pd.read_csv(RESULTS / name)
        assert df["n"].max() <= n_hotels, name


@pytest.mark.skipif(not (RESULTS / "REPORT.md").exists(), reason="results/ not generated yet")
def test_report_has_no_placeholder_text():
    assert "There are no comments available" not in (RESULTS / "REPORT.md").read_text()
