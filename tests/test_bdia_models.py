"""BDIA indices, legacy reproduction (on the committed raw data), agreement and small-sample statistics."""

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import biophilic
from biophilic import agreement, bdia, config, io
from biophilic.models import stats

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def bdia_data():
    return io.read_bdia()


def test_raw_bdia_shape_and_groups(bdia_data):
    coding, hotels = bdia_data
    assert len(coding) == 108 and len(hotels) == 36
    assert hotels["group"].value_counts().to_dict() == {"waikiki_4star_model": 26, "hawaii_4star_test": 5,
                                                        "lower_star_test": 5}
    assert set(coding[bdia.item_codes()].stack().unique()) <= {0, 1}


def test_legacy_columns_reproduced_exactly(bdia_data):
    err = bdia.legacy_reproduction(*bdia_data)
    assert err["scenario_max_abs_err"] < 1e-9
    assert err["index_max_abs_err"] < 1e-9
    assert err["prediction_max_abs_err"] < 1e-9


def test_dimension_balanced_counts_dimensions_equally():
    codes = bdia.item_codes()
    row = {c: 0 for c in codes}
    for c in [c for c in codes if c.startswith("B")]:  # all 6 natural-material items, nothing else
        row[c] = 1
    df = pd.DataFrame([row])
    assert bdia.scenario_scores(df, "equal_items").iloc[0] == pytest.approx(6 / 29)
    assert bdia.scenario_scores(df, "dimension_balanced").iloc[0] == pytest.approx(1 / 7)


def test_kr20_perfectly_consistent_items():
    X = np.array([[1, 1, 1], [0, 0, 0], [1, 1, 1], [0, 0, 0]])
    assert bdia.kr20(X) == pytest.approx(1.0)


def test_agreement_statistics():
    a = [1, 1, 0, 0, 1, 0, 1, 1]
    assert agreement.cohen_kappa(a, a) == pytest.approx(1.0)
    assert agreement.gwet_ac1(a, a) == pytest.approx(1.0)
    b = [1, 1, 1, 1, 1, 1, 1, 0]
    # prevalence skew: kappa is low while AC1 stays interpretable
    assert agreement.cohen_kappa(a, b) < agreement.gwet_ac1(a, b)


def test_bh_fdr_known_values():
    p = [0.01, 0.04, 0.03, 0.20]
    # sorted 0.01, 0.03, 0.04, 0.20 → raw 0.04, 0.06, 0.0533, 0.20 → monotone 0.04, 0.0533, 0.0533, 0.20
    assert np.allclose(stats.bh_fdr(p), [0.04, 0.16 / 3, 0.16 / 3, 0.20])


def test_loocv_r2_is_out_of_sample():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(30, 1))
    y = rng.normal(size=30)  # pure noise → predictive R² must not be positive in expectation
    assert stats.loocv_r2_ols(X, y) < 0.05


def test_single_version_source():
    text = (ROOT / "pyproject.toml").read_text()
    assert 'dynamic = ["version"]' in text and "biophilic.__version__" in text
    top = (ROOT / "docs" / "CHANGELOG.md").read_text().split("\n## ")[1]
    assert re.match(rf"v?{re.escape(biophilic.__version__)}\b", top)


def test_config_has_29_items_7_dimensions():
    items = config.bdia_items()
    assert len(items) == 29 and len({i.code for i in items}) == 29
    assert {i.dimension for i in items} == set("ABCDEFG")
