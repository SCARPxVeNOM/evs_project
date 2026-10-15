"""The ISRIC SoilGrids soil layer and the cross-check it provides.

The notable thing pinned here is a negative: the classification must *not* be
driving the infiltration rate. See ``scripts/ingest_soilgrids.py`` for why -
across all 22 hoblis it resolves to two argic classes at 32-55% confidence, and
using it would collapse a 23 mm/hr spread into 6 and stop the rule engine's
5 mm/hr pit screen from discriminating anything.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from app.config import DATA_DIR
from app.datasets import load_hoblis, load_soil_wrb, site_context_for_hobli

ROOT = Path(__file__).resolve().parents[1]


def _load_script():
    path = ROOT / "scripts" / "ingest_soilgrids.py"
    spec = importlib.util.spec_from_file_location("ingest_soilgrids", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest():
    return _load_script()


# ------------------------------------------------------------------ raw source


def test_the_raw_classification_is_vendored():
    raw = DATA_DIR / "raw" / "isric_soilgrids_wrb_hoblis.json"
    assert raw.exists()
    assert raw.stat().st_size > 2_000


def test_every_hobli_has_a_classification():
    soil = load_soil_wrb()
    assert len(soil) == 22
    for key in load_hoblis()["hobli_key"]:
        assert key in soil, f"{key} has no SoilGrids row"


def test_the_classification_reports_its_own_low_confidence():
    """
    Confidence is genuinely poor. Asserting the ceiling keeps anyone from later
    treating this as authoritative: if ISRIC ever returns sharper classes this
    test fails and the decision to only cross-check should be revisited.
    """
    confidences = [row["wrb_confidence_pct"] for row in load_soil_wrb().values()]
    assert min(confidences) >= 20
    assert max(confidences) <= 70


def test_only_argic_classes_are_returned_here():
    classes = {row["wrb_class"] for row in load_soil_wrb().values()}
    assert classes == {"Luvisols", "Lixisols"}


def test_the_ingestion_reports_no_problems(ingest):
    data = ingest.read_raw()
    with (DATA_DIR / "hoblis.csv").open(newline="", encoding="utf-8") as fh:
        hoblis = list(csv.DictReader(fh))
    assert ingest.check(ingest.build(data, hoblis)) == []


# -------------------------------------------------- the rate is NOT taken from it


def test_infiltration_still_comes_from_the_texture_estimate():
    """The spread the texture estimates give is the signal that must survive."""
    rates = sorted(load_hoblis()["infiltration_mm_per_hr"].unique())
    assert len(rates) >= 5
    assert max(rates) - min(rates) >= 20


def test_the_soil_table_records_the_assumed_rate_it_checks_against():
    soil = load_soil_wrb()
    hoblis = load_hoblis().set_index("hobli_key")
    for key, row in soil.items():
        assert row["assumed_infiltration_mm_per_hr"] == pytest.approx(
            float(hoblis.loc[key, "infiltration_mm_per_hr"])
        )


# ----------------------------------------------------------------- cross-check


def test_some_but_not_all_hoblis_are_disputed():
    """A flag that fires everywhere or nowhere would carry no information."""
    flags = [row["infiltration_disputed"] for row in load_soil_wrb().values()]
    disputed = sum(1 for f in flags if str(f).lower() == "yes")
    assert 0 < disputed < 22


def test_a_disputed_site_warns_about_the_percolation_test():
    site = site_context_for_hobli("Attibele")
    assert site.soil_infiltration_disputed is True
    joined = " ".join(site.notes)
    assert "shrink-swell clay" in joined
    assert "percolation test matters more than usual" in joined


def test_an_undisputed_site_does_not_warn():
    site = site_context_for_hobli("Varthur")
    assert site.soil_infiltration_disputed is False
    assert "shrink-swell clay" not in " ".join(site.notes)


def test_a_clay_site_is_not_disputed_because_it_already_assumes_slow_soil():
    """
    Varthur carries high clay probability *and* a slow assumed rate, so the two
    agree. The flag must key on disagreement, not on clay alone.
    """
    soil = load_soil_wrb()["varthur"]
    assert soil["vertisol_probability_pct"] > 12
    assert soil["assumed_infiltration_mm_per_hr"] < 20
    assert str(soil["infiltration_disputed"]).lower() == "no"


def test_the_soil_evidence_travels_out_through_the_api():
    payload = site_context_for_hobli("Kengeri").as_dict()
    assert payload["soil_wrb_class"] == "Lixisols"
    assert payload["soil_wrb_confidence_pct"] > 0
    assert "SoilGrids" in payload["soil_source_ref"]


def test_a_missing_soil_table_is_tolerated(monkeypatch):
    """The app must still run for anyone who has not fetched the soil data."""
    import app.datasets as datasets

    datasets.load_soil_wrb.cache_clear()
    monkeypatch.setattr(datasets, "load_soil_wrb", lambda: {})
    site = datasets.site_context_for_hobli("Kengeri")
    assert site.soil_wrb_class is None
    assert site.soil_infiltration_disputed is False
    # monkeypatch restores the real lru_cache'd loader on teardown; its cache was
    # cleared above and simply repopulates on next use.
