"""The KSNDMC ingestion, and the provenance it attaches to the data it writes.

These tests read the vendored raw table in ``data/raw/`` rather than the network,
so they pin that what shipped in ``data/`` is actually what KSNDMC published.
"""

from __future__ import annotations

import csv
import importlib.util
from pathlib import Path

import pytest

from app.config import DATA_DIR
from app.datasets import load_hoblis, load_rainfall_monthly, site_context_for_hobli

ROOT = Path(__file__).resolve().parents[1]
KSNDMC_GRADES = {"ksndmc_hobli_normal", "ksndmc_taluk_normal"}


def _load_script():
    """Import scripts/ingest_ksndmc.py, which is not an installed package."""
    path = ROOT / "scripts" / "ingest_ksndmc.py"
    spec = importlib.util.spec_from_file_location("ingest_ksndmc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def ingest():
    return _load_script()


@pytest.fixture(scope="module")
def normals(ingest):
    gauges, taluks = ingest.read_normals()
    return ingest.hobli_normals(gauges, taluks)


# ----------------------------------------------------------------- raw source


def test_the_raw_ksndmc_table_is_vendored():
    """The pipeline must be reproducible without network access."""
    raw = DATA_DIR / "raw" / "ksndmc_karnataka_hobli_normals.csv"
    assert raw.exists(), "data/raw/ksndmc_karnataka_hobli_normals.csv is missing"
    assert raw.stat().st_size > 10_000


def test_every_covered_hobli_resolves_to_a_ksndmc_normal(normals):
    for row in csv.DictReader((DATA_DIR / "hoblis.csv").open(encoding="utf-8")):
        assert row["hobli"] in normals, f"{row['hobli']} has no KSNDMC normal"


def test_the_three_seasons_sum_to_the_annual_normal(normals):
    """
    Karnataka splits the year into pre-monsoon (Jan-May), south-west (Jun-Sep)
    and north-east (Oct-Dec). That those three sum to the annual normal is what
    confirms the Jan-May convention, so it is asserted rather than assumed.
    """
    for name, record in normals.items():
        seasons = record["pre"] + record["swm"] + record["nem"]
        assert seasons == pytest.approx(record["annual"], rel=0.02), name


def test_the_ingestion_reports_no_problems(ingest, normals):
    hoblis = list(csv.DictReader((DATA_DIR / "hoblis.csv").open(encoding="utf-8")))
    assert ingest.check(normals, hoblis) == []


def test_within_season_shares_sum_to_one_per_season(ingest):
    rain, _ = ingest.within_season_shares()
    for season, months in ingest.SEASONS.items():
        assert sum(rain[m] for m in months) == pytest.approx(1.0), season


def test_the_two_kasaba_hoblis_fall_back_to_their_taluk(normals):
    for hobli in ("Bengaluru East (Kasaba)", "Bengaluru South (Kasaba)"):
        assert normals[hobli]["grade"] == "ksndmc_taluk_normal"


def test_gauge_backed_hoblis_average_more_than_one_gauge(normals):
    """Most hoblis carry several telemetric gauges; the normal is their mean."""
    gauge_backed = [r for r in normals.values() if r["grade"] == "ksndmc_hobli_normal"]
    assert len(gauge_backed) == 20
    assert sum(r["gauges"] for r in gauge_backed) > 40


# ------------------------------------------------------------ shipped tables


def test_shipped_rainfall_is_graded_as_ksndmc():
    grades = set(load_rainfall_monthly()["data_grade"].unique())
    assert grades <= KSNDMC_GRADES, f"unexpected grades: {grades - KSNDMC_GRADES}"


def test_shipped_hoblis_carry_rainfall_provenance():
    hoblis = load_hoblis()
    assert set(hoblis["rainfall_data_grade"].unique()) <= KSNDMC_GRADES
    assert (hoblis["rainfall_source_ref"].str.contains("KSNDMC")).all()


def test_annual_normals_match_the_raw_table(normals):
    """Guard against hoblis.csv drifting away from the source it was built from."""
    for _, row in load_hoblis().iterrows():
        expected = normals[row["hobli"]]["annual"]
        assert float(row["annual_rainfall_mm"]) == pytest.approx(expected, abs=0.1)


def test_seasonality_now_differs_between_hoblis():
    """
    The whole point of the ingestion: before it, every hobli shared one IMD
    seasonality curve, so the SW-monsoon share was identical across all 22.
    """
    rain = load_rainfall_monthly()
    swm = rain[rain["month"].isin([6, 7, 8, 9])].groupby("hobli_key")["rainfall_mm"].sum()
    total = rain.groupby("hobli_key")["rainfall_mm"].sum()
    share = (swm / total).round(4)
    assert share.nunique() > 10, "seasonality looks like one shared curve again"
    assert 0.45 <= share.min() and share.max() <= 0.70


# ----------------------------------------------------------------- site notes


def test_a_gauge_backed_site_says_its_rainfall_is_measured():
    site = site_context_for_hobli("Kengeri")
    assert site.rainfall_data_grade == "ksndmc_hobli_normal"
    joined = " ".join(site.notes)
    assert "rain-gauge normal" in joined
    assert "within each season" in joined


def test_a_taluk_backed_site_admits_the_coarser_source():
    site = site_context_for_hobli("Bengaluru South (Kasaba)")
    assert site.rainfall_data_grade == "ksndmc_taluk_normal"
    assert "no rain gauge of its own" in " ".join(site.notes)


def test_soil_and_groundwater_are_still_flagged_as_estimates():
    site = site_context_for_hobli("Kengeri")
    assert site.data_grade == "reference_estimate"
    assert "percolation test" in " ".join(site.notes)


def test_provenance_travels_out_through_the_api_payload():
    payload = site_context_for_hobli("Varthur").as_dict()
    assert payload["rainfall_data_grade"] == "ksndmc_hobli_normal"
    assert "KSNDMC" in payload["rainfall_source_ref"]
