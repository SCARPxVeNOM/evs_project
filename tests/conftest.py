"""Shared fixtures."""

from __future__ import annotations

import pytest

from app.core.engine import HouseholdInputs, recommend
from app.datasets import site_context_for_hobli


@pytest.fixture(scope="session")
def jayanagar():
    """A typical BBMP-core site: red loamy soil, deep over-exploited aquifer."""
    return site_context_for_hobli("Bengaluru South (Kasaba)")


@pytest.fixture(scope="session")
def varthur():
    """Clay-loam soil in the east, where percolation is slow and the bore is deep."""
    return site_context_for_hobli("Varthur")


@pytest.fixture(scope="session")
def kengeri():
    """Lateritic gravel in the west - the best infiltration in the dataset."""
    return site_context_for_hobli("Kengeri")


@pytest.fixture
def typical_inputs() -> HouseholdInputs:
    """A 1200 sq ft roof on a 2400 sq ft plot - the common Bengaluru case."""
    return HouseholdInputs(
        roof_area_m2=111.0,
        roof_type="rcc_flat",
        open_area_m2=40.0,
        household_size=4,
        house_type="independent",
        water_source="cauvery_and_borewell",
        has_borewell=True,
        plot_area_m2=223.0,
        engine="rules",
    )


@pytest.fixture
def typical_report(jayanagar, typical_inputs):
    return recommend(jayanagar, typical_inputs)


@pytest.fixture(scope="session")
def client():
    from fastapi.testclient import TestClient

    from app.main import app

    with TestClient(app) as test_client:
        yield test_client
