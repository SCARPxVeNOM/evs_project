"""The classifier layer.

The model artifact is optional - the service runs on rules alone - so the tests
that need it skip when it has not been trained.
"""

from __future__ import annotations

import pytest

from app.config import param
from app.core import ml as ml_layer
from app.core.engine import HouseholdInputs, recommend
from app.core.ml import FEATURE_NAMES, feature_row, features_from_context
from app.core.recommender import ALL_LABELS, DecisionContext, decide_structure


def context(**overrides) -> DecisionContext:
    args = {
        "roof_area_m2": 120.0,
        "roof_runoff_coeff": 0.85,
        "open_area_m2": 40.0,
        "infiltration_mm_per_hr": 20.0,
        "gw_depth_m": 25.0,
        "gw_category": "over_exploited",
        "annual_rainfall_mm": 950.0,
        "annual_roof_harvest_litres": 82_000.0,
        "annual_non_potable_demand_litres": 108_000.0,
        "daily_non_potable_demand_litres": 297.0,
        "persons": 4,
        "house_type": "independent",
        "water_source": "cauvery_and_borewell",
        "has_borewell": True,
    }
    args.update(overrides)
    return DecisionContext(**args)


@pytest.fixture(scope="module")
def model():
    loaded = ml_layer.load_model()
    if loaded is None:
        pytest.skip("no trained model; run 'python -m ml.train'")
    return loaded


# ------------------------------------------------------------------ features


def test_feature_row_matches_the_declared_names():
    row = feature_row(context())
    assert len(row) == len(FEATURE_NAMES)
    assert all(isinstance(value, float) for value in row)


def test_features_are_all_numeric():
    for value in features_from_context(context()).values():
        assert isinstance(value, float)


def test_categories_are_ordinal_encoded():
    safe = features_from_context(context(gw_category="safe"))["gw_category_code"]
    over = features_from_context(context(gw_category="over_exploited"))["gw_category_code"]
    assert over > safe


def test_an_unknown_category_gets_its_own_code():
    code = features_from_context(context(gw_category="unsurveyed"))["gw_category_code"]
    assert code == 4


def test_borewell_flag_is_binary():
    assert features_from_context(context(has_borewell=True))["has_borewell"] == 1.0
    assert features_from_context(context(has_borewell=False))["has_borewell"] == 0.0


# --------------------------------------------------------------------- model


def test_model_predicts_a_known_label(model):
    prediction = model.predict(context())
    assert prediction.label in ALL_LABELS
    assert 0 <= prediction.probability <= 1
    assert pytest.approx(sum(prediction.probabilities.values()), abs=0.01) == 1.0


def test_model_reproduces_the_rules_on_a_typical_site(model):
    ctx = context()
    assert model.predict(ctx).label == decide_structure(ctx).label


def test_model_meets_the_accuracy_floor(model):
    metrics = ml_layer.read_metrics()
    assert metrics, "metrics.json missing"
    assert metrics["test_accuracy"] >= param("ml", "min_accuracy")
    assert metrics["cv_mean_accuracy"] >= param("ml", "min_accuracy")


def test_model_learned_every_label(model):
    metrics = ml_layer.read_metrics()
    assert set(metrics["labels"]) == set(ALL_LABELS)


def test_feature_importances_are_reported_and_sum_to_one(model):
    importances = model.feature_importances()
    assert len(importances) == len(FEATURE_NAMES)
    assert sum(f["importance"] for f in importances) == pytest.approx(1.0, abs=0.01)
    assert importances[0]["importance"] >= importances[-1]["importance"]


def test_the_most_important_features_are_the_physical_ones(model):
    top = {f["feature"] for f in model.feature_importances()[:5]}
    assert top & {"roof_area_m2", "demand_coverage", "annual_roof_harvest_litres"}


def test_model_info_is_serialisable(model):
    info = model.info()
    assert info["available"] is True
    assert info["labels"]
    import json

    json.loads(json.dumps(info))


def test_model_info_without_an_artifact_explains_itself(monkeypatch):
    monkeypatch.setattr(ml_layer, "load_model", lambda: None)
    info = ml_layer.model_info()
    assert info["available"] is False
    assert "ml.train" in info["reason"]


# ------------------------------------------------- integration with the engine


def test_auto_mode_reports_agreement(jayanagar, model):
    report = recommend(jayanagar, HouseholdInputs(roof_area_m2=120, open_area_m2=40, engine="auto"))
    assert report["ml"]["used"] == "rules"
    assert "agrees_with_rules" in report["ml"]
    assert report["ml"]["prediction"]["label"] in ALL_LABELS


def test_ml_mode_is_allowed_to_decide(jayanagar, model):
    report = recommend(jayanagar, HouseholdInputs(roof_area_m2=120, open_area_m2=40, engine="ml"))
    assert report["meta"]["engine_used"] in {"ml", "rules"}
    assert report["recommendation"]["label"] in ALL_LABELS


def test_ml_mode_cannot_recommend_an_unbuildable_structure(jayanagar, model):
    """A site with no borewell must never be given a borewell recharge structure."""
    report = recommend(
        jayanagar,
        HouseholdInputs(
            roof_area_m2=120,
            open_area_m2=40,
            has_borewell=False,
            water_source="cauvery",
            engine="ml",
        ),
    )
    assert "borewell_recharge" not in report["recommendation"]["components"]


def test_a_corrupt_artifact_does_not_break_the_service(tmp_path, monkeypatch):
    bad = tmp_path / "broken.joblib"
    bad.write_bytes(b"not a model")
    monkeypatch.setattr("app.core.ml.resolve_ml_path", lambda key: bad)
    ml_layer.clear_model_cache()
    try:
        assert ml_layer.load_model() is None
    finally:
        ml_layer.clear_model_cache()
