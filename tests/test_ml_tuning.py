"""Hyperparameter tuning, calibration and the metrics they produce.

These tests read the shipped artifacts rather than retraining - a full search
takes minutes. They skip when no model has been trained.
"""

from __future__ import annotations

import numpy as np
import pytest

from app.config import param
from app.core import ml as ml_layer
from app.core.ml import FEATURE_NAMES
from app.core.recommender import ALL_LABELS


@pytest.fixture(scope="module")
def metrics():
    data = ml_layer.read_metrics()
    if not data:
        pytest.skip("no metrics.json; run 'python -m ml.train'")
    return data


@pytest.fixture(scope="module")
def model():
    loaded = ml_layer.load_model()
    if loaded is None:
        pytest.skip("no trained model; run 'python -m ml.train'")
    return loaded


# ------------------------------------------------------------------- selection


def test_selection_is_on_macro_f1_not_accuracy(metrics):
    """
    Accuracy is dominated by one label that is over half the data, so selecting
    on it would ignore how the rare structures are predicted.
    """
    assert metrics["selection_score"] == "f1_macro"


def test_every_family_was_compared(metrics):
    families = set(metrics["candidates"])
    assert "RandomForestClassifier" in families
    assert "DecisionTreeClassifier" in families
    if metrics["tuned"]:
        assert len(families) >= 4


def test_the_winner_really_had_the_best_cross_validated_score(metrics):
    best = metrics["best_model"]
    scores = {name: entry["cv_score"] for name, entry in metrics["candidates"].items()}
    assert best in scores
    assert scores[best] == max(scores.values())


def test_the_search_recorded_the_hyperparameters_it_chose(metrics):
    if not metrics["tuned"]:
        pytest.skip("trained with --fast, no search performed")
    assert metrics["search_iterations_per_family"] > 0
    assert metrics["best_params"], "the winning configuration was not recorded"


def test_imbalanced_classes_are_all_still_learned(metrics):
    """Tuning for macro F1 is pointless if a rare label ends up never predicted."""
    assert set(metrics["labels"]) == set(ALL_LABELS)
    for label, scores in metrics["per_label"].items():
        assert scores["support"] > 0
        assert scores["recall"] > 0.5, f"{label} is barely being found"


def test_balanced_accuracy_is_reported_alongside_plain_accuracy(metrics):
    assert 0 < metrics["test_balanced_accuracy"] <= 1
    assert 0 < metrics["test_f1_macro"] <= 1
    assert metrics["test_accuracy"] >= param("ml", "min_accuracy")


# ----------------------------------------------------------------- calibration


def test_calibration_was_attempted_and_recorded(metrics):
    cal = metrics["calibration"]
    assert "applied" in cal
    assert cal["ece_before"] >= 0
    assert cal["ece_after"] >= 0
    assert cal["note"]


def test_calibration_was_only_kept_if_it_helped(metrics):
    """The decision rule itself: never ship a calibration that made ECE worse."""
    cal = metrics["calibration"]
    if cal["applied"]:
        assert cal["ece_after"] < cal["ece_before"]
        assert cal["method"] == "isotonic"
    else:
        assert cal["ece_after"] >= cal["ece_before"]


def test_a_calibrated_model_states_confidence_worth_believing(metrics):
    cal = metrics["calibration"]
    if not cal["applied"]:
        pytest.skip("calibration did not improve ECE, so none was applied")
    # The API shows this probability to users as a confidence figure.
    assert cal["ece_after"] < 0.05


def test_probabilities_still_form_a_distribution(model):
    from app.core.recommender import DecisionContext

    ctx = DecisionContext(
        roof_area_m2=120.0, roof_runoff_coeff=0.85, open_area_m2=40.0,
        infiltration_mm_per_hr=20.0, gw_depth_m=25.0, gw_category="over_exploited",
        annual_rainfall_mm=880.0, annual_roof_harvest_litres=78_000.0,
        annual_non_potable_demand_litres=108_000.0, daily_non_potable_demand_litres=297.0,
        persons=4, house_type="independent", water_source="cauvery_and_borewell",
        has_borewell=True,
    )
    prediction = model.predict(ctx)
    assert sum(prediction.probabilities.values()) == pytest.approx(1.0, abs=0.01)
    assert 0 <= prediction.probability <= 1
    assert prediction.probability == max(prediction.probabilities.values())


def test_importances_survive_a_calibrated_estimator(model, metrics):
    """
    A calibrated wrapper exposes no feature_importances_, so the trainer stores
    them in the bundle. The API contract needs a full set that sums to one either
    way - this is the regression guard for that.
    """
    importances = model.feature_importances()
    assert len(importances) == len(FEATURE_NAMES)
    assert sum(f["importance"] for f in importances) == pytest.approx(1.0, abs=0.01)
    assert {f["feature"] for f in importances} == set(FEATURE_NAMES)
    if metrics["calibration"]["applied"]:
        assert not hasattr(model.estimator, "feature_importances_")


# --------------------------------------------------------------- learning curve


def test_the_learning_curve_is_recorded_and_monotone_in_size(metrics):
    curve = metrics.get("learning_curve")
    if curve is None:
        pytest.skip("trained with --no-learning-curve")
    sizes = [point["train_rows"] for point in curve]
    assert sizes == sorted(sizes)
    assert len(curve) >= 3
    for point in curve:
        assert 0 < point["validation_score"] <= 1


def test_more_data_helps_less_than_it_did_at_the_start(metrics):
    """A curve that still climbs steeply at the end would mean the set is too small."""
    curve = metrics.get("learning_curve")
    if curve is None or len(curve) < 3:
        pytest.skip("no usable learning curve")
    first_gain = curve[1]["validation_score"] - curve[0]["validation_score"]
    last_gain = curve[-1]["validation_score"] - curve[-2]["validation_score"]
    assert last_gain <= max(first_gain, 0.01)


# ------------------------------------------------------------------- honesty


def test_the_metrics_file_carries_its_own_caveat(metrics):
    """
    The figures measure imitation of the rule engine, not correctness of the
    advice. That has to travel with the numbers, not just sit in the README.
    """
    caveat = metrics["caveat"]
    assert "rule engine" in caveat
    assert "not whether the recommendations are correct" in caveat


def test_label_noise_is_reported(metrics):
    noise = metrics["label_noise_in_dataset"]
    assert noise is None or 0 <= noise <= 0.2


def test_the_confusion_matrix_accounts_for_the_whole_test_split(metrics):
    rows = np.array(metrics["confusion_matrix"]["rows"])
    assert rows.shape == (len(metrics["labels"]), len(metrics["labels"]))
    total = rows.sum()
    support = sum(s["support"] for s in metrics["per_label"].values())
    assert total == support
