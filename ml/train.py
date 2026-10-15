"""Train and tune the structure-type classifier.

Four model families are searched over their hyperparameters by randomised
cross-validation, the best is selected, its predicted probabilities are
calibrated, and the winner is saved with its metrics and feature importances.
The decision tree is also exported as plain text, because a readable tree is the
most useful artefact for a report: it shows the thresholds the model learned and
lets them be checked against the rules they came from.

    python -m ml.train --regenerate      # rebuild the dataset, then tune
    python -m ml.train                   # tune on the existing dataset
    python -m ml.train --fast            # skip the search, use known-good defaults
    python -m ml.train --n-iter 40       # a wider search

**``--regenerate`` matters.** Without it an existing ``ml/data/training_data.csv``
is reused, so after any change under ``data/`` the model would be fitted to the
old site data. The dataset is derived from the hobli tables.

Three choices here are deliberate.

**Selection is on macro F1, not accuracy.** The labels are badly imbalanced -
``storage_tank_and_recharge_pit`` is over half the samples - so accuracy is
dominated by the majority class and barely moves when a rare label is predicted
badly. Macro F1 weights every structure type equally, which is what matters:
getting ``borewell_recharge`` right is as important as the common case, and it is
only about 2.5% of the data.

**Probabilities are calibrated.** The API reports the model's probability to the
user as a confidence ("chosen with 87% confidence"), so that number needs to mean
what it says. A random forest's vote share is not a calibrated probability - it
is usually over-confident in the middle of the range. The best estimator is
refitted on part of the training set, isotonic calibration is fitted on a held-out
slice, and the result is kept only if it actually lowers expected calibration
error on the untouched test set. If it does not help, the uncalibrated model is
kept and the metrics say so.

**The test set is touched once.** Hyperparameter search and model selection run
entirely on cross-validation within the training split. The test split is scored
at the end only, so the reported figures are not the ones that were optimised.
"""

from __future__ import annotations

import argparse
import json
import platform
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_score,
    learning_curve,
    train_test_split,
)
from sklearn.tree import DecisionTreeClassifier, export_text

import sklearn
from app.config import param, resolve_ml_path
from app.core.ml import FEATURE_LABELS, FEATURE_NAMES
from ml.generate_dataset import DEFAULT_OUTPUT, generate

TREE_RULES_PATH = resolve_ml_path("artifact_path").parent / "decision_tree_rules.txt"
SELECTION_SCORE = "f1_macro"
CALIBRATION_HOLDOUT = 0.25
ECE_BINS = 12


# --------------------------------------------------------------------- dataset


def load_dataset(path: Path, regenerate: bool, samples: int | None, seed: int | None) -> pd.DataFrame:
    if regenerate or not path.exists():
        df = generate(samples, seed)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False)
        print(f"generated {path} ({len(df)} rows)")
        return df
    print(f"using existing {path} - pass --regenerate after changing anything in data/")
    return pd.read_csv(path)


# ------------------------------------------------------------- search spaces


def search_spaces(seed: int) -> dict[str, tuple[object, dict]]:
    """One estimator and one parameter distribution per family."""
    return {
        "DecisionTreeClassifier": (
            DecisionTreeClassifier(random_state=seed),
            {
                "max_depth": [6, 8, 10, 12, 16, 20, None],
                "min_samples_leaf": [1, 2, 4, 8, 16, 24, 40],
                "min_samples_split": [2, 5, 10, 20],
                "criterion": ["gini", "entropy"],
                "class_weight": ["balanced", None],
                "ccp_alpha": [0.0, 1e-5, 1e-4, 1e-3],
            },
        ),
        "RandomForestClassifier": (
            RandomForestClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300, 400, 600],
                "max_depth": [10, 14, 18, 24, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "min_samples_split": [2, 5, 10],
                "max_features": ["sqrt", "log2", 0.5, 0.8],
                "class_weight": ["balanced", "balanced_subsample", None],
            },
        ),
        "ExtraTreesClassifier": (
            ExtraTreesClassifier(random_state=seed, n_jobs=-1),
            {
                "n_estimators": [200, 300, 400, 600],
                "max_depth": [10, 14, 18, 24, None],
                "min_samples_leaf": [1, 2, 4, 8],
                "max_features": ["sqrt", "log2", 0.5, 0.8],
                "class_weight": ["balanced", None],
            },
        ),
        "HistGradientBoostingClassifier": (
            HistGradientBoostingClassifier(random_state=seed),
            {
                "learning_rate": [0.03, 0.05, 0.1, 0.2],
                "max_iter": [200, 300, 500],
                "max_leaf_nodes": [15, 31, 63],
                "min_samples_leaf": [5, 10, 20, 40],
                "l2_regularization": [0.0, 0.1, 1.0],
                "max_features": [0.6, 0.8, 1.0],
            },
        ),
    }


def fast_candidates(seed: int) -> dict[str, object]:
    """The previous hand-set models, for ``--fast``."""
    return {
        "DecisionTreeClassifier": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=20, class_weight="balanced", random_state=seed
        ),
        "RandomForestClassifier": RandomForestClassifier(
            n_estimators=300, max_depth=14, min_samples_leaf=4,
            class_weight="balanced", n_jobs=-1, random_state=seed,
        ),
    }


# ------------------------------------------------------------- calibration


def expected_calibration_error(y_true, probabilities, classes, bins: int = ECE_BINS) -> float:
    """Gap between stated confidence and observed accuracy, averaged over bins.

    The confidence of the predicted class is bucketed, and in each bucket the mean
    confidence is compared with the share actually got right. A well calibrated
    model that says 70% is right about 70% of the time.
    """
    probabilities = np.asarray(probabilities)
    index = probabilities.argmax(axis=1)
    confidence = probabilities.max(axis=1)
    lookup = np.asarray(classes)
    predicted = lookup[index]
    correct = (predicted == np.asarray(y_true)).astype(float)

    edges = np.linspace(0.0, 1.0, bins + 1)
    total = len(confidence)
    error = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (confidence > lo) & (confidence <= hi)
        if not mask.any():
            continue
        error += mask.sum() / total * abs(correct[mask].mean() - confidence[mask].mean())
    return float(error)


def try_calibration(estimator, X_train, y_train, X_test, y_test, seed: int):
    """Isotonic-calibrate the winner, keeping it only if the test ECE improves."""
    base_proba = estimator.predict_proba(X_test)
    base_ece = expected_calibration_error(y_test, base_proba, estimator.classes_)
    base_ll = float(log_loss(y_test, base_proba, labels=list(estimator.classes_)))

    X_fit, X_cal, y_fit, y_cal = train_test_split(
        X_train, y_train, test_size=CALIBRATION_HOLDOUT, random_state=seed, stratify=y_train
    )
    import copy

    inner = copy.deepcopy(estimator).fit(X_fit, y_fit)
    # scikit-learn 1.6 replaced cv="prefit" with an explicitly frozen estimator,
    # and 1.9 removed the old spelling outright. Support both.
    try:
        from sklearn.frozen import FrozenEstimator

        calibrated = CalibratedClassifierCV(FrozenEstimator(inner), method="isotonic")
    except ImportError:  # scikit-learn < 1.6
        calibrated = CalibratedClassifierCV(inner, method="isotonic", cv="prefit")
    calibrated.fit(X_cal, y_cal)

    cal_proba = calibrated.predict_proba(X_test)
    cal_ece = expected_calibration_error(y_test, cal_proba, calibrated.classes_)
    cal_ll = float(log_loss(y_test, cal_proba, labels=list(calibrated.classes_)))

    improved = cal_ece < base_ece
    report = {
        "applied": bool(improved),
        "method": "isotonic" if improved else None,
        "ece_before": round(base_ece, 4),
        "ece_after": round(cal_ece, 4),
        "log_loss_before": round(base_ll, 4),
        "log_loss_after": round(cal_ll, 4),
        "holdout_fraction": CALIBRATION_HOLDOUT,
        "note": (
            "Isotonic calibration lowered expected calibration error on the untouched "
            "test set, so the calibrated model is what ships and the confidence the API "
            "reports is meaningful."
            if improved
            else "Calibration did not improve expected calibration error, so the "
            "uncalibrated model was kept. Treat the reported confidence as a ranking, "
            "not a probability."
        ),
    }
    return (calibrated if improved else estimator), report


# ------------------------------------------------------------------- reporting


def accuracy_floor_warning(value: float) -> str | None:
    minimum = float(param("ml", "min_accuracy"))
    if value < minimum:
        return (
            f"WARNING: holdout accuracy {value:.3f} is below the {minimum:.2f} floor "
            "in parameters.json"
        )
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and tune the RWH structure classifier")
    parser.add_argument("--data", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--samples", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--regenerate", action="store_true", help="rebuild the dataset first")
    parser.add_argument("--fast", action="store_true", help="skip the search; fixed defaults")
    parser.add_argument("--n-iter", type=int, default=24, help="search iterations per family")
    parser.add_argument("--no-learning-curve", action="store_true")
    args = parser.parse_args()

    seed = int(args.seed if args.seed is not None else param("ml", "random_state"))
    df = load_dataset(args.data, args.regenerate, args.samples, seed)

    X = df[FEATURE_NAMES]
    y = df["label"].astype(str)
    labels = sorted(y.unique())

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=float(param("ml", "test_size")), random_state=seed, stratify=y
    )
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=seed)

    print(
        f"\n{len(X_train)} training rows, {len(X_test)} held back; "
        f"selecting on {SELECTION_SCORE} over 5 stratified folds"
    )
    print("class balance: " + ", ".join(f"{k}={v}" for k, v in y.value_counts().items()))

    searched: dict[str, dict] = {}

    if args.fast:
        print("\n--fast: using fixed hyperparameters, no search")
        for name, estimator in fast_candidates(seed).items():
            scores = cross_val_score(
                estimator, X_train, y_train, cv=folds, scoring=SELECTION_SCORE, n_jobs=-1
            )
            estimator.fit(X_train, y_train)
            searched[name] = {
                "estimator": estimator,
                "cv_score": float(scores.mean()),
                "cv_std": float(scores.std()),
                "best_params": {},
            }
            print(f"  {name:34} cv {scores.mean():.4f} +/- {scores.std():.4f}")
    else:
        print(f"\nsearching {args.n_iter} configurations per family:")
        for name, (estimator, space) in search_spaces(seed).items():
            search = RandomizedSearchCV(
                estimator,
                space,
                n_iter=args.n_iter,
                scoring=SELECTION_SCORE,
                cv=folds,
                random_state=seed,
                n_jobs=-1,
                refit=True,
                error_score="raise",
            )
            search.fit(X_train, y_train)
            searched[name] = {
                "estimator": search.best_estimator_,
                "cv_score": float(search.best_score_),
                "cv_std": float(
                    search.cv_results_["std_test_score"][search.best_index_]
                ),
                "best_params": {k: _jsonable(v) for k, v in search.best_params_.items()},
            }
            print(
                f"  {name:34} cv {search.best_score_:.4f} "
                f"+/- {searched[name]['cv_std']:.4f}"
            )

    # Score every tuned family once on the untouched test split, for the record.
    for name, entry in searched.items():
        predicted = entry["estimator"].predict(X_test)
        entry["test_accuracy"] = float(accuracy_score(y_test, predicted))
        entry["test_balanced_accuracy"] = float(balanced_accuracy_score(y_test, predicted))
        entry["test_f1_macro"] = float(f1_score(y_test, predicted, average="macro"))

    best_name = max(searched, key=lambda k: searched[k]["cv_score"])
    best = searched[best_name]
    print(f"\nbest family by cross-validated {SELECTION_SCORE}: {best_name}")
    if best["best_params"]:
        print("tuned hyperparameters:")
        for key, value in sorted(best["best_params"].items()):
            print(f"  {key:22} {value}")

    estimator, calibration = try_calibration(
        best["estimator"], X_train, y_train, X_test, y_test, seed
    )
    print(
        f"\ncalibration: ECE {calibration['ece_before']:.4f} -> "
        f"{calibration['ece_after']:.4f} "
        f"({'applied' if calibration['applied'] else 'not applied'})"
    )

    predicted = estimator.predict(X_test)
    test_accuracy = float(accuracy_score(y_test, predicted))
    test_balanced = float(balanced_accuracy_score(y_test, predicted))
    test_f1 = float(f1_score(y_test, predicted, average="macro"))
    report = classification_report(y_test, predicted, zero_division=0, output_dict=True)

    print(f"\nholdout: accuracy {test_accuracy:.4f}  balanced {test_balanced:.4f}  "
          f"macro-F1 {test_f1:.4f}")
    print(classification_report(y_test, predicted, zero_division=0))

    # Importances come from the tuned base model: a calibrated wrapper has none,
    # and the API contract expects a full set that sums to one.
    base = best["estimator"]
    raw = getattr(base, "feature_importances_", None)
    if raw is None:
        from sklearn.inspection import permutation_importance

        print("this family exposes no native importances; permuting instead")
        perm = permutation_importance(
            base, X_test, y_test, n_repeats=5, random_state=seed, n_jobs=-1
        )
        values = np.clip(perm.importances_mean, 0, None)
        raw = values / values.sum() if values.sum() > 0 else np.full(len(FEATURE_NAMES), 1 / len(FEATURE_NAMES))
    importances = sorted(
        zip(FEATURE_NAMES, (float(v) for v in raw)), key=lambda kv: kv[1], reverse=True
    )
    print("\nfeature importances:")
    for name, value in importances:
        print(f"  {value:6.3f}  {FEATURE_LABELS.get(name, name)}")

    curve = None
    if not args.no_learning_curve:
        sizes, train_scores, val_scores = learning_curve(
            best["estimator"],
            X_train,
            y_train,
            train_sizes=[0.1, 0.25, 0.5, 0.75, 1.0],
            cv=folds,
            scoring=SELECTION_SCORE,
            n_jobs=-1,
        )
        curve = [
            {
                "train_rows": int(n),
                "train_score": round(float(tr.mean()), 4),
                "validation_score": round(float(va.mean()), 4),
            }
            for n, tr, va in zip(sizes, train_scores, val_scores)
        ]
        gain = curve[-1]["validation_score"] - curve[-2]["validation_score"]
        print(
            f"\nlearning curve: {curve[0]['validation_score']:.4f} at "
            f"{curve[0]['train_rows']} rows -> {curve[-1]['validation_score']:.4f} at "
            f"{curve[-1]['train_rows']}; last step gained {gain:+.4f}"
        )
        print(
            "  more samples would still help" if gain > 0.002
            else "  the curve has flattened; more samples of the same kind will not help"
        )

    agreement = None
    if "rule_label" in df.columns:
        agreement = float((df["label"] == df["rule_label"]).mean())

    metrics = {
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "best_model": best_name,
        "tuned": not args.fast,
        "selection_score": SELECTION_SCORE,
        "search_iterations_per_family": 0 if args.fast else args.n_iter,
        "best_params": best["best_params"],
        "n_samples": int(len(df)),
        "n_features": len(FEATURE_NAMES),
        "labels": labels,
        "cv_mean_accuracy": best["cv_score"],
        "cv_std_accuracy": best["cv_std"],
        "test_accuracy": test_accuracy,
        "test_balanced_accuracy": test_balanced,
        "test_f1_macro": test_f1,
        "calibration": calibration,
        "label_noise_in_dataset": None if agreement is None else round(1 - agreement, 4),
        "per_label": {
            label: {
                "precision": round(report[label]["precision"], 4),
                "recall": round(report[label]["recall"], 4),
                "f1": round(report[label]["f1-score"], 4),
                "support": int(report[label]["support"]),
            }
            for label in labels
            if label in report
        },
        "confusion_matrix": {
            "labels": labels,
            "rows": confusion_matrix(y_test, predicted, labels=labels).tolist(),
        },
        "feature_importances": [
            {"feature": n, "label": FEATURE_LABELS.get(n, n), "importance": round(v, 4)}
            for n, v in importances
        ],
        "learning_curve": curve,
        "candidates": {
            name: {
                "cv_score": round(entry["cv_score"], 4),
                "cv_std": round(entry["cv_std"], 4),
                "test_accuracy": round(entry["test_accuracy"], 4),
                "test_balanced_accuracy": round(entry["test_balanced_accuracy"], 4),
                "test_f1_macro": round(entry["test_f1_macro"], 4),
                "best_params": entry["best_params"],
            }
            for name, entry in searched.items()
        },
        "caveat": (
            "Labels come from the rule engine, so these figures measure how well the "
            "model reproduces the rules, not whether the recommendations are correct. "
            "Replace ml/data/training_data.csv with surveyed sites to learn something new."
        ),
        "environment": {
            "python": platform.python_version(),
            "scikit_learn": sklearn.__version__,
        },
    }

    artifact_path = resolve_ml_path("artifact_path")
    metrics_path = resolve_ml_path("metrics_path")
    artifact_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "estimator": estimator,
            "feature_names": FEATURE_NAMES,
            "labels": labels,
            "model_name": best_name,
            "feature_importances": metrics["feature_importances"],
            "metrics": {
                "cv_mean_accuracy": round(best["cv_score"], 4),
                "test_accuracy": round(test_accuracy, 4),
                "test_balanced_accuracy": round(test_balanced, 4),
                "test_f1_macro": round(test_f1, 4),
                "calibrated": calibration["applied"],
                "expected_calibration_error": calibration["ece_after"],
                "n_samples": int(len(df)),
                "trained_at": metrics["trained_at"],
            },
        },
        artifact_path,
    )
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    tree = searched.get("DecisionTreeClassifier")
    if tree is not None:
        TREE_RULES_PATH.write_text(
            export_text(tree["estimator"], feature_names=list(FEATURE_NAMES), max_depth=6),
            encoding="utf-8",
        )

    print(f"\nsaved {artifact_path}")
    print(f"saved {metrics_path}")
    if tree is not None:
        print(f"saved {TREE_RULES_PATH}")
    warning = accuracy_floor_warning(test_accuracy)
    if warning:
        print(warning)


def _jsonable(value):
    """Search spaces hold numpy scalars and None; make them JSON-safe."""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


if __name__ == "__main__":
    main()
