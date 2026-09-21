from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline

from ner_landslide.config import Settings
from ner_landslide.features import MODEL_FEATURES

LOGGER = logging.getLogger(__name__)


@dataclass
class LandslideRiskModel:
    estimator: Pipeline
    calibrator: LogisticRegression | None
    features: list[str]
    warning_threshold: float
    trained_at: str

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        raw = self.estimator.predict_proba(features[self.features])[:, 1]
        if self.calibrator is None:
            return np.asarray(raw, dtype=float)
        return np.asarray(
            self.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
            dtype=float,
        )


def _metric_summary(
    labels: pd.Series, probabilities: np.ndarray, threshold: float
) -> dict[str, Any]:
    predictions = probabilities >= threshold
    unique = labels.nunique()
    return {
        "rows": int(len(labels)),
        "events": int(labels.sum()),
        "event_rate": float(labels.mean()),
        "pr_auc": float(average_precision_score(labels, probabilities)) if unique > 1 else None,
        "roc_auc": float(roc_auc_score(labels, probabilities)) if unique > 1 else None,
        "brier_score": float(brier_score_loss(labels, probabilities)),
        "recall": float(recall_score(labels, predictions, zero_division=0)),
        "precision": float(precision_score(labels, predictions, zero_division=0)),
        "confusion_matrix": confusion_matrix(labels, predictions, labels=[0, 1]).tolist(),
    }


def _select_threshold(
    labels: pd.Series,
    probabilities: np.ndarray,
    minimum_recall: float,
) -> float:
    if labels.nunique() < 2:
        return 0.5
    precision, recall, thresholds = precision_recall_curve(labels, probabilities)
    eligible = np.flatnonzero(recall[:-1] >= minimum_recall)
    if not len(eligible):
        return 0.5
    best = eligible[np.argmax(precision[:-1][eligible])]
    return float(thresholds[best])


def _prepare(frame: pd.DataFrame) -> pd.DataFrame:
    missing = set(MODEL_FEATURES + ["label", "date", "state"]) - set(frame)
    if missing:
        raise ValueError(f"Training frame is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["date"] = pd.to_datetime(result["date"])
    result[MODEL_FEATURES] = result[MODEL_FEATURES].replace([np.inf, -np.inf], np.nan)
    return result


def train_model(
    settings: Settings,
    features: pd.DataFrame,
) -> tuple[LandslideRiskModel, dict[str, Any]]:
    frame = _prepare(features)
    active_features = [feature for feature in MODEL_FEATURES if frame[feature].notna().any()]
    if len(frame) < 20 or frame["label"].nunique() < 2:
        raise RuntimeError("At least 20 rows containing events and controls are required")

    temporal_cutoff = frame["date"].quantile(1 - settings.training.temporal_holdout_fraction)
    temporal_mask = frame["date"] >= temporal_cutoff
    state_mask = frame["state"].isin(settings.training.holdout_states)
    test_mask = temporal_mask | state_mask
    development = frame.loc[~test_mask].sort_values("date")
    test = frame.loc[test_mask].copy()
    if len(development) < 10 or development["label"].nunique() < 2:
        LOGGER.warning("Configured holdouts are too large; using time-forward split only")
        state_mask = pd.Series(False, index=frame.index)
        test_mask = temporal_mask
        development = frame.loc[~test_mask].sort_values("date")
        test = frame.loc[test_mask].copy()

    validation_start = development["date"].quantile(1 - settings.training.temporal_holdout_fraction)
    validation_mask = development["date"] >= validation_start
    training = development.loc[~validation_mask]
    validation = development.loc[validation_mask]
    if training["label"].nunique() < 2 or validation["label"].nunique() < 2:
        split = max(2, int(len(development) * 0.8))
        training = development.iloc[:split]
        validation = development.iloc[split:]
    if training["label"].nunique() < 2:
        raise RuntimeError("The chronological training partition contains only one class")

    transformer = ColumnTransformer(
        [("numeric", SimpleImputer(strategy="median", add_indicator=True), active_features)],
        remainder="drop",
    )
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.06,
        max_iter=300,
        max_leaf_nodes=31,
        min_samples_leaf=5,
        l2_regularization=1.0,
        early_stopping=True,
        random_state=settings.project.seed,
    )
    estimator = Pipeline([("prepare", transformer), ("classifier", classifier)])
    labels = training["label"].astype(int)
    positive_weight = max(1.0, float((labels == 0).sum() / max((labels == 1).sum(), 1)))
    sample_weight = np.where(labels == 1, positive_weight, 1.0)
    estimator.fit(training[active_features], labels, classifier__sample_weight=sample_weight)

    calibration_raw = estimator.predict_proba(validation[active_features])[:, 1]
    calibrator: LogisticRegression | None = None
    validation_class_counts = validation["label"].value_counts()
    if (
        len(validation) >= 30
        and len(validation_class_counts) == 2
        and validation_class_counts.min() >= 10
    ):
        calibrator = LogisticRegression(random_state=settings.project.seed)
        calibrator.fit(calibration_raw.reshape(-1, 1), validation["label"])
        calibration_probability = calibrator.predict_proba(calibration_raw.reshape(-1, 1))[:, 1]
    else:
        calibration_probability = calibration_raw
    threshold = _select_threshold(
        validation["label"],
        calibration_probability,
        settings.training.min_recall_for_threshold,
    )
    model = LandslideRiskModel(
        estimator=estimator,
        calibrator=calibrator,
        features=active_features,
        warning_threshold=threshold,
        trained_at=datetime.now(UTC).isoformat(),
    )

    test_probability = model.predict_proba(test)
    report: dict[str, Any] = {
        "trained_at": model.trained_at,
        "temporal_cutoff": pd.Timestamp(temporal_cutoff).isoformat(),
        "holdout_states": settings.training.holdout_states,
        "warning_threshold": threshold,
        "partitions": {
            "training_rows": len(training),
            "validation_rows": len(validation),
            "test_rows": len(test),
        },
        "overall": _metric_summary(test["label"], test_probability, threshold),
        "temporal_holdout": _metric_summary(
            frame.loc[temporal_mask, "label"],
            model.predict_proba(frame.loc[temporal_mask]),
            threshold,
        ),
        "state_holdout": {},
        "per_state": {},
    }
    state_test = frame.loc[state_mask]
    if len(state_test):
        report["state_holdout"] = _metric_summary(
            state_test["label"], model.predict_proba(state_test), threshold
        )
    for state, state_frame in test.groupby("state"):
        report["per_state"][state] = _metric_summary(
            state_frame["label"],
            model.predict_proba(state_frame),
            threshold,
        )

    if len(test) and test["label"].nunique() > 1:
        importance = permutation_importance(
            estimator,
            test[active_features],
            test["label"],
            scoring="average_precision",
            n_repeats=5,
            random_state=settings.project.seed,
        )
        report["permutation_importance"] = dict(
            sorted(
                zip(active_features, importance.importances_mean, strict=True),
                key=lambda item: item[1],
                reverse=True,
            )
        )
        fraction_positive, mean_prediction = calibration_curve(
            test["label"], test_probability, n_bins=5, strategy="quantile"
        )
        report["calibration_curve"] = {
            "mean_prediction": mean_prediction.tolist(),
            "fraction_positive": fraction_positive.tolist(),
        }

    artifact_path = settings.project.artifacts_dir / "landslide_model.joblib"
    report_path = settings.project.artifacts_dir / "evaluation.json"
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, artifact_path)
    report_path.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    LOGGER.info("Saved model to %s and evaluation to %s", artifact_path, report_path)
    return model, report


def load_model(path: Path | str) -> LandslideRiskModel:
    model = joblib.load(path)
    if not isinstance(model, LandslideRiskModel):
        raise TypeError(f"{path} is not a LandslideRiskModel artifact")
    return model
