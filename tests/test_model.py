from __future__ import annotations

import numpy as np
import pandas as pd

from ner_landslide.config import Settings
from ner_landslide.features import MODEL_FEATURES
from ner_landslide.model import load_model, train_model


def test_model_training_persists_evaluation(settings: Settings) -> None:
    rng = np.random.default_rng(42)
    rows = 300
    labels = (np.arange(rows) % 5 == 0).astype(int)
    frame = pd.DataFrame({feature: rng.normal(size=rows) for feature in MODEL_FEATURES})
    frame["rain_3d_mm"] = labels * 100 + rng.normal(10, 2, rows)
    frame["soil_moisture_best"] = labels * 0.3 + rng.normal(0.2, 0.02, rows)
    frame["label"] = labels
    frame["date"] = pd.date_range("2020-01-01", periods=rows, freq="D")
    frame["state"] = np.where(np.arange(rows) % 7 == 0, "Sikkim", "Assam")

    model, report = train_model(settings, frame)

    assert settings.project.artifacts_dir.joinpath("landslide_model.joblib").exists()
    assert settings.project.artifacts_dir.joinpath("evaluation.json").exists()
    assert report["overall"]["rows"] > 0
    probabilities = model.predict_proba(frame.iloc[:3])
    assert np.all((probabilities >= 0) & (probabilities <= 1))
    loaded = load_model(settings.project.artifacts_dir / "landslide_model.joblib")
    np.testing.assert_allclose(loaded.predict_proba(frame.iloc[:3]), probabilities)
