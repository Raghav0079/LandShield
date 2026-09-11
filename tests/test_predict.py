from __future__ import annotations

from datetime import date

import geopandas as gpd
import numpy as np
import pandas as pd
import pytest
from shapely.geometry import box

from ner_landslide.config import Settings
from ner_landslide.data.weather import WEATHER_FEATURES, OpenMeteoConnector
from ner_landslide.predict import generate_predictions


class FakeModel:
    warning_threshold = 0.5

    def predict_proba(self, features: pd.DataFrame) -> np.ndarray:
        return np.clip(features["rain_3d_mm"].to_numpy() / 100, 0, 1)


def test_prediction_outputs(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grid = gpd.GeoDataFrame(
        {
            "grid_id": ["G1", "G2"],
            "state": ["Assam", "Meghalaya"],
            "latitude": [25.0, 25.5],
            "longitude": [91.0, 91.5],
        },
        geometry=[box(90.95, 24.95, 91.05, 25.05), box(91.45, 25.45, 91.55, 25.55)],
        crs=4326,
    )
    terrain = pd.DataFrame(
        {
            "grid_id": ["G1", "G2"],
            "elevation_m": [100, 1000],
            "slope_degrees": [5, 35],
            "aspect_degrees": [90, 180],
        }
    )
    satellite = pd.DataFrame({"grid_id": ["G1", "G2"], "ndvi": [0.4, 0.7], "ndmi": [0.1, 0.4]})

    def fake_forecast(*args: object, **kwargs: object) -> pd.DataFrame:
        rows = []
        for grid_id, rain in [("G1", 20.0), ("G2", 80.0)]:
            for offset in (1, 2):
                row = {
                    "grid_id": grid_id,
                    "target_date": pd.Timestamp("2026-09-09") + pd.Timedelta(days=offset),
                }
                row.update(dict.fromkeys(WEATHER_FEATURES, 0.3))
                row["rain_3d_mm"] = rain
                rows.append(row)
        return pd.DataFrame(rows)

    monkeypatch.setattr(OpenMeteoConnector, "forecast_features", fake_forecast)
    forecast = generate_predictions(
        settings,
        FakeModel(),  # type: ignore[arg-type]
        grid,
        terrain,
        satellite,
        as_of=date(2026, 9, 9),
    )

    assert len(forecast) == 4
    assert set(forecast["risk_category"]) == {"low", "severe"}
    assert settings.project.outputs_dir.joinpath("risk_forecast.geojson").exists()
    assert settings.project.outputs_dir.joinpath("risk_peak.geojson").exists()
    assert settings.project.outputs_dir.joinpath("risk_ranked.csv").exists()
