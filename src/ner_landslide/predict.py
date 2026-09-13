from __future__ import annotations

import json
import logging
import os
from datetime import UTC, date, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
from dotenv import load_dotenv

from ner_landslide.config import Settings
from ner_landslide.data.weather import OpenMeteoConnector
from ner_landslide.features import assemble_features
from ner_landslide.model import LandslideRiskModel

LOGGER = logging.getLogger(__name__)


def _dispatch_alerts(forecast: gpd.GeoDataFrame, output_dir: Path) -> None:
    load_dotenv(
        Path(__file__).resolve().parents[2] / "LandShield_Alert_System" / ".env"
    )
    if os.getenv("LANDSHIELD_ALERTS_ENABLED", "false").lower() != "true":
        return

    from LandShield_Alert_System.alert_engine import dispatch_forecast_alerts

    results = dispatch_forecast_alerts(forecast, output_dir / ".alert_sent.json")
    LOGGER.info("Dispatched %d new landslide alerts", len(results))


def _risk_category(probability: float, settings: Settings) -> str:
    thresholds = settings.prediction.thresholds
    if probability >= thresholds.severe:
        return "severe"
    if probability >= thresholds.high:
        return "high"
    if probability >= thresholds.medium:
        return "medium"
    return "low"


def _drivers(row: pd.Series) -> str:
    """Return transparent trigger indicators; these are not local SHAP values."""
    scores = {
        "steep_slope": float(row.get("slope_degrees", 0) or 0) / 45,
        "3_day_rain": float(row.get("rain_3d_mm", 0) or 0) / 100,
        "7_day_rain": float(row.get("rain_7d_mm", 0) or 0) / 200,
        "saturated_surface": float(row.get("soil_moisture_best", 0) or 0) / 0.45,
        "wet_vegetation": max(float(row.get("ndmi", 0) or 0), 0) / 0.5,
    }
    ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
    return json.dumps(
        [name for name, score in ranked[:3] if np.isfinite(score) and score > 0]
    )


def generate_predictions(
    settings: Settings,
    model: LandslideRiskModel,
    grid: gpd.GeoDataFrame,
    terrain: pd.DataFrame,
    satellite: pd.DataFrame,
    *,
    as_of: date | None = None,
    sensor_path: Path | None = None,
    refresh_weather: bool = True,
) -> gpd.GeoDataFrame:
    as_of_date = as_of or date.today()
    weather = OpenMeteoConnector(settings).forecast_features(
        grid[["grid_id", "latitude", "longitude"]],
        as_of=as_of_date,
        refresh=refresh_weather,
    )
    base = weather[["grid_id", "target_date"]].merge(
        pd.DataFrame(grid.drop(columns="geometry")),
        on="grid_id",
        how="left",
        validate="many_to_one",
    )
    features = assemble_features(
        base,
        weather,
        terrain,
        satellite,
        sensor_path=sensor_path,
    )
    features["risk_probability"] = model.predict_proba(features)
    features["risk_category"] = features["risk_probability"].map(
        lambda value: _risk_category(float(value), settings)
    )
    features["warning"] = features["risk_probability"] >= model.warning_threshold
    features["drivers"] = features.apply(_drivers, axis=1)
    generated_at = datetime.now(UTC).isoformat()
    features["generated_at"] = generated_at
    features["weather_source"] = "Open-Meteo forecast/reanalysis"
    features["terrain_source"] = "Copernicus DEM GLO-30"
    features["imagery_source"] = "Sentinel-2 L2A composite"
    features["driver_method"] = (
        "ranked trigger indicators, not local feature attribution"
    )

    geometry = grid[["grid_id", "geometry"]]
    forecast = gpd.GeoDataFrame(
        features.merge(geometry, on="grid_id", how="left", validate="many_to_one"),
        geometry="geometry",
        crs=grid.crs,
    )
    output_dir = settings.project.outputs_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    forecast.to_parquet(output_dir / "risk_forecast.parquet", index=False)
    forecast.to_file(output_dir / "risk_forecast.geojson", driver="GeoJSON")

    peak_indices = forecast.groupby("grid_id")["risk_probability"].idxmax()
    peak = forecast.loc[peak_indices].sort_values("risk_probability", ascending=False)
    export_columns = [
        "grid_id",
        "state",
        "latitude",
        "longitude",
        "target_date",
        "risk_probability",
        "risk_category",
        "warning",
        "drivers",
        "generated_at",
    ]
    peak[export_columns].to_csv(output_dir / "risk_ranked.csv", index=False)
    peak.to_file(output_dir / "risk_peak.geojson", driver="GeoJSON")
    _dispatch_alerts(forecast, output_dir)
    LOGGER.info(
        "Saved %d forecast rows and %d peak cell risks to %s",
        len(forecast),
        len(peak),
        output_dir,
    )
    return forecast
