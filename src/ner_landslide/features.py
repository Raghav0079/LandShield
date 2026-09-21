from __future__ import annotations

import logging
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd

from ner_landslide.config import Settings
from ner_landslide.data.sensors import attach_sensor_measurements, read_sensor_csv
from ner_landslide.data.weather import WEATHER_FEATURES, OpenMeteoConnector

LOGGER = logging.getLogger(__name__)

STATIC_FEATURES = [
    "latitude",
    "longitude",
    "elevation_m",
    "slope_degrees",
    "aspect_sin",
    "aspect_cos",
    "ndvi",
    "ndmi",
]
DERIVED_FEATURES = [
    "month_sin",
    "month_cos",
    "soil_moisture_best",
    "sensor_observed",
]
MODEL_FEATURES = WEATHER_FEATURES + STATIC_FEATURES + DERIVED_FEATURES


def _engineer_features(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    date_column = "date" if "date" in result else "target_date"
    dates = pd.to_datetime(result[date_column])
    angle = 2 * np.pi * dates.dt.month / 12
    result["month_sin"] = np.sin(angle)
    result["month_cos"] = np.cos(angle)

    aspect = np.radians(pd.to_numeric(result["aspect_degrees"], errors="coerce"))
    result["aspect_sin"] = np.sin(aspect)
    result["aspect_cos"] = np.cos(aspect)
    if "sensor_soil_moisture" not in result:
        result["sensor_soil_moisture"] = np.nan
    result["sensor_observed"] = result["sensor_soil_moisture"].notna().astype(int)
    result["soil_moisture_best"] = result["sensor_soil_moisture"].fillna(
        result["soil_moisture_surface"]
    )
    return result


def _upsert_training_features(
    existing: pd.DataFrame | None,
    incoming: pd.DataFrame,
    *,
    updated_at: pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Update overlapping cell-date rows and retain non-overlapping history."""
    timestamp = updated_at if updated_at is not None else pd.Timestamp.now(tz="UTC")
    keys = ["grid_id", "date"]
    fresh = incoming.copy()
    fresh["date"] = pd.to_datetime(fresh["date"]).dt.normalize()
    fresh = fresh.drop_duplicates(keys, keep="last")

    if existing is None or existing.empty:
        fresh["first_ingested_at"] = timestamp
        fresh["last_updated_at"] = timestamp
        combined = fresh
    else:
        history = existing.copy()
        history["date"] = pd.to_datetime(history["date"]).dt.normalize()
        if "first_ingested_at" not in history:
            history["first_ingested_at"] = timestamp
        if "last_updated_at" not in history:
            history["last_updated_at"] = timestamp
        history["first_ingested_at"] = pd.to_datetime(history["first_ingested_at"], utc=True)
        history["last_updated_at"] = pd.to_datetime(history["last_updated_at"], utc=True)
        history = history.drop_duplicates(keys, keep="last")

        first_seen = history.set_index(keys)["first_ingested_at"].to_dict()
        fresh["first_ingested_at"] = [
            first_seen.get((row.grid_id, row.date), timestamp)
            for row in fresh[keys].itertuples(index=False)
        ]
        fresh["last_updated_at"] = timestamp

        fresh_keys = pd.MultiIndex.from_frame(fresh[keys])
        history_keys = pd.MultiIndex.from_frame(history[keys])
        unchanged_history = history.loc[~history_keys.isin(fresh_keys)]
        combined = pd.concat([unchanged_history, fresh], ignore_index=True)

    combined = combined.sort_values(["date", "state", "grid_id"]).reset_index(drop=True)
    combined["sample_id"] = [f"S{number:07d}" for number in range(len(combined))]
    ordered = ["sample_id", *[column for column in combined if column != "sample_id"]]
    return combined[ordered]


def assemble_features(
    base: pd.DataFrame,
    weather: pd.DataFrame,
    terrain: pd.DataFrame,
    satellite: pd.DataFrame,
    *,
    sensor_path: Path | None = None,
) -> pd.DataFrame:
    weather_key = "sample_id" if "sample_id" in base else "grid_id"
    merge_keys = [weather_key]
    if weather_key == "grid_id" and "target_date" in weather:
        merge_keys.append("target_date")
    frame = base.merge(weather, on=merge_keys, how="left", validate="one_to_one")
    frame = frame.merge(terrain, on="grid_id", how="left", validate="many_to_one")
    frame = frame.merge(satellite, on="grid_id", how="left", validate="many_to_one")
    if sensor_path:
        frame = attach_sensor_measurements(frame, read_sensor_csv(sensor_path))
    return _engineer_features(frame)


def build_training_features(
    settings: Settings,
    samples: pd.DataFrame,
    grid: gpd.GeoDataFrame,
    terrain: pd.DataFrame,
    satellite: pd.DataFrame,
    *,
    sensor_path: Path | None = None,
    refresh: bool = False,
) -> pd.DataFrame:
    output = settings.project.processed_dir / "training_features.parquet"
    existing: pd.DataFrame | None = None
    if output.exists():
        existing = pd.read_parquet(output)
    if output.exists() and not refresh and sensor_path is None:
        cached = existing
        if cached is None:
            raise RuntimeError(f"Unable to read existing training features from {output}")
        identity_columns = ["sample_id", "grid_id", "date", "label"]
        cached_identity = cached[identity_columns].copy()
        sample_identity = samples[identity_columns].copy()
        cached_identity["date"] = pd.to_datetime(cached_identity["date"])
        sample_identity["date"] = pd.to_datetime(sample_identity["date"])
        has_merge_metadata = {"first_ingested_at", "last_updated_at"}.issubset(cached)
        if cached_identity.equals(sample_identity) and has_merge_metadata:
            return cached
        LOGGER.info("Training samples changed; rebuilding cached feature table")

    weather = OpenMeteoConnector(settings).historical_features(samples, refresh=refresh)
    base = samples.copy()
    grid_coordinates = grid[["grid_id", "latitude", "longitude"]].rename(
        columns={"latitude": "grid_latitude", "longitude": "grid_longitude"}
    )
    base = base.merge(grid_coordinates, on="grid_id", how="left", validate="many_to_one")
    base["latitude"] = base["grid_latitude"].fillna(base["latitude"])
    base["longitude"] = base["grid_longitude"].fillna(base["longitude"])
    base = base.drop(columns=["grid_latitude", "grid_longitude"])
    incoming = assemble_features(
        base,
        weather,
        terrain,
        satellite,
        sensor_path=sensor_path,
    )
    frame = _upsert_training_features(existing, incoming)
    minimum = settings.training.minimum_training_records
    if minimum and len(frame) < minimum:
        raise RuntimeError(
            f"Upsert produced {len(frame)} unique rows, below configured minimum {minimum}"
        )
    frame.to_parquet(output, index=False)
    frame.to_csv(output.with_suffix(".csv"), index=False)
    missing_weather = frame[WEATHER_FEATURES].isna().all(axis=1).sum()
    if missing_weather:
        LOGGER.warning("%d samples have no weather features", missing_weather)
    LOGGER.info("Saved %d training feature rows to %s", len(frame), output)
    return frame
