from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REQUIRED_COLUMNS = {
    "sensor_id",
    "timestamp",
    "latitude",
    "longitude",
    "soil_moisture",
}


def read_sensor_csv(path: Path | str) -> pd.DataFrame:
    """Read normalized field measurements without assuming a hardware vendor."""
    frame = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(frame)
    if missing:
        raise ValueError(f"Sensor CSV is missing required columns: {sorted(missing)}")
    frame = frame.copy()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True).dt.tz_localize(None)
    numeric = ["latitude", "longitude", "soil_moisture"]
    frame[numeric] = frame[numeric].apply(pd.to_numeric, errors="coerce")
    frame = frame.dropna(subset=["timestamp", *numeric])
    if not frame["soil_moisture"].between(0, 1).all():
        raise ValueError("soil_moisture must be volumetric water content between 0 and 1")
    return frame.sort_values("timestamp").reset_index(drop=True)


def attach_sensor_measurements(
    samples: pd.DataFrame,
    sensors: pd.DataFrame,
    *,
    maximum_distance_km: float = 25,
    maximum_age_hours: int = 24,
) -> pd.DataFrame:
    """Attach the nearest recent sensor value, retaining modeled data as fallback."""
    if sensors.empty:
        return samples.assign(sensor_soil_moisture=np.nan, sensor_distance_km=np.nan)
    output = samples.copy()
    date_column = "date" if "date" in output else "target_date"
    targets = pd.to_datetime(output[date_column]).dt.tz_localize(None)
    sensor_coords = np.radians(sensors[["latitude", "longitude"]].to_numpy())
    tree = BallTree(sensor_coords, metric="haversine")
    sample_coords = np.radians(output[["latitude", "longitude"]].to_numpy())
    distances, indices = tree.query(sample_coords, k=min(5, len(sensors)))

    values: list[float] = []
    selected_distances: list[float] = []
    for row_number, (candidate_distances, candidate_indices) in enumerate(
        zip(distances, indices, strict=True)
    ):
        chosen_value = np.nan
        chosen_distance = np.nan
        for angular_distance, sensor_index in zip(
            candidate_distances, candidate_indices, strict=True
        ):
            distance_km = float(angular_distance * 6371.0088)
            age = targets.iloc[row_number] - sensors.iloc[sensor_index]["timestamp"]
            if distance_km <= maximum_distance_km and pd.Timedelta(0) <= age <= pd.Timedelta(
                hours=maximum_age_hours
            ):
                chosen_value = float(sensors.iloc[sensor_index]["soil_moisture"])
                chosen_distance = distance_km
                break
        values.append(chosen_value)
        selected_distances.append(chosen_distance)
    output["sensor_soil_moisture"] = values
    output["sensor_distance_km"] = selected_distances
    return output
