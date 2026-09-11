from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from ner_landslide.data.sensors import attach_sensor_measurements, read_sensor_csv


def test_sensor_csv_and_nearest_attachment(tmp_path: Path) -> None:
    path = tmp_path / "sensors.csv"
    path.write_text(
        "sensor_id,timestamp,latitude,longitude,soil_moisture\n"
        "A,2025-06-01T06:00:00Z,25.50,91.50,0.44\n",
        encoding="utf-8",
    )
    sensors = read_sensor_csv(path)
    samples = pd.DataFrame(
        {
            "date": [pd.Timestamp("2025-06-01 12:00")],
            "latitude": [25.51],
            "longitude": [91.51],
        }
    )
    result = attach_sensor_measurements(samples, sensors)
    assert result.loc[0, "sensor_soil_moisture"] == 0.44
    assert result.loc[0, "sensor_distance_km"] < 5


def test_sensor_values_must_be_fractional(tmp_path: Path) -> None:
    path = tmp_path / "invalid.csv"
    path.write_text(
        "sensor_id,timestamp,latitude,longitude,soil_moisture\n"
        "A,2025-06-01T06:00:00Z,25.50,91.50,44\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="between 0 and 1"):
        read_sensor_csv(path)
