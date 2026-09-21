from __future__ import annotations

import numpy as np
import pandas as pd

from ner_landslide.data.weather import _hourly_frame, _summarize


def test_weather_window_summaries() -> None:
    times = pd.date_range("2025-01-01", "2025-01-31 23:00", freq="h")
    payload = {
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "precipitation": np.ones(len(times)).tolist(),
            "soil_moisture_0_to_7cm": np.full(len(times), 0.4).tolist(),
            "soil_moisture_7_to_28cm": np.full(len(times), 0.3).tolist(),
        }
    }
    result = _summarize(_hourly_frame(payload), pd.Timestamp("2025-01-31"))

    assert result["rain_1d_mm"] == 24
    assert result["rain_3d_mm"] == 72
    assert result["rain_30d_mm"] == 720
    assert result["rain_max_24h_mm"] == 24
    assert result["soil_moisture_surface"] == 0.4
    assert result["soil_moisture_root"] == 0.3
