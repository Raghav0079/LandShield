from __future__ import annotations

import os

import pandas as pd
import pytest
import requests

from ner_landslide.config import Settings
from ner_landslide.data.landslides import _parse_event_dates
from ner_landslide.data.weather import OpenMeteoConnector


def test_coolr_millisecond_date_parsing() -> None:
    frame = pd.DataFrame({"event_date": [1_735_689_600_000]})
    parsed = _parse_event_dates(frame)
    assert parsed.iloc[0] == pd.Timestamp("2025-01-01")


def test_historical_weather_connector_with_mock(
    settings: Settings,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector = OpenMeteoConnector(settings)
    times = pd.date_range("2025-05-01", "2025-05-31 23:00", freq="h")
    location = {
        "hourly": {
            "time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
            "precipitation": [0.5] * len(times),
            "soil_moisture_0_to_7cm": [0.4] * len(times),
            "soil_moisture_7_to_28cm": [0.3] * len(times),
        }
    }
    monkeypatch.setattr(connector.client, "get_json", lambda *args, **kwargs: location)
    samples = pd.DataFrame(
        {
            "sample_id": ["S1"],
            "date": ["2025-05-31"],
            "latitude": [25.5],
            "longitude": [91.5],
        }
    )
    result = connector.historical_features(samples)
    assert result.loc[0, "sample_id"] == "S1"
    assert result.loc[0, "rain_1d_mm"] == 12


@pytest.mark.network
@pytest.mark.skipif(
    os.getenv("RUN_NETWORK_TESTS") != "1",
    reason="set RUN_NETWORK_TESTS=1 to contact public APIs",
)
def test_open_meteo_live_contract(settings: Settings) -> None:
    response = requests.get(
        settings.weather.archive_url,
        params={
            "latitude": 25.57,
            "longitude": 91.88,
            "start_date": "2025-06-01",
            "end_date": "2025-06-02",
            "hourly": "precipitation,soil_moisture_0_to_7cm",
            "timezone": "UTC",
        },
        timeout=30,
    )
    response.raise_for_status()
    assert "hourly" in response.json()
