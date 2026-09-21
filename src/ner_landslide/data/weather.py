from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, cast

import numpy as np
import pandas as pd

from ner_landslide.config import Settings
from ner_landslide.utils import CachedHttpClient, chunks

LOGGER = logging.getLogger(__name__)

HOURLY_VARIABLES = [
    "precipitation",
    "soil_moisture_0_to_7cm",
    "soil_moisture_7_to_28cm",
]
WEATHER_FEATURES = [
    "rain_1d_mm",
    "rain_3d_mm",
    "rain_7d_mm",
    "rain_14d_mm",
    "rain_30d_mm",
    "rain_max_1h_mm",
    "rain_max_24h_mm",
    "soil_moisture_surface",
    "soil_moisture_root",
    "soil_moisture_7d_mean",
]


def _locations(payload: Any) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict) and "hourly" in payload:
        return [payload]
    message = payload.get("reason") if isinstance(payload, dict) else str(payload)
    raise RuntimeError(f"Unexpected Open-Meteo response: {message}")


def _hourly_frame(location: dict[str, Any]) -> pd.DataFrame:
    hourly = location.get("hourly", {})
    if "time" not in hourly:
        raise RuntimeError(f"Open-Meteo location has no hourly data: {location}")
    frame = pd.DataFrame(hourly)
    frame["time"] = pd.to_datetime(frame["time"], utc=True).dt.tz_localize(None)
    for column in HOURLY_VARIABLES:
        if column in frame:
            frame[column] = pd.to_numeric(frame[column], errors="coerce")
        else:
            frame[column] = np.nan
    return frame


def _summarize(frame: pd.DataFrame, target: pd.Timestamp) -> dict[str, float]:
    target_end = target.normalize() + pd.Timedelta(days=1)
    available = frame.loc[frame["time"] < target_end].copy()
    rain = available.set_index("time")["precipitation"].fillna(0)

    def rainfall(days: int) -> float:
        start = target_end - pd.Timedelta(days=days)
        return float(rain.loc[rain.index >= start].sum())

    last_24h = rain.loc[rain.index >= target_end - pd.Timedelta(days=1)]
    rolling_24h = rain.rolling(24, min_periods=1).sum()
    soil_week = available.loc[
        available["time"] >= target_end - pd.Timedelta(days=7),
        "soil_moisture_0_to_7cm",
    ]
    surface = available["soil_moisture_0_to_7cm"].dropna()
    root = available["soil_moisture_7_to_28cm"].dropna()
    return {
        "rain_1d_mm": rainfall(1),
        "rain_3d_mm": rainfall(3),
        "rain_7d_mm": rainfall(7),
        "rain_14d_mm": rainfall(14),
        "rain_30d_mm": rainfall(30),
        "rain_max_1h_mm": float(last_24h.max()) if len(last_24h) else 0.0,
        "rain_max_24h_mm": float(rolling_24h.max()) if len(rolling_24h) else 0.0,
        "soil_moisture_surface": float(surface.iloc[-1]) if len(surface) else np.nan,
        "soil_moisture_root": float(root.iloc[-1]) if len(root) else np.nan,
        "soil_moisture_7d_mean": float(soil_week.mean()) if len(soil_week) else np.nan,
    }


def _power_daily_frame(payload: dict[str, Any]) -> pd.DataFrame:
    parameters = payload.get("properties", {}).get("parameter", {})
    required = {"PRECTOTCORR", "GWETTOP", "GWETROOT"}
    if not required.issubset(parameters):
        raise RuntimeError(
            f"NASA POWER response is missing variables: {required - set(parameters)}"
        )
    frame = pd.DataFrame(parameters)
    frame.index = pd.to_datetime(frame.index, format="%Y%m%d")
    frame = frame.replace(-999, np.nan).sort_index()
    return frame


def _summarize_power(frame: pd.DataFrame, target: pd.Timestamp) -> dict[str, float]:
    target = target.normalize()
    available = frame.loc[frame.index <= target]
    precipitation = available["PRECTOTCORR"].clip(lower=0).fillna(0)

    def rainfall(days: int) -> float:
        start = target - pd.Timedelta(days=days - 1)
        return float(precipitation.loc[precipitation.index >= start].sum())

    recent_month = precipitation.loc[precipitation.index >= target - pd.Timedelta(days=29)]
    surface = available["GWETTOP"].dropna()
    root = available["GWETROOT"].dropna()
    surface_week = surface.loc[surface.index >= target - pd.Timedelta(days=6)]
    return {
        "rain_1d_mm": rainfall(1),
        "rain_3d_mm": rainfall(3),
        "rain_7d_mm": rainfall(7),
        "rain_14d_mm": rainfall(14),
        "rain_30d_mm": rainfall(30),
        "rain_max_1h_mm": np.nan,
        "rain_max_24h_mm": float(recent_month.max()) if len(recent_month) else np.nan,
        "soil_moisture_surface": float(surface.iloc[-1]) if len(surface) else np.nan,
        "soil_moisture_root": float(root.iloc[-1]) if len(root) else np.nan,
        "soil_moisture_7d_mean": float(surface_week.mean()) if len(surface_week) else np.nan,
    }


class OpenMeteoConnector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = CachedHttpClient(
            settings.project.cache_dir,
            timeout=settings.weather.timeout_seconds,
        )

    def historical_features(
        self,
        samples: pd.DataFrame,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        """Fetch antecedent weather for sample rows with sample_id, date, latitude, longitude."""
        required = {"sample_id", "date", "latitude", "longitude"}
        missing = required - set(samples)
        if missing:
            raise ValueError(f"Historical weather samples are missing columns: {sorted(missing)}")
        if len(samples) >= 1000 and "grid_id" in samples:
            return self._power_historical_features(samples, refresh=refresh)

        results: list[dict[str, Any]] = []
        work = samples.copy()
        work["date"] = pd.to_datetime(work["date"]).dt.normalize()
        for raw_target, dated_samples in work.groupby("date", sort=True):
            target = pd.Timestamp(cast(Any, raw_target))
            records = dated_samples.to_dict("records")
            for batch in chunks(records, self.settings.weather.coordinate_batch_size):
                params = {
                    "latitude": ",".join(str(row["latitude"]) for row in batch),
                    "longitude": ",".join(str(row["longitude"]) for row in batch),
                    "start_date": (target - pd.Timedelta(days=30)).date().isoformat(),
                    "end_date": target.date().isoformat(),
                    "hourly": ",".join(HOURLY_VARIABLES),
                    "timezone": "UTC",
                }
                payload = self.client.get_json(
                    self.settings.weather.archive_url,
                    params=params,
                    cache_namespace="weather_archive",
                    refresh=refresh,
                )
                locations = _locations(payload)
                if len(locations) != len(batch):
                    raise RuntimeError("Open-Meteo returned a different number of locations")
                for row, location in zip(batch, locations, strict=True):
                    results.append(
                        {
                            "sample_id": row["sample_id"],
                            **_summarize(_hourly_frame(location), target),
                        }
                    )
        LOGGER.info("Built historical weather features for %d samples", len(results))
        return pd.DataFrame(results)

    def _power_historical_features(
        self,
        samples: pd.DataFrame,
        *,
        refresh: bool,
    ) -> pd.DataFrame:
        """Use cached NASA POWER daily series for large historical feature sets."""
        work = samples.copy()
        work["date"] = pd.to_datetime(work["date"]).dt.normalize()
        start = (work["date"].min() - pd.Timedelta(days=30)).strftime("%Y%m%d")
        end = work["date"].max().strftime("%Y%m%d")
        results: list[dict[str, Any]] = []
        for _grid_id, point_samples in work.groupby("grid_id", sort=True):
            point = point_samples.iloc[0]
            payload = self.client.get_json(
                self.settings.weather.power_url,
                params={
                    "parameters": "PRECTOTCORR,GWETTOP,GWETROOT",
                    "community": "AG",
                    "longitude": float(point["longitude"]),
                    "latitude": float(point["latitude"]),
                    "start": start,
                    "end": end,
                    "format": "JSON",
                },
                cache_namespace="nasa_power_daily",
                refresh=refresh,
            )
            daily = _power_daily_frame(payload)
            for sample in point_samples.itertuples(index=False):
                results.append(
                    {
                        "sample_id": sample.sample_id,
                        **_summarize_power(daily, pd.Timestamp(cast(Any, sample.date))),
                    }
                )
        LOGGER.info(
            "Built NASA POWER historical features for %d samples across %d cells",
            len(results),
            work["grid_id"].nunique(),
        )
        return pd.DataFrame(results)

    def forecast_features(
        self,
        points: pd.DataFrame,
        *,
        as_of: date | None = None,
        refresh: bool = True,
    ) -> pd.DataFrame:
        """Create one feature row per grid cell and forecast target day."""
        required = {"grid_id", "latitude", "longitude"}
        missing = required - set(points)
        if missing:
            raise ValueError(f"Forecast points are missing columns: {sorted(missing)}")
        today = pd.Timestamp(as_of or date.today())
        horizon = self.settings.prediction.warning_horizon_days
        records = points.to_dict("records")
        results: list[dict[str, Any]] = []
        for batch in chunks(records, self.settings.weather.coordinate_batch_size):
            params = {
                "latitude": ",".join(str(row["latitude"]) for row in batch),
                "longitude": ",".join(str(row["longitude"]) for row in batch),
                "hourly": ",".join(HOURLY_VARIABLES),
                "past_days": 30,
                "forecast_days": max(horizon + 1, self.settings.weather.forecast_days),
                "timezone": "UTC",
            }
            payload = self.client.get_json(
                self.settings.weather.forecast_url,
                params=params,
                cache_namespace=f"weather_forecast_{today.date().isoformat()}",
                refresh=refresh,
            )
            locations = _locations(payload)
            if len(locations) != len(batch):
                raise RuntimeError("Open-Meteo returned a different number of locations")
            for row, location in zip(batch, locations, strict=True):
                hourly = _hourly_frame(location)
                for offset in range(1, horizon + 1):
                    target = today + pd.Timedelta(days=offset)
                    results.append(
                        {
                            "grid_id": row["grid_id"],
                            "target_date": target,
                            **_summarize(hourly, target),
                        }
                    )
        LOGGER.info(
            "Built %d weather forecast feature rows through %s",
            len(results),
            (today + timedelta(days=horizon)).date(),
        )
        return pd.DataFrame(results)
