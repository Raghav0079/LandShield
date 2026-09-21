from __future__ import annotations

import logging
from typing import Any

import geopandas as gpd
import pandas as pd
import requests
from shapely.geometry import Point, shape

from ner_landslide.config import Settings
from ner_landslide.utils import CachedHttpClient

LOGGER = logging.getLogger(__name__)


def _parse_event_dates(frame: pd.DataFrame) -> pd.Series:
    candidates = (
        "event_date",
        "eventdate",
        "eventDate",
        "date",
        "landslide_date",
    )
    column = next((name for name in candidates if name in frame.columns), None)
    if column is None:
        raise RuntimeError(f"COOLR response has no recognized date field: {list(frame.columns)}")
    values = frame[column]
    if pd.api.types.is_numeric_dtype(values):
        return pd.to_datetime(values, unit="ms", errors="coerce", utc=True).dt.tz_localize(None)
    return pd.to_datetime(values, errors="coerce", utc=True, format="mixed").dt.tz_localize(None)


def _download_fallback_csv(settings: Settings, refresh: bool) -> gpd.GeoDataFrame:
    path = settings.project.cache_dir / "coolr" / "global_landslide_catalog.csv"
    if refresh or not path.exists():
        LOGGER.warning(
            "The live NASA COOLR service is unavailable; downloading its public catalog export"
        )
        response = requests.get(settings.landslides.fallback_csv_url, timeout=180)
        response.raise_for_status()
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
    frame = pd.read_csv(path)
    required = {"latitude", "longitude"}
    if not required.issubset(frame):
        raise RuntimeError("Fallback landslide catalog has no latitude/longitude fields")
    geometry = [
        Point(longitude, latitude)
        for longitude, latitude in zip(frame["longitude"], frame["latitude"], strict=True)
    ]
    return gpd.GeoDataFrame(frame, geometry=geometry, crs=4326)


def download_landslide_events(
    settings: Settings,
    boundaries: gpd.GeoDataFrame,
    *,
    refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Query NASA COOLR events intersecting the NER bounding box."""
    output = settings.project.processed_dir / "landslide_events.parquet"
    if output.exists() and not refresh:
        return gpd.read_parquet(output)

    minx, miny, maxx, maxy = boundaries.total_bounds
    client = CachedHttpClient(settings.project.cache_dir, timeout=120)
    features: list[dict[str, Any]] = []
    try:
        offset = 0
        while True:
            params = {
                "where": "1=1",
                "geometry": f"{minx},{miny},{maxx},{maxy}",
                "geometryType": "esriGeometryEnvelope",
                "inSR": "4326",
                "spatialRel": "esriSpatialRelIntersects",
                "outFields": "*",
                "returnGeometry": "true",
                "outSR": "4326",
                "resultOffset": offset,
                "resultRecordCount": settings.landslides.page_size,
                "f": "geojson",
            }
            payload = client.get_json(
                settings.landslides.query_url,
                params=params,
                cache_namespace="coolr",
                refresh=refresh,
            )
            page = payload.get("features", [])
            features.extend(page)
            if len(page) < settings.landslides.page_size:
                break
            offset += len(page)
        if not features:
            raise RuntimeError("NASA COOLR returned no events")
        records = [dict(feature.get("properties", {})) for feature in features]
        geometries = [shape(feature["geometry"]) for feature in features]
        events = gpd.GeoDataFrame(records, geometry=geometries, crs=4326)
    except RuntimeError:
        events = _download_fallback_csv(settings, refresh)
    events["event_date"] = _parse_event_dates(events).dt.normalize()
    events = events.dropna(subset=["event_date"]).copy()
    events = gpd.sjoin(events, boundaries[["state", "geometry"]], predicate="within", how="inner")
    events = events.drop(columns=["index_right"], errors="ignore")

    start = pd.Timestamp(settings.training.start_date)
    end = pd.Timestamp(settings.training.end_date)
    events = events.loc[events["event_date"].between(start, end)].copy()
    events = events.sort_values("event_date").reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    events.to_parquet(output, index=False)
    LOGGER.info("Saved %d NER landslide events to %s", len(events), output)
    return events
