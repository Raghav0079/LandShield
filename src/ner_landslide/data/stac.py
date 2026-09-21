from __future__ import annotations

import logging
import math
from collections import defaultdict
from typing import Any

import geopandas as gpd
import numpy as np
import pandas as pd
import planetary_computer
import pystac_client
import rasterio
from rasterio.warp import transform
from rasterio.windows import Window

from ner_landslide.config import Settings

LOGGER = logging.getLogger(__name__)
RASTER_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "AWS_NO_SIGN_REQUEST": "YES",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif,.TIF",
    "GDAL_HTTP_CONNECTTIMEOUT": "15",
    "GDAL_HTTP_TIMEOUT": "60",
    "GDAL_HTTP_MAX_RETRY": "3",
    "GDAL_HTTP_RETRY_DELAY": "1",
}


def _asset(item: Any, candidates: tuple[str, ...]) -> str:
    for key in candidates:
        if key in item.assets:
            return str(item.assets[key].href)
    raise KeyError(f"None of assets {candidates} exist in STAC item {item.id}")


def _points_in_bbox(points: pd.DataFrame, bbox: list[float] | tuple[float, ...]) -> pd.DataFrame:
    minx, miny, maxx, maxy = bbox
    return points.loc[
        points["longitude"].between(minx, maxx) & points["latitude"].between(miny, maxy)
    ]


def _sample_band(url: str, points: pd.DataFrame) -> np.ndarray:
    with rasterio.Env(**RASTER_ENV), rasterio.open(url) as source:
        longitudes = points["longitude"].tolist()
        latitudes = points["latitude"].tolist()
        if source.crs and source.crs.to_epsg() != 4326:
            xs, ys = transform("EPSG:4326", source.crs, longitudes, latitudes)
        else:
            xs, ys = longitudes, latitudes
        values = np.array([value[0] for value in source.sample(zip(xs, ys, strict=True))])
        invalid = ~np.isfinite(values)
        if source.nodata is not None:
            invalid |= values == source.nodata
        values = values.astype(float)
        values[invalid] = np.nan
        return values


def _terrain_at_point(
    source: rasterio.DatasetReader,
    lon: float,
    lat: float,
) -> tuple[float, float, float]:
    if source.crs and source.crs.to_epsg() != 4326:
        xs, ys = transform("EPSG:4326", source.crs, [lon], [lat])
        x, y = xs[0], ys[0]
    else:
        x, y = lon, lat
    row, column = source.index(x, y)
    window = source.read(
        1,
        window=Window(column - 1, row - 1, 3, 3),
        boundless=True,
        fill_value=np.nan,
    ).astype(float)
    if source.nodata is not None:
        window[window == source.nodata] = np.nan
    if window.shape != (3, 3) or not np.isfinite(window[1, 1]):
        return np.nan, np.nan, np.nan

    if source.crs and source.crs.is_geographic:
        x_resolution = abs(source.transform.a) * 111_320 * max(math.cos(math.radians(lat)), 0.1)
        y_resolution = abs(source.transform.e) * 110_540
    else:
        x_resolution = abs(source.transform.a)
        y_resolution = abs(source.transform.e)
    filled = np.where(np.isfinite(window), window, np.nanmedian(window))
    dz_dy, dz_dx = np.gradient(filled, y_resolution, x_resolution)
    gradient_x = float(dz_dx[1, 1])
    gradient_y = float(dz_dy[1, 1])
    slope = math.degrees(math.atan(math.hypot(gradient_x, gradient_y)))
    aspect = (math.degrees(math.atan2(-gradient_x, gradient_y)) + 360) % 360
    return float(window[1, 1]), slope, aspect


class PlanetaryComputerConnector:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.catalog = pystac_client.Client.open(
            settings.stac.url,
            modifier=planetary_computer.sign_inplace,
        )

    def terrain_features(
        self,
        grid: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        output = self.settings.project.processed_dir / "terrain_features.parquet"
        if output.exists() and not refresh:
            return pd.read_parquet(output)

        points = grid[["grid_id", "latitude", "longitude"]].copy()
        values: dict[str, tuple[float, float, float]] = {}
        search = self.catalog.search(
            collections=[self.settings.stac.dem_collection],
            bbox=boundaries.total_bounds.tolist(),
        )
        for item in search.items():
            if item.bbox is None:
                continue
            candidates = _points_in_bbox(points, item.bbox).loc[
                lambda frame: ~frame["grid_id"].isin(values)
            ]
            if candidates.empty:
                continue
            try:
                url = _asset(item, ("data", "dem", "elevation"))
                with rasterio.Env(**RASTER_ENV), rasterio.open(url) as source:
                    for row in candidates.to_dict("records"):
                        grid_id = str(row["grid_id"])
                        values[grid_id] = _terrain_at_point(
                            source,
                            float(row["longitude"]),
                            float(row["latitude"]),
                        )
            except (KeyError, OSError, rasterio.errors.RasterioError) as error:
                LOGGER.warning("Skipping DEM item %s: %s", item.id, error)

        terrain = points[["grid_id"]].copy()
        terrain[["elevation_m", "slope_degrees", "aspect_degrees"]] = [
            values.get(grid_id, (np.nan, np.nan, np.nan)) for grid_id in terrain["grid_id"]
        ]
        output.parent.mkdir(parents=True, exist_ok=True)
        terrain.to_parquet(output, index=False)
        LOGGER.info(
            "Built terrain features for %d grid cells", terrain["elevation_m"].notna().sum()
        )
        return pd.DataFrame(terrain)

    def satellite_features(
        self,
        grid: gpd.GeoDataFrame,
        boundaries: gpd.GeoDataFrame,
        *,
        refresh: bool = False,
    ) -> pd.DataFrame:
        output = self.settings.project.processed_dir / "satellite_features.parquet"
        if output.exists() and not refresh:
            return pd.read_parquet(output)

        points = grid[["grid_id", "state", "latitude", "longitude"]].copy()
        ndvi: dict[str, list[float]] = defaultdict(list)
        ndmi: dict[str, list[float]] = defaultdict(list)
        interval = (
            f"{self.settings.stac.sentinel_start_date.isoformat()}/"
            f"{self.settings.stac.sentinel_end_date.isoformat()}"
        )
        for boundary in boundaries.itertuples(index=False):
            search = self.catalog.search(
                collections=[self.settings.stac.sentinel_collection],
                bbox=list(boundary.geometry.bounds),
                datetime=interval,
                query={"eo:cloud_cover": {"lt": self.settings.stac.max_cloud_cover}},
                max_items=self.settings.stac.max_sentinel_items,
            )
            items = sorted(
                search.item_collection(),
                key=lambda item: float(item.properties.get("eo:cloud_cover", 100)),
            )
            state_points = points.loc[points["state"] == boundary.state]
            for item in items:
                if item.bbox is None:
                    continue
                candidates = _points_in_bbox(state_points, item.bbox)
                if candidates.empty:
                    continue
                try:
                    red = _sample_band(_asset(item, ("B04", "red")), candidates)
                    nir = _sample_band(_asset(item, ("B08", "nir")), candidates)
                    swir = _sample_band(_asset(item, ("B11", "swir16")), candidates)
                except (KeyError, OSError, rasterio.errors.RasterioError) as error:
                    LOGGER.warning("Skipping Sentinel item %s: %s", item.id, error)
                    continue
                denominator_ndvi = nir + red
                denominator_ndmi = nir + swir
                scene_ndvi = np.divide(
                    nir - red,
                    denominator_ndvi,
                    out=np.full_like(nir, np.nan),
                    where=denominator_ndvi != 0,
                )
                scene_ndmi = np.divide(
                    nir - swir,
                    denominator_ndmi,
                    out=np.full_like(nir, np.nan),
                    where=denominator_ndmi != 0,
                )
                for grid_id, vegetation, moisture in zip(
                    candidates["grid_id"], scene_ndvi, scene_ndmi, strict=True
                ):
                    if np.isfinite(vegetation):
                        ndvi[grid_id].append(float(vegetation))
                    if np.isfinite(moisture):
                        ndmi[grid_id].append(float(moisture))

        result = points[["grid_id"]].copy()
        result["ndvi"] = result["grid_id"].map(
            lambda key: float(np.median(ndvi[key])) if ndvi[key] else np.nan
        )
        result["ndmi"] = result["grid_id"].map(
            lambda key: float(np.median(ndmi[key])) if ndmi[key] else np.nan
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        result.to_parquet(output, index=False)
        LOGGER.info("Built Sentinel-2 indices for %d grid cells", result["ndvi"].notna().sum())
        return pd.DataFrame(result)
