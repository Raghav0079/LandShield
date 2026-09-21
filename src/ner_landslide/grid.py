from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
from shapely.geometry import box

from ner_landslide.config import Settings

LOGGER = logging.getLogger(__name__)


def build_grid(
    settings: Settings,
    boundaries: gpd.GeoDataFrame,
    *,
    refresh: bool = False,
) -> gpd.GeoDataFrame:
    output = settings.project.processed_dir / "ner_grid.parquet"
    if output.exists() and not refresh:
        cached = gpd.read_parquet(output)
        if "grid_resolution_degrees" in cached and np.isclose(
            cached["grid_resolution_degrees"].iloc[0],
            settings.region.grid_resolution_degrees,
        ):
            return cached
        LOGGER.info("Grid resolution changed; rebuilding cached grid")

    resolution = settings.region.grid_resolution_degrees
    rows: list[dict[str, object]] = []
    for state_row in boundaries.itertuples(index=False):
        minx, miny, maxx, maxy = state_row.geometry.bounds
        longitude_starts = np.arange(
            np.floor(minx / resolution) * resolution,
            maxx,
            resolution,
        )
        latitude_starts = np.arange(
            np.floor(miny / resolution) * resolution,
            maxy,
            resolution,
        )
        for longitude in longitude_starts:
            for latitude in latitude_starts:
                cell = box(
                    longitude,
                    latitude,
                    longitude + resolution,
                    latitude + resolution,
                )
                clipped = cell.intersection(state_row.geometry)
                if clipped.is_empty or clipped.area < resolution * resolution * 0.01:
                    continue
                point = clipped.representative_point()
                rows.append(
                    {
                        "grid_id": (f"{state_row.state[:3].upper()}_{point.y:.5f}_{point.x:.5f}"),
                        "state": state_row.state,
                        "latitude": point.y,
                        "longitude": point.x,
                        "grid_resolution_degrees": resolution,
                        "geometry": clipped,
                    }
                )

    grid = gpd.GeoDataFrame(rows, crs=4326)
    grid = grid.sort_values(["state", "latitude", "longitude"]).reset_index(drop=True)
    output.parent.mkdir(parents=True, exist_ok=True)
    grid.to_parquet(output, index=False)
    LOGGER.info("Built %d cells at %.3f-degree resolution", len(grid), resolution)
    return grid
