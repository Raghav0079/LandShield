from __future__ import annotations

import geopandas as gpd
from shapely.geometry import Point, box

from ner_landslide.config import Settings
from ner_landslide.grid import build_grid
from ner_landslide.labels import build_training_samples


def test_grid_and_control_sampling(settings: Settings) -> None:
    settings = settings.model_copy(
        update={
            "region": settings.region.model_copy(
                update={"states": ["Assam"], "grid_resolution_degrees": 0.25}
            )
        }
    )
    boundaries = gpd.GeoDataFrame(
        {"state": ["Assam"]},
        geometry=[box(90, 25, 91, 26)],
        crs=4326,
    )
    grid = build_grid(settings, boundaries)
    assert len(grid) == 16
    assert grid["grid_id"].is_unique

    events = gpd.GeoDataFrame(
        {"event_date": ["2025-06-01"], "state": ["Assam"]},
        geometry=[Point(90.125, 25.125)],
        crs=4326,
    )
    events["event_date"] = events["event_date"].astype("datetime64[ns]")
    samples = build_training_samples(settings, grid, events)

    assert samples["label"].sum() == 1
    assert (samples["label"] == 0).sum() == 2
    assert samples["sample_id"].is_unique
    assert set(samples["state"]) == {"Assam"}

    expanded_settings = settings.model_copy(
        update={"training": settings.training.model_copy(update={"minimum_training_records": 20})}
    )
    expanded = build_training_samples(expanded_settings, grid, events)
    assert len(expanded) == 20
    assert expanded["label"].sum() == 1
    assert "background_control" in set(expanded["sample_type"])
