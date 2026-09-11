from __future__ import annotations

import logging

import geopandas as gpd
import numpy as np
import pandas as pd

from ner_landslide.config import Settings

LOGGER = logging.getLogger(__name__)


def _haversine_km(
    latitude: np.ndarray,
    longitude: np.ndarray,
    event_latitude: float,
    event_longitude: float,
) -> np.ndarray:
    lat1 = np.radians(latitude)
    lon1 = np.radians(longitude)
    lat2 = np.radians(event_latitude)
    lon2 = np.radians(event_longitude)
    delta_latitude = lat1 - lat2
    delta_longitude = lon1 - lon2
    value = (
        np.sin(delta_latitude / 2) ** 2
        + np.cos(lat1) * np.cos(lat2) * np.sin(delta_longitude / 2) ** 2
    )
    return np.asarray(2 * 6371.0088 * np.arcsin(np.sqrt(value)), dtype=float)


def _eligible_control_cells(
    grid: gpd.GeoDataFrame,
    event_locations: gpd.GeoDataFrame,
    target_date: pd.Timestamp,
    window: pd.Timedelta,
    exclusion_radius_km: float,
) -> gpd.GeoDataFrame:
    allowed = np.ones(len(grid), dtype=bool)
    nearby_events = event_locations.loc[
        event_locations["event_date"].between(target_date - window, target_date + window)
    ]
    for event in nearby_events.itertuples(index=False):
        distances = _haversine_km(
            grid["latitude"].to_numpy(),
            grid["longitude"].to_numpy(),
            event.event_latitude,
            event.event_longitude,
        )
        allowed &= distances >= exclusion_radius_km
    return grid.loc[allowed]


def build_training_samples(
    settings: Settings,
    grid: gpd.GeoDataFrame,
    events: gpd.GeoDataFrame,
    *,
    refresh: bool = False,
) -> pd.DataFrame:
    """Map events to cells and sample geographically separated same-day controls."""
    output = settings.project.processed_dir / "training_samples.parquet"
    if output.exists() and not refresh:
        cached = pd.read_parquet(output)
        cached_minimum = settings.training.minimum_training_records or 0
        dates_are_valid = (
            pd.to_datetime(cached["date"]).max() <= pd.to_datetime(events["event_date"]).max()
        )
        keys_are_unique = not cached.duplicated(["grid_id", "date"]).any()
        if (
            len(cached) >= cached_minimum
            and dates_are_valid
            and keys_are_unique
            and set(cached["grid_id"]).issubset(set(grid["grid_id"]))
        ):
            return cached
        LOGGER.info("Training size or grid changed; rebuilding cached samples")
    if events.empty:
        raise RuntimeError("No landslide events are available for training")

    all_events = events.copy()
    if settings.training.max_events and len(events) > settings.training.max_events:
        state_names = sorted(events["state"].unique())
        quota, remainder = divmod(settings.training.max_events, len(state_names))
        selected = []
        for index, state in enumerate(state_names):
            state_events = events.loc[events["state"] == state]
            count = min(len(state_events), quota + (index < remainder))
            selected.append(
                state_events.sample(
                    n=count,
                    random_state=settings.project.seed + index,
                )
            )
        events = gpd.GeoDataFrame(pd.concat(selected, ignore_index=True), crs=events.crs)

    event_points = events[["event_date", "state", "geometry"]].copy()
    joined = gpd.sjoin(
        event_points,
        grid[["grid_id", "state", "latitude", "longitude", "geometry"]].rename(
            columns={"state": "grid_state"}
        ),
        predicate="intersects",
        how="inner",
    )
    positives = (
        joined.rename(columns={"event_date": "date"})[
            ["grid_id", "state", "date", "latitude", "longitude"]
        ]
        .drop_duplicates(["grid_id", "date"])
        .reset_index(drop=True)
    )
    positives["label"] = 1
    positives["sample_type"] = "event"

    event_locations = all_events.copy()
    event_locations["event_latitude"] = event_locations.geometry.y
    event_locations["event_longitude"] = event_locations.geometry.x
    rng = np.random.default_rng(settings.project.seed)
    controls: list[dict[str, object]] = []
    window = pd.Timedelta(days=settings.training.event_window_days)
    for positive in positives.itertuples(index=False):
        candidates = grid.loc[grid["state"] == positive.state].copy()
        candidates = _eligible_control_cells(
            candidates,
            event_locations,
            positive.date,
            window,
            settings.training.exclusion_radius_km,
        )
        count = min(settings.training.controls_per_event, len(candidates))
        if count == 0:
            LOGGER.warning("No eligible control cells for %s on %s", positive.state, positive.date)
            continue
        indices = rng.choice(len(candidates), size=count, replace=False)
        for candidate in candidates.iloc[indices].itertuples(index=False):
            controls.append(
                {
                    "grid_id": candidate.grid_id,
                    "state": candidate.state,
                    "date": positive.date,
                    "latitude": candidate.latitude,
                    "longitude": candidate.longitude,
                    "label": 0,
                    "sample_type": "control",
                }
            )

    samples = pd.concat([positives, pd.DataFrame(controls)], ignore_index=True)
    samples["date"] = pd.to_datetime(samples["date"]).dt.normalize()
    samples = samples.drop_duplicates(["grid_id", "date"], keep="first").reset_index(drop=True)

    minimum = settings.training.minimum_training_records
    if minimum and len(samples) < minimum:
        existing = set(zip(samples["grid_id"], samples["date"], strict=True))
        event_dates = pd.DatetimeIndex(event_locations["event_date"].dropna().unique())
        background_dates = pd.date_range(
            settings.training.start_date,
            event_locations["event_date"].max(),
            freq="7D",
        )
        candidate_dates = event_dates.union(background_dates)
        background: list[dict[str, object]] = []
        for target_date in candidate_dates:
            candidates = _eligible_control_cells(
                grid,
                event_locations,
                pd.Timestamp(target_date),
                window,
                settings.training.exclusion_radius_km,
            )
            for candidate in candidates.itertuples(index=False):
                key = (candidate.grid_id, pd.Timestamp(target_date))
                if key in existing:
                    continue
                background.append(
                    {
                        "grid_id": candidate.grid_id,
                        "state": candidate.state,
                        "date": pd.Timestamp(target_date),
                        "latitude": candidate.latitude,
                        "longitude": candidate.longitude,
                        "label": 0,
                        "sample_type": "background_control",
                    }
                )
        rng.shuffle(background)
        needed = minimum - len(samples)
        if len(background) < needed:
            raise RuntimeError(
                f"Only {len(samples) + len(background)} unique safe samples are available; "
                f"cannot reach requested minimum of {minimum}"
            )
        samples = pd.concat([samples, pd.DataFrame(background[:needed])], ignore_index=True)

    samples.insert(0, "sample_id", [f"S{number:07d}" for number in range(len(samples))])
    samples.to_parquet(output, index=False)
    LOGGER.info(
        "Built %d samples: %d events and %d controls",
        len(samples),
        int(samples["label"].sum()),
        int((samples["label"] == 0).sum()),
    )
    return pd.DataFrame(samples)
