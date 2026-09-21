from __future__ import annotations

import logging
import unicodedata

import geopandas as gpd
import requests

from ner_landslide.config import Settings
from ner_landslide.utils import CachedHttpClient

LOGGER = logging.getLogger(__name__)


def _normalized_name(value: str) -> str:
    ascii_name = unicodedata.normalize("NFKD", value).encode("ascii", "ignore").decode()
    return " ".join(ascii_name.casefold().split())


def download_ner_boundaries(
    settings: Settings,
    *,
    refresh: bool = False,
) -> gpd.GeoDataFrame:
    """Download open geoBoundaries ADM1 data and retain the configured NER states."""
    output = settings.project.processed_dir / "ner_states.geojson"
    if output.exists() and not refresh:
        return gpd.read_file(output)

    client = CachedHttpClient(settings.project.cache_dir)
    metadata = client.get_json(
        settings.region.boundary_url,
        cache_namespace="boundaries",
        refresh=refresh,
    )
    geometry_url = metadata.get("gjDownloadURL") or metadata.get("simplifiedGeometryGeoJSON")
    if not geometry_url:
        raise RuntimeError("geoBoundaries response did not contain a GeoJSON download URL")

    raw_path = settings.project.cache_dir / "boundaries" / "india_adm1.geojson"
    if refresh or not raw_path.exists():
        response = requests.get(geometry_url, timeout=120)
        response.raise_for_status()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        raw_path.write_bytes(response.content)

    frame = gpd.read_file(raw_path).to_crs(4326)
    name_column = next(
        (column for column in ("shapeName", "shape_name", "NAME_1", "state") if column in frame),
        None,
    )
    if name_column is None:
        raise RuntimeError(f"Cannot identify state-name column; available columns: {list(frame)}")

    canonical = {_normalized_name(name): name for name in settings.region.states}
    frame["state"] = (
        frame[name_column].astype(str).map(lambda value: canonical.get(_normalized_name(value)))
    )
    frame = frame.loc[frame["state"].notna(), ["state", "geometry"]].copy()
    missing = set(settings.region.states) - set(frame["state"])
    if missing:
        raise RuntimeError(f"Boundary source is missing configured states: {sorted(missing)}")

    frame = frame.dissolve(by="state", as_index=False)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_file(output, driver="GeoJSON")
    LOGGER.info("Saved %d NER state boundaries to %s", len(frame), output)
    return frame
