"""Run the NER landslide pipeline directly through its Python API."""

from __future__ import annotations

import argparse
from pathlib import Path

import geopandas as gpd
import pandas as pd

from ner_landslide.config import Settings, load_settings
from ner_landslide.data.boundaries import download_ner_boundaries
from ner_landslide.data.landslides import download_landslide_events
from ner_landslide.data.stac import PlanetaryComputerConnector
from ner_landslide.features import build_training_features
from ner_landslide.grid import build_grid
from ner_landslide.labels import build_training_samples
from ner_landslide.model import LandslideRiskModel, load_model, train_model
from ner_landslide.predict import generate_predictions


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/smoke.yaml"),
        help="Pipeline YAML configuration.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Retrain from training_features.csv before predicting.",
    )
    parser.add_argument(
        "--rebuild-data",
        action="store_true",
        help="Redownload source data and rebuild all intermediate files.",
    )
    return parser.parse_args()


def ensure_data(
    settings: Settings,
    *,
    require_training: bool,
    rebuild: bool = False,
) -> tuple[gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    """Download and build any missing files needed for prediction or training."""
    processed = settings.project.processed_dir
    boundaries = download_ner_boundaries(settings, refresh=rebuild)
    grid = build_grid(settings, boundaries, refresh=rebuild)

    terrain_path = processed / "terrain_features.parquet"
    satellite_path = processed / "satellite_features.parquet"
    connector: PlanetaryComputerConnector | None = None
    if rebuild or not terrain_path.exists():
        print("Downloading Copernicus DEM terrain features...")
        connector = PlanetaryComputerConnector(settings)
        terrain = connector.terrain_features(grid, boundaries, refresh=rebuild)
    else:
        terrain = pd.read_parquet(terrain_path)

    if rebuild or not satellite_path.exists():
        print("Downloading Sentinel-2 imagery features...")
        connector = connector or PlanetaryComputerConnector(settings)
        satellite = connector.satellite_features(grid, boundaries, refresh=rebuild)
    else:
        satellite = pd.read_parquet(satellite_path)

    if require_training:
        csv_path = processed / "training_features.csv"
        parquet_path = processed / "training_features.parquet"
        if rebuild or not csv_path.exists():
            if parquet_path.exists() and not rebuild:
                pd.read_parquet(parquet_path).to_csv(csv_path, index=False)
            else:
                print("Downloading landslide events and historical weather...")
                events = download_landslide_events(settings, boundaries, refresh=rebuild)
                samples = build_training_samples(settings, grid, events, refresh=rebuild)
                build_training_features(
                    settings,
                    samples,
                    grid,
                    terrain,
                    satellite,
                    refresh=rebuild,
                )
    return grid, terrain, satellite


def get_model(
    settings: Settings,
    *,
    retrain: bool,
    rebuild_data: bool,
) -> LandslideRiskModel:
    """Load the saved model or train it after preparing missing source data."""
    model_path = settings.project.artifacts_dir / "landslide_model.joblib"
    if retrain or rebuild_data or not model_path.exists():
        ensure_data(settings, require_training=True, rebuild=rebuild_data)
        features_path = settings.project.processed_dir / "training_features.csv"
        features = pd.read_csv(features_path)
        model, report = train_model(settings, features)
        print(
            f"Trained with {len(features):,} records; test PR-AUC={report['overall']['pr_auc']:.4f}"
        )
    else:
        model = load_model(model_path)
        print(f"Loaded model: {model_path}")
    return model


def main() -> None:
    args = parse_args()
    settings = load_settings(args.config)
    settings.create_directories()
    model = get_model(
        settings,
        retrain=args.train,
        rebuild_data=args.rebuild_data,
    )
    grid, terrain, satellite = ensure_data(
        settings,
        require_training=False,
        rebuild=args.rebuild_data,
    )
    forecast = generate_predictions(settings, model, grid, terrain, satellite)

    columns = ["state", "target_date", "risk_probability", "risk_category", "warning"]
    print(forecast[columns].sort_values("risk_probability", ascending=False).head(10))
    print(f"CSV output: {settings.project.outputs_dir / 'risk_ranked.csv'}")


if __name__ == "__main__":
    main()
