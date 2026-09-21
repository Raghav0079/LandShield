from __future__ import annotations

import json
from pathlib import Path

import geopandas as gpd
import pandas as pd
import typer

from ner_landslide.config import Settings, load_settings
from ner_landslide.data.boundaries import download_ner_boundaries
from ner_landslide.data.landslides import download_landslide_events
from ner_landslide.data.stac import PlanetaryComputerConnector
from ner_landslide.features import build_training_features
from ner_landslide.grid import build_grid
from ner_landslide.labels import build_training_samples
from ner_landslide.model import load_model, train_model
from ner_landslide.predict import generate_predictions
from ner_landslide.utils import configure_logging

app = typer.Typer(
    help="Train and run the North East India landslide-risk pipeline.",
    no_args_is_help=True,
)


def _settings(config: Path, resolution: float | None) -> Settings:
    settings = load_settings(config)
    if resolution is not None:
        settings = settings.model_copy(
            update={
                "region": settings.region.model_copy(update={"grid_resolution_degrees": resolution})
            }
        )
    settings.create_directories()
    return settings


def _static_features(
    settings: Settings,
    grid: gpd.GeoDataFrame,
    boundaries: gpd.GeoDataFrame,
    refresh: bool,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    terrain_path = settings.project.processed_dir / "terrain_features.parquet"
    satellite_path = settings.project.processed_dir / "satellite_features.parquet"
    if terrain_path.exists() and satellite_path.exists() and not refresh:
        terrain = pd.read_parquet(terrain_path)
        satellite = pd.read_parquet(satellite_path)
        expected = set(grid["grid_id"])
        if set(terrain["grid_id"]) == expected and set(satellite["grid_id"]) == expected:
            return terrain, satellite
        refresh = True
    connector = PlanetaryComputerConnector(settings)
    terrain = connector.terrain_features(grid, boundaries, refresh=refresh)
    if settings.stac.enabled:
        satellite = connector.satellite_features(grid, boundaries, refresh=refresh)
    else:
        satellite = grid[["grid_id"]].copy()
        satellite["ndvi"] = float("nan")
        satellite["ndmi"] = float("nan")
        satellite.to_parquet(satellite_path, index=False)
    return terrain, satellite


def _prepare(
    settings: Settings,
    *,
    refresh: bool = False,
) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame, gpd.GeoDataFrame, pd.DataFrame, pd.DataFrame]:
    boundaries = download_ner_boundaries(settings, refresh=refresh)
    grid = build_grid(settings, boundaries, refresh=refresh)
    events = download_landslide_events(settings, boundaries, refresh=refresh)
    terrain, satellite = _static_features(settings, grid, boundaries, refresh)
    return boundaries, grid, events, terrain, satellite


@app.command("download")
def download_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
    refresh: bool = typer.Option(False, help="Ignore cached source data."),
    resolution: float | None = typer.Option(None, min=0.01, max=2),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Download boundaries, event inventory, terrain, and imagery features."""
    configure_logging(verbose)
    settings = _settings(config, resolution)
    _, grid, events, terrain, satellite = _prepare(settings, refresh=refresh)
    typer.echo(
        f"Prepared {len(grid)} cells, {len(events)} events, "
        f"{terrain['elevation_m'].notna().sum()} DEM cells, "
        f"and {satellite['ndvi'].notna().sum()} imagery cells."
    )


@app.command("build-features")
def build_features_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
    sensor_csv: Path | None = typer.Option(None, exists=True),
    refresh: bool = typer.Option(False),
    resolution: float | None = typer.Option(None, min=0.01, max=2),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Build labeled event/control rows and their antecedent features."""
    configure_logging(verbose)
    settings = _settings(config, resolution)
    _, grid, events, terrain, satellite = _prepare(settings, refresh=refresh)
    samples = build_training_samples(settings, grid, events, refresh=refresh)
    features = build_training_features(
        settings,
        samples,
        grid,
        terrain,
        satellite,
        sensor_path=sensor_csv,
        refresh=refresh,
    )
    typer.echo(f"Saved {len(features)} training rows.")


@app.command("train")
def train_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
    features_path: Path | None = typer.Option(None, exists=True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Train, calibrate, evaluate, and save the risk model."""
    configure_logging(verbose)
    settings = _settings(config, None)
    path = features_path or settings.project.processed_dir / "training_features.parquet"
    features = pd.read_parquet(path)
    model, report = train_model(settings, features)
    typer.echo(
        f"Saved model trained at {model.trained_at}; test PR-AUC={report['overall']['pr_auc']}."
    )


@app.command("evaluate")
def evaluate_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
) -> None:
    """Print the persisted spatial-temporal evaluation report."""
    settings = _settings(config, None)
    report_path = settings.project.artifacts_dir / "evaluation.json"
    if not report_path.exists():
        raise typer.BadParameter("No evaluation report exists; run the train command first")
    typer.echo(json.dumps(json.loads(report_path.read_text(encoding="utf-8")), indent=2))


@app.command("predict")
def predict_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
    model_path: Path | None = typer.Option(None, exists=True),
    sensor_csv: Path | None = typer.Option(None, exists=True),
    refresh_weather: bool = typer.Option(True),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Generate the current all-NER forecast risk map."""
    configure_logging(verbose)
    settings = _settings(config, None)
    boundaries = download_ner_boundaries(settings)
    grid = build_grid(settings, boundaries)
    terrain, satellite = _static_features(settings, grid, boundaries, refresh=False)
    model = load_model(model_path or settings.project.artifacts_dir / "landslide_model.joblib")
    forecast = generate_predictions(
        settings,
        model,
        grid,
        terrain,
        satellite,
        sensor_path=sensor_csv,
        refresh_weather=refresh_weather,
    )
    typer.echo(f"Saved {len(forecast)} forecast rows.")


@app.command("run")
def run_command(
    config: Path = typer.Option(Path("configs/ner.yaml"), exists=True),
    sensor_csv: Path | None = typer.Option(None, exists=True),
    refresh: bool = typer.Option(False),
    resolution: float | None = typer.Option(None, min=0.01, max=2),
    verbose: bool = typer.Option(False, "--verbose", "-v"),
) -> None:
    """Run ingestion, feature engineering, training, and prediction."""
    configure_logging(verbose)
    settings = _settings(config, resolution)
    _, grid, events, terrain, satellite = _prepare(settings, refresh=refresh)
    samples = build_training_samples(settings, grid, events, refresh=refresh)
    features = build_training_features(
        settings,
        samples,
        grid,
        terrain,
        satellite,
        sensor_path=sensor_csv,
        refresh=refresh,
    )
    model, _ = train_model(settings, features)
    forecast = generate_predictions(
        settings,
        model,
        grid,
        terrain,
        satellite,
        sensor_path=sensor_csv,
        refresh_weather=True,
    )
    typer.echo(f"Pipeline complete: {len(features)} training and {len(forecast)} forecast rows.")
