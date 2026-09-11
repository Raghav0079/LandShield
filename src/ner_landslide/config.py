from __future__ import annotations

from datetime import date
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ProjectConfig(StrictModel):
    seed: int = 42
    cache_dir: Path = Path("data/cache")
    processed_dir: Path = Path("data/processed")
    artifacts_dir: Path = Path("artifacts")
    outputs_dir: Path = Path("outputs")


class RegionConfig(StrictModel):
    states: list[str]
    grid_resolution_degrees: float = Field(gt=0, le=2)
    boundary_url: str


class TrainingConfig(StrictModel):
    start_date: date
    end_date: date
    max_events: int | None = Field(default=None, ge=8)
    minimum_training_records: int | None = Field(default=None, ge=20)
    controls_per_event: int = Field(ge=1, le=100)
    exclusion_radius_km: float = Field(gt=0)
    event_window_days: int = Field(ge=0, le=14)
    holdout_states: list[str] = Field(default_factory=list)
    temporal_holdout_fraction: float = Field(gt=0, lt=0.5)
    min_recall_for_threshold: float = Field(gt=0, le=1)

    @model_validator(mode="after")
    def dates_are_ordered(self) -> TrainingConfig:
        if self.start_date >= self.end_date:
            raise ValueError("training.start_date must be earlier than end_date")
        return self


class WeatherConfig(StrictModel):
    archive_url: str
    forecast_url: str
    power_url: str
    coordinate_batch_size: int = Field(ge=1, le=100)
    timeout_seconds: int = Field(ge=5)
    forecast_days: int = Field(ge=1, le=16)


class StacConfig(StrictModel):
    url: str
    dem_collection: str
    sentinel_collection: str
    sentinel_start_date: date
    sentinel_end_date: date
    max_sentinel_items: int = Field(ge=1)
    max_cloud_cover: float = Field(ge=0, le=100)
    enabled: bool = True


class LandslideConfig(StrictModel):
    query_url: str
    fallback_csv_url: str
    page_size: int = Field(ge=1, le=2000)


class RiskThresholds(StrictModel):
    medium: float
    high: float
    severe: float

    @model_validator(mode="after")
    def thresholds_are_ordered(self) -> RiskThresholds:
        if not 0 < self.medium < self.high < self.severe < 1:
            raise ValueError("risk thresholds must satisfy 0 < medium < high < severe < 1")
        return self


class PredictionConfig(StrictModel):
    warning_horizon_days: int = Field(ge=1, le=16)
    thresholds: RiskThresholds


class Settings(StrictModel):
    project: ProjectConfig
    region: RegionConfig
    training: TrainingConfig
    weather: WeatherConfig
    stac: StacConfig
    landslides: LandslideConfig
    prediction: PredictionConfig

    def create_directories(self) -> None:
        for path in (
            self.project.cache_dir,
            self.project.processed_dir,
            self.project.artifacts_dir,
            self.project.outputs_dir,
        ):
            path.mkdir(parents=True, exist_ok=True)


def load_settings(path: Path | str = "configs/ner.yaml") -> Settings:
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as stream:
        raw = yaml.safe_load(stream)
    return Settings.model_validate(raw)
