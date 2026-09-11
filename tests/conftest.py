from __future__ import annotations

from pathlib import Path

import pytest

from ner_landslide.config import Settings, load_settings


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    configured = load_settings(Path(__file__).parents[1] / "configs" / "ner.yaml")
    project = configured.project.model_copy(
        update={
            "cache_dir": tmp_path / "cache",
            "processed_dir": tmp_path / "processed",
            "artifacts_dir": tmp_path / "artifacts",
            "outputs_dir": tmp_path / "outputs",
        }
    )
    training = configured.training.model_copy(
        update={
            "holdout_states": ["Sikkim"],
            "minimum_training_records": None,
            "controls_per_event": 2,
            "exclusion_radius_km": 5,
        }
    )
    result = configured.model_copy(update={"project": project, "training": training})
    result.create_directories()
    return result
