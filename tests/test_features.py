from __future__ import annotations

import pandas as pd

from ner_landslide.features import _upsert_training_features


def test_training_feature_upsert_updates_and_appends() -> None:
    first_seen = pd.Timestamp("2026-01-01", tz="UTC")
    refreshed = pd.Timestamp("2026-02-01", tz="UTC")
    existing = pd.DataFrame(
        {
            "sample_id": ["S0000000", "S0000001"],
            "grid_id": ["A", "C"],
            "state": ["Assam", "Assam"],
            "date": ["2025-06-01", "2025-06-02"],
            "label": [0, 0],
            "rain_1d_mm": [1.0, 3.0],
            "first_ingested_at": [first_seen, first_seen],
            "last_updated_at": [first_seen, first_seen],
        }
    )
    incoming = pd.DataFrame(
        {
            "sample_id": ["N1", "N2"],
            "grid_id": ["A", "B"],
            "state": ["Assam", "Assam"],
            "date": ["2025-06-01", "2025-06-03"],
            "label": [1, 0],
            "rain_1d_mm": [10.0, 2.0],
        }
    )

    merged = _upsert_training_features(existing, incoming, updated_at=refreshed)

    assert len(merged) == 3
    assert merged["sample_id"].is_unique
    updated = merged.loc[merged["grid_id"] == "A"].iloc[0]
    assert updated["label"] == 1
    assert updated["rain_1d_mm"] == 10
    assert updated["first_ingested_at"] == first_seen
    assert updated["last_updated_at"] == refreshed
    unchanged = merged.loc[merged["grid_id"] == "C"].iloc[0]
    assert unchanged["last_updated_at"] == first_seen
