# NER Landslide Prediction Pipeline

An ML-only decision-support pipeline for rainfall-triggered landslide risk across
Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, and Tripura.
It downloads public data, learns from reported landslides, and produces cell-level
forecast probabilities and GIS-ready warning layers.

This is an engineering prototype, not a certified public-warning system. A high score
means that the model recognizes conditions similar to catalogued landslides; it does not
guarantee that a landslide will occur.

## One-command setup on a new computer

Copy or clone the complete project folder, open a terminal inside it, and run:

```bash
python3 setup_and_run.py
```

On Windows, use:

```powershell
py setup_and_run.py
```

`setup_and_run.py` creates `.venv`, installs the project and dashboard dependencies,
downloads any missing public data, trains a model when no model exists, retrieves the
current forecast, and writes the risk outputs. Existing downloads are cached and reused.

Python 3.11 or newer is required. If the `uv` package manager is installed, the setup
script can obtain and manage Python 3.11 automatically.

To prepare the data and open the real-time GIS dashboard:

```bash
python3 setup_and_run.py --dashboard
```

The first run uses `configs/smoke.yaml`: real public data for all eight NER states on a
coarse grid. It can take several minutes because DEM, satellite, event and weather data
must be downloaded. Later runs are much faster.

## Use the prepared project

The repository already contains a trained 10,000-record smoke dataset and model. From a
terminal opened in this folder:

```bash
cd /Users/akshay.sharma/SIH
source .venv/bin/activate
```

The most useful existing files are:

- `data/smoke/training_features.csv`: model features and the `label` target.
- `artifacts/smoke/evaluation.json`: train/validation/test metrics.
- `outputs/smoke/risk_ranked.csv`: one peak forecast per grid cell, highest risk first.
- `outputs/smoke/risk_forecast.geojson`: all forecast days for GIS/QGIS.

To fetch the latest weather forecast and update risk outputs without retraining:

```bash
ner-landslide predict --config configs/smoke.yaml
```

To retrain the model from the existing 10,000 records and then generate predictions:

```bash
ner-landslide train --config configs/smoke.yaml
ner-landslide predict --config configs/smoke.yaml
```

To rebuild the 10,000-record feature dataset from cached/public sources:

```bash
ner-landslide build-features --config configs/smoke.yaml
```

This writes both `training_features.parquet` and `training_features.csv`. To rerun every
stage, use:

```bash
ner-landslide run --config configs/smoke.yaml
```

Training feature rebuilds use upsert semantics. `grid_id + date` is the natural key:
overlapping rows are refreshed, previously unseen rows are appended, and duplicate keys
are removed. `first_ingested_at` records when a row first entered the dataset and
`last_updated_at` records its most recent refresh. Forecast outputs remain latest
snapshots and are overwritten on each prediction run.

In `risk_ranked.csv`, `risk_probability` is the model score, `risk_category` is
low/medium/high/severe, `warning` applies the learned alert threshold, and `drivers`
lists the strongest physical trigger indicators. Open `risk_peak.geojson` in QGIS to
view the cell polygons on a map.

## Run directly with Python

[`run_python.py`](run_python.py) uses the package API directly, without invoking the
`ner-landslide` CLI.

Generate current predictions with the existing model:

```bash
.venv/bin/python run_python.py
```

If required data or the model is missing, the script downloads/builds it automatically.

Retrain from `data/smoke/training_features.csv` first, then predict:

```bash
.venv/bin/python run_python.py --train
```

Select another configuration:

```bash
.venv/bin/python run_python.py --config configs/smoke.yaml
```

The script loads the YAML settings, training data or saved model, grid, terrain and
satellite features. It retrieves current Open-Meteo forecast data, displays the ten
highest-risk rows, and writes:

- `outputs/smoke/risk_ranked.csv`
- `outputs/smoke/risk_forecast.parquet`
- `outputs/smoke/risk_forecast.geojson`
- `outputs/smoke/risk_peak.geojson`

The same modules can be imported in a notebook:

```python
from ner_landslide.config import load_settings
from ner_landslide.model import load_model, train_model
from ner_landslide.predict import generate_predictions
```

Use `--rebuild-data` to deliberately ignore processed data and reconstruct it:

```bash
.venv/bin/python run_python.py --rebuild-data
```

## Real-time GIS dashboard

Launch the dashboard after setup:

```bash
.venv/bin/streamlit run dashboard.py
```

Or use the one-command setup and launcher:

```bash
python3 setup_and_run.py --dashboard
```

The dashboard provides:

- GIS risk polygons and an interpolated risk heatmap;
- forecast-date, state, category and probability filters;
- live Open-Meteo forecast refresh and optional automatic refresh;
- peak-risk, warning and state-coverage indicators;
- state-level risk summaries and a ranked priority-location list;
- source freshness and decision-support warnings.

Use **Refresh forecast now** to retrieve current weather and regenerate risks. Enabling
**Automatic live refresh** repeats this at the selected interval. The dashboard uses
Carto map tiles and therefore requires an internet connection for the basemap.

## Data flow

- Rainfall and modeled soil moisture: [Open-Meteo Historical and Forecast APIs](https://open-meteo.com/en/docs/historical-weather-api), used for rainfall intensity and soil saturation.
- Bulk historical rainfall and soil wetness: [NASA POWER Daily API](https://power.larc.nasa.gov/docs/services/api/temporal/daily/), used when large training tables would exceed hourly API limits.
- Elevation: [Copernicus DEM GLO-30](https://planetarycomputer.microsoft.com/dataset/cop-dem-glo-30), used for elevation, slope, and aspect.
- Multispectral imagery: [Sentinel-2 L2A](https://planetarycomputer.microsoft.com/dataset/sentinel-2-l2a), used for median NDVI and NDMI.
- Historical events: [NASA COOLR](https://gis.earthdata.nasa.gov/gis05/rest/services/Landslides/COOLR_Events_Points/FeatureServer), used as positive labels.
- State boundaries: [geoBoundaries](https://www.geoboundaries.org/) (CC BY 4.0), used for NER clipping and state evaluation.
- Optional field measurements: an authority-supplied normalized CSV, used to prefer recent nearby sensor soil moisture.

Open-Meteo soil moisture is reanalysis/model output, not a physical sensor reading. The
CSV adapter is the integration point for real deployments because sensor vendors and
government endpoints do not share one public schema.

The connector tries NASA's live COOLR feature service first. If that service is
unavailable, it uses the configured public mirror of NASA's catalog export and logs the
fallback. The fallback is real catalog data, but is less current than live COOLR.

## Quick start

Python 3.11 or newer and GDAL-compatible wheels are required.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev]"

# See available stages.
ner-landslide --help

# Full configured all-NER workflow.
ner-landslide run --config configs/ner.yaml
```

The default 0.1-degree grid covers all eight states. Public raster reads and weather API
calls can take one to several hours depending on network throttling, and cached source
data can require several GB. A first smoke run can use a coarser grid:

```bash
ner-landslide run --config configs/smoke.yaml
```

The smoke configuration still uses real public data across all eight states. It limits
training events and Sentinel scenes and writes results under `outputs/smoke/`; it does
not create synthetic observations.

Each stage is resumable:

```bash
ner-landslide download
ner-landslide build-features
ner-landslide train
ner-landslide evaluate
ner-landslide predict
```

Use `--refresh` only when source data should be downloaded again. Change date ranges,
resolution, holdout states, source URLs, and warning thresholds in
`configs/ner.yaml`.

## Outputs

Downloaded responses are content-addressed under `data/cache/`; normalized data is under
`data/processed/`.

- `artifacts/landslide_model.joblib`: preprocessing, classifier, probability calibrator,
  feature schema, and learned warning threshold.
- `artifacts/evaluation.json`: time-forward and state-held-out metrics, per-state metrics,
  calibration points, and permutation importance.
- `outputs/risk_forecast.parquet` and `.geojson`: one row per cell and forecast day.
- `outputs/risk_peak.geojson`: highest forecast risk for each cell.
- `outputs/risk_ranked.csv`: cells ranked by peak probability.

Forecast rows include `risk_probability`, `risk_category`, `warning`, `target_date`,
source names, generation time, and `drivers`. Drivers are ranked physical trigger
indicators and are explicitly not local SHAP explanations.

## Field sensor contract

Pass `--sensor-csv measurements.csv` to `build-features`, `predict`, or `run`.
The file must contain:

```csv
sensor_id,timestamp,latitude,longitude,soil_moisture
SM-001,2026-09-09T10:00:00Z,25.5788,91.8933,0.42
```

`soil_moisture` is volumetric water content from 0 to 1. The nearest observation within
25 km and 24 hours is used; otherwise modeled surface moisture remains the fallback.

## Modeling and validation

NASA events are mapped to grid cells. Control cells are sampled in the same state and on
the same dates, but at least the configured exclusion distance from any temporally nearby
event. This case-control design avoids materializing tens of millions of all-NER cell-day
rows and prevents the model from learning simple seasonal differences between events and
controls.

The model is a class-weighted histogram gradient booster with median imputation and
sigmoid probability calibration. Evaluation combines:

- a forward-in-time holdout;
- fully held-out configured states;
- per-state precision, recall, PR-AUC, ROC-AUC, Brier score, and confusion matrices.

PR-AUC and calibrated probability quality matter more than raw accuracy because
landslides are rare.

## Important limitations

- COOLR is a reported-event inventory. Remote and low-impact failures are underreported,
  event dates/locations can be uncertain, and absence from the catalogue is not proof of
  a true negative.
- Case-control probabilities can differ from real-world incidence. Before operational
  use, recalibrate against a complete regional inventory and local ground observations.
- The configured Sentinel composite is a susceptibility covariate, not live change
  detection. Cloudy monsoon scenes may be missing.
- A 0.1-degree weather grid cannot represent individual cut slopes, drainage failures,
  road works, geology, or building-scale hazards.
- Warning thresholds require review by geologists and disaster-management authorities.
  An operational service also needs redundant feeds, data-quality alarms, human review,
  communication protocols, and field validation.

## Development

```bash
ruff check .
ruff format --check .
mypy src
pytest
pytest -m network  # optional live-service smoke tests
```