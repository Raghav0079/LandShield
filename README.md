# 🛡️ LandShield — AI-Based Landslide Early Warning & Risk Monitoring System in NER

> **An AI/ML-driven early warning and risk monitoring solution for the North-Eastern Region (NER) of India**
> **Target Region:** All 8 North-Eastern States (Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura)
> **Built for:** Smart India Hackathon 2026 — Problem Statement **SIH26001** (Disaster Management) · Team **Kavach**

---

## 📌 Executive Summary

**LandShield** is an end-to-end, data-driven AI early warning system engineered specifically for the complex topography of the North-Eastern Region of India. By fusing static geospatial terrain features with dynamic real-time meteorological and satellite telemetry on a unified $0.1^\circ \times 0.1^\circ$ spatial cell grid, LandShield generates **0 to 72-hour rolling risk forecasts** with high precision.

The system combines a **static terrain susceptibility model** (what makes a location prone to landslides) with a **dynamic environmental trigger model** (what makes it likely *right now*) to produce calibrated, explainable risk forecasts at **6h / 24h / 72h** horizons.

Equipped with a **0 ms zero-latency offline KDTree spatial proximity engine**, LandShield maps raw latitude/longitude coordinates to complete administrative addresses (Village, Circle/Tehsil, District, State, and 6-digit PIN Code) and translates risk probabilities into plain-language **Explainable AI (SHAP)** reasoning for emergency response agencies like NDRF, SDRF, and local authorities.

---

## ✨ Key Features & Innovations

- **Unified Micro-Cell Grid Fusion:** Combines static terrain/geology datasets (Copernicus DEM, slope, soil wetness index) with dynamic telemetry feeds (precipitation, live satellite indices) on a common grid.
- **Multi-Horizon Probabilistic Forecasting:** Rolling 3-hourly forecasts across three time horizons — **Nowcast (0–24h)**, **Medium (24–48h)**, and **Extended (48–72h)** — aligned to the 6h/24h/72h decision windows used by response teams.
- **Static + Dynamic Risk Fusion:** A terrain susceptibility layer (slope, geology, soil saturation) is combined with a live environmental trigger layer (rainfall accumulation, satellite moisture indices) so the model captures both *where* landslides are likely and *when*.
- **Offline KDTree Micro-Geocoding:** 0 ms instant administrative lookup using spatial KD-Trees, with no external API dependency or network latency during emergencies.
- **Explainable AI (SHAP):** Unpacks model predictions to show decision-makers *why* a grid cell is at high risk (e.g., 3-day accumulated rainfall, saturated surface, steep slope angle).
- **NDRF Action Dispatch Table:** Auto-generates prioritized emergency response tables with exact PIN codes, risk severity levels, and key environmental triggers.
- **Interactive 3D GIS Dashboard:** Built with WebGL-powered **PyDeck** and **Streamlit** for real-time risk heatmaps, risk polygon overlays, and state-wide administrative filtering.

---

## 🛠️ Technology Stack

| Layer                          | Technologies & Tools                                               |
| ------------------------------ | -------------------------------------------------------------------|
| **Language & Environment**     | Python 3.10+, `uv` (fast dependency manager)                       |
| **Machine Learning & AI**      | LightGBM, HistGradientBoosting, Scikit-Learn                       |
| **Explainable AI (XAI)**       | SHAP (SHapley Additive exPlanations)                               |
| **Spatial & Geocoding Engine** | SciPy (`scipy.spatial.KDTree`), GeoPandas, Shapely, PyArrow         |
| **GIS & Frontend Dashboard**   | Streamlit, PyDeck (Deck.gl 3D WebGL), Carto / OSM Tiles             |
| **Data Pipelines & Storage**   | Pandas, NumPy, Parquet, Open-Meteo API, Copernicus DEM, Sentinel-2  |

---

## 🔬 Project Flow

The pipeline moves through four stages, from raw data to an actionable field alert:

```
[ Data Sources ] ──────> [ Ingestion & Preprocessing ] ──────> [ Feature Engineering ]
 • IMD / Open-Meteo        • Cleaning & temporal alignment        • 3-day / 7-day rainfall
 • Copernicus DEM          • 0.1° × 0.1° spatial grid mapping     • Soil moisture index
 • Sentinel satellites     • Administrative boundary clipping     • Terrain slope angle
                                                                          │
                                                                          ▼
[ Output & Alerts ] <────── [ Explainability & Action ] <────── [ Ensemble AI Engine ]
 • PyDeck 3D heatmap          • SHAP factor contribution            • LightGBM / HistGBM
 • NDRF dispatch table        • 0 ms KDTree geocoding                • Calibrated probability
 • CAP / SMS warnings         • Admin address + PIN code             • 6h / 24h / 72h horizons
```

**Step by step:**

1. **Ingest** — Pull static terrain layers (DEM, slope, soil) once, and dynamic telemetry (rainfall, satellite indices) on a rolling schedule from IMD, Open-Meteo, and Copernicus/Sentinel sources.
2. **Grid & align** — Snap every source onto the shared 0.1°×0.1° cell grid and clip to NER administrative boundaries.
3. **Engineer features** — Derive rainfall accumulation windows, soil saturation, and slope-based susceptibility per cell.
4. **Predict** — Run the LightGBM/HistGradientBoosting ensemble to produce a calibrated risk probability per cell, per horizon (6h/24h/72h).
5. **Explain** — Attach SHAP values so each prediction comes with a plain-language "why" (e.g., "72h accumulated rainfall + steep slope").
6. **Geocode** — Resolve the flagged grid cell to a Village/Circle/District/PIN code instantly via the offline KDTree engine.
7. **Act** — Surface results as a 3D risk heatmap in the Streamlit/PyDeck dashboard and as a prioritized NDRF dispatch table for emergency responders.

---

## 📁 Repository Directory Structure

```
LandShield/
│
├── .streamlit/                # Streamlit app configuration
├── configs/                   # Configuration files (smoke.yaml, full_run.yaml)
├── data/
│   └── smoke/                 # Smoke-test datasets & precomputed features
├── src/
│   └── ner_landslide/         # Core package: ingestion, feature engineering, model, geocoding
├── tests/                     # Unit & integration tests
│
├── dashboard.py                # Main interactive Streamlit/PyDeck GIS dashboard
├── setup_and_run.py            # One-command orchestration script
├── run_python.py               # Pipeline entry point / runner
├── pyproject.toml              # Project metadata & dependencies (uv/pip)
├── .gitignore                  # Version control exclusions
└── README.md                   # System documentation
```

---

## 🚀 Quickstart & Installation Guide

### 1. Clone the Repository

```bash
git clone https://github.com/Raghav0079/LandShield.git
cd LandShield
```

### 2. Set Up Virtual Environment & Activate

**Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

**Windows (CMD / PowerShell):**
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -e .
```
> This project uses `pyproject.toml` for dependency management. If you prefer `uv`:
> ```bash
> uv sync
> ```

### 4. Run the One-Command Pipeline & Launch Dashboard

```bash
python3 setup_and_run.py --dashboard
```

Alternatively, launch the dashboard directly via Streamlit:
```bash
streamlit run dashboard.py
```

---

## 📜 License & Acknowledgments

All open datasets used are credited to **IMD**, **ISRO Bhuvan**, **Sentinel Copernicus**, and **Open-Meteo**.

Built by **Team Kavach** for Smart India Hackathon 2026 (SIH26001 — AI-Based Early Warning and Landslide Risk Monitoring System in NER).
