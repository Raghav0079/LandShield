# 🛡️ LandShield — AI-Based Landslide Early Warning & Risk Monitoring System in NER

> **An AI/ML-driven early warning and risk monitoring solution for the North-Eastern Region (NER) of India**  
> **Target Region:** All 8 North-Eastern States (Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, Nagaland, Sikkim, Tripura)

---

## 📌 Executive Summary

**LandShield** is an end-to-end, data-driven AI early warning system engineered specifically for the complex topography of the North-Eastern Region of India. By fusing static geospatial terrain features with dynamic real-time meteorological and satellite telemetry on a unified $0.1^\circ \times 0.1^\circ$ spatial cell grid, LandShield generates **0 to 72-hour rolling risk forecasts** with high precision.

Equipped with a **$0\text{ ms}$ zero-latency offline KDTree spatial proximity engine**, LandShield maps raw latitude/longitude coordinates to complete administrative addresses (Village, Circle/Tehsil, District, State, and 6-digit PIN Code) and translates risk probabilities into plain-language **Explainable AI (SHAP)** reasoning for emergency response agencies like NDRF, SDRF, and local authorities.

---

## ✨ Key Features & Innovations

- **Unified Micro-Cell Grid Fusion:** Combines static terrain/geology datasets (Copernicus DEM, slope, soil wetness index) with dynamic telemetry feeds (precipitation, live satellite indices) on a common grid.
- **Multi-Horizon Probabilistic Forecasting:** Provides rolling 3-hourly forecasts across 3 time horizons: **Nowcast (0–24h)**, **Medium (24–48h)**, and **Extended (48–72h)**.
- **Offline KDTree Micro-Geocoding:** $0\text{ ms}$ instant administrative lookup using spatial KD-Trees without external API dependencies or network latency during emergencies.
- **Explainable AI (SHAP):** Unpacks model predictions to show decision-makers *why* a grid cell is at high risk (e.g., 3-day accumulated rainfall, saturated surface, steep slope angle).
- **NDRF Action Dispatch Table:** Auto-generates prioritized emergency response tables with exact PIN codes, risk severity levels, and key environmental triggers.
- **Interactive 3D GIS Dashboard:** Built with WebGL-powered **PyDeck** and **Streamlit** for real-time risk heatmaps, risk polygon overlays, and state-wide administrative filtering.

---

## 🛠️ Technology Stack

| Layer | Technologies & Tools |
| :--- | :--- |
| **Language & Environment** | Python 3.10+, `uv` (Fast dependency manager) |
| **Machine Learning & AI** | LightGBM, HistGradientBoosting, Scikit-Learn |
| **Explainable AI (XAI)** | SHAP (SHapley Additive exPlanations) |
| **Spatial & Geocoding Engine** | SciPy (`scipy.spatial.KDTree`), GeoPandas, Shapely, PyArrow |
| **GIS & Frontend Dashboard** | Streamlit, PyDeck (Deck.gl 3D WebGL), Carto / OSM Tiles |
| **Data Pipelines & Storage** | Pandas, NumPy, Parquet, Open-Meteo API, Copernicus DEM, Sentinel-2 |

---

## 📁 Repository Directory Structure

```text
AI-Ml-Based-landslide-Early-warning-Detection-Model-in-North-East-Region/
│
├── configs/                  # Configuration files (smoke.yaml, full_run.yaml)
├── data/                     # Raw and processed GIS & weather telemetry data
│   ├── smoke/                # Pre-trained smoke datasets & features
│   └── geocodes/             # NER Administrative boundary & PIN code datasets
├── models/                   # Saved ML models & feature transformers
├── outputs/                  # Forecast risk GeoJSONs & ranked risk CSVs
│
├── dashboard.py              # Main Interactive Streamlit/PyDeck GIS Dashboard
├── setup_and_run.py          # One-command orchestration script
├── requirements.txt          # Python dependencies list
├── .gitignore                # Version control exclusions
└── README.md                 # System documentation

🚀 Quickstart & Installation Guide

1. Clone the Repository

git clone [https://github.com/vansh9696/AI-Ml-Based-landslide-Early-warning-Detection-Model-in-North-East-Region.git](https://github.com/vansh9696/AI-Ml-Based-landslide-Early-warning-Detection-Model-in-North-East-Region.git)
cd AI-Ml-Based-landslide-Early-warning-Detection-Model-in-North-East-Region

2. Set Up Virtual Environment & Activate
Linux / macOS:

python3 -m venv .venv
source .venv/bin/activate
Windows (CMD / PowerShell):

DOS
python -m venv .venv
.venv\Scripts\activate

3. Install Dependencies
pip install --upgrade pip
pip install -r requirements.txt
4. Run One-Command Pipeline & Launch Dashboard
Execute the execution script to load/fetch telemetry data, evaluate model metrics, and spin up the Streamlit interface:


python3 setup_and_run.py --dashboard
Alternatively, launch the dashboard directly via Streamlit:
streamlit run dashboard.py

🔬 Model Architecture & Data Flow


[ Data Sources ] ──> [ Ingestion & Preprocessing ] ──> [ Feature Engineering ]
 • IMD / Open-Meteo      • Cleaning & Alignment        • 3-Day / 7-Day Rainfall
 • Copernicus DEM        • Spatial Grid Mapping        • Soil Moisture Index
 • Sentinel Satellites   • Boundary Clipping           • Terrain Slope Angle
                                                                │
                                                                ▼
[ Output & Alerts ] <── [ Explainability & Action ] <── [ Ensemble AI Engine ]
 • PyDeck 3D Heatmap     • SHAP Factor Contribution     • LightGBM / HistGBM
 • NDRF Dispatch Table   • 0ms KDTree Geocoding         • Calibrated Probability
 • CAP / SMS Warnings    • Admin Address + PIN Code     • 0-72h Risk Horizons



📜 License & Acknowledgments
All open datasets used are credited to IMD, ISRO Bhuvan, Sentinel Copernicus, and Open-Meteo.