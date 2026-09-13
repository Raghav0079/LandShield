"""Real-time GIS dashboard for NER landslide-risk forecasts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import geopandas as gpd
import numpy as np
import pandas as pd
import pydeck as pdk
import shap
import streamlit as st
from streamlit_autorefresh import st_autorefresh

from ner_landslide.config import load_settings
from ner_landslide.model import LandslideRiskModel
from ner_landslide.predict import generate_predictions
from run_python import ensure_data, get_model

st.set_page_config(
    page_title="LandShield",
    page_icon="⛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(145deg, #07131f 0%, #0c1f2b 55%, #102f32 100%);
    }
    [data-testid="stMetric"] {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.13);
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 10px 30px rgba(0,0,0,0.18);
    }
    [data-testid="stSidebar"] {
        background: #071923;
    }
    .status-pill {
        display: inline-block;
        padding: 6px 12px;
        border-radius: 999px;
        background: rgba(34,197,94,0.15);
        color: #86efac;
        border: 1px solid rgba(34,197,94,0.35);
        font-size: 0.85rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

COLORS = {
    "low": [34, 197, 94, 155],
    "medium": [250, 204, 21, 175],
    "high": [249, 115, 22, 195],
    "severe": [239, 68, 68, 220],
}


def load_forecast(path: Path) -> gpd.GeoDataFrame:
    frame = gpd.read_parquet(path)
    frame["target_date"] = pd.to_datetime(frame["target_date"])
    return frame


def refresh_forecast(config_path: Path) -> gpd.GeoDataFrame:
    settings = load_settings(config_path)
    settings.create_directories()
    model = get_model(settings, retrain=False, rebuild_data=False)
    grid, terrain, satellite = ensure_data(
        settings,
        require_training=False,
        rebuild=False,
    )
    return generate_predictions(settings, model, grid, terrain, satellite)


def shap_reasons(model: LandslideRiskModel, frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame()
    estimator = model.estimator
    transformer = estimator.named_steps["prepare"]
    classifier = estimator.named_steps["classifier"]
    transformed = transformer.transform(frame[model.features])
    shap_values = np.asarray(shap.TreeExplainer(classifier)(transformed).values)
    if getattr(shap_values, "ndim", 2) == 3:
        shap_values = shap_values[:, :, 1]
    transformed_names = [str(name) for name in transformer.get_feature_names_out()]
    reason_rows = []
    for (_, row), values in zip(frame.iterrows(), shap_values, strict=True):
        contributions = {}
        for name, value in zip(transformed_names, values, strict=True):
            feature = next(
                (
                    candidate
                    for candidate in model.features
                    if name.endswith(candidate)
                    or name.endswith(f"missingindicator_{candidate}")
                ),
                name,
            )
            contributions[feature] = contributions.get(feature, 0.0) + float(value)
        positive = sorted(
            ((feature, value) for feature, value in contributions.items() if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )[:3]
        total_positive = sum(value for _, value in positive)
        result = {
            "state": row["state"],
            "latitude": row["latitude"],
            "longitude": row["longitude"],
            "target_date": row["target_date"],
            "risk_probability": row["risk_probability"],
            "risk_category": row["risk_category"],
        }
        for index in range(3):
            feature, value = positive[index] if index < len(positive) else ("-", 0.0)
            result[f"reason_{index + 1}"] = feature
            result[f"reason_{index + 1}_impact"] = (
                value / total_positive if total_positive else 0.0
            )
        reason_rows.append(result)
    return pd.DataFrame(reason_rows)


st.sidebar.title("Control centre")
dataset = st.sidebar.selectbox(
    "Configuration",
    ["Smoke — fast, all 8 states", "Full — 0.1° production grid"],
)
config_path = Path(
    "configs/smoke.yaml" if dataset.startswith("Smoke") else "configs/ner.yaml"
)
settings = load_settings(config_path)
forecast_path = settings.project.outputs_dir / "risk_forecast.parquet"
model = get_model(settings, retrain=False, rebuild_data=False)

auto_refresh = st.sidebar.toggle("Automatic live refresh", value=False)
refresh_minutes = st.sidebar.slider("Refresh interval (minutes)", 5, 60, 15, 5)
refresh_count = (
    st_autorefresh(interval=refresh_minutes * 60 * 1000, key="forecast_refresh")
    if auto_refresh
    else 0
)
manual_refresh = st.sidebar.button(
    "Refresh forecast now", type="primary", use_container_width=True
)

if manual_refresh or refresh_count > 0 or not forecast_path.exists():
    with st.spinner("Fetching weather and calculating fresh landslide risks…"):
        forecast = refresh_forecast(config_path)
    st.toast("Forecast updated from live weather data")
else:
    forecast = load_forecast(forecast_path)

st.title("LandShield")
st.caption(
    "Landslide risk monitoring and early warning across "
    "Arunachal Pradesh, Assam, Manipur, Meghalaya, Mizoram, "
    "Nagaland, Sikkim and Tripura."
)

generated = pd.to_datetime(forecast["generated_at"], utc=True).max()
age_minutes = max(
    (datetime.now(UTC) - generated.to_pydatetime()).total_seconds() / 60, 0
)
st.markdown(
    f'<span class="status-pill">Live data · updated {age_minutes:.0f} minutes ago</span>',
    unsafe_allow_html=True,
)

dates = sorted(forecast["target_date"].dt.date.unique())
selected_date = st.sidebar.select_slider("Forecast date", options=dates, value=dates[0])
states = sorted(forecast["state"].unique())
selected_states = st.sidebar.multiselect("States", states, default=states)
categories = st.sidebar.multiselect(
    "Risk categories",
    ["low", "medium", "high", "severe"],
    default=["low", "medium", "high", "severe"],
)
minimum_probability = st.sidebar.slider("Minimum probability", 0.0, 1.0, 0.0, 0.05)
map_mode = st.sidebar.radio("Map layer", ["Risk polygons", "Heatmap"])

filtered = forecast.loc[
    (forecast["target_date"].dt.date == selected_date)
    & forecast["state"].isin(selected_states)
    & forecast["risk_category"].isin(categories)
    & (forecast["risk_probability"] >= minimum_probability)
].copy()
filtered["fill_color"] = filtered["risk_category"].map(COLORS)

metric_columns = st.columns(4)
metric_columns[0].metric(
    "Peak probability",
    f"{filtered['risk_probability'].max():.1%}" if len(filtered) else "—",
)
metric_columns[1].metric(
    "High / severe cells",
    int(filtered["risk_category"].isin(["high", "severe"]).sum()),
)
metric_columns[2].metric("Active warnings", int(filtered["warning"].sum()))
metric_columns[3].metric("States covered", int(filtered["state"].nunique()))

if filtered.empty:
    st.warning("No cells match the selected filters.")
else:
    view = pdk.ViewState(
        latitude=float(filtered["latitude"].mean()),
        longitude=float(filtered["longitude"].mean()),
        zoom=5.2,
        pitch=30,
    )
    if map_mode == "Risk polygons":
        geojson = json.loads(filtered.to_json(default=str))
        layer = pdk.Layer(
            "GeoJsonLayer",
            geojson,
            pickable=True,
            stroked=True,
            filled=True,
            get_fill_color="properties.fill_color",
            get_line_color=[220, 235, 240, 130],
            line_width_min_pixels=0.5,
        )
    else:
        layer = pdk.Layer(
            "HeatmapLayer",
            filtered,
            get_position="[longitude, latitude]",
            get_weight="risk_probability",
            radius_pixels=70,
            intensity=1.4,
            threshold=0.05,
            pickable=True,
        )

    deck = pdk.Deck(
        layers=[layer],
        initial_view_state=view,
        map_provider="carto",
        map_style="dark",
        tooltip={
            "html": (
                "<b>{state}</b><br/>Risk: {risk_category}<br/>"
                "Probability: {risk_probability}<br/>Target: {target_date}"
            ),
            "style": {"backgroundColor": "#071923", "color": "white"},
        },
    )
    st.pydeck_chart(deck, use_container_width=True, height=610)

left, right = st.columns([1, 1.4])
with left:
    st.subheader("Average risk by state")
    state_summary = (
        filtered.groupby("state", as_index=False)["risk_probability"]
        .mean()
        .sort_values("risk_probability", ascending=False)
    )
    st.bar_chart(state_summary, x="state", y="risk_probability", horizontal=True)

st.subheader("SHAP reasons for elevated risk")
shap_input = (
    filtered.loc[filtered["risk_category"].isin(["high", "severe"])]
    .sort_values("risk_probability", ascending=False)
    .head(100)
)
if shap_input.empty:
    st.info("No high or severe regions match the current filters.")
else:
    reasons = shap_reasons(model, shap_input)
    st.dataframe(
        reasons,
        hide_index=True,
        use_container_width=True,
        column_config={
            "risk_probability": st.column_config.ProgressColumn(
                "Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            ),
            "reason_1_impact": st.column_config.NumberColumn(
                "Reason 1 impact", format="%.1f%%"
            ),
            "reason_2_impact": st.column_config.NumberColumn(
                "Reason 2 impact", format="%.1f%%"
            ),
            "reason_3_impact": st.column_config.NumberColumn(
                "Reason 3 impact", format="%.1f%%"
            ),
        },
    )

with right:
    st.subheader("Priority locations")
    display_columns = [
        "state",
        "latitude",
        "longitude",
        "risk_probability",
        "risk_category",
        "warning",
        "drivers",
    ]
    priority = filtered.sort_values("risk_probability", ascending=False)[
        display_columns
    ].head(25)
    st.dataframe(
        priority,
        hide_index=True,
        use_container_width=True,
        column_config={
            "risk_probability": st.column_config.ProgressColumn(
                "Probability",
                min_value=0.0,
                max_value=1.0,
                format="%.2f",
            )
        },
    )

with st.expander("Data sources and interpretation"):
    st.write(
        "Forecast weather: Open-Meteo · Historical weather: NASA POWER · "
        "Terrain: Copernicus DEM · Imagery: Sentinel-2 · Events: NASA COOLR/GLC."
    )
    st.warning(
        "Decision-support prototype only. Scores require validation and human review "
        "before public warnings or emergency action."
    )
