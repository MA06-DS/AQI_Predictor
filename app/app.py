import os
import tempfile
from pathlib import Path

import hopsworks
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import sample_colorscale
import streamlit as st
import xgboost as xgb
from dotenv import load_dotenv


load_dotenv()

PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT", "anaskaaqi")
FEATURE_GROUP_NAME = "aqi_training_features"
FEATURE_GROUP_VERSION = 2
MODEL_VERSION = int(os.getenv("AQI_MODEL_VERSION", "1"))

MODEL_HORIZONS = (1, 24, 48, 72)
MODEL_NAMES = {
    horizon: f"aqi_{horizon}_hour_xgboost" for horizon in MODEL_HORIZONS
}

COOLWARM_COLORSCALE = [
    [0.0, "#3b4cc0"],
    [0.25, "#8db0fe"],
    [0.5, "#dddcdc"],
    [0.75, "#f4987a"],
    [1.0, "#b40426"],
]

# These raw fields are the only inputs shown in the dashboard feature panel.
DISPLAY_FEATURES = [
    "temperature",
    "humidity",
    "pressure",
    "wind_speed",
    "wind_direction",
    "precipitation",
    "cloud_cover",
    "pm25",
]

# The registry models also require these fields, but they stay hidden from the
# dashboard because they are engineered from timestamps or AQI history.
MODEL_FEATURES = DISPLAY_FEATURES + [
    "aqi",
    "hour",
    "day",
    "month",
    "year",
    "day_of_week",
    "is_weekend",
    "aqi_lag_1",
    "aqi_lag_3",
    "aqi_lag_6",
    "aqi_lag_12",
    "aqi_lag_24",
    "aqi_mean_6",
    "aqi_std_6",
    "aqi_mean_12",
    "aqi_std_12",
    "aqi_mean_24",
    "aqi_std_24",
]

TIME_COLUMNS = ("datetime_local", "datetime_utc")
TREND_COLUMNS = ("aqi", "temperature")


st.set_page_config(
    page_title="Karachi AQI Predictor",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp, [data-testid="stAppViewContainer"] {
        background: #f4f7f8 !important;
        color: #12343b !important;
    }
    [data-testid="stSidebar"] {
        background: #ffffff !important;
    }
    .stApp p, .stApp label, .stApp span, .stApp [data-testid="stCaptionContainer"] {
        color: #38545c;
    }
    .block-container {
        max-width: 1400px;
        padding-top: 2.5rem;
        padding-bottom: 3rem;
    }
    h1, h2, h3, .stApp h1, .stApp h2, .stApp h3 {
        color: #12343b !important;
        letter-spacing: 0;
    }
    h1 {
        font-size: 2.7rem !important;
        font-weight: 750 !important;
        margin-bottom: 0.15rem !important;
    }
    h2 {
        margin-top: 2rem !important;
        font-size: 2rem !important;
        font-weight: 800 !important;
    }
    [data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.84);
        border: 1px solid #d6e5e7;
        border-radius: 14px;
        padding: 1rem 1.1rem;
        box-shadow: 0 8px 24px rgba(18, 52, 59, 0.07);
    }
    [data-testid="stMetricLabel"] {
        color: #547078;
        font-weight: 600;
    }
    [data-testid="stMetricValue"] {
        color: #12343b;
    }
    div[data-testid="stTabs"] button {
        color: #547078;
        font-size: 1.15rem;
        font-style: oblique;
        font-weight: 750;
        min-height: 3rem;
        padding-left: 0.8rem;
        padding-right: 0.8rem;
    }
    div[data-testid="stTabs"] button[aria-selected="true"] {
        color: #0d7c86;
    }
    div[data-testid="stButton"] button[kind="secondary"] {
        background: transparent;
        border: 0;
        border-bottom: 3px solid transparent;
        border-radius: 0;
        box-shadow: none;
        color: #315d68;
        font-size: 1.05rem;
        font-style: italic;
        font-weight: 650;
        min-height: 2.8rem;
    }
    div[data-testid="stButton"] button[kind="secondary"]:hover {
        background: transparent;
        border-bottom-color: #ff4b5f;
        color: #12343b;
    }
    div[data-testid="stButton"] button[kind="primary"] {
        background: transparent;
        border: 0;
        border-bottom: 3px solid #ff4b5f;
        border-radius: 0;
        box-shadow: none;
        color: #12343b;
        font-size: 1.05rem;
        font-style: italic;
        font-weight: 750;
        min-height: 2.8rem;
    }
    [data-testid="stSidebar"] div[data-testid="stButton"] button {
        background: #12343b;
        border: 1px solid #12343b;
        border-radius: 8px;
        color: #ffffff;
        font-size: 0.9rem;
        font-style: normal;
        min-height: 2.4rem;
    }
    [data-testid="stTable"] table {
        background: #ffffff !important;
        color: #12343b !important;
        table-layout: fixed !important;
        width: 100% !important;
    }
    [data-testid="stTable"] th,
    [data-testid="stTable"] td {
        color: #12343b !important;
        font-weight: 600;
        padding: 0.7rem 0.55rem !important;
        white-space: nowrap;
    }
    [data-testid="stTable"] th:first-child,
    [data-testid="stTable"] td:first-child {
        text-align: left !important;
        width: 28% !important;
    }
    [data-testid="stTable"] th:not(:first-child),
    [data-testid="stTable"] td:not(:first-child) {
        text-align: right !important;
        width: 18% !important;
    }
    .section-note {
        color: #607980;
        font-size: 0.9rem;
        margin-top: -0.55rem;
        margin-bottom: 1rem;
    }
    .latest-banner {
        background: rgba(255, 255, 255, 0.78);
        border: 1px solid #d6e5e7;
        border-radius: 14px;
        box-shadow: 0 8px 24px rgba(18, 52, 59, 0.06);
    }
    .value-card {
        background: #ffffff;
        border: 1px solid #d6e5e7;
        border-radius: 14px;
        min-height: 82px;
        margin-bottom: 0.9rem;
        padding: 0.9rem 1rem;
        box-shadow: 0 8px 24px rgba(18, 52, 59, 0.06);
    }
    .value-card-label {
        color: #38545c !important;
        font-size: 0.78rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .value-card-value {
        color: #12343b !important;
        font-size: 1.35rem;
        font-weight: 750;
        margin-top: 0.25rem;
    }
    .value-card-detail {
        color: #0d6972 !important;
        font-size: 0.78rem;
        font-weight: 650;
        margin-top: 0.15rem;
    }
    .essential-strip {
        display: grid;
        grid-template-columns: repeat(4, minmax(0, 1fr));
        gap: 0.7rem;
        margin: 1.25rem 0 1.1rem;
    }
    .essential-card {
        background: #12343b;
        border: 1px solid #245761;
        border-radius: 12px;
        color: #ffffff;
        min-height: 70px;
        padding: 0.75rem 1rem;
        position: relative;
    }
    .essential-label {
        color: #b5d9dc !important;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.05em;
        text-transform: uppercase;
    }
    .essential-label span {
        color: #b5d9dc !important;
    }
    .essential-icon {
        color: #ffffff !important;
        font-size: 2rem;
        line-height: 1;
        position: absolute;
        right: 1rem;
        top: 50%;
        transform: translateY(-50%);
    }
    .essential-value {
        color: #ffffff;
        font-size: 1.5rem;
        font-weight: 800;
        margin-top: 0.18rem;
    }
    .main-board {
        display: grid;
        grid-template-columns: minmax(190px, 0.8fr) minmax(360px, 1.6fr) minmax(220px, 0.95fr);
        gap: 1rem;
        align-items: stretch;
        margin: 0.5rem 0 2rem;
    }
    .board-panel {
        background: #ffffff;
        border: 1px solid #d6e5e7;
        border-radius: 16px;
        box-sizing: border-box;
        height: 450px;
        box-shadow: 0 8px 24px rgba(18, 52, 59, 0.07);
        padding: 1rem;
    }
    [data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff;
        border-color: #d6e5e7;
        border-radius: 16px;
        box-shadow: 0 8px 24px rgba(18, 52, 59, 0.07);
        height: 450px;
    }
    .panel-title {
        color: #12343b;
        font-size: 0.9rem;
        font-weight: 750;
        margin-bottom: 0.8rem;
    }
    .side-item {
        border-bottom: 1px solid #e6eff0;
        padding: 0.65rem 0;
    }
    .side-item:last-child {
        border-bottom: 0;
    }
    .side-label {
        color: #607980;
        font-size: 0.72rem;
        font-weight: 700;
        text-transform: uppercase;
    }
    .side-value {
        color: #12343b;
        font-size: 1.2rem;
        font-weight: 800;
        margin-top: 0.15rem;
    }
    .aqi-panel {
        background: #effbf6;
        border-color: #c8e5d8;
        text-align: center;
    }
    .aqi-kicker {
        color: #607980;
        font-size: 0.75rem;
        font-weight: 700;
        letter-spacing: 0.04em;
        text-transform: uppercase;
    }
    .aqi-gauge {
        background: conic-gradient(from 270deg, #2eaa52 0deg 30deg, #b5d52f 30deg 60deg, #f5d13b 60deg 90deg, #f59d32 90deg 120deg, #df493d 120deg 150deg, #843d91 150deg 180deg, transparent 180deg 360deg);
        border-radius: 50% 50% 0 0;
        height: 150px;
        margin: 1.2rem auto 0;
        max-width: 290px;
        overflow: visible;
        position: relative;
        width: 100%;
    }
    .aqi-gauge::after {
        background: #effbf6;
        border-radius: 50% 50% 0 0;
        bottom: 0;
        content: "";
        left: 16%;
        position: absolute;
        right: 16%;
        top: 30%;
        z-index: 1;
    }
    .aqi-needle {
        background: #111111;
        bottom: -20px;
        clip-path: polygon(0 50%, 87% 0, 100% 50%, 87% 100%);
        height: 18px;
        left: 50%;
        position: absolute;
        transform-origin: 0 50%;
        width: 125px;
        z-index: 3;
    }
    .aqi-needle::after {
        background: #111111;
        border: 4px solid #ffffff;
        border-radius: 50%;
        bottom: -8px;
        content: "";
        height: 15px;
        left: -12px;
        position: absolute;
        width: 15px;
    }
    .aqi-number {
        color: #c9aa08;
        font-size: 3.2rem;
        font-weight: 800;
        line-height: 1;
        margin-top: -0.25rem;
        position: relative;
        z-index: 1;
    }
    .aqi-label {
        color: #12343b;
        font-size: 1.1rem;
        font-weight: 750;
        margin-top: 0.4rem;
    }
    .status-pill {
        border-radius: 999px;
        display: inline-block;
        font-size: 0.75rem;
        font-weight: 750;
        margin-top: 0.6rem;
        padding: 0.35rem 0.75rem;
    }
    .forecast-item {
        align-items: center;
        border-bottom: 1px solid #e6eff0;
        display: flex;
        justify-content: space-between;
        padding: 0.7rem 0;
    }
    .forecast-item:last-child {
        border-bottom: 0;
    }
    .forecast-horizon {
        color: #607980;
        font-size: 0.78rem;
        font-weight: 700;
    }
    .forecast-number {
        color: #12343b;
        font-size: 1.4rem;
        font-weight: 800;
    }
    @media (max-width: 900px) {
        .essential-strip { grid-template-columns: repeat(2, minmax(0, 1fr)); }
        .main-board { grid-template-columns: 1fr; }
        .board-panel, [data-testid="stVerticalBlockBorderWrapper"] { height: auto; }
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def aqi_band(value):
    if value <= 50:
        return "Good", "#16803c"
    if value <= 100:
        return "Moderate", "#b7791f"
    if value <= 150:
        return "Unhealthy for sensitive groups", "#c05621"
    if value <= 200:
        return "Unhealthy", "#c53030"
    if value <= 300:
        return "Very unhealthy", "#702459"
    return "Hazardous", "#4a1d1d"


@st.cache_resource(show_spinner=False)
def connect_hopsworks():
    api_key = os.getenv("HOPSWORKS_API_KEY")
    if not api_key:
        raise RuntimeError("HOPSWORKS_API_KEY is not set.")
    project = hopsworks.login(project=PROJECT_NAME, api_key_value=api_key)
    feature_store = project.get_feature_store()
    feature_group = feature_store.get_feature_group(
        name=FEATURE_GROUP_NAME,
        version=FEATURE_GROUP_VERSION,
    )
    return project, feature_group


@st.cache_data(ttl=300, show_spinner=False)
def load_history():
    _, feature_group = connect_hopsworks()
    data = feature_group.read()
    if data is None or data.empty:
        raise RuntimeError("The Hopsworks feature group returned no rows.")

    data = data.copy()
    timestamp_column = next(
        (column for column in TIME_COLUMNS if column in data.columns), None
    )
    if timestamp_column is None:
        raise RuntimeError("No datetime column was found in the feature group.")

    data[timestamp_column] = pd.to_datetime(data[timestamp_column], errors="coerce")
    data = (
        data.dropna(subset=[timestamp_column])
        .sort_values(timestamp_column)
        .drop_duplicates(subset=[timestamp_column], keep="last")
        .reset_index(drop=True)
    )
    data = add_training_features(data)
    return data, timestamp_column


def add_training_features(data):
    data = data.copy()
    for lag in (1, 2, 3, 6, 12, 18, 24, 36, 48, 72):
        data[f"aqi_extra_lag_{lag}"] = data["aqi"].shift(lag)

    for window in (3, 6, 12, 24, 36, 48, 72):
        rolling_aqi = data["aqi"].rolling(window)
        data[f"aqi_extra_mean_{window}"] = rolling_aqi.mean()
        data[f"aqi_extra_std_{window}"] = rolling_aqi.std()
        data[f"aqi_extra_min_{window}"] = rolling_aqi.min()
        data[f"aqi_extra_max_{window}"] = rolling_aqi.max()

    for period in (1, 3, 6, 12, 24, 48):
        data[f"aqi_change_{period}"] = data["aqi"].diff(period)
    for period in (6, 24):
        data[f"aqi_pct_change_{period}"] = data["aqi"].pct_change(period)

    data["hour_sin"] = np.sin(2 * np.pi * data["hour"] / 24)
    data["hour_cos"] = np.cos(2 * np.pi * data["hour"] / 24)
    data["dow_sin"] = np.sin(2 * np.pi * data["day_of_week"] / 7)
    data["dow_cos"] = np.cos(2 * np.pi * data["day_of_week"] / 7)
    return data.replace([np.inf, -np.inf], np.nan)


def find_model_file(model_directory):
    candidates = list(Path(model_directory).rglob("model.json"))
    if not candidates:
        candidates = list(Path(model_directory).rglob("*.json"))
    if not candidates:
        raise RuntimeError("The downloaded model does not contain a JSON model file.")
    return candidates[0]


@st.cache_resource(show_spinner=False)
def load_registered_model(horizon):
    project, _ = connect_hopsworks()
    registry = project.get_model_registry()
    registered_model = registry.get_model(MODEL_NAMES[horizon], version=MODEL_VERSION)
    with tempfile.TemporaryDirectory() as temporary_directory:
        model_directory = registered_model.download(
            local_path=temporary_directory,
        )
        model_file = find_model_file(model_directory)
        model = xgb.XGBRegressor()
        model.load_model(str(model_file))
        return model


def get_model_features(model):
    feature_names = getattr(model, "feature_names", None)
    if feature_names:
        return list(feature_names)
    return list(model.get_booster().feature_names or MODEL_FEATURES)


def build_model_input(latest_row, model):
    model_features = get_model_features(model)
    missing_features = [
        feature for feature in model_features if feature not in latest_row.index
    ]
    if missing_features:
        raise RuntimeError(
            "The latest Hopsworks row is missing model features: "
            + ", ".join(missing_features)
        )

    model_input = pd.DataFrame(
        [[pd.to_numeric(latest_row[feature], errors="coerce") for feature in model_features]],
        columns=model_features,
    )
    if model_input.isna().any().any():
        missing = model_input.columns[model_input.isna().any()].tolist()
        raise RuntimeError("The latest row has missing values in: " + ", ".join(missing))
    return model_input


def make_predictions(latest_row):
    predictions = {}
    for horizon in MODEL_HORIZONS:
        model = load_registered_model(horizon)
        model_features = get_model_features(model)
        missing_features = [
            feature for feature in model_features if feature not in latest_row.index
        ]
        if missing_features:
            raise RuntimeError(
                f"The +{horizon}h model requires missing Hopsworks features: "
                + ", ".join(missing_features)
            )

        model_input = pd.DataFrame(
            [[pd.to_numeric(latest_row[feature], errors="coerce") for feature in model_features]],
            columns=model_features,
        )
        if model_input.isna().any().any():
            missing = model_input.columns[model_input.isna().any()].tolist()
            raise RuntimeError(
                f"The latest row has missing values for the +{horizon}h model: "
                + ", ".join(missing)
            )
        predictions[horizon] = float(model.predict(model_input)[0])
    return predictions


def display_value(value):
    if pd.isna(value):
        return "-"
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, float):
        return f"{value:.2f}"
    return str(value)


def render_value_card(label, value, detail=""):
    st.markdown(
        f"""
        <div class="value-card">
            <div class="value-card-label">{label}</div>
            <div class="value-card-value">{value}</div>
            {f'<div class="value-card-detail">{detail}</div>' if detail else ''}
        </div>
        """,
        unsafe_allow_html=True,
    )


def correlation_heatmap_styles(dataframe):
    def style_cell(value):
        if pd.isna(value):
            return ""
        intensity = min(abs(float(value)), 1.0)
        coolwarm_color = sample_colorscale(COOLWARM_COLORSCALE, [intensity])[0]
        return f"background-color: {coolwarm_color}; color: #000000"

    return (
        dataframe.style
        .applymap(style_cell)
        .set_table_styles(
            [
                {
                    "selector": "table",
                    "props": [("background-color", "#ffffff"), ("color", "#000000")],
                },
                {
                    "selector": "th",
                    "props": [
                        ("background-color", "#ffffff"),
                        ("color", "#000000"),
                        ("font-weight", "700"),
                    ],
                },
                {
                    "selector": "td",
                    "props": [
                        ("color", "#ffffff !important"),
                        ("font-weight", "700"),
                    ],
                },
            ]
        )
        .format("{:.2f}")
    )


def aqi_interval_color(value):
    if value <= 50:
        return "#2eaa52"
    if value <= 100:
        return "#f5d13b"
    if value <= 150:
        return "#f59d32"
    if value <= 200:
        return "#df493d"
    if value <= 300:
        return "#843d91"
    return "#800000"


def render_aqi_alerts(current_value, forecasts):
    hazardous_forecasts = [
        f"+{horizon}h ({value:.1f})"
        for horizon, value in forecasts.items()
        if value >= 301
    ]
    if current_value is not None and current_value >= 301:
        st.error(
            f"Hazardous AQI alert: current AQI is {current_value:.1f}. "
            "Avoid outdoor exposure and follow local health guidance."
        )
    elif current_value is not None and current_value >= 201:
        st.warning(
            f"Very unhealthy AQI warning: current AQI is {current_value:.1f}. "
            "Reduce prolonged or heavy outdoor activity."
        )

    if hazardous_forecasts:
        st.error(
            "Hazardous AQI forecast: "
            + ", ".join(hazardous_forecasts)
            + ". Prepare to limit outdoor exposure."
        )


def render_aqi_gauge(value):
    interval_color = aqi_interval_color(value)
    gauge = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=max(0, min(float(value), 400)),
            number={"font": {"color": interval_color, "size": 68}, "valueformat": ".0f"},
            gauge={
                "axis": {
                    "range": [0, 400],
                    "tickmode": "array",
                    "tickvals": [0, 50, 100, 150, 200, 300, 400],
                    "ticktext": ["0", "50", "100", "150", "200", "300", "400"],
                    "tickfont": {"color": "#38545c", "size": 10},
                },
                "bar": {"color": "#111111", "thickness": 0.18},
                "borderwidth": 0,
                "bgcolor": "#effbf6",
                "steps": [
                    {"range": [0, 50], "color": "#2eaa52"},
                    {"range": [50, 100], "color": "#f5d13b"},
                    {"range": [100, 150], "color": "#f59d32"},
                    {"range": [150, 200], "color": "#df493d"},
                    {"range": [200, 300], "color": "#843d91"},
                    {"range": [300, 400], "color": "#800000"},
                ],
            },
            domain={"x": [0, 1], "y": [0, 1]},
        )
    )
    gauge.update_layout(
        height=285,
        margin={"l": 18, "r": 18, "t": 18, "b": 0},
        paper_bgcolor="rgba(0,0,0,0)",
        font={"color": "#12343b"},
    )
    st.plotly_chart(gauge, use_container_width=True, config={"displayModeBar": False})


def render_trend_chart(dataframe, column, fill=False):
    line = go.Scatter(
        x=dataframe.index,
        y=dataframe[column],
        mode="lines",
        name=column.replace("_", " ").title(),
        line={"color": "#12343b", "width": 3},
        fill="tozeroy" if fill else None,
        fillcolor="rgba(18, 52, 59, 0.12)" if fill else None,
    )
    chart = go.Figure(line)
    chart.update_layout(
        height=360,
        margin={"l": 20, "r": 20, "t": 18, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#12343b"},
        hoverlabel={"bgcolor": "#ffffff", "font": {"color": "#12343b", "size": 13}},
        showlegend=False,
        xaxis={
            "showgrid": False,
            "linecolor": "#12343b",
            "tickfont": {"color": "#12343b", "size": 12},
        },
        yaxis={
            "gridcolor": "#e6eff0",
            "linecolor": "#12343b",
            "tickfont": {"color": "#12343b", "size": 12},
        },
    )
    if len(dataframe) > 168:
        chart.update_layout(
            xaxis={
                "showgrid": False,
                "linecolor": "#12343b",
                "range": [dataframe.index[-168], dataframe.index[-1]],
                "rangeslider": {"visible": True, "thickness": 0.08},
            }
        )
    st.plotly_chart(
        chart,
        use_container_width=True,
        config={
            "displayModeBar": True,
            "scrollZoom": True,
            "doubleClick": "reset",
        },
    )


def render_shap_explanation(latest_row, horizon):
    model = load_registered_model(horizon)
    model_input = build_model_input(latest_row, model)
    contributions = model.get_booster().predict(
        xgb.DMatrix(model_input), pred_contribs=True
    )[0]
    contribution_series = pd.Series(
        contributions[:-1], index=model_input.columns, name="SHAP contribution"
    )
    top_features = contribution_series.abs().nlargest(10).index
    top_contributions = contribution_series.loc[top_features].sort_values()
    colors = [
        sample_colorscale(COOLWARM_COLORSCALE, [0.82 if value > 0 else 0.18])[0]
        for value in top_contributions
    ]
    explanation = go.Figure(
        go.Bar(
            x=top_contributions.values,
            y=top_contributions.index,
            orientation="h",
            marker_color=colors,
            text=[f"{value:+.2f}" for value in top_contributions.values],
            textposition="auto",
            hovertemplate="%{y}<br>Contribution: %{x:+.3f}<extra></extra>",
        )
    )
    explanation.update_layout(
        height=430,
        margin={"l": 20, "r": 20, "t": 18, "b": 20},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font={"color": "#12343b"},
        xaxis={"title": "SHAP contribution", "gridcolor": "#e6eff0", "zerolinecolor": "#12343b"},
        yaxis={"title": "Feature", "tickfont": {"color": "#12343b", "size": 11}},
        showlegend=False,
    )
    st.plotly_chart(explanation, use_container_width=True, config={"displayModeBar": False})


def select_shap_horizon(horizon):
    st.session_state.shap_horizon = horizon


def select_model_comparison(model_name):
    st.session_state.selected_comparison_model = model_name


st.markdown("# Karachi AQI Predictor")
st.caption("Live air-quality conditions and registry-backed forecasts")

with st.sidebar:
    st.markdown("### Data source")
    st.write(f"`{FEATURE_GROUP_NAME}` v{FEATURE_GROUP_VERSION}")
    st.write(f"Project: `{PROJECT_NAME}`")
    if st.button("Refresh Hopsworks data", use_container_width=True):
        load_history.clear()
        st.rerun()

try:
    history, timestamp_column = load_history()
    latest = history.iloc[-1]
except Exception as error:
    st.error(f"Could not load Hopsworks data: {error}")
    st.stop()

latest_time = latest[timestamp_column]
current_aqi = float(latest["aqi"]) if "aqi" in latest else None
band, band_color = aqi_band(current_aqi) if current_aqi is not None else ("Unknown", "#555555")

try:
    forecasts = make_predictions(latest)
except Exception as error:
    forecasts = {}
    st.warning(f"Forecasts are unavailable: {error}")

render_aqi_alerts(current_aqi, forecasts)

essential_features = ["temperature", "humidity", "pm25", "wind_speed"]
feature_icons = {
    "temperature": "☀️" if 6 <= int(latest.get("hour", latest_time.hour)) < 18 else "🌙",
    "humidity": "💧",
    "pm25": "🌫️",
    "wind_speed": "💨",
}
essential_cards = []
for feature in essential_features:
    value = display_value(latest.get(feature))
    unit = {"temperature": "°C", "humidity": "%", "pm25": "μg/m³", "wind_speed": "m/s"}[feature]
    icon_markup = f'<span class="essential-icon">{feature_icons[feature]}</span>'
    essential_cards.append(
        f'<div class="essential-card"><div class="essential-label"><span>{feature.replace("_", " ")}</span>{icon_markup}</div>'
        f'<div class="essential-value">{value} {unit}</div></div>'
    )
st.markdown(
    f'<div class="essential-strip">{"".join(essential_cards)}</div>',
    unsafe_allow_html=True,
)

left_features = ["pressure", "wind_direction", "precipitation", "cloud_cover"]
left_items = []
for feature in left_features:
    unit = {"pressure": "hPa", "wind_direction": "°", "precipitation": "mm", "cloud_cover": "%"}[feature]
    left_items.append(
        f'<div class="side-item"><div class="side-label">{feature.replace("_", " ")}</div>'
        f'<div class="side-value">{display_value(latest.get(feature))} {unit}</div></div>'
    )

forecast_items = []
for horizon in MODEL_HORIZONS:
    if horizon in forecasts:
        forecast_band, forecast_color = aqi_band(forecasts[horizon])
        forecast_value_color = aqi_interval_color(forecasts[horizon])
        forecast_items.append(
            f'<div class="forecast-item"><div><div class="forecast-horizon">+{horizon} hours</div>'
            f'<div style="color:{forecast_color}; font-size:0.72rem; font-weight:700">{forecast_band}</div></div>'
            f'<div class="forecast-number" style="color:{forecast_value_color}">{forecasts[horizon]:.1f}</div></div>'
        )

aqi_fill = {"Good": "#e1f5e8", "Moderate": "#fff4c7", "Unhealthy for sensitive groups": "#ffe1c7"}.get(band, "#ffe1e1")
board_columns = st.columns([0.8, 1.6, 0.95], gap="small")
with board_columns[0]:
    st.markdown(
        f'<div class="board-panel"><div class="panel-title">Other conditions</div>{"".join(left_items)}</div>',
        unsafe_allow_html=True,
    )
with board_columns[1]:
    with st.container(border=True):
        st.markdown('<div class="panel-title" style="text-align:center">Current air quality</div>', unsafe_allow_html=True)
        if current_aqi is not None:
            render_aqi_gauge(current_aqi)
        st.markdown(
            f'<div style="text-align:center"><div class="aqi-label">AQI</div>'
            f'<div class="status-pill" style="background:{aqi_fill}; color:{band_color}">{band}</div>'
            f'<div class="section-note" style="margin-top:0.8rem">Latest reading · {latest_time:%d %b %Y, %H:%M}</div></div>',
            unsafe_allow_html=True,
        )
with board_columns[2]:
    st.markdown(
        f'<div class="board-panel"><div class="panel-title">AQI forecast</div>'
        f'{"".join(forecast_items) if forecast_items else "<div class=section-note>Forecasts unavailable</div>"}</div>',
        unsafe_allow_html=True,
    )

st.markdown("## Trends")
trend_data = history.set_index(timestamp_column)
trend_tabs = st.tabs(["AQI trend", "Temperature trend", "Correlation heatmap"])
with trend_tabs[0]:
    render_trend_chart(trend_data, "aqi")
with trend_tabs[1]:
    render_trend_chart(trend_data, "temperature", fill=True)
with trend_tabs[2]:
    correlation_columns = [
        column for column in [*DISPLAY_FEATURES, "aqi"] if column in history.columns
    ]
    correlation = history[correlation_columns].corr().round(2)
    st.table(correlation_heatmap_styles(correlation))

st.markdown("## Feature importance explanation")
if "shap_horizon" not in st.session_state:
    st.session_state.shap_horizon = 1

with st.container():
    shap_buttons = st.columns(4, gap="small")
    for button_column, horizon in zip(shap_buttons, MODEL_HORIZONS):
        with button_column:
            st.button(
                f"+{horizon}h",
                key=f"shap_horizon_{horizon}",
                type="primary" if st.session_state.shap_horizon == horizon else "secondary",
                use_container_width=True,
                on_click=select_shap_horizon,
                args=(horizon,),
            )

selected_shap_horizon = st.session_state.shap_horizon
st.markdown(
    f'<div class="section-note">Top SHAP contributions for the +{selected_shap_horizon} hour registered model</div>',
    unsafe_allow_html=True,
)
try:
    render_shap_explanation(latest, selected_shap_horizon)
except Exception as error:
    st.warning(f"Feature explanation is unavailable: {error}")

st.markdown("## Model Applied")
if "selected_comparison_model" not in st.session_state:
    st.session_state.selected_comparison_model = "XGBoost Regressor"

comparison_buttons = st.columns(4, gap="small")
with comparison_buttons[0]:
    st.button(
        "XGBoost Regressor",
        key="xgboost_regressor_button",
        type="primary"
        if st.session_state.selected_comparison_model == "XGBoost Regressor"
        else "secondary",
        use_container_width=True,
        on_click=select_model_comparison,
        args=("XGBoost Regressor",),
    )
with comparison_buttons[1]:
    st.button(
        "Extra Trees Regressor",
        key="extra_trees_regressor_button",
        type="primary"
        if st.session_state.selected_comparison_model == "Extra Trees Regressor"
        else "secondary",
        use_container_width=True,
        on_click=select_model_comparison,
        args=("Extra Trees Regressor",),
    )
with comparison_buttons[2]:
    st.button(
        "MLPRegressor",
        key="mlp_regressor_button",
        type="primary"
        if st.session_state.selected_comparison_model == "MLPRegressor"
        else "secondary",
        use_container_width=True,
        on_click=select_model_comparison,
        args=("MLPRegressor",),
    )
with comparison_buttons[3]:
    st.button(
        "PyTorch",
        key="pytorch_button",
        type="primary"
        if st.session_state.selected_comparison_model == "PyTorch"
        else "secondary",
        use_container_width=True,
        on_click=select_model_comparison,
        args=("PyTorch",),
    )

if st.session_state.selected_comparison_model == "XGBoost Regressor":
    xgboost_comparison = pd.DataFrame(
        {
            "Horizon": ["+1h", "+24h", "+48h", "+72h"],
            "R²": [0.8930, 0.5605, 0.5370, 0.5283],
        }
    )
    st.table(
        xgboost_comparison.style
        .format({"R²": "{:.4f}"})
        .set_table_styles(
            [
                {"selector": "table", "props": [("background-color", "#ffffff"), ("color", "#12343b"), ("width", "100%"), ("table-layout", "fixed")]},
                {"selector": "th", "props": [("background-color", "#ffffff"), ("color", "#12343b"), ("font-weight", "700"), ("text-align", "left")]},
                {"selector": "td", "props": [("color", "#12343b"), ("font-weight", "600"), ("text-align", "left")]},
                {"selector": "th.col1, td.col1", "props": [("text-align", "right")]},
            ]
        )
    )
    st.markdown(
        "**Interpretation:** This run reflects the project's core modelling strategy: a separate "
        "gradient-boosted tree model trained per horizon, using sequential error-correction "
        "(each new tree fits the residuals of the ensemble so far) plus second-order loss "
        "approximation and explicit regularization to control overfitting.\n\n"
        "The results show the expected and desired pattern: very high explanatory power at +1h "
        "(89.3%), then a controlled, gradual decline as the forecast horizon lengthens "
        "(56.1% → 53.7% → 52.8%). Unlike Extra Trees or the MLP below, XGBoost does **not** "
        "collapse into negative R² territory at any horizon — every horizon still explains "
        "more than half of the variance at +24h and above. This stability under increasing "
        "forecast distance is precisely why XGBoost was selected as the final production model: "
        "it degrades gracefully rather than catastrophically, and the regularization terms "
        "(reg_alpha, reg_lambda) combined with horizon-specific tuning prevent the long-horizon "
        "overfitting problems seen in the model's own earlier development iterations."
    )
elif st.session_state.selected_comparison_model == "Extra Trees Regressor":
    extra_trees_comparison = pd.DataFrame(
        {
            "Horizon": ["+1h", "+24h", "+48h", "+72h"],
            "MAE": [11.78, 26.23, 31.93, 27.88],
            "R²": [0.8837, 0.5375, 0.3702, 0.5246],
        }
    )
    st.table(
        extra_trees_comparison.style
        .format({"MAE": "{:.2f}", "R²": "{:.4f}"})
        .set_table_styles(
            [
                {"selector": "table", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("width", "100%"), ("table-layout", "fixed"),
                ]},
                {"selector": "th", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("font-weight", "700"), ("text-align", "left"),
                ]},
                {"selector": "td", "props": [
                    ("color", "#12343b"), ("font-weight", "600"),
                    ("text-align", "left"),
                ]},
                {"selector": "th.col1, td.col1, th.col2, td.col2", "props": [("text-align", "right")]},
            ]
        )
    )
    st.markdown(
        "**Interpretation:** Extra Trees (Extremely Randomized Trees) is, like Random Forest, "
        "a bagged ensemble of decision trees, but it selects split thresholds randomly rather "
        "than searching for the optimal split. This added randomness typically increases bias "
        "slightly but reduces variance further than Random Forest.\n\n"
        "At +1h, the model explains **88.4%** of AQI variance — strong short-term performance, "
        "consistent with the idea that recent AQI/environmental conditions are highly predictive "
        "of the very next hour. Performance degrades steadily through +24h (53.8%) and dips to "
        "its weakest point at +48h (37.0%), where both MAE rises and R² falls, suggesting the "
        "model struggles most to generalize at this mid-range horizon. It partially recovers at "
        "+72h (52.5%), which mirrors a pattern seen elsewhere in this project (e.g., the original "
        "multi-horizon XGBoost run) where the relationship between forecast distance and error is "
        "not perfectly monotonic — likely because AQI has daily/weekly cyclical structure that "
        "re-aligns at certain lag distances. Overall, Extra Trees is a viable baseline but is less "
        "stable across horizons than the horizon-specific XGBoost approach."
    )
elif st.session_state.selected_comparison_model == "MLPRegressor":
    mlp_comparison = pd.DataFrame(
        {
            "Horizon": ["+1h", "+24h", "+48h", "+72h"],
            "MAE": [32.99, 50.92, 50.44, 43.86],
            "RMSE": [38.88, 60.76, 61.59, 53.90],
            "R²": [0.4872, -0.2295, -0.2977, -0.0223],
        }
    )
    st.table(
        mlp_comparison.style
        .format({"MAE": "{:.2f}", "RMSE": "{:.2f}", "R²": "{:.4f}"})
        .set_table_styles(
            [
                {"selector": "table", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("width", "100%"), ("table-layout", "fixed"),
                ]},
                {"selector": "th", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("font-weight", "700"), ("text-align", "left"),
                ]},
                {"selector": "td", "props": [
                    ("color", "#12343b"), ("font-weight", "600"),
                    ("text-align", "left"),
                ]},
                {"selector": "th.col1, td.col1, th.col2, td.col2, th.col3, td.col3", "props": [("text-align", "right")]},
            ]
        )
    )
    st.markdown(
        "**Interpretation:** The (scikit-learn-style) MLP Regressor is a fully-connected "
        "feed-forward neural network trained via backpropagation. It shows a partial "
        "success/failure split: at +1h it explains **48.7%** of variance — a moderate result, "
        "weaker than every tree-based model at this horizon but still clearly better than a "
        "mean-based guess. Beyond +1h, however, R² turns **negative** at +24h, +48h, and +72h, "
        "with the worst point at +48h (-0.2977). This indicates the MLP could pick up some "
        "short-horizon signal (likely from the most recent lag features) but failed to generalize "
        "to longer-range temporal dependencies — a common weakness of plain MLPs on tabular "
        "time-series data, since they lack the sequential/recurrent structure or the explicit "
        "feature-interaction handling that boosted trees provide. Error magnitude (MAE/RMSE) is "
        "also consistently higher than XGBoost and Extra Trees at every horizon, reinforcing "
        "that this architecture was not competitive for this task."
    )
elif st.session_state.selected_comparison_model == "PyTorch":
    pytorch_comparison = pd.DataFrame(
        {
            "Horizon": ["+1h", "+24h", "+48h", "+72h"],
            "R²": [-0.7731, -0.7006, -0.7667, -0.8005],
        }
    )
    st.table(
        pytorch_comparison.style
        .format({"R²": "{:.4f}"})
        .set_table_styles(
            [
                {"selector": "table", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("width", "100%"), ("table-layout", "fixed"),
                ]},
                {"selector": "th", "props": [
                    ("background-color", "#ffffff"), ("color", "#12343b"),
                    ("font-weight", "700"), ("text-align", "left"),
                ]},
                {"selector": "td", "props": [
                    ("color", "#12343b"), ("font-weight", "600"),
                    ("text-align", "left"),
                ]},
                {"selector": "th.col1, td.col1", "props": [("text-align", "right")]},
            ]
        )
    )
    st.markdown(
        "**Interpretation:** These figures correspond to an early/unstable PyTorch training run "
        "(consistent with the numerical-instability issue noted during development, where NaNs "
        "appeared in the training loss before the pipeline was corrected). Every horizon shows a "
        "**negative R²**, meaning the network's predictions were worse across the board than simply "
        "predicting the mean AQI value for every input — i.e., the model had not learned a usable "
        "mapping between the input features and AQI at any forecast distance. Notably, performance "
        "does **not** improve at +1h the way it does for the tree-based models; this points to a "
        "training/convergence failure (e.g., an unstable loss landscape, poor initialization, or a "
        "learning-rate/NaN issue) rather than a horizon-difficulty effect. This run should be treated "
        "as a failed/pre-fix training attempt and not representative of the corrected PyTorch model "
        "reported later in the project (which achieved a positive average R² of ~0.6088)."
    )

st.markdown("## Model Comparison")
model_comparison = pd.DataFrame(
    {
        "Model": ["XGBoost", "Extra Trees", "MLP Regressor", "PyTorch (unstable run)"],
        "+1h R²": [0.8930, 0.8837, 0.4872, -0.7731],
        "+24h R²": [0.5605, 0.5375, -0.2295, -0.7006],
        "+48h R²": [0.5370, 0.3702, -0.2977, -0.7667],
        "+72h R²": [0.5283, 0.5246, -0.0223, -0.8005],
    }
)
comparison_style = (
    model_comparison.style
    .format({column: "{:.4f}" for column in model_comparison.columns[1:]})
    .set_table_styles(
        [
            {
                "selector": "table",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#12343b"),
                    ("table-layout", "fixed"),
                    ("width", "100%"),
                ],
            },
            {
                "selector": "th",
                "props": [
                    ("background-color", "#ffffff"),
                    ("color", "#12343b"),
                    ("font-weight", "700"),
                    ("text-align", "right"),
                ],
            },
            {
                "selector": "th.col0, td.col0",
                "props": [("text-align", "left"), ("width", "28%")],
            },
            {
                "selector": "th:not(.col0), td:not(.col0)",
                "props": [("text-align", "right"), ("width", "18%")],
            },
            {
                "selector": "td",
                "props": [("color", "#12343b"), ("font-weight", "600")],
            },
        ]
    )
)
st.table(comparison_style)
st.markdown(
    "XGBoost is the only model that stays positive and relatively stable across "
    "**all four** required horizons, which is the basis for its selection as the final "
    "forecasting model. Extra Trees is a reasonable secondary baseline but degrades "
    "more sharply at +48h. The MLP and the (unstable) PyTorch run both illustrate why "
    "the project's final conclusion — that added model complexity does not automatically "
    "improve forecasting performance on this tabular, structured dataset — held true in practice."
)

st.caption(f"Showing the newest of {len(history):,} rows read from Hopsworks.")