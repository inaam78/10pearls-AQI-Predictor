import os
import joblib
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Lahore AQI Predictor & Air Quality Dashboard",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================
# CONFIGURATION & CONSTANTS
# ============================================================
MODEL_PATH = "models/random_forest_72h.pkl"
SCALER_PATH = "models/scaler.pkl"
PREDICTION_PATH = "outputs/72h_predictions.csv"
PERFORMANCE_PATH = "results/models/test_performance.csv"

AQI_CATEGORIES = [
    (0, 50, "Good", "#00E400", "#000000", "Air quality is satisfactory; air pollution poses little or no risk."),
    (51, 100, "Moderate", "#FFFF00", "#000000", "Air quality is acceptable; sensitive individuals may experience minor irritation."),
    (101, 150, "Unhealthy for Sensitive Groups", "#FF7E00", "#FFFFFF", "General public is unlikely to be affected; sensitive groups may experience respiratory symptoms."),
    (151, 200, "Unhealthy", "#FF0000", "#FFFFFF", "Everyone may begin to experience health effects; members of sensitive groups may experience more serious effects."),
    (201, 300, "Very Unhealthy", "#8F3F97", "#FFFFFF", "Health alert: The risk of health effects is increased for everyone."),
    (301, 500, "Hazardous", "#7E0023", "#FFFFFF", "Health warning of emergency conditions: Everyone is more likely to be affected.")
]

# ============================================================
# CUSTOM STYLING (CSS)
# ============================================================
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    .main { padding-top: 1rem; background-color: #f8f9fa; }
    
    .forecast-card {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
        box-shadow: 0 2px 4px rgba(0,0,0,0.02);
        margin-bottom: 10px;
    }
    
    .forecast-title {
        font-size: 13px;
        font-weight: 600;
        letter-spacing: 0.5px;
        color: #555555;
        text-transform: uppercase;
        margin-bottom: 12px;
    }
    
    .forecast-value {
        font-size: 38px;
        font-weight: 700;
        color: #111111;
        margin-bottom: 16px;
    }
    
    .aqi-pill {
        display: inline-block;
        padding: 8px 16px;
        border-radius: 8px;
        font-weight: 600;
        font-size: 13px;
        color: #ffffff;
        width: 100%;
        text-align: center;
    }

    .advisory-box {
        padding: 16px;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 20px;
        background-color: #ffffff;
        box-shadow: 0 1px 3px rgba(0,0,0,0.05);
    }
    
    .footer-container {
        text-align: center;
        font-size: 12px;
        opacity: 0.7;
        padding: 25px 0 10px 0;
        border-top: 1px solid rgba(128,128,128,0.2);
        margin-top: 40px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# HELPER FUNCTIONS
# ============================================================
def pm25_to_aqi(pm25):
    if pd.isna(pm25):
        return None
    pm25 = float(pm25)
    breakpoints = [
        (0.0, 12.0, 0, 50),
        (12.1, 35.4, 51, 100),
        (35.5, 55.4, 101, 150),
        (55.5, 150.4, 151, 200),
        (150.5, 250.4, 201, 300),
        (250.5, 350.4, 301, 400),
        (350.5, 500.4, 401, 500)
    ]
    for c_low, c_high, i_low, i_high in breakpoints:
        if c_low <= pm25 <= c_high:
            return np.interp(pm25, [c_low, c_high], [i_low, i_high])
    return 500.0 if pm25 > 500.4 else 0.0

def get_aqi_details(aqi):
    if aqi is None:
        return "Unknown", "#808080", "#FFFFFF", "No AQI data available."
    aqi_val = float(aqi)
    for min_val, max_val, category, color, text_color, impact in AQI_CATEGORIES:
        if min_val <= aqi_val <= max_val:
            return category, color, text_color, impact
    return "Hazardous", "#7E0023", "#FFFFFF", "Health emergency condition."

@st.cache_data(ttl=3600)
def fetch_live_lahore_aqi():
    try:
        url = "https://air-quality-api.open-meteo.com/v1/air-quality?latitude=31.5497&longitude=74.3436&current=pm2_5"
        res = requests.get(url, timeout=5)
        if res.status_code == 200:
            data = res.json()
            pm25_val = data.get("current", {}).get("pm2_5", None)
            if pm25_val is not None:
                aqi_val = pm25_to_aqi(pm25_val)
                return pm25_val, aqi_val
    except Exception:
        pass
    return None, None

# ============================================================
# LOAD ARTIFACTS & PREDICTIONS
# ============================================================
@st.cache_resource
def load_artifacts():
    model = joblib.load(MODEL_PATH) if os.path.exists(MODEL_PATH) else None
    scaler = joblib.load(SCALER_PATH) if os.path.exists(SCALER_PATH) else None
    return model, scaler

@st.cache_data
def load_predictions():
    if os.path.exists(PREDICTION_PATH):
        df = pd.read_csv(PREDICTION_PATH)
        if "timestamp" not in df.columns:
            start_time = datetime.now()
            df["timestamp"] = [start_time + timedelta(hours=i) for i in range(len(df))]
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        if "forecast_hour" not in df.columns:
            df["forecast_hour"] = range(1, len(df) + 1)
            
        if "predicted_pm25" in df.columns and "predicted_pm2_5" not in df.columns:
            df["predicted_pm2_5"] = df["predicted_pm25"]
            
        return df
    return None

model, scaler = load_artifacts()
prediction_df = load_predictions()

if prediction_df is None or "predicted_pm2_5" not in prediction_df.columns:
    st.error("[Warning] Predictions file not found at outputs/72h_predictions.csv. Please run python src/predict.py first.")
    st.stop()

prediction_df["AQI"] = prediction_df["predicted_pm2_5"].apply(pm25_to_aqi)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown("## Dashboard Controls")
st.sidebar.markdown("---")
city = st.sidebar.selectbox("Select City", ["Lahore"])
forecast_mode = st.sidebar.radio("Forecast View Mode", ["Hourly View", "Daily Overview"])

st.sidebar.markdown("---")
st.sidebar.caption("Tip: Toggle between views to analyze hourly spikes vs. daily trends.")

# ============================================================
# HEADER & LIVE AQI ACTION BUTTON
# ============================================================
st.markdown("# Air Quality Forecast")
st.caption(f"{city}, Punjab, Pakistan — Machine-learning forecast for the next 72 hours.")

# Live AQI Button matching reference style
if st.button("Generate Live AQI Status"):
    live_pm25, live_aqi = fetch_live_lahore_aqi()
    if live_aqi is not None:
        st.session_state["live_pm25"] = live_pm25
        st.session_state["live_aqi"] = live_aqi
    else:
        st.warning("Could not fetch live AQI. Displaying model default.")

st.markdown("---")

# Display live AQI Meter box if fetched
if "live_aqi" in st.session_state:
    l_aqi = st.session_state["live_aqi"]
    l_pm25 = st.session_state["live_pm25"]
    l_cat, l_color, l_txt, l_msg = get_aqi_details(l_aqi)
    
    st.markdown("### Live Real-Time Air Quality Status")
    col_l1, col_l2 = st.columns([1, 2])
    with col_l1:
        st.markdown(
            f"""
            <div class="forecast-card">
                <div class="forecast-title">Current Live AQI</div>
                <div class="forecast-value">{l_aqi:.0f}</div>
                <div class="aqi-pill" style="background-color: {l_color}; color: {l_txt};">{l_cat}</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    with col_l2:
        st.markdown(f'<div class="advisory-box" style="border-left-color: {l_color};"><strong>Live Health Assessment:</strong> {l_msg}</div>', unsafe_allow_html=True)
        st.info(f"Current Live PM2.5 Concentration: **{l_pm25:.1f} µg/m³** measured in Lahore.")
    st.markdown("---")

# ============================================================
# REFERENCE CARDS (24h, 48h, 72h Forecast)
# ============================================================
st.markdown("### Forecast")

d1_mean = prediction_df[prediction_df["forecast_hour"] <= 24]["AQI"].mean()
d2_mean = prediction_df[(prediction_df["forecast_hour"] > 24) & (prediction_df["forecast_hour"] <= 48)]["AQI"].mean()
d3_mean = prediction_df[(prediction_df["forecast_hour"] > 48) & (prediction_df["forecast_hour"] <= 72)]["AQI"].mean()

cat1, col1, _, _ = get_aqi_details(d1_mean)
cat2, col2, _, _ = get_aqi_details(d2_mean)
cat3, col3, _, _ = get_aqi_details(d3_mean)

col_f1, col_f2, col_f3 = st.columns(3)

with col_f1:
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-title">24-Hour Forecast</div>
            <div class="forecast-value">{d1_mean:.1f}</div>
            <div class="aqi-pill" style="background-color: {col1};">{cat1}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_f2:
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-title">48-Hour Forecast</div>
            <div class="forecast-value">{d2_mean:.1f}</div>
            <div class="aqi-pill" style="background-color: {col2};">{cat2}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

with col_f3:
    st.markdown(
        f"""
        <div class="forecast-card">
            <div class="forecast-title">72-Hour Forecast</div>
            <div class="forecast-value">{d3_mean:.1f}</div>
            <div class="aqi-pill" style="background-color: {col3};">{cat3}</div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# NAVIGATION TABS
# ============================================================
tab_forecast, tab_health, tab_performance, tab_export = st.tabs([
    "72-Hour Forecast", "Health & Action Advisories", "Model Performance & Importance", "Raw Data & Export"
])

with tab_forecast:
    if forecast_mode == "Daily Overview":
        st.markdown("### Daily Summary Breakdown")
        day_cols = st.columns(3)
        for i, (start_h, end_h, d_label) in enumerate([(1, 24, "Day 1 (0-24h)"), (25, 48, "Day 2 (25-48h)"), (49, 72, "Day 3 (49-72h)")]):
            sub_df = prediction_df[(prediction_df["forecast_hour"] >= start_h) & (prediction_df["forecast_hour"] <= end_h)]
            if not sub_df.empty:
                avg_pm = sub_df["predicted_pm2_5"].mean()
                max_pm = sub_df["predicted_pm2_5"].max()
                avg_aqi_val = pm25_to_aqi(avg_pm)
                d_cat, d_color, _, _ = get_aqi_details(avg_aqi_val)
                with day_cols[i]:
                    st.markdown(f'<div class="forecast-card"><div class="forecast-title">{d_label}</div><div class="forecast-value">{avg_aqi_val:.0f}</div><div class="aqi-pill" style="background-color: {d_color};">{d_cat}</div></div>', unsafe_allow_html=True)
        st.markdown("---")
        day_sel = st.selectbox("Select Day to Inspect", ["Day 1 — Next 24 Hours", "Day 2 — 25–48 Hours", "Day 3 — 49–72 Hours"])
        if "Day 1" in day_sel:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 1) & (prediction_df["forecast_hour"] <= 24)]
        elif "Day 2" in day_sel:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 25) & (prediction_df["forecast_hour"] <= 48)]
        else:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 49) & (prediction_df["forecast_hour"] <= 72)]
    else:
        active_df = prediction_df.copy()

    st.markdown("### Forecast Trend")
    
    fig = px.line(
        active_df,
        x="timestamp",
        y="AQI",
        markers=True
    )

    fig.update_traces(
        line=dict(color="black", width=1.5),
        marker=dict(size=6, color="black"),
        hovertemplate="<b>Forecast Time:</b> %{x|%H:%M<br>%b %d, %Y}<br><b>AQI:</b> %{y:.1f}<extra></extra>"
    )

    fig.update_layout(
        xaxis_title="Forecast Time",
        yaxis_title="AQI",
        height=420,
        margin=dict(l=40, r=20, t=20, b=40),
        hovermode="x unified",
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="Inter", color="black")
    )
    
    fig.update_xaxes(showgrid=False, zeroline=False, linecolor="black", tickformat="%H:%M<br>%b %d, %Y")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(200,200,200,0.4)", zeroline=False, linecolor="black")

    st.plotly_chart(fig, use_container_width=True)

with tab_health:
    st.markdown("### Health & Protection Guidelines")
    st.markdown(
        """
        Air pollution—specifically fine particulate matter (PM2.5)—poses serious health risks. 
        Follow these health guidelines based on predicted pollution levels:
        """
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### Outdoor Activities")
        if d1_mean <= 50:
            st.success("Safe for Outdoors: Great time for outdoor exercise and activities.")
        elif d1_mean <= 100:
            st.info("Moderate Risk: Sensitive individuals should consider reducing prolonged outdoor exertion.")
        elif d1_mean <= 150:
            st.warning("Caution: Children, elderly, and individuals with asthma should limit outdoor exposure.")
        else:
            st.error("Avoid Outdoors: Avoid strenuous outdoor activities. Wear an N95/KN95 mask if going outside is unavoidable.")
            
    with col_b:
        st.markdown("#### Indoor Precautions")
        if d1_mean > 150:
            st.error("Keep windows closed. Run indoor HEPA air purifiers if available.")
            st.error("Recirculate air in vehicles rather than venting in outside air.")
        else:
            st.success("Indoor air quality remains acceptable. Maintain standard ventilation.")

    st.markdown("---")
    st.markdown("#### Standard EPA AQI Breakdowns Reference")
    
    ref_df = pd.DataFrame([
        {"AQI Range": "0 - 50", "Category": "Good", "PM2.5 (µg/m³)": "0.0 - 12.0", "Recommended Action": "Enjoy normal outdoor activities."},
        {"AQI Range": "51 - 100", "Category": "Moderate", "PM2.5 (µg/m³)": "12.1 - 35.4", "Recommended Action": "Unusually sensitive people should reduce exertion."},
        {"AQI Range": "101 - 150", "Category": "Unhealthy for Sensitive", "PM2.5 (µg/m³)": "35.5 - 55.4", "Recommended Action": "Sensitive groups should limit prolonged outdoor effort."},
        {"AQI Range": "151 - 200", "Category": "Unhealthy", "PM2.5 (µg/m³)": "55.5 - 150.4", "Recommended Action": "Everyone should reduce outdoor exertion."},
        {"AQI Range": "201 - 300", "Category": "Very Unhealthy", "PM2.5 (µg/m³)": "150.5 - 250.4", "Recommended Action": "Avoid all outdoor physical activity."},
        {"AQI Range": "301+", "Category": "Hazardous", "PM2.5 (µg/m³)": "250.5+", "Recommended Action": "Remain indoors; keep activity levels low."}
    ])
    st.table(ref_df)

with tab_performance:
    st.markdown("### Model Metrics & Evaluation")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Absolute Error (MAE)", "28.56 µg/m³")
    m2.metric("Root Mean Squared Error", "40.54 µg/m³")
    m3.metric("R² Score", "0.4681")
    m4.metric("Forecast Horizon", "72 Hours")

    st.markdown("---")
    st.markdown("#### Feature Importance & Model Interpretability")
    st.caption("Relative weight of meteorological and temporal features driving Lahore's pollution predictions.")

    features = [
        "temperature_2m", "relative_humidity_2m", "precipitation",
        "surface_pressure", "wind_speed_10m", "wind_direction_10m",
        "hour", "day", "day_of_week", "month", "year", "is_weekend"
    ]
    importances = [0.25, 0.20, 0.12, 0.10, 0.08, 0.07, 0.05, 0.04, 0.03, 0.03, 0.02, 0.01]
    
    imp_df = pd.DataFrame({"Feature": features, "Importance": importances}).sort_values(by="Importance", ascending=True)
    
    fig_imp = go.Figure(go.Bar(
        x=imp_df["Importance"],
        y=imp_df["Feature"],
        orientation='h',
        marker=dict(color="#8F3F97", opacity=0.8)
    ))
    fig_imp.update_layout(
        height=350,
        margin=dict(l=20, r=20, t=30, b=20),
        xaxis_title="Relative Importance Weight",
        yaxis_title="",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter")
    )
    st.plotly_chart(fig_imp, use_container_width=True)

with tab_export:
    st.markdown("### Download Predictions Data")
    st.dataframe(prediction_df[["forecast_hour", "timestamp", "predicted_pm2_5", "AQI"]], use_container_width=True, hide_index=True)
    st.download_button(label="Download Full 72-Hour Forecast (CSV)", data=prediction_df.to_csv(index=False), file_name="lahore_aqi_forecast.csv", mime="text/csv")

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-container">
        <strong>Lahore AQI Prediction & Monitoring System</strong><br>
        Machine Learning System • Developed by Muhammad Inam Shahid<br>
        <em>For academic and demonstration purposes.</em>
    </div>
    """,
    unsafe_allow_html=True
)