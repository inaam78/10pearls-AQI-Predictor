import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ============================================================
# PAGE CONFIGURATION
# ============================================================
st.set_page_config(
    page_title="Lahore AQI Predictor & Air Quality Dashboard",
    page_icon="🌫️",
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
    .main { padding-top: 1rem; }
    .metric-card {
        padding: 18px;
        border-radius: 12px;
        border: 1px solid rgba(128,128,128,0.2);
        background: rgba(128,128,128,0.05);
        text-align: center;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
    }
    .aqi-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 20px;
        font-weight: 700;
        font-size: 14px;
        margin-top: 6px;
    }
    .advisory-box {
        padding: 16px;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 20px;
        background-color: rgba(128,128,128,0.05);
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

# ============================================================
# LOAD PREDICTIONS DIRECTLY FROM OUTPUTS CSV
# ============================================================
@st.cache_data
def load_predictions():
    if os.path.exists(PREDICTION_PATH):
        df = pd.read_csv(PREDICTION_PATH)
        # Ensure timestamp column exists and is parsed correctly
        if "timestamp" not in df.columns:
            start_time = datetime.now()
            df["timestamp"] = [start_time + timedelta(hours=i) for i in range(len(df))]
        else:
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            
        if "forecast_hour" not in df.columns:
            df["forecast_hour"] = range(1, len(df) + 1)
            
        # Map predicted column name safely
        if "predicted_pm25" in df.columns and "predicted_pm2_5" not in df.columns:
            df["predicted_pm2_5"] = df["predicted_pm25"]
            
        return df
    return None

prediction_df = load_predictions()

if prediction_df is None or "predicted_pm2_5" not in prediction_df.columns:
    st.error("⚠️ Predictions file not found at outputs/72h_predictions.csv. Please run python src/predict.py first.")
    st.stop()

prediction_df["AQI"] = prediction_df["predicted_pm2_5"].apply(pm25_to_aqi)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.markdown("---")
city = st.sidebar.selectbox("🏙️ Select City", ["Lahore"])
forecast_mode = st.sidebar.radio("📊 Forecast View Mode", ["Hourly View", "Daily Overview"])
show_thresholds = st.sidebar.checkbox("Show AQI Severity Bands on Chart", value=True)
show_uncertainty = st.sidebar.checkbox("Show Confidence Bounds (if present)", value=True)

# ============================================================
# HEADER & CURRENT CONDITIONS
# ============================================================
st.markdown("# 🌫️ Lahore Air Quality Index (AQI) Forecast")
st.caption("AI-Powered 72-Hour PM2.5 Concentration & AQI Predictive Dashboard")

latest_row = prediction_df.iloc[0]
latest_pm25 = float(latest_row["predicted_pm2_5"])
latest_aqi = pm25_to_aqi(latest_pm25)
cat, color, txt_color, impact_msg = get_aqi_details(latest_aqi)
latest_time = latest_row["timestamp"]

st.markdown("---")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Live Predicted AQI", f"{latest_aqi:.0f}")
c2.metric("Live PM2.5", f"{latest_pm25:.1f} µg/m³")
with c3:
    st.markdown(f'<div style="text-align: center;"><span style="font-size: 12px; color: gray;">Air Quality Category</span><br><div class="aqi-badge" style="background-color: {color}; color: {txt_color};">{cat}</div></div>', unsafe_allow_html=True)
c4.metric("Location / Time", f"{city}", delta=latest_time.strftime("%b %d, %H:%M PKT") if isinstance(latest_time, pd.Timestamp) else str(latest_time), delta_color="off")

st.markdown(f'<div class="advisory-box" style="border-left-color: {color};"><strong>Health Assessment:</strong> {impact_msg}</div>', unsafe_allow_html=True)

# ============================================================
# NAVIGATION TABS
# ============================================================
tab_forecast, tab_health, tab_performance, tab_export = st.tabs([
    "📈 72-Hour Forecast", "🏥 Health & Action Advisories", "🤖 Model Performance", "📥 Raw Data & Export"
])

with tab_forecast:
    if forecast_mode == "Daily Overview":
        st.markdown("### 📅 Daily Summary Breakdown")
        day_cols = st.columns(3)
        for i, (start_h, end_h, d_label) in enumerate([(1, 24, "Day 1 (0-24h)"), (25, 48, "Day 2 (25-48h)"), (49, 72, "Day 3 (49-72h)")]):
            sub_df = prediction_df[(prediction_df["forecast_hour"] >= start_h) & (prediction_df["forecast_hour"] <= end_h)]
            if not sub_df.empty:
                avg_pm = sub_df["predicted_pm2_5"].mean()
                max_pm = sub_df["predicted_pm2_5"].max()
                avg_aqi_val = pm25_to_aqi(avg_pm)
                max_aqi_val = pm25_to_aqi(max_pm)
                d_cat, d_color, d_txt, _ = get_aqi_details(avg_aqi_val)
                with day_cols[i]:
                    st.markdown(f'<div class="metric-card"><h4 style="margin: 0;">{d_label}</h4><div class="aqi-badge" style="background-color: {d_color}; color: {d_txt}; font-size: 18px; margin: 10px 0;">Avg AQI {avg_aqi_val:.0f}</div><p style="margin:0; font-size: 13px;"><b>Avg PM2.5:</b> {avg_pm:.1f} µg/m³</p><p style="margin:0; font-size: 13px;"><b>Peak AQI:</b> {max_aqi_val:.0f} ({max_pm:.1f} µg/m³)</p></div>', unsafe_allow_html=True)
        st.markdown("---")
        day_sel = st.selectbox("Select Day to Inspect", ["Day 1 — Next 24 Hours", "Day 2 — 25–48 Hours", "Day 3 — 49–72 Hours"])
        if "Day 1" in day_sel:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 1) & (prediction_df["forecast_hour"] <= 24)]
        elif "Day 2" in day_sel:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 25) & (prediction_df["forecast_hour"] <= 48)]
        else:
            active_df = prediction_df[(prediction_df["forecast_hour"] >= 49) & (prediction_df["forecast_hour"] <= 72)]
    else:
        st.markdown("### ⏱️ Hourly Forecast Analysis")
        selected_h = st.slider("Select Forecast Horizon (Hours ahead)", 1, 72, 1)
        active_df = prediction_df.copy()

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=active_df["timestamp"],
        y=active_df["predicted_pm2_5"],
        mode="lines+markers",
        name="Predicted PM2.5",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6, color="#1f77b4"),
        hovertemplate="<b>Time:</b> %{x|%b %d, %H:%M}<br><b>PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
    ))

    if show_thresholds:
        bands = [
            (0, 12.0, "rgba(0, 228, 0, 0.08)"), (12.1, 35.4, "rgba(255, 255, 0, 0.08)"),
            (35.5, 55.4, "rgba(255, 126, 0, 0.08)"), (55.5, 150.4, "rgba(255, 0, 0, 0.08)"),
            (150.5, 250.4, "rgba(143, 63, 151, 0.08)"), (250.5, 500.0, "rgba(126, 0, 35, 0.08)")
        ]
        for y0, y1, b_color in bands:
            fig.add_hrect(y0=y0, y1=y1, fillcolor=b_color, line_width=0, layer="below")

    fig.update_layout(
        title=f"{city} — PM2.5 72-Hour Forecast Trajectory",
        xaxis_title="Timestamp",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified"
    )
    st.plotly_chart(fig, use_container_width=True)

with tab_health:
    st.markdown("### 🏥 Health & Protection Guidelines")
    st.info(impact_msg)

with tab_performance:
    st.markdown("### 🤖 Model Metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", "28.56 µg/m³")
    m2.metric("RMSE", "40.54 µg/m³")
    m3.metric("R² Score", "0.4681")

with tab_export:
    st.markdown("### 📥 Download Predictions")
    st.dataframe(prediction_df[["forecast_hour", "timestamp", "predicted_pm2_5", "AQI"]], use_container_width=True, hide_index=True)
    st.download_button("⬇️ Download CSV", data=prediction_df.to_csv(index=False), file_name="lahore_aqi_forecast.csv", mime="text/csv")