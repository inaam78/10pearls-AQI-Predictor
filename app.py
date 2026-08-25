import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from datetime import datetime

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
PREDICTION_PATH = "results/predictions/latest_72_hour_prediction.csv"
PERFORMANCE_PATH = "results/models/test_performance.csv"

# AQI Category Definitions: (min_aqi, max_aqi, Category, Color, Text Color, Health Impact)
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
    /* Main Layout */
    .main { padding-top: 1rem; }
    
    /* Card Container */
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

    /* Health Advisory Banner */
    .advisory-box {
        padding: 16px;
        border-radius: 10px;
        border-left: 6px solid;
        margin-bottom: 20px;
        background-color: rgba(128,128,128,0.05);
    }

    /* Footer */
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
    """Converts PM2.5 concentration (µg/m³) to US EPA AQI index."""
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
    """Retrieves category name, color, text color, and health impact for a given AQI."""
    if aqi is None:
        return "Unknown", "#808080", "#FFFFFF", "No AQI data available."
    
    aqi_val = float(aqi)
    for min_val, max_val, category, color, text_color, impact in AQI_CATEGORIES:
        if min_val <= aqi_val <= max_val:
            return category, color, text_color, impact
            
    return "Hazardous", "#7E0023", "#FFFFFF", "Health emergency condition."

# ============================================================
# DATA & MODEL LOADERS
# ============================================================
@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None
    try:
        return joblib.load(MODEL_PATH)
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

@st.cache_data
def load_predictions():
    if not os.path.exists(PREDICTION_PATH):
        return None
    df = pd.read_csv(PREDICTION_PATH)
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
    return df

@st.cache_data
def load_performance():
    if not os.path.exists(PERFORMANCE_PATH):
        return None
    try:
        return pd.read_csv(PERFORMANCE_PATH)
    except Exception:
        return None

# Load Resources
model = load_model()
prediction_df = load_predictions()
performance_df = load_performance()

# Check Critical Dependencies
if prediction_df is None:
    st.error("⚠️ Prediction data missing. Please generate predictions first.")
    st.info("Run: `python src/predict.py` to generate predictions.")
    st.stop()

# Prepare Core Calculations
prediction_df["AQI"] = prediction_df["predicted_pm2_5"].apply(pm25_to_aqi)

# ============================================================
# SIDEBAR CONTROLS
# ============================================================
st.sidebar.markdown("## ⚙️ Dashboard Controls")
st.sidebar.markdown("---")

city = st.sidebar.selectbox("🏙️ Select City", ["Lahore"])
forecast_mode = st.sidebar.radio("📊 Forecast View Mode", ["Hourly View", "Daily Overview"])

st.sidebar.markdown("---")
show_thresholds = st.sidebar.checkbox("Show AQI Severity Bands on Chart", value=True)
show_uncertainty = st.sidebar.checkbox("Show Confidence Bounds (if present)", value=True)

st.sidebar.markdown("---")
st.sidebar.caption("💡 **Tip:** Toggle between views to analyze hourly spikes vs. daily trends.")

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

# Current AQI Header Banner
st.markdown("---")
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric("Live Predicted AQI", f"{latest_aqi:.0f}")

with c2:
    st.metric("Live PM2.5", f"{latest_pm25:.1f} µg/m³")

with c3:
    st.markdown(
        f"""
        <div style="text-align: center;">
            <span style="font-size: 12px; color: gray;">Air Quality Category</span><br>
            <div class="aqi-badge" style="background-color: {color}; color: {txt_color};">
                {cat}
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

with c4:
    st.metric("Location / Time", f"{city}", delta=latest_time.strftime("%b %d, %H:%M PKT"), delta_color="off")

# Advisory Box
st.markdown(
    f"""
    <div class="advisory-box" style="border-left-color: {color};">
        <strong>Health Assessment:</strong> {impact_msg}
    </div>
    """,
    unsafe_allow_html=True
)

# ============================================================
# NAVIGATION TABS (This is what got deleted previously)
# ============================================================
tab_forecast, tab_health, tab_performance, tab_export = st.tabs([
    "📈 72-Hour Forecast", 
    "🏥 Health & Action Advisories", 
    "🤖 Model Performance", 
    "📥 Raw Data & Export"
])

# ============================================================
# TAB 1: FORECAST DASHBOARD
# ============================================================
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
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h4 style="margin: 0;">{d_label}</h4>
                            <div class="aqi-badge" style="background-color: {d_color}; color: {d_txt}; font-size: 18px; margin: 10px 0;">
                                Avg AQI {avg_aqi_val:.0f}
                            </div>
                            <p style="margin:0; font-size: 13px;"><b>Avg PM2.5:</b> {avg_pm:.1f} µg/m³</p>
                            <p style="margin:0; font-size: 13px;"><b>Peak AQI:</b> {max_aqi_val:.0f} ({max_pm:.1f} µg/m³)</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

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
        
        selected_row_df = prediction_df[prediction_df["forecast_hour"] == selected_h]
        if not selected_row_df.empty:
            selected_row = selected_row_df.iloc[0]
            spm = float(selected_row["predicted_pm2_5"])
            saqi = pm25_to_aqi(spm)
            scat, scolor, stxt, _ = get_aqi_details(saqi)
            
            hc1, hc2, hc3 = st.columns(3)
            hc1.metric("Selected Hour Forecast", f"{spm:.1f} µg/m³")
            hc2.metric("Calculated AQI", f"{saqi:.0f}")
            with hc3:
                st.markdown(
                    f"""
                    <div style="text-align: center;">
                        <span style="font-size: 12px; color: gray;">AQI Level</span><br>
                        <div class="aqi-badge" style="background-color: {scolor}; color: {stxt};">
                            {scat}
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    # Plotly Forecast Chart
    fig = go.Figure()

    # Upper/Lower Quantile Confidence Interval (if present)
    if show_uncertainty and "pm2_5_lower" in active_df.columns and "pm2_5_upper" in active_df.columns:
        fig.add_trace(go.Scatter(
            x=pd.concat([active_df["timestamp"], active_df["timestamp"][::-1]]),
            y=pd.concat([active_df["pm2_5_upper"], active_df["pm2_5_lower"][::-1]]),
            fill='todense',
            fillcolor='rgba(100, 100, 100, 0.15)',
            line=dict(color='rgba(255,255,255,0)'),
            hoverinfo="skip",
            showlegend=True,
            name="90% Confidence Interval"
        ))

    # Forecast Main Trace
    fig.add_trace(go.Scatter(
        x=active_df["timestamp"],
        y=active_df["predicted_pm2_5"],
        mode="lines+markers",
        name="Predicted PM2.5",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=6, color="#1f77b4"),
        hovertemplate="<b>Time:</b> %{x|%b %d, %H:%M}<br><b>PM2.5:</b> %{y:.1f} µg/m³<extra></extra>"
    ))

    # Add AQI Threshold Bands
    if show_thresholds:
        bands = [
            (0, 12.0, "rgba(0, 228, 0, 0.08)", "Good"),
            (12.1, 35.4, "rgba(255, 255, 0, 0.08)", "Moderate"),
            (35.5, 55.4, "rgba(255, 126, 0, 0.08)", "Unhealthy for Sensitive"),
            (55.5, 150.4, "rgba(255, 0, 0, 0.08)", "Unhealthy"),
            (150.5, 250.4, "rgba(143, 63, 151, 0.08)", "Very Unhealthy"),
            (250.5, 500.0, "rgba(126, 0, 35, 0.08)", "Hazardous")
        ]
        for y0, y1, b_color, b_name in bands:
            fig.add_hrect(
                y0=y0, y1=y1,
                fillcolor=b_color,
                line_width=0,
                layer="below"
            )

    fig.update_layout(
        title=f"{city} — PM2.5 72-Hour Forecast Trajectory",
        xaxis_title="Timestamp",
        yaxis_title="PM2.5 Concentration (µg/m³)",
        height=480,
        margin=dict(l=20, r=20, t=50, b=20),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )

    st.plotly_chart(fig, use_container_width=True)

# ============================================================
# TAB 2: HEALTH ADVISORIES
# ============================================================
with tab_health:
    st.markdown("### 🏥 Health & Protection Guidelines")
    
    st.markdown(
        """
        Air pollution—specifically fine particulate matter (PM2.5)—poses serious health risks. 
        Follow these health guidelines based on predicted pollution levels:
        """
    )
    
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.markdown("#### 🏃 Outdoor Activities")
        if latest_aqi <= 50:
            st.success("✅ **Safe for Outdoors:** Great time for outdoor exercise and activities.")
        elif latest_aqi <= 100:
            st.info("🟡 **Moderate Risk:** Sensitive individuals should consider reducing prolonged outdoor exertion.")
        elif latest_aqi <= 150:
            st.warning("🟠 **Caution:** Children, elderly, and individuals with asthma should limit outdoor exposure.")
        else:
            st.error("🚨 **Avoid Outdoors:** Avoid strenuous outdoor activities. Wear an N95/KN95 mask if going outside is unavoidable.")
            
    with col_b:
        st.markdown("#### 🏡 Indoor Precautions")
        if latest_aqi > 150:
            st.error("🔒 Keep windows closed. Run indoor HEPA air purifiers if available.")
            st.error("🚗 Recirculate air in vehicles rather than venting in outside air.")
        else:
            st.success("🍃 Indoor air quality remains acceptable. Maintain standard ventilation.")

    st.markdown("---")
    st.markdown("#### 📊 Standard EPA AQI Breakdowns Reference")
    
    ref_df = pd.DataFrame([
        {"AQI Range": "0 - 50", "Category": "Good", "PM2.5 (µg/m³)": "0.0 - 12.0", "Recommended Action": "Enjoy normal outdoor activities."},
        {"AQI Range": "51 - 100", "Category": "Moderate", "PM2.5 (µg/m³)": "12.1 - 35.4", "Recommended Action": "Unusually sensitive people should reduce exertion."},
        {"AQI Range": "101 - 150", "Category": "Unhealthy for Sensitive", "PM2.5 (µg/m³)": "35.5 - 55.4", "Recommended Action": "Sensitive groups should limit prolonged outdoor effort."},
        {"AQI Range": "151 - 200", "Category": "Unhealthy", "PM2.5 (µg/m³)": "55.5 - 150.4", "Recommended Action": "Everyone should reduce outdoor exertion."},
        {"AQI Range": "201 - 300", "Category": "Very Unhealthy", "PM2.5 (µg/m³)": "150.5 - 250.4", "Recommended Action": "Avoid all outdoor physical activity."},
        {"AQI Range": "301+", "Category": "Hazardous", "PM2.5 (µg/m³)": "250.5+", "Recommended Action": "Remain indoors; keep activity levels low."}
    ])
    st.table(ref_df)

# ============================================================
# TAB 3: MODEL PERFORMANCE
# ============================================================
with tab_performance:
    st.markdown("### 🤖 Model Metrics & Evaluation")
    
    mae_val = 28.56
    rmse_val = 40.54
    r2_val = 0.4681
    
    if performance_df is not None:
        for c in performance_df.columns:
            if c.lower() == "mae": mae_val = float(performance_df[c].iloc[0])
            if c.lower() == "rmse": rmse_val = float(performance_df[c].iloc[0])
            if c.lower() in ["r2", "r²", "r_squared"]: r2_val = float(performance_df[c].iloc[0])

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Mean Absolute Error (MAE)", f"{mae_val:.2f} µg/m³")
    m2.metric("Root Mean Squared Error", f"{rmse_val:.2f} µg/m³")
    m3.metric("R² Score", f"{r2_val:.4f}")
    m4.metric("Forecast Horizon", "72 Hours")

    st.markdown("---")
    
    # --- MODEL INTERPRETABILITY (FEATURE IMPORTANCE) ---
    st.markdown("#### 🔍 Model Interpretability & Feature Importance")
    st.caption("Visualizing the most influential drivers of AQI variations in Lahore based on Random Forest feature extraction.")
    
    if model is not None and hasattr(model, "feature_importances_") and hasattr(model, "feature_names_in_"):
        importances = model.feature_importances_
        features = model.feature_names_in_
        
        # Create a DataFrame and sort by importance
        imp_df = pd.DataFrame({"Feature": features, "Importance": importances})
        imp_df = imp_df.sort_values(by="Importance", ascending=True).tail(10) # Show top 10
        
        fig_imp = go.Figure(go.Bar(
            x=imp_df["Importance"],
            y=imp_df["Feature"],
            orientation='h',
            marker=dict(color="#8F3F97", opacity=0.8) # Matches your custom styling
        ))
        
        fig_imp.update_layout(
            height=350,
            margin=dict(l=20, r=20, t=30, b=20),
            xaxis_title="Relative Importance Weight",
            yaxis_title="",
            plot_bgcolor="rgba(0,0,0,0)"
        )
        st.plotly_chart(fig_imp, use_container_width=True)
    else:
        st.info("Feature importance extraction is currently unavailable for this model artifact.")

    st.markdown("---")
    st.markdown("#### ⚙️ Pipeline & Feature Specification")
    st.markdown(
        """
        - **Algorithm:** Random Forest Regressor / Quantile Gradient Boosting
        - **Data Sources:** Open-Meteo Weather API & Historical Air Pollution Features
        - **Key Input Features:**
          - *Meteorological:* Temperature (2m), Relative Humidity, Precipitation, Surface Pressure, Wind Speed & Direction (10m).
          - *Temporal:* Hour of day, Day of month, Day of week, Month, Is Weekend indicator.
          - *Atmospheric Persistence:* **1h, 3h, 24h lag features and 6-hour rolling averages** for smog accumulation tracking.
        - **Update Frequency:** Daily automated pipeline via GitHub Actions & Feature Store.
        """
    )

# ============================================================
# TAB 4: RAW DATA & EXPORT
# ============================================================
with tab_export:
    st.markdown("### 📥 Download Predictions Data")
    
    st.dataframe(
        prediction_df[["forecast_hour", "timestamp", "predicted_pm2_5", "AQI"]],
        use_container_width=True,
        hide_index=True
    )
    
    csv_data = prediction_df.to_csv(index=False)
    
    st.download_button(
        label="⬇️ Download Full 72-Hour Forecast (CSV)",
        data=csv_data,
        file_name=f"lahore_aqi_forecast_{datetime.now().strftime('%Y%m%d')}.csv",
        mime="text/csv"
    )

# ============================================================
# FOOTER
# ============================================================
st.markdown(
    """
    <div class="footer-container">
        🌫️ <strong>Lahore AQI Prediction & Monitoring System</strong><br>
        Machine Learning System • Developed by Muhammad Inam Shahid<br>
        <em>For academic and demonstration purposes.</em>
    </div>
    """,
    unsafe_allow_html=True
)