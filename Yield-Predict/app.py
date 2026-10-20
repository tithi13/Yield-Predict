"""
app.py
======
Streamlit Web Application: Crop Yield Prediction using Weather and Soil Data
Polished, modern agricultural analytics dashboard adhering strictly to project ML pipeline.
"""

import os
import json
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

# ---------------------------------------------------------
# Page Configuration & Styling (Agriculture-Inspired Palette)
# ---------------------------------------------------------
st.set_page_config(
    page_title="Crop Yield Prediction | Weather & Soil ML",
    page_icon="🌾",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern agricultural analytics dashboard
st.markdown("""
<style>
    /* Global background and base typography */
    .stApp {
        background-color: #FAF8F5;
        color: #1F2937;
        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    }

    /* Main container text visibility */
    .main .block-container {
        color: #1F2937;
    }
    .main p, .main span, .main label, .main div {
        color: inherit;
    }

    /* Streamlit Input Labels (Sliders, Selectboxes, NumberInputs, Radio, etc.) */
    .main label[data-baseweb="label"],
    .main [data-testid="stWidgetLabel"],
    .main [data-testid="stWidgetLabel"] label,
    .main [data-testid="stWidgetLabel"] p,
    .main [data-testid="stWidgetLabel"] span {
        color: #1F2937 !important;
        font-weight: 600 !important;
        font-size: 0.93rem !important;
    }

    /* Slider values, thumb numbers, min/max captions */
    .main [data-testid="stSlider"] div[data-testid="stTickBarMin"],
    .main [data-testid="stSlider"] div[data-testid="stTickBarMax"],
    .main [data-testid="stSlider"] div[data-testid="stThumbValue"],
    .main [data-testid="stSlider"] [data-testid="stWidgetLabel"] + div div {
        color: #1F2937 !important;
        font-weight: 600 !important;
    }

    /* Tooltip & Help Icons */
    .main [data-testid="stTooltipIcon"] {
        color: #2E7D32 !important;
    }
    .main [data-testid="stTooltipIcon"] svg {
        fill: #2E7D32 !important;
    }

    /* Selectbox dropdown container & selected text */
    .main [data-baseweb="select"] {
        background-color: #FFFFFF !important;
        border-color: #D1D5DB !important;
    }
    .main [data-baseweb="select"] span,
    .main [data-baseweb="select"] div,
    .main [data-baseweb="select"] input {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }

    /* Streamlit Radios in main area */
    .main [data-testid="stRadio"] label,
    .main [data-testid="stRadio"] label p,
    .main [data-testid="stRadio"] label div {
        color: #1F2937 !important;
        font-weight: 500 !important;
    }

    /* Streamlit Tab Headers */
    .main [data-baseweb="tab-list"] button {
        color: #4B5563 !important;
        font-weight: 600 !important;
    }
    .main [data-baseweb="tab-list"] button[aria-selected="true"] {
        color: #1E4620 !important;
        border-bottom-color: #2E7D32 !important;
        font-weight: 700 !important;
    }

    /* Captions and Subtext in main body */
    .main .stCaption,
    .main [data-testid="stCaptionContainer"],
    .main [data-testid="stCaptionContainer"] p {
        color: #4B5563 !important;
        font-size: 0.86rem !important;
    }

    /* Markdown text in main container */
    .main [data-testid="stMarkdownContainer"] p,
    .main [data-testid="stMarkdownContainer"] li {
        color: #1F2937;
    }

    /* Sidebar isolation: preserve readable light text in dark sidebar */
    section[data-testid="stSidebar"] {
        background-color: #172F19;
        color: #F3F4F6;
    }
    section[data-testid="stSidebar"] h1,
    section[data-testid="stSidebar"] h2,
    section[data-testid="stSidebar"] h3,
    section[data-testid="stSidebar"] h4 {
        color: #FFFFFF !important;
    }
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] span,
    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] li,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
    section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] span {
        color: #E2E8F0 !important;
    }
    section[data-testid="stSidebar"] .stCaption,
    section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p {
        color: #CBD5E1 !important;
    }
    section[data-testid="stSidebar"] [data-testid="stRadio"] label p {
        color: #F8FAFC !important;
        font-weight: 500 !important;
    }

    /* Headers */
    .hero-header-box {
        background: linear-gradient(135deg, #1E4620 0%, #2D5A27 100%);
        border-radius: 12px;
        padding: 24px 30px;
        margin-bottom: 24px;
        color: #FFFFFF !important;
        box-shadow: 0 4px 14px rgba(30, 70, 32, 0.15);
        border-left: 6px solid #D97706;
    }
    .hero-title {
        font-size: 2.1rem;
        font-weight: 700;
        margin: 0;
        letter-spacing: -0.02em;
        color: #FFFFFF !important;
    }
    .hero-subtitle {
        font-size: 1.0rem;
        margin-top: 6px;
        margin-bottom: 0;
        color: #E2E8F0 !important;
        opacity: 0.95;
    }
    
    /* Section headers */
    .section-title {
        font-size: 1.3rem;
        font-weight: 700;
        color: #1E4620 !important;
        margin-top: 10px;
        margin-bottom: 12px;
        display: flex;
        align-items: center;
        gap: 8px;
    }
    
    /* Polished Cards */
    .custom-card {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 10px;
        padding: 20px 24px;
        margin-bottom: 18px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        color: #1F2937;
    }
    
    .card-title {
        font-size: 1.05rem;
        font-weight: 600;
        color: #1E4620 !important;
        margin-bottom: 14px;
        border-bottom: 1px solid #F3F4F6;
        padding-bottom: 8px;
    }

    /* Prediction Outcome Cards */
    .pred-card-selected {
        background: #FFFFFF;
        border: 2px solid #2E7D32;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(46, 125, 50, 0.12);
        position: relative;
        color: #1F2937;
    }
    .badge-selected {
        background-color: #2E7D32;
        color: #FFFFFF !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .badge-baseline {
        background-color: #E5E7EB;
        color: #374151 !important;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        padding: 4px 10px;
        border-radius: 20px;
        display: inline-block;
        margin-bottom: 10px;
    }
    .pred-card-baseline {
        background: #FFFFFF;
        border: 1px solid #D1D5DB;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 2px 6px rgba(0, 0, 0, 0.04);
        color: #1F2937;
    }
    .yield-number-selected {
        font-size: 2.5rem;
        font-weight: 800;
        color: #1E4620 !important;
        line-height: 1.1;
        margin: 6px 0;
    }
    .yield-number-baseline {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1F2937 !important;
        line-height: 1.1;
        margin: 6px 0;
    }
    .yield-unit {
        font-size: 1.0rem;
        font-weight: 500;
        color: #4B5563 !important;
    }
    
    /* Fertilizer Advisory Box */
    .advisory-container {
        background-color: #FFFFFF;
        border: 1px solid #FDE68A;
        border-left: 6px solid #D97706;
        border-radius: 8px;
        padding: 18px 22px;
        margin-top: 18px;
        box-shadow: 0 2px 6px rgba(217, 119, 6, 0.06);
        color: #1F2937;
    }
    .advisory-header {
        font-size: 1.05rem;
        font-weight: 700;
        color: #92400E !important;
        margin-bottom: 6px;
        display: flex;
        align-items: center;
        gap: 6px;
    }
    .disclaimer-banner {
        background-color: #FEF3C7;
        color: #78350F !important;
        padding: 8px 12px;
        border-radius: 6px;
        font-size: 0.88rem;
        line-height: 1.4;
        margin-bottom: 12px;
        border: 1px dashed #F59E0B;
    }
    
    /* Metric pill boxes */
    .metric-pill {
        background-color: #FFFFFF;
        border: 1px solid #E5E7EB;
        border-radius: 8px;
        padding: 12px 16px;
        text-align: center;
        box-shadow: 0 1px 4px rgba(0,0,0,0.03);
    }
    .metric-pill-title {
        font-size: 0.82rem;
        text-transform: uppercase;
        color: #4B5563 !important;
        font-weight: 600;
    }
    .metric-pill-val {
        font-size: 1.45rem;
        font-weight: 700;
        color: #1E4620 !important;
        margin-top: 4px;
    }
    .metric-pill-sub {
        font-size: 0.80rem;
        color: #2E7D32 !important;
        font-weight: 600;
    }
    
    /* Primary Predict Button refinement */
    div.stButton > button:first-child {
        background-color: #2E7D32 !important;
        color: #FFFFFF !important;
        font-weight: 600 !important;
        border: none !important;
        border-radius: 8px !important;
        padding: 10px 24px !important;
        font-size: 1.05rem !important;
        transition: all 0.2s ease-in-out !important;
        box-shadow: 0 2px 6px rgba(46, 125, 50, 0.25) !important;
    }
    div.stButton > button:first-child:hover {
        background-color: #1E4620 !important;
        color: #FFFFFF !important;
        box-shadow: 0 4px 10px rgba(30, 70, 32, 0.35) !important;
        transform: translateY(-1px) !important;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Load Artifacts and Metadata
# ---------------------------------------------------------
MODELS_DIR = "models"
METADATA_FILE = os.path.join(MODELS_DIR, "model_metadata.json")
LR_FILE = os.path.join(MODELS_DIR, "linear_regression_pipeline.joblib")
RF_FILE = os.path.join(MODELS_DIR, "random_forest_pipeline.joblib")
TEST_PRED_FILE = os.path.join(MODELS_DIR, "test_predictions_sample.csv")

@st.cache_resource
def load_models_and_metadata():
    if not os.path.exists(METADATA_FILE):
        return None, None, None, None
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    lr_model = joblib.load(LR_FILE)
    rf_model = joblib.load(RF_FILE)
    test_preds = pd.read_csv(TEST_PRED_FILE) if os.path.exists(TEST_PRED_FILE) else None
    return metadata, lr_model, rf_model, test_preds

metadata, lr_model, rf_model, test_preds = load_models_and_metadata()

if metadata is None:
    st.error("Model artifacts not found! Please run `python train_model.py` first to generate models and metadata.")
    st.stop()

crops_available = metadata.get("crops", [])
yield_units = metadata.get("yield_units", "Tonnes per Hectare (t/ha)")
metrics_lr = metadata["evaluation_metrics"]["Linear_Regression"]
metrics_rf = metadata["evaluation_metrics"]["Random_Forest"]
better_model = metadata.get("better_model", "Random Forest Regressor")

# ---------------------------------------------------------
# Advisory Helper Functions
# ---------------------------------------------------------
def get_fertilizer_advisory(crop, n, p, k, ph):
    """
    Rule-based nutrient balance check.
    Clearly labeled as an educational heuristic advisory, not a clinical prescription.
    """
    notes = []
    
    # Soil pH advisory
    if ph < 5.5:
        notes.append(f"**Soil pH ({ph:.2f}) is strongly acidic:** Nutrient solubility (Phosphorus, Calcium) decreases. Consider applying agricultural lime (calcium carbonate) or dolomite to condition soil toward pH 6.0–6.8.")
    elif ph > 7.5:
        notes.append(f"**Soil pH ({ph:.2f}) is moderately alkaline:** Micronutrient availability (Zinc, Iron, Manganese) may be restricted. Consider gypsum or well-decomposed organic manure to buffer soil reaction.")
    else:
        notes.append(f"**Soil pH ({ph:.2f}) is within the optimal agronomic range (5.5–7.5)** for nutrient availability.")

    # Nitrogen (N) heuristic (Typical guideline 60 - 120 kg/ha)
    if n < 50:
        notes.append(f"**Nitrogen ({n:.0f} kg/ha) is low:** Leaf chlorophyll synthesis and vegetative tillering may be constrained. Consider split application of nitrogenous fertilizers (e.g., Urea or Ammonium Sulphate).")
    elif n > 140:
        notes.append(f"**Nitrogen ({n:.0f} kg/ha) is high:** Excessive nitrogen may induce lodging, delay vegetative maturity, and heighten vulnerability to foliar pathogens. Minimize supplementary N fertilizer.")
    else:
        notes.append(f"**Nitrogen ({n:.0f} kg/ha) is balanced** within typical cereal and legume vegetative requirements.")

    # Phosphorus (P) heuristic (Typical guideline 25 - 60 kg/ha)
    if p < 25:
        notes.append(f"**Phosphorus ({p:.0f} kg/ha) is low:** Early root elongation and seed establishment benefit from basal application of Single Superphosphate (SSP) or Diammonium Phosphate (DAP).")
    elif p > 70:
        notes.append(f"**Phosphorus ({p:.0f} kg/ha) is ample to high:** Sufficient for root development; routine basal maintenance is adequate.")
    else:
        notes.append(f"**Phosphorus ({p:.0f} kg/ha) is moderate and well-balanced** for root anchorage and flowering.")

    # Potassium (K) heuristic (Typical guideline 20 - 50 kg/ha)
    if k < 20:
        notes.append(f"**Potassium ({k:.0f} kg/ha) is low:** Potassium controls stomatal opening, drought resilience, and grain filling. Application of Muriate of Potash (MOP) prior to flowering is recommended.")
    else:
        notes.append(f"**Potassium ({k:.0f} kg/ha) is adequate** for osmotic regulation, pest resistance, and harvest firmness.")

    return notes

# ---------------------------------------------------------
# Sidebar: Navigation & Project Summary
# ---------------------------------------------------------
with st.sidebar:
    st.markdown("### 🌾 Agrowise Analytics")
    st.caption("College ML Prototype | 5th Semester IT Viva")
    st.markdown("---")
    
    app_mode = st.radio(
        "Dashboard Navigation:",
        [
            "1. Yield Prediction & Fertilizer Advisory",
            "2. Model Performance & Comparison",
            "3. Dataset Provenance & Limitations"
        ]
    )
    
    st.markdown("---")
    st.markdown("#### 📋 Project Abstract")
    st.markdown("**Title:** Crop Yield Prediction using Weather and Soil Data")
    st.markdown(f"**Target Metric:** Historical Crop Yield (`{yield_units}`)")
    st.markdown(f"**Selected Model:** `{better_model}` (R² = {metrics_rf['R2']:.4f})")
    st.markdown(f"**Benchmark Model:** `Linear Regression` (R² = {metrics_lr['R2']:.4f})")
    
    st.markdown("---")
    st.markdown("#### ⚙️ Feature Summary")
    st.markdown("""
    - **Climate:** Rainfall, Temperature, Humidity
    - **Soil:** Soil pH, Nitrogen, Phosphorus, Potassium
    - **Category:** Crop Variety (54 classes)
    """)

# ---------------------------------------------------------
# SECTION 1: YIELD PREDICTION & FERTILIZER ADVISORY
# ---------------------------------------------------------
if app_mode == "1. Yield Prediction & Fertilizer Advisory":
    # Header Banner
    st.markdown("""
    <div class="hero-header-box">
        <h1 class="hero-title">🌾 Crop Yield Prediction & Soil Advisory</h1>
        <p class="hero-subtitle">Interactive machine learning dashboard forecasting historical crop productivity based on soil fertility and regional climate indicators.</p>
    </div>
    """, unsafe_allow_html=True)

    col_input, col_pred = st.columns([1.15, 1], gap="large")

    with col_input:
        st.markdown('<div class="section-title">🌱 1. Enter Soil & Environmental Inputs</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="custom-card">
            <div class="card-title">Agricultural & Meteorological Parameters</div>
        """, unsafe_allow_html=True)

        c1, c2 = st.columns(2, gap="medium")
        with c1:
            crop_type = st.selectbox(
                "Crop Variety",
                crops_available,
                index=crops_available.index("Rice") if "Rice" in crops_available else 0,
                help="Select target agricultural crop variety from the 54 historical Indian varieties."
            )
            rainfall = st.slider(
                "Annual Rainfall (mm)",
                min_value=100.0,
                max_value=3500.0,
                value=1150.0,
                step=25.0,
                help="Total annual regional precipitation in millimeters."
            )
            temperature = st.slider(
                "Temperature (°C)",
                min_value=10.0,
                max_value=45.0,
                value=26.5,
                step=0.5,
                help="Mean ambient temperature during growing season in degrees Celsius."
            )
            humidity = st.slider(
                "Relative Humidity (%)",
                min_value=20.0,
                max_value=100.0,
                value=72.0,
                step=1.0,
                help="Mean relative atmospheric humidity percentage."
            )

        with c2:
            soil_ph = st.slider(
                "Soil pH Level",
                min_value=3.5,
                max_value=9.5,
                value=6.5,
                step=0.1,
                help="Soil reaction scale: <5.5 acidic, 6.0-7.2 neutral/optimal, >7.5 alkaline."
            )
            nitrogen = st.slider(
                "Nitrogen (N) [kg/ha]",
                min_value=10.0,
                max_value=200.0,
                value=78.0,
                step=1.0,
                help="Available Nitrogen nutrient ratio in topsoil in kg/hectare."
            )
            phosphorus = st.slider(
                "Phosphorus (P) [kg/ha]",
                min_value=5.0,
                max_value=120.0,
                value=35.0,
                step=1.0,
                help="Available Phosphorus nutrient ratio in topsoil in kg/hectare."
            )
            potassium = st.slider(
                "Potassium (K) [kg/ha]",
                min_value=5.0,
                max_value=120.0,
                value=28.0,
                step=1.0,
                help="Available Potassium nutrient ratio in topsoil in kg/hectare."
            )

        predict_btn = st.button("🚀 Predict Crop Yield", type="primary", use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    with col_pred:
        st.markdown('<div class="section-title">📊 2. Comparative Model Forecast</div>', unsafe_allow_html=True)
        
        # Prepare feature DataFrame for Scikit-Learn Pipeline
        input_data = pd.DataFrame([{
            "rainfall": rainfall,
            "temperature": temperature,
            "humidity": humidity,
            "pH": soil_ph,
            "N": nitrogen,
            "P": phosphorus,
            "K": potassium,
            "crop_type": crop_type
        }])

        try:
            pred_lr = float(lr_model.predict(input_data)[0])
            pred_rf = float(rf_model.predict(input_data)[0])
            pred_lr_disp = max(0.0, pred_lr)
            pred_rf_disp = max(0.0, pred_rf)
        except Exception as e:
            st.error(f"Inference validation error: {str(e)}")
            st.stop()

        # Selected Model Card (Random Forest Regressor)
        st.markdown(f"""
        <div class="pred-card-selected">
            <span class="badge-selected">★ Preferred Model (Highest Test R²: {metrics_rf['R2']:.4f})</span>
            <div style="color: #4B5563; font-weight: 600; font-size: 0.95rem;">Random Forest Regressor Prediction</div>
            <div class="yield-number-selected">{pred_rf_disp:.2f} <span class="yield-unit">{yield_units}</span></div>
            <div style="font-size: 0.85rem; color: #4B5563; margin-top: 4px;">
                Evaluated Test MAE: <strong>{metrics_rf['MAE']:.2f} t/ha</strong> &bull; Non-linear tree ensemble capturing nutrient-climate interactions.
            </div>
            <div style="font-size: 0.78rem; color: #6B7280; margin-top: 8px; border-top: 1px solid #E5E7EB; padding-top: 6px;">
                <em>Statistical model forecast for {crop_type}; does not guarantee agricultural harvest outcomes.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

        st.write("")

        # Baseline Model Card (Linear Regression)
        diff_val = pred_lr_disp - pred_rf_disp
        st.markdown(f"""
        <div class="pred-card-baseline">
            <span class="badge-baseline">Baseline Benchmark (Test R²: {metrics_lr['R2']:.4f})</span>
            <div style="color: #4B5563; font-weight: 600; font-size: 0.95rem;">Linear Regression Prediction</div>
            <div class="yield-number-baseline">{pred_lr_disp:.2f} <span class="yield-unit">{yield_units}</span></div>
            <div style="font-size: 0.85rem; color: #4B5563; margin-top: 4px;">
                Difference from Preferred: <strong>{diff_val:+.2f} t/ha</strong> &bull; Evaluated Test MAE: <strong>{metrics_lr['MAE']:.2f} t/ha</strong>
            </div>
            <div style="font-size: 0.78rem; color: #6B7280; margin-top: 8px; border-top: 1px solid #F3F4F6; padding-top: 6px;">
                <em>Ordinary least squares additive linear model.</em>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # -----------------------------------------------------
    # Fertilizer Advisory Section
    # -----------------------------------------------------
    st.markdown('<div class="section-title">🧪 3. Soil Nutrient & Fertilizer Advisory</div>', unsafe_allow_html=True)
    advisory_notes = get_fertilizer_advisory(crop_type, nitrogen, phosphorus, potassium, soil_ph)

    st.markdown("""
    <div class="advisory-container">
        <div class="advisory-header">⚠️ Heuristic Nutrient Guidance & Advisory Notice</div>
        <div class="disclaimer-banner">
            <strong>Mandatory Academic Disclaimer:</strong> This section provides a basic educational heuristic advisory based on 
            general soil science thresholds. It is <strong>NOT a scientifically validated fertilizer prescription</strong> and 
            must never be used for real-world farming interventions without physical soil testing and official agronomic extension clearance.
        </div>
    """, unsafe_allow_html=True)

    adv_cols = st.columns(2)
    with adv_cols[0]:
        st.markdown(f"**Soil pH Assessment:**\n- {advisory_notes[0]}")
        st.markdown(f"**Nitrogen (N) Balance:**\n- {advisory_notes[1]}")
    with adv_cols[1]:
        st.markdown(f"**Phosphorus (P) Balance:**\n- {advisory_notes[2]}")
        st.markdown(f"**Potassium (K) Balance:**\n- {advisory_notes[3]}")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------------------------------------------------
# SECTION 2: MODEL PERFORMANCE & COMPARISON
# ---------------------------------------------------------
elif app_mode == "2. Model Performance & Comparison":
    st.markdown("""
    <div class="hero-header-box">
        <h1 class="hero-title">📊 Model Performance & Comparative Evidence</h1>
        <p class="hero-subtitle">Rigorous benchmark of Linear Regression and Random Forest Regressor on identical held-out test data (20% partition, n = 3,904).</p>
    </div>
    """, unsafe_allow_html=True)

    # Metric summary pill cards
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">R² Score (Variance Explained)</div>
            <div class="metric-pill-val">{metrics_rf['R2']:.4f}</div>
            <div class="metric-pill-sub">RF vs LR ({metrics_lr['R2']:.4f}) &bull; +{metrics_rf['R2']-metrics_lr['R2']:.4f}</div>
        </div>
        """, unsafe_allow_html=True)
    with p2:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Mean Absolute Error (MAE)</div>
            <div class="metric-pill-val">{metrics_rf['MAE']:.2f} t/ha</div>
            <div class="metric-pill-sub">56.3% lower error than LR ({metrics_lr['MAE']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    with p3:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Root Mean Squared Error (RMSE)</div>
            <div class="metric-pill-val">{metrics_rf['RMSE']:.2f} t/ha</div>
            <div class="metric-pill-sub">23.5% lower error than LR ({metrics_lr['RMSE']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)
    with p4:
        st.markdown(f"""
        <div class="metric-pill">
            <div class="metric-pill-title">Mean Squared Error (MSE)</div>
            <div class="metric-pill-val">{metrics_rf['MSE']:.2f}</div>
            <div class="metric-pill-sub">RF ({metrics_rf['MSE']:.2f}) vs LR ({metrics_lr['MSE']:.2f})</div>
        </div>
        """, unsafe_allow_html=True)

    st.write("")

    col_chart, col_rationale = st.columns([1.1, 1], gap="large")

    with col_chart:
        st.markdown('<div class="section-title">📈 Metric Comparison Chart</div>', unsafe_allow_html=True)
        metrics_df = pd.DataFrame({
            "Metric": ["MAE (t/ha)", "RMSE (t/ha)", "MSE", "R² Score"],
            "Linear Regression": [metrics_lr["MAE"], metrics_lr["RMSE"], metrics_lr["MSE"], metrics_lr["R2"]],
            "Random Forest": [metrics_rf["MAE"], metrics_rf["RMSE"], metrics_rf["MSE"], metrics_rf["R2"]]
        })

        fig_metrics = go.Figure(data=[
            go.Bar(
                name='Linear Regression (Baseline)',
                x=metrics_df["Metric"],
                y=metrics_df["Linear Regression"],
                marker_color='#93C5FD',
                text=[f"{v:.2f}" for v in metrics_df["Linear Regression"]],
                textposition='outside'
            ),
            go.Bar(
                name='Random Forest Regressor (Preferred)',
                x=metrics_df["Metric"],
                y=metrics_df["Random Forest"],
                marker_color='#2E7D32',
                text=[f"{v:.2f}" for v in metrics_df["Random Forest"]],
                textposition='outside'
            )
        ])
        fig_metrics.update_layout(
            barmode='group',
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937', family='sans-serif'),
            margin=dict(l=20, r=20, t=30, b=20),
            yaxis_title="Measured Metric Value",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_metrics, use_container_width=True)

    with col_rationale:
        st.markdown('<div class="section-title">🧠 Agronomic & ML Rationale</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #1E4620; margin-top: 0;">Why Random Forest Outperforms Linear Regression</h4>
            <ul style="color: #374151; font-size: 0.93rem; line-height: 1.6;">
                <li><strong>Non-Linear Diminishing Returns:</strong> Biological crop yield does not grow indefinitely with rainfall or temperature. Excess rainfall causes root waterlogging and excessive heat causes flower drop. Tree splits capture these plateau curves naturally.</li>
                <li><strong>Complex Nutrient Synergy:</strong> Soil pH dictates whether Nitrogen, Phosphorus, and Potassium are chemically bio-available to roots. Random Forest splits recursively on pH thresholds before testing NPK values, capturing multi-way interaction effects.</li>
                <li><strong>Zero Data Leakage:</strong> Both models were evaluated on the same 20% held-out test set with standard scaling and one-hot encoding fitted strictly on training data.</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    # Actual vs Predicted Scatter Visualization
    st.markdown("---")
    st.markdown('<div class="section-title">🎯 Actual vs. Predicted Yield on Held-Out Test Data</div>', unsafe_allow_html=True)

    if test_preds is not None:
        c_filter, _ = st.columns([1, 2])
        with c_filter:
            model_view = st.radio(
                "Display predictions for:",
                ["Random Forest Regressor", "Linear Regression", "Both Models"],
                horizontal=True
            )

        fig_scatter = go.Figure()
        max_val = max(test_preds["Actual_Yield"].max(), test_preds["Predicted_RF"].max(), test_preds["Predicted_LR"].max())

        # 45-degree ideal fit line
        fig_scatter.add_trace(go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#DC2626", dash="dash", width=2),
            name="Ideal 1:1 Prediction Line"
        ))

        if model_view in ["Random Forest Regressor", "Both Models"]:
            fig_scatter.add_trace(go.Scatter(
                x=test_preds["Actual_Yield"],
                y=test_preds["Predicted_RF"],
                mode="markers",
                marker=dict(color="#2E7D32", opacity=0.75, size=6),
                name="Random Forest Regressor",
                text=test_preds["crop_type"],
                hovertemplate="<b>Crop:</b> %{text}<br><b>Actual:</b> %{x:.2f} t/ha<br><b>Predicted:</b> %{y:.2f} t/ha<extra></extra>"
            ))

        if model_view in ["Linear Regression", "Both Models"]:
            fig_scatter.add_trace(go.Scatter(
                x=test_preds["Actual_Yield"],
                y=test_preds["Predicted_LR"],
                mode="markers",
                marker=dict(color="#2563EB", opacity=0.55, size=6),
                name="Linear Regression",
                text=test_preds["crop_type"],
                hovertemplate="<b>Crop:</b> %{text}<br><b>Actual:</b> %{x:.2f} t/ha<br><b>Predicted:</b> %{y:.2f} t/ha<extra></extra>"
            ))

        fig_scatter.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937'),
            xaxis=dict(title="Actual Historical Yield (Tonnes per Hectare)", gridcolor="#F3F4F6"),
            yaxis=dict(title="Predicted Yield (Tonnes per Hectare)", gridcolor="#F3F4F6"),
            height=460,
            margin=dict(l=20, r=20, t=30, b=20),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # -----------------------------------------------------
    # Yield Sensitivity Visualization (Requirement #4)
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown('<div class="section-title">🧪 Yield Sensitivity Simulation (Trained Model Response)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size: 0.90rem; color: #4B5563; margin-bottom: 12px;">
        <strong>Methodology Note:</strong> This chart calculates how the existing trained <code>Random Forest</code> model output changes when 
        sweeping one continuous variable across realistic bounds while holding all other inputs at their selected baseline values. 
        <strong>This illustrates model mathematical behavior, not empirical causal biological effects.</strong>
    </div>
    """, unsafe_allow_html=True)

    sc1, sc2 = st.columns([1, 2.5], gap="large")
    with sc1:
        sens_param = st.selectbox(
            "Select Variable to Sweep:",
            ["Annual Rainfall (mm)", "Temperature (°C)", "Relative Humidity (%)", "Soil pH", "Nitrogen (N)"]
        )
        sens_crop = st.selectbox("Select Crop for Simulation:", ["Rice", "Wheat", "Maize", "Cotton(lint)", "Sugarcane"])

    with sc2:
        # Define sweep ranges
        param_configs = {
            "Annual Rainfall (mm)": ("rainfall", np.linspace(300, 3000, 30)),
            "Temperature (°C)": ("temperature", np.linspace(12, 42, 30)),
            "Relative Humidity (%)": ("humidity", np.linspace(30, 95, 30)),
            "Soil pH": ("pH", np.linspace(4.5, 8.5, 30)),
            "Nitrogen (N)": ("N", np.linspace(20, 180, 30))
        }

        col_name, sweep_vals = param_configs[sens_param]
        
        # Base vector
        base_vector = {
            "rainfall": 1150.0,
            "temperature": 26.5,
            "humidity": 72.0,
            "pH": 6.5,
            "N": 78.0,
            "P": 35.0,
            "K": 28.0,
            "crop_type": sens_crop
        }

        sim_rows = []
        for val in sweep_vals:
            row = base_vector.copy()
            row[col_name] = val
            sim_rows.append(row)
        
        sim_df = pd.DataFrame(sim_rows)
        sim_preds = rf_model.predict(sim_df)

        fig_sens = px.line(
            x=sweep_vals,
            y=sim_preds,
            labels={"x": sens_param, "y": f"Predicted Yield ({yield_units})"},
            title=f"Predicted Yield Sensitivity to {sens_param} for {sens_crop}"
        )
        fig_sens.update_traces(line_color="#2E7D32", line_width=3)
        fig_sens.update_layout(
            plot_bgcolor='#FFFFFF',
            paper_bgcolor='#FFFFFF',
            font=dict(color='#1F2937'),
            height=340,
            margin=dict(l=20, r=20, t=40, b=20)
        )
        st.plotly_chart(fig_sens, use_container_width=True)

# ---------------------------------------------------------
# SECTION 3: DATASET PROVENANCE & LIMITATIONS
# ---------------------------------------------------------
elif app_mode == "3. Dataset Provenance & Limitations":
    st.markdown("""
    <div class="hero-header-box">
        <h1 class="hero-title">📚 Dataset Provenance, Architecture & Limitations</h1>
        <p class="hero-subtitle">Comprehensive academic documentation for Information Technology 5th-semester project evaluation and viva examination.</p>
    </div>
    """, unsafe_allow_html=True)

    tab_prov, tab_setup, tab_limits = st.tabs(["1. Dataset Provenance", "2. Installation & Run Guide", "3. Project Scope & Limitations"])

    with tab_prov:
        st.markdown('<div class="section-title">Dataset Origins and Schema</div>', unsafe_allow_html=True)
        st.markdown("""
        - **Source:** Kaggle Agricultural Benchmark Dataset (`anshumish/crop-yield-data-with-soil-and-weather-dataset`).
        - **Underlying Provenance:** Historical state and district statistics compiled by the **Ministry of Agriculture & Farmers Welfare, Government of India**, merged with regional India Meteorological Department (IMD) meteorological observations.
        - **Dataset Size:** 19,517 clean records across 54 major Indian crop varieties spanning 1997 through 2020.
        - **Target Variable:** `yield` (Historical Crop Yield).
        - **Target Units:** **Tonnes per Hectare (t/ha)** ($\text{Production in Tonnes} / \text{Cultivated Area in Hectares}$).
        """)

        st.write("")
        st.markdown("#### Input Feature Specifications:")
        features_table = pd.DataFrame([
            {"Feature": "rainfall", "Label": "Annual Rainfall", "Units": "Millimeters (mm)", "Domain": "Meteorology", "Type": "Continuous"},
            {"Feature": "temperature", "Label": "Average Temperature", "Units": "Degrees Celsius (°C)", "Domain": "Meteorology", "Type": "Continuous"},
            {"Feature": "humidity", "Label": "Relative Humidity", "Units": "Percentage (%)", "Domain": "Meteorology", "Type": "Continuous"},
            {"Feature": "pH", "Label": "Soil pH", "Units": "pH scale (3.5 - 9.5)", "Domain": "Soil Health", "Type": "Continuous"},
            {"Feature": "N", "Label": "Nitrogen", "Units": "kg/ha", "Domain": "Soil Health", "Type": "Continuous"},
            {"Feature": "P", "Label": "Phosphorus", "Units": "kg/ha", "Domain": "Soil Health", "Type": "Continuous"},
            {"Feature": "K", "Label": "Potassium", "Units": "kg/ha", "Domain": "Soil Health", "Type": "Continuous"},
            {"Feature": "crop_type", "Label": "Crop Variety", "Units": "54 categories", "Domain": "Agronomy", "Type": "Categorical (OHE)"}
        ])
        st.dataframe(features_table, use_container_width=True)

    with tab_setup:
        st.markdown('<div class="section-title">Exact Windows Execution Commands</div>', unsafe_allow_html=True)
        st.markdown("Run the following terminal commands to replicate or inspect the prototype:")
        
        st.code("""# 1. Open Windows PowerShell and navigate to workspace:
cd e:\\Yield-Predict

# 2. Install Python dependencies:
pip install -r requirements.txt

# 3. Execute automated pipeline test suite:
python test_pipeline.py

# 4. Retrain models and re-evaluate on test data:
python train_model.py

# 5. Launch the Streamlit dashboard:
streamlit run app.py
""", language="powershell")

    with tab_limits:
        st.markdown('<div class="section-title">Documented Project Limitations</div>', unsafe_allow_html=True)
        st.markdown("""
        <div class="custom-card">
            <h4 style="color: #92400E; margin-top: 0;">Academic Prototype Notice</h4>
            <p style="color: #4B5563; font-size: 0.93rem;">
                This system is an academic machine learning prototype designed for a college-level capstone viva. 
                It illustrates preprocessing, pipeline architecture, and comparative regression modeling. 
                It is <strong>not suitable for commercial agricultural decision-making</strong> without real-world agronomic validation.
            </p>
            <h4 style="color: #1E4620; margin-top: 14px;">Technical Limitations:</h4>
            <ol style="color: #374151; font-size: 0.91rem; line-height: 1.6;">
                <li><strong>Regional Soil Aggregation:</strong> Soil nutrient indicators reflect regional baseline testing surveys. Sub-field variances (micro-topography, drainage) cannot be resolved without field-level GPS soil core tests.</li>
                <li><strong>Exogenous Agronomic Factors:</strong> Yield variability driven by seed cultivars (hybrid vs traditional), pest outbreaks, micro-irrigation scheduling, and unseasonal weather disasters are not tracked in the tabular matrix.</li>
                <li><strong>Unit Boundary Handling:</strong> Coconut records were isolated from the continuous tonne-based regression benchmark to prevent mixing unit counts (thousand nuts/ha) with weight (tonnes/ha).</li>
                <li><strong>Heuristic Advisory:</strong> The fertilizer advisory is an educational rule-based heuristic and not a chemical prescription.</li>
            </ol>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #6B7280; font-size: 0.85rem; padding-bottom: 20px;'>"
    "Crop Yield Prediction using Weather and Soil Data &bull; Department of Information Technology &bull; 5th Semester Viva Demonstration"
    "</div>",
    unsafe_allow_html=True
)
