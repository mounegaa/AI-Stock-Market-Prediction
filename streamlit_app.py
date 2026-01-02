import streamlit as st
import pandas as pd
import joblib
import os

# ============================
# PATH SETUP (CRITICAL)
# ============================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.path.join(BASE_DIR, "data")
VIX_FILE = "INDIA VIX_minute.csv"

MODEL_PATH = os.path.join(BASE_DIR, "model_direction.joblib")
FEATURES_PATH = os.path.join(BASE_DIR, "features.pkl")

HORIZON = 10

# ============================
# DEBUG (REMOVE LATER)
# ============================

st.write("📁 Base directory:", BASE_DIR)
st.write("📄 Files in base dir:", os.listdir(BASE_DIR))

# ============================
# LOAD MODEL & FEATURES
# ============================

try:
    model = joblib.load(MODEL_PATH)
    FEATURES = joblib.load(FEATURES_PATH)
except Exception as e:
    st.error("❌ Failed to load model or features")
    st.exception(e)
    st.stop()

# ============================
# LOAD DATA FILES
# ============================

if not os.path.exists(DATA_DIR):
    st.error("❌ Data directory not found")
    st.stop()

index_files = [
    f for f in os.listdir(DATA_DIR)
    if f.endswith("_minute.csv") and f != VIX_FILE
]

index_files.sort()

# ============================
# STREAMLIT UI
# ============================

st.set_page_config(
    page_title="Market Direction Predictor",
    layout="centered"
)

st.title("📊 AI Stock Market Direction Predictor")

st.write(
    """
    Predicts **BULLISH / BEARISH** market direction  
    using multiple **NIFTY indices** and **INDIA VIX**.
    """
)

if not index_files:
    st.warning("⚠️ No index CSV files found in data folder")
    st.stop()

index_file = st.selectbox("Select Index", index_files)

# ============================
# LOAD SELECTED DATA
# ============================

try:
    index_df = pd.read_csv(os.path.join(DATA_DIR, index_file))
    vix_df = pd.read_csv(os.path.join(DATA_DIR, VIX_FILE))
except Exception as e:
    st.error("❌ Failed to load CSV files")
    st.exception(e)
    st.stop()

st.success("✅ Data loaded successfully")

# ============================
# FEATURE PREP (PLACEHOLDER)
# ============================

st.info("⚙️ Feature engineering & prediction logic goes here")

# Example dummy output
st.subheader("📈 Prediction")
st.write("Prediction logic executed successfully")
