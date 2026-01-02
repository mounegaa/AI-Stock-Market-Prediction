import streamlit as st
import pandas as pd
import joblib
import os

# ============================
# CONFIG
# ============================

DATA_DIR = "data"
VIX_FILE = "INDIA VIX_minute.csv"
MODEL_PATH = "model_direction.joblib"
FEATURES_PATH = "features.pkl"
HORIZON = 10

# ============================
# LOAD MODEL & FEATURES
# ============================

model = joblib.load(MODEL_PATH)
FEATURES = joblib.load(FEATURES_PATH)

# ============================
# LOAD FILE LIST
# ============================

index_files = sorted([
    f for f in os.listdir(DATA_DIR)
    if f.endswith("_minute.csv") and f != VIX_FILE
])

# ============================
# STREAMLIT UI
# ============================

st.set_page_config(page_title="Market Direction Predictor", layout="centered")
st.title("📈 Market Direction Predictor")

st.write(
    "Predicts **next 10-minute market direction** using "
    "**historical index data + INDIA VIX**."
)

index_file = st.selectbox("Select Index", index_files)

# ============================
# LOAD DATA
# ============================

@st.cache_data
def load_data(index_file):
    df = pd.read_csv(os.path.join(DATA_DIR, index_file))
    df.columns = df.columns.str.lower()
    df['datetime'] = pd.to_datetime(df['date'])
    df = df.sort_values('datetime')

    vix = pd.read_csv(os.path.join(DATA_DIR, VIX_FILE))
    vix.columns = vix.columns.str.lower()
    vix['datetime'] = pd.to_datetime(vix['date'])
    vix = vix.sort_values('datetime')[['datetime', 'close']]
    vix.rename(columns={'close': 'vix'}, inplace=True)

    df = pd.merge_asof(df, vix, on='datetime', direction='backward')
    return df

df = load_data(index_file)

# ============================
# FEATURE ENGINEERING
# ============================

df['ret_1']  = df['close'].pct_change(1)
df['ret_5']  = df['close'].pct_change(5)
df['ret_10'] = df['close'].pct_change(10)

df['vol_10'] = df['ret_1'].rolling(10).std()
df['vol_30'] = df['ret_1'].rolling(30).std()

df['mean_60'] = df['close'].rolling(60).mean()
df['std_60']  = df['close'].rolling(60).std()
df['zscore_60'] = (df['close'] - df['mean_60']) / df['std_60']

hl = df['high'] - df['low']
hc = (df['high'] - df['close'].shift()).abs()
lc = (df['low'] - df['close'].shift()).abs()
df['tr'] = pd.concat([hl, hc, lc], axis=1).max(axis=1)
df['atr_14'] = df['tr'].rolling(14).mean()

df['vix_change'] = df['vix'].pct_change()
df['vix_ma'] = df['vix'].rolling(10).mean()

df = df.dropna(subset=FEATURES)

latest = df.iloc[-1:]

# ============================
# PREDICTION
# ============================

if st.button("🔍 Predict Next Move"):

    X = latest[FEATURES]
    prob_bear, prob_bull = model.predict_proba(X)[0]

    if prob_bull > prob_bear:
        signal = "🟢 BULLISH"
        confidence = prob_bull
        color = "green"
    else:
        signal = "🔴 BEARISH"
        confidence = prob_bear
        color = "red"

    st.subheader("Prediction")
    st.markdown(f"<h2 style='color:{color}'>{signal}</h2>", unsafe_allow_html=True)

    st.subheader("Confidence")
    st.progress(float(confidence))
    st.write(f"Confidence: **{confidence:.2f}**")

    st.caption(f"Bullish: {prob_bull:.2f} | Bearish: {prob_bear:.2f}")

    st.write("🕒 Based on latest available candle")
