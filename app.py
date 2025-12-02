# ============================
# Player Market Value Predictor - FINAL WORKING APP
# ============================

import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

# -------------------------------------------
# 1️⃣ MUST BE FIRST STREAMLIT COMMAND
# -------------------------------------------
st.set_page_config(page_title="Player Market Value Predictor", layout="wide")

# -------------------------------------------
# 2️⃣ CONFIG
# -------------------------------------------
MODEL_PATH = "xgb_final_model (1).pkl"
# Main page setup
st.markdown(
    """
    <h1 style='
        font-size: 100px;
        text-align: center;
        font-weight: 900;
        color: #0b3d91;
        margin-top: -10px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    '>
        ⚽ Player Market Value Prediction Dashboard
    </h1>
    """,
    unsafe_allow_html=True
)
st.image("pic2.jpg", use_container_width=True)

# Features (ORDER MUST MATCH YOUR MODEL)
features = [
    'born', 'avgsentimentscore', 'positive', 'age', 'neutral', 'min',
    'avg_days_injured', 'total_days_injured', 'recent_injury_days',
    'pasprog', 'off', 'tklatt3rd', 'tkldef3rd', 'blocks', 'negative',
    'fld', 'carprgdist', 'sca', 'car3rd', 'rec', 'carries', '90s',
    'passhoatt', 'max_days_injured', 'toudefpen', 'pasblocks', 'toumid3rd'
]

# Default values
default_vals = {f: 0.0 for f in features}
default_vals.update({
    "age": 25,
    "min": 1800,
    "90s": 20,
    "pasprog": 5.0,
    "sca": 1.0,
    "rec": 10.0,
    "carprgdist": 100.0
})

# -------------------------------------------
# 3️⃣ LOAD MODEL
# -------------------------------------------
@st.cache_resource
def load_model():
    try:
        with open(MODEL_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Model loading failed: {e}")
        return None

model = load_model()
# -------------------------------------------
# 5️⃣ TITLE
# -------------------------------------------
#st.title("⚽ Player Market Value Prediction Dashboard")

# -------------------------------------------
# 6️⃣ LAYOUT (LEFT = Inputs, RIGHT = Charts)
# -------------------------------------------
left_col, right_col = st.columns([1.1, 1])

# -------------------------------------------
# 7️⃣ INPUT SECTION
# -------------------------------------------
with left_col:
    st.header("🔧 Player Inputs")

    n_per_row = 5
    inputs = {}

    for i in range(0, len(features), n_per_row):
        cols = st.columns(n_per_row)
        for j, key in enumerate(features[i:i+n_per_row]):
            with cols[j]:
                default_val = default_vals[key]

                val = st.number_input(
                    key,
                    value=float(default_val),
                    format="%.4f",
                    key=f"input_{key}"         # unique key
                )
                inputs[key] = val

    st.markdown("---")
    predict_btn = st.button("🔮 Predict Market Value")

# -------------------------------------------
# 8️⃣ RIGHT SIDE VISUAL PLACEHOLDERS
# -------------------------------------------
with right_col:
    st.header("📊 Visualizations")
    trend_plot = st.empty()
    radar_plot = st.empty()

# -------------------------------------------
# 9️⃣ PREDICTION & CHARTS
# -------------------------------------------
def plot_trend(df_row):
    fig = go.Figure(go.Bar(x=df_row.index, y=df_row.values))
    fig.update_layout(title="Player Feature Trend", xaxis_tickangle=-45, height=350)
    return fig

def plot_radar(df_row, n_features=8):
    labels = df_row.index[:n_features]
    values = df_row.values[:n_features]
    fig = go.Figure(go.Scatterpolar(r=values, theta=labels, fill="toself"))
    fig.update_layout(height=380, showlegend=False)
    return fig

if predict_btn:
    X = pd.DataFrame([[inputs[f] for f in features]], columns=features)

    if model:
        try:
            pred = model.predict(X)[0]
            st.success(f"Predicted Market Value: €{pred:,.2f}")

            trend_plot.plotly_chart(
                plot_trend(X.loc[0]),
                use_container_width=True
            )

            radar_plot.plotly_chart(
                plot_radar(X.loc[0]),
                use_container_width=True
            )

        except Exception as e:
            st.error(f"Prediction failed: {e}")

    else:
        st.error("Model not loaded!")

# -------------------------------------------
# 🔟 EXPORT INPUTS
# -------------------------------------------
st.markdown("---")
if st.button("Download Inputs as CSV"):
    df_export = pd.DataFrame([inputs])
    st.download_button(
        "Download CSV File",
        df_export.to_csv(index=False),
        file_name="player_inputs.csv",
        mime="text/csv"
    )