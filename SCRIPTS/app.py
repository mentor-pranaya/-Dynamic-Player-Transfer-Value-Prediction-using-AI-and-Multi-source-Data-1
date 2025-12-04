# app_player_value.py
import streamlit as st
st.set_page_config(page_title="Player Transfer Value", page_icon="⚽", layout="wide")

import os
import math
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
#ui background
st.markdown(
    """
    <style>
    /* full-page background */
    .stApp {
        background: radial-gradient(circle at 0% 0%, #e0f2ff 0, #ffffff 45%),
                    radial-gradient(circle at 100% 100%, #ffe4f3 0, #ffffff 40%);
    }

    /* main Streamlit block container as card */
    .block-container {
        padding-top: 2rem !important;
        padding-bottom: 2rem !important;
        max-width: 1150px !important;
        margin: 0 auto !important;
        background: rgba(255, 255, 255, 0.96);
        border-radius: 18px;
        box-shadow: 0 18px 45px rgba(15, 23, 42, 0.18);
    }

    .section-title {
        font-size: 1.2rem;
        font-weight: 700;
        margin-top: 1.5rem;
        margin-bottom: 0.75rem;
    }

    .sub-card {
        padding: 1rem 1.25rem;
        border-radius: 14px;
        background-color: #f9fafb;
        border: 1px solid rgba(148, 163, 184, 0.35);
        margin-bottom: 1rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# ----PATH-----
CSV_PATH = r"D:\INFOSYS\DATA\processed\final_afe_dataset.csv"
MODEL_PATH = r"D:\INFOSYS\scripts\xgb_tuned_final.pkl"

TEXT_COL = "player_name"   # column for player names
TARGET_COL = "log_value"   # model predicts this


# ---------- LOAD DATA & MODEL ----------
@st.cache_data
def load_data(path):
    if not os.path.exists(path):
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        df = pd.read_csv(path, encoding="latin1")
    return df


@st.cache_resource
def load_model(path):
    if not os.path.exists(path):
        return None
    try:
        return joblib.load(path)
    except Exception:
        return None


df = load_data(CSV_PATH)
model = load_model(MODEL_PATH)

if df is None:
    st.error(f"Could not find dataset at {CSV_PATH}")
    st.stop()

# quick check
if TEXT_COL not in df.columns:
    st.error(f"Column '{TEXT_COL}' not found in dataset. Check your CSV.")
    st.stop()

# --------- AGE HANDLING (fix the weird values) ----------
AGE_COL = "Age" if "Age" in df.columns else None
AGE_IS_SCALED = False
AGE_MEAN = 25.0
AGE_STD = 6.0

if AGE_COL is not None:
    try:
        age_series = pd.to_numeric(df[AGE_COL], errors="coerce")
        age_series = age_series[np.isfinite(age_series)]
        if len(age_series) > 10:
            m = float(age_series.mean())
            s = float(age_series.std())
            # detect standardized z-score age
            if abs(m) < 0.6 and 0.7 < s < 1.4:
                AGE_IS_SCALED = True
                AGE_MEAN = 25.0    
                AGE_STD = 6.0
    except Exception:
        pass


def get_display_age(row):
    """Return age as a nice integer string like '26'."""
    if AGE_COL is None:
        return "N/A"
    try:
        raw = float(row.get(AGE_COL))
    except Exception:
        return "N/A"

    # if not scaled and in normal range, use directly
    if not AGE_IS_SCALED and 10 <= raw <= 80:
        return str(int(round(raw)))

    # if scaled, map back to approximate years
    if AGE_IS_SCALED:
        est = raw * AGE_STD + AGE_MEAN
        return f"{int(round(est))}"

    # fallback
    return str(int(round(raw)))


# ---------- PSEUDO MARKET VALUE SCALING ----------
MIN_PV, MAX_PV = 0.0, 1.0
if TARGET_COL in df.columns:
    try:
        all_log = pd.to_numeric(df[TARGET_COL], errors="coerce")
        all_log = all_log[np.isfinite(all_log)]
        if len(all_log) > 0:
            pseudo = np.exp(all_log)
            MIN_PV, MAX_PV = float(pseudo.min()), float(pseudo.max())
    except Exception:
        pass


def log_to_million(log_val):
    try:
        pv = math.exp(float(log_val))
    except Exception:
        pv = 1.0
    if MAX_PV - MIN_PV <= 0:
        return 1.0
    norm = (pv - MIN_PV) / (MAX_PV - MIN_PV)
    # map into roughly 1–150M
    return 1.0 + norm * (150.0 - 1.0)


# ---------- PLOTS ----------
def radar_plot(row):
    labels = [
        "score",
        "sentiment_score",
        "vader_compound_mean",
        "tb_polarity_mean",
        "kw_score_mean",
        "popularity_total",
    ]
    vals = []
    for col in labels:
        try:
            vals.append(float(row.get(col, 0.0)))
        except Exception:
            vals.append(0.0)

    import math as _math
    N = len(labels)
    angles = np.linspace(0, 2 * _math.pi, N, endpoint=False)
    vals = vals + vals[:1]
    angles = np.concatenate([angles, [angles[0]]])

    fig, ax = plt.subplots(figsize=(5, 4), subplot_kw={"polar": True})
    fig.patch.set_alpha(0.0)
    ax.set_facecolor("#f9fafb")
    ax.plot(angles, vals, marker="o")
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_title("Player Attribute Radar", fontsize=11)
    return fig


def similar_players_chart(df, row, top_n=8):
    league_col = "League" if "League" in df.columns else "League "
    if league_col in df.columns:
        same = df[df[league_col] == row.get(league_col)]
    else:
        same = df.copy()

    sort_col = "log_value" if "log_value" in same.columns else "score"
    same = same.copy()
    same = same.sort_values(sort_col, ascending=False).head(top_n)

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.barh(same[TEXT_COL].astype(str), same[sort_col])
    ax.set_xlabel(sort_col)
    ax.invert_yaxis()
    ax.set_title("Top Similar Players by Value (same league)")
    return fig


def value_trend_chart(df, player_name):
    sub = df[df[TEXT_COL] == player_name].copy()
    if sub.shape[0] <= 1:
        return None
    if "season_id" in sub.columns:
        sub = sub.sort_values("season_id")
        x = sub["season_id"]
    elif "season_name" in sub.columns:
        x = sub["season_name"]
    else:
        x = range(1, sub.shape[0] + 1)

    if TARGET_COL in sub.columns:
        y = np.exp(sub[TARGET_COL])
    else:
        return None

    fig, ax = plt.subplots(figsize=(7, 3))
    ax.plot(x, y, marker="o")
    ax.set_title("Value Trend (player history)")
    ax.set_xlabel("Season / record")
    ax.set_ylabel("Relative value")
    return fig


# ---------- UI ----------
st.markdown(
    "<h1 style='font-weight:700; margin-bottom:0.5rem;'>⚽ Player Transfer Value Prediction System</h1>",
    unsafe_allow_html=True,
)
st.write("Select a player to see their details, model prediction and visual insights.")

# Dropdown of players
all_players = sorted(df[TEXT_COL].astype(str).unique())
selected_player = st.selectbox("Choose a player", all_players)

# Filter to rows for that player 
player_rows = df[df[TEXT_COL] == selected_player]
row = player_rows.iloc[0]  # use first row for details/prediction

# ----- INFO TABLE & METRICS -----
st.markdown("<div class='section-title'>📌 Player Overview</div>", unsafe_allow_html=True)

age_display = get_display_age(row)
club = row.get("Club", row.get("club", "N/A"))
league = row.get("League", row.get("League Country", "N/A"))
position = row.get("Position", row.get("position", "N/A"))

c1, c2, c3, c4 = st.columns(4)
with c1:
    st.metric("Player", selected_player)
with c2:
    st.metric("Age", age_display)
with c3:
    st.metric("Club", str(club))
with c4:
    st.metric("League", str(league))

# small table of extra info - only useful things
pretty_stats = {}
pretty_stats["Player Nationality"] = row.get("Player Nationality", "N/A")
pretty_stats["Transfer Type"]      = row.get("Transfer Type", "N/A")
pretty_stats["Club"]               = club
pretty_stats["League"]             = league
pretty_stats["Position"]           = position

if "Fee" in df.columns:
    try:
        pretty_stats["Transfer Fee (scaled)"] = round(float(row.get("Fee", 0)), 4)
    except Exception:
        pass

if "score" in df.columns:
    try:
        pretty_stats["Performance score"] = round(float(row.get("score", 0)), 4)
    except Exception:
        pass

if "sentiment_score" in df.columns:
    try:
        s_val = float(row.get("sentiment_score", 0) or 0)
        if abs(s_val) > 0.001:
            pretty_stats["Sentiment score"] = round(s_val, 4)
    except Exception:
        pass

if "popularity_total" in df.columns:
    try:
        pop_val = float(row.get("popularity_total", 0) or 0)
        if pop_val != 0:
            pretty_stats["Social popularity (count)"] = int(pop_val)
    except Exception:
        pass

stats_df = pd.DataFrame.from_dict(pretty_stats, orient="index", columns=["Value"])

#  make Value column 
def _fmt_val(v):
    try:
        f = float(v)
        return f"{f:.4f}"
    except Exception:
        return str(v)

stats_df["Value"] = stats_df["Value"].apply(_fmt_val)

st.markdown("<div class='sub-card'>", unsafe_allow_html=True)
st.table(stats_df)
st.markdown("</div>", unsafe_allow_html=True)

# ----- PREDICTION -----
st.markdown("<div class='section-title'>💰 Predicted Market Value</div>", unsafe_allow_html=True)

if model is not None:
    if hasattr(model, "feature_names_in_"):
        feat_names = list(model.feature_names_in_)
    else:
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        feat_names = [c for c in num_cols if c not in [TARGET_COL, "row_index"]]

    x_dict = {}
    for f in feat_names:
        x_dict[f] = row.get(f, 0)
    X = pd.DataFrame([x_dict], columns=feat_names)

    try:
        pred_log = float(model.predict(X)[0])
    except Exception:
        pred_log = float(row.get(TARGET_COL, 0.0))
else:
    pred_log = float(row.get(TARGET_COL, 0.0))

pred_value_m = log_to_million(pred_log)

m1, m2 = st.columns(2)
with m1:
    st.metric("Model output (log scale)", f"{pred_log:.4f}")
with m2:
    st.success(f"Predicted Market Value: € {pred_value_m:,.2f} M")

# ----- CHARTS -----
st.markdown("<div class='section-title'>📊 Visual Insights</div>", unsafe_allow_html=True)

col_left, col_right = st.columns(2)

with col_left:
    fig_radar = radar_plot(row)
    st.pyplot(fig_radar)

with col_right:
    fig_sim = similar_players_chart(df, row, top_n=8)
    st.pyplot(fig_sim)

trend_fig = value_trend_chart(df, selected_player)
if trend_fig is not None:
    st.markdown("<div class='section-title'>📈 Value Trend (player history)</div>", unsafe_allow_html=True)
    st.pyplot(trend_fig)

# download one-row + prediction
out = row.to_dict()
out["pred_log_value"] = pred_log
out["pred_market_value_million"] = pred_value_m
out_df = pd.DataFrame([out])
csv_bytes = out_df.to_csv(index=False).encode("utf-8")
st.download_button(
    "Download this player's data + prediction (CSV)",
    data=csv_bytes,
    file_name=f"{selected_player.replace(' ', '_')}_prediction.csv",
)
