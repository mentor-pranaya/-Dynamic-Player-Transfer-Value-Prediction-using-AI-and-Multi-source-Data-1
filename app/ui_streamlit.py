# app/ui_streamlit.py
"""
TransferIQ — All-in-One Streamlit App (complete)
- Home
- Predict Player Value (fuzzy search + suggestions + charts)
- Player Performance Lookup (trend + compare)
- Upload dataset or use demo
- Charts: radar, feature importance, league distribution, similar players,
  attribute comparison, performance comparison
Notes:
 - Place your model at models/final_xgb_model.pkl (joblib)
 - Place datasets under data/processed/ or upload in UI
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from difflib import get_close_matches, SequenceMatcher
from matplotlib import rcParams

rcParams.update({'figure.autolayout': True})
st.set_page_config(page_title="TransferIQ", layout="wide", page_icon="⚽")

# ---------------------------
# Paths (change if needed)
# ---------------------------
MODEL_PATH = "models/final_xgb_model.pkl"
PLAYERS_CSV = "data/processed/realistic_football_data.csv"
PERF_CSV = "data/processed/performance.csv"

# ---------------------------
# Helper: load data with fallback demo
# ---------------------------
@st.cache_data(show_spinner=False)
def load_players(path):
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            if "name" not in df.columns:
                st.warning(f"{path} found but 'name' column missing.")
            return df
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    # demo fallback
    demo = pd.DataFrame([
        {"name":"Lionel Messi","age":36,"position":"RW","club":"PSG","league":"Ligue 1",
         "overall":93,"potential":93,"pace":85,"shooting":92,"passing":91,"dribbling":95,"physical":65,
         "matches":30,"goals":20,"assists":12,"minutes":2400,"injury_count":0,"value":60.0},
        {"name":"Kylian Mbappé","age":24,"position":"ST","club":"PSG","league":"Ligue 1",
         "overall":91,"potential":95,"pace":97,"shooting":90,"passing":78,"dribbling":89,"physical":82,
         "matches":28,"goals":25,"assists":8,"minutes":2300,"injury_count":1,"value":140.0},
        {"name":"Erling Haaland","age":24,"position":"ST","club":"Man City","league":"Premier League",
         "overall":90,"potential":95,"pace":89,"shooting":93,"passing":65,"dribbling":78,"physical":88,
         "matches":26,"goals":27,"assists":6,"minutes":2100,"injury_count":1,"value":120.0},
        {"name":"Cristiano Ronaldo","age":40,"position":"ST","club":"Al Nassr","league":"Saudi Pro",
         "overall":89,"potential":89,"pace":78,"shooting":91,"passing":82,"dribbling":80,"physical":85,
         "matches":30,"goals":18,"assists":5,"minutes":2500,"injury_count":0,"value":20.0},
    ])
    return demo

@st.cache_data(show_spinner=False)
def load_performance(path):
    if os.path.exists(path):
        try:
            return pd.read_csv(path)
        except Exception as e:
            st.warning(f"Failed to load {path}: {e}")
    # demo perf
    rows = []
    for name in ["Lionel Messi","Kylian Mbappé","Erling Haaland","Cristiano Ronaldo"]:
        for m in range(1,13):
            rows.append({"name":name,"match":m,"rating":float(np.clip(6 + np.random.randn()*0.5 + (m%5)/5.0,5,9))})
    return pd.DataFrame(rows)

@st.cache_resource(show_spinner=False)
def load_model(path):
    if os.path.exists(path):
        try:
            m = joblib.load(path)
            return m
        except Exception as e:
            st.warning("Model file found but failed to load: " + str(e))
            return None
    return None

# ---------------------------
# String matching utilities (strong fuzzy)
# ---------------------------
def seq_ratio(a, b):
    return SequenceMatcher(None, a, b).ratio()

def find_player(name, df, cutoff_seq=0.45, cutoff_get=0.3, max_matches=6):
    """
    Returns (exact_df, suggestions_df)
    - exact_df: DataFrame with exact (case-insensitive) matches (can be empty)
    - suggestions_df: DataFrame with close matches (can be None)
    Uses SequenceMatcher ratio plus get_close_matches fallback.
    """
    name_clean = name.strip().lower()
    if name_clean == "":
        return pd.DataFrame(), None

    # exact (case-insensitive)
    exact = df[df["name"].str.lower() == name_clean]
    if not exact.empty:
        return exact, None

    # SequenceMatcher across all names (tolerant)
    temp = df.copy()
    temp["__ratio__"] = temp["name"].str.lower().apply(lambda x: seq_ratio(x, name_clean))
    strong = temp[temp["__ratio__"] >= cutoff_seq].sort_values("__ratio__", ascending=False)
    if not strong.empty:
        return pd.DataFrame(), strong.drop(columns="__ratio__").head(max_matches)[["name","club","league","position","value"]]

    # get_close_matches fallback (lower cutoff to be tolerant)
    all_names = df["name"].str.lower().tolist()
    close = get_close_matches(name_clean, all_names, n=max_matches, cutoff=cutoff_get)
    if close:
        suggestions = df[df["name"].str.lower().isin(close)].drop_duplicates(subset=["name"])
        return pd.DataFrame(), suggestions[["name","club","league","position","value"]].head(max_matches)

    return pd.DataFrame(), None

# ---------------------------
# Chart functions
# ---------------------------
def make_radar(player_row):
    labels = ["overall","potential","pace","shooting","passing","dribbling","physical"]
    vals = [float(player_row.get(l, 0) or 0) for l in labels]
    angles = np.linspace(0, 2*np.pi, len(labels), endpoint=False).tolist()
    vals = vals + vals[:1]
    angles = angles + angles[:1]
    fig, ax = plt.subplots(figsize=(5,5), subplot_kw=dict(polar=True))
    ax.plot(angles, vals, 'o-', linewidth=2)
    ax.fill(angles, vals, alpha=0.25)
    ax.set_thetagrids(np.degrees(angles[:-1]), labels)
    ax.set_ylim(0,100)
    return fig

def feature_importance_plot(model, max_features=10):
    try:
        import xgboost as xgb
        fig, ax = plt.subplots(figsize=(6,4))
        xgb.plot_importance(model, ax=ax, max_num_features=max_features)
        return fig
    except Exception:
        fig, ax = plt.subplots(figsize=(6,4))
        ax.text(0.5,0.5,"Feature importance unavailable", ha="center")
        ax.axis('off')
        return fig

def similar_players_bar(df, player_row, top_n=12):
    pos = player_row.get("position", None)
    if pos and "position" in df.columns:
        cands = df[df["position"].str.lower()==str(pos).lower()].nlargest(top_n, "value")
    else:
        cands = df.nlargest(top_n, "value")
    if cands.empty:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"No similar players found", ha="center")
        ax.axis("off")
        return fig
    fig, ax = plt.subplots(figsize=(10,3))
    ax.bar(cands["name"].astype(str), cands["value"])
    ax.set_xticklabels(cands["name"].astype(str), rotation=45, ha='right')
    ax.set_ylabel("Market Value (M €)")
    return fig

def league_value_hbar(df):
    if "league" in df.columns and "value" in df.columns:
        s = df.groupby("league")["value"].mean().sort_values()
        fig, ax = plt.subplots(figsize=(8,4))
        s.plot(kind="barh", ax=ax)
        ax.set_xlabel("Average Market Value (M €)")
        return fig
    else:
        fig, ax = plt.subplots()
        ax.text(0.5,0.5,"League/value columns not found", ha='center')
        ax.axis('off')
        return fig

def compare_player_attributes(df, player_row, top_n=4):
    pos = player_row.get("position", None)
    if pos and "position" in df.columns:
        cands = df[df["position"].str.lower()==str(pos).lower()].nlargest(top_n+1, "value")
    else:
        cands = df.nlargest(top_n+1, "value")
    cands = cands[cands["name"] != player_row.get("name", "")].head(top_n)
    players = [player_row.get("name")] + cands["name"].tolist()
    attributes = ["overall","potential","pace","shooting","passing","dribbling","physical"]
    data = { "Player": players }
    for attr in attributes:
        vals = [player_row.get(attr, 0)] + list(cands.get(attr, 0).fillna(0))
        data[attr] = vals
    comp_df = pd.DataFrame(data).set_index("Player")
    fig, ax = plt.subplots(figsize=(12,5))
    comp_df.plot(kind="bar", ax=ax)
    ax.set_ylabel("Rating")
    ax.set_title(f"Attribute Comparison: {player_row.get('name')} vs Others")
    ax.legend(loc='upper left', bbox_to_anchor=(1,1))
    return fig

def compare_performance_trend(perf_df, player_name, top_players=3):
    pf = perf_df[perf_df["name"].str.lower() == player_name.lower()]
    if pf.empty:
        return None
    others = [n for n in perf_df["name"].unique() if n.lower() != player_name.lower()][:top_players]
    fig, ax = plt.subplots(figsize=(10,4))
    ax.plot(pf["match"], pf["rating"], label=player_name, linewidth=3, marker='o')
    for p in others:
        pf2 = perf_df[perf_df["name"]==p]
        if not pf2.empty:
            ax.plot(pf2["match"], pf2["rating"], label=p, alpha=0.5)
    ax.set_title(f"Performance Trend Comparison: {player_name}")
    ax.set_xlabel("Match")
    ax.set_ylabel("Rating")
    ax.legend()
    return fig

# ---------------------------
# Prediction helper
# ---------------------------
def predict_from_features(model, df_row):
    """
    model: loaded model or None
    df_row: dict-like or pd.Series
    returns predicted market value (float, in M €)
    """
    if model is None:
        # fallback heuristic
        overall = df_row.get("overall", 60)
        potential = df_row.get("potential", overall)
        goals = df_row.get("goals", 0)
        matches = df_row.get("matches", 1) or 1
        gpm = goals / matches if matches>0 else 0
        age = df_row.get("age", 25)
        base = overall*0.6 + potential*0.3 + gpm*10
        age_factor = 1.0 if age<=28 else max(0.5, 1 - (age-28)*0.03)
        return round(base * age_factor / 1.5, 2)
    else:
        try:
            if hasattr(model, "feature_names_in_"):
                features = list(model.feature_names_in_)
                X = pd.DataFrame([{f: df_row.get(f, 0) for f in features}])
            else:
                numeric = {k:v for k,v in dict(df_row).items() if isinstance(v,(int,float,np.number))}
                X = pd.DataFrame([numeric])
            pred = model.predict(X)[0]
            return float(pred)
        except Exception:
            return predict_from_features(None, df_row)

# ---------------------------
# Load resources
# ---------------------------
players_df = load_players(PLAYERS_CSV)
perf_df = load_performance(PERF_CSV)
model = load_model(MODEL_PATH)

# Inform user if session lost previous uploads
if (not os.path.exists(PLAYERS_CSV)) or (not os.path.exists(PERF_CSV)):
    st.info("If your original uploaded files are not present in this session, re-upload them on the Upload Dataset page or place them under data/processed/ to use your exact files.")

# ---------------------------
# Sidebar & Navigation
# ---------------------------
st.sidebar.title("TransferIQ")
page = st.sidebar.radio("Navigation", ["Home","Predict Player Value","Player Performance Lookup","Upload Dataset"])

# ---------------------------
# HOME
# ---------------------------
if page == "Home":
    st.title("⚽ TransferIQ — AI Powered Player Market Value Prediction")
    st.markdown(
        """
        **TransferIQ** predicts football player market values using machine learning.
        Use the sidebar to navigate:
        - **Predict Player Value** — type a player name and get a realistic prediction (auto-filled).
        - **Player Performance Lookup** — view match-by-match ratings/trends.
        - **Upload Dataset** — upload your CSV or use the demo dataset.
        """
    )
    left, right = st.columns([2,1])
    with left:
        st.subheader("Quick Preview (Top players)")
        st.dataframe(players_df.head(10), use_container_width=True)
    with right:
        st.subheader("Status")
        st.write("Model loaded:", "✅ Yes" if model is not None else "⚠️ No (fallback used)")
        st.write("Players in dataset:", len(players_df))
        st.write("Performance rows:", len(perf_df))

# ---------------------------
# PREDICT PLAYER VALUE
# ---------------------------
elif page == "Predict Player Value":
    st.title("🔮 Predict Player Market Value")
    st.write("Type a player's full name (case-insensitive). Typos are accepted — app will suggest close matches.")

    name_input = st.text_input("Player name (e.g., Lionel Messi)")
    match_cutoff = st.slider("Fuzzy sensitivity (lower = more tolerant)", min_value=0.2, max_value=0.8, value=0.45, step=0.05)
    if st.button("Fetch & Predict"):
        if not name_input.strip():
            st.warning("Please enter a player name.")
        else:
            exact_df, suggestions = find_player(name_input, players_df, cutoff_seq=match_cutoff, cutoff_get=0.25)
            if not exact_df.empty:
                player = exact_df.iloc[0].to_dict()
                st.success(f"Found player: {player.get('name')}")
            else:
                if suggestions is not None and not suggestions.empty:
                    st.warning("Exact match not found. Did you mean one of these?")
                    st.table(suggestions.reset_index(drop=True))
                    choice = st.selectbox("Pick suggestion to continue", suggestions["name"].tolist())
                    if choice:
                        player = suggestions[suggestions["name"]==choice].iloc[0].to_dict()
                        st.success(f"Using: {player.get('name')}")
                    else:
                        st.stop()
                else:
                    st.error("No similar player found. Try different spelling or upload dataset that contains the player.")
                    st.stop()

            # Display summary and metrics
            st.subheader("Player Summary")
            cols = st.columns(3)
            cols[0].metric("Age", player.get("age", "N/A"))
            cols[1].metric("Position", player.get("position", "N/A"))
            cols[2].metric("Club", player.get("club", "N/A"))

            display_keys = ["name","age","position","club","league","matches","goals","assists","minutes","injury_count","value"]
            display_present = [k for k in display_keys if k in player]
            st.table(pd.DataFrame([ {k:player.get(k, '') for k in display_present} ]).T.rename(columns={0:"Value"}))

            # Predict
            pred_value = predict_from_features(model, player)
            st.header(f"💰 Predicted Market Value: € {pred_value:,.2f} M")

            # Visual insights
            st.markdown("### Visual Insights")
            a,b = st.columns([1,1])
            with a:
                st.subheader("🌀 Attribute Radar")
                fig_radar = make_radar(player)
                st.pyplot(fig_radar)
            with b:
                st.subheader("🔥 Feature Importance")
                fig_fi = feature_importance_plot(model)
                st.pyplot(fig_fi)

            st.subheader("📊 Similar Players (by position)")
            fig_sim = similar_players_bar(players_df, player)
            st.pyplot(fig_sim)

            st.subheader("🌍 Average Market Value per League")
            fig_league = league_value_hbar(players_df)
            st.pyplot(fig_league)

            st.subheader("📊 Attribute Comparison (vs similar players)")
            fig_comp = compare_player_attributes(players_df, player, top_n=4)
            st.pyplot(fig_comp)

            st.subheader("📈 Performance Comparison")
            perf_fig = compare_performance_trend(perf_df, player.get("name",""), top_players=3)
            if perf_fig:
                st.pyplot(perf_fig)
            else:
                st.info("No match-level performance data available for this player.")

# ---------------------------
# PERFORMANCE LOOKUP
# ---------------------------
elif page == "Player Performance Lookup":
    st.title("📈 Player Performance Lookup")
    q = st.text_input("Enter player name for performance trend", "")
    if st.button("Load Performance"):
        if not q.strip():
            st.warning("Enter a name")
        else:
            pf = perf_df[perf_df["name"].str.lower() == q.strip().lower()]
            if pf.empty:
                _, suggestions = find_player(q, players_df, cutoff_seq=0.45)
                if suggestions is not None:
                    st.info("No match in performance table. But found player(s) in players dataset:")
                    st.table(suggestions[["name","club","league","position"]])
                else:
                    st.error("No performance records for this player in performance dataset.")
            else:
                st.dataframe(pf)
                fig, ax = plt.subplots(figsize=(10,4))
                sns.lineplot(x="match", y="rating", data=pf, marker="o", ax=ax)
                ax.set_title(f"Performance Trend: {q.title()}")
                ax.set_xlabel("Match")
                ax.set_ylabel("Rating")
                st.pyplot(fig)

# ---------------------------
# UPLOAD DATASET
# ---------------------------
elif page == "Upload Dataset":
    st.title("📂 Upload custom dataset or load demo")
    st.write("Upload a players CSV (columns: name, age, position, club, league, overall, potential, pace, shooting, passing, dribbling, physical, matches, goals, assists, minutes, injury_count, value)")
    uploaded = st.file_uploader("Upload players CSV", type=["csv"])
    uploaded_perf = st.file_uploader("Upload performance CSV", type=["csv"], key="perf")

    col1, col2 = st.columns(2)
    with col1:
        if st.button("Load demo dataset"):
            players_df = load_players(PLAYERS_CSV)
            perf_df = load_performance(PERF_CSV)
            st.success("Demo dataset loaded (in-memory).")
            st.dataframe(players_df.head(10))
    if uploaded:
        try:
            df_new = pd.read_csv(uploaded)
            st.session_state["players_df_uploaded"] = df_new
            players_df = df_new
            st.success("Players file loaded into session (use Predict page now).")
            st.dataframe(df_new.head(10))
        except Exception as e:
            st.error("Failed to read players CSV: " + str(e))
    if uploaded_perf:
        try:
            perf_new = pd.read_csv(uploaded_perf)
            st.session_state["perf_df_uploaded"] = perf_new
            perf_df = perf_new
            st.success("Performance file loaded into session.")
            st.dataframe(perf_new.head(10))
        except Exception as e:
            st.error("Failed to read performance CSV: " + str(e))

st.markdown("---")
st.markdown("Made by **Akshitha Velugu** • Infosys Springboard Virtual Internship 6.0")
