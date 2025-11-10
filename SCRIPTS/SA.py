
import pandas as pd
import numpy as np

# try to import VADER and TextBlob, if not installed explain to user
try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    have_vader = True
except Exception:
    print("VADER not installed (pip install vaderSentiment). Falling back to TextBlob only.")
    have_vader = False

try:
    from textblob import TextBlob
    have_textblob = True
except Exception:
    print("TextBlob not installed (pip install textblob). TextBlob features will be skipped.")
    have_textblob = False

# load files 
FE_PATH = "../DATA/processed/final_cleaned.csv"   # main features (or change)
SENT_PATH = "../DATA/processed/combined_sentiment.csv"  # posts/tweets

print("loading files...")
fe = pd.read_csv(FE_PATH, encoding="latin1")
posts = pd.read_csv(SENT_PATH, encoding="latin1")
print("fe:", fe.shape, "posts:", posts.shape)

# find text & player column
# detect text column
text_col = None
for c in ["text","Tweet","tweet","content","message","title"]:
    if c in posts.columns:
        text_col = c
        break
if text_col is None:
    print("No text column found in posts file. Add a text column and re-run.")
    raise SystemExit()

# detect player column in posts (optional)
player_col = None
for c in ["player","player_name","name","playerName","Label"]:
    if c in posts.columns:
        player_col = c
        break

# if no player column, we will compute global sentiment and attach as global feature
if player_col is None:
    print("No player column found in posts. Script will compute global sentiment to attach to all players.")
else:
    print("found player column in posts:", player_col)

# ensure text is string
posts[text_col] = posts[text_col].astype(str).fillna("")

#  per-post sentiment using VADER & TextBlob 
if have_vader:
    print("computing VADER scores...")
    analyzer = SentimentIntensityAnalyzer()
    posts["vader_compound"] = posts[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["compound"])
    posts["vader_pos"] = posts[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["pos"])
    posts["vader_neu"] = posts[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["neu"])
    posts["vader_neg"] = posts[text_col].apply(lambda t: analyzer.polarity_scores(str(t))["neg"])
else:
    # fill with NaN so later aggregations still work
    posts["vader_compound"] = np.nan
    posts["vader_pos"] = np.nan
    posts["vader_neu"] = np.nan
    posts["vader_neg"] = np.nan

if have_textblob:
    print("computing TextBlob scores...")
    posts["tb_polarity"] = posts[text_col].apply(lambda t: TextBlob(str(t)).sentiment.polarity)
    posts["tb_subjectivity"] = posts[text_col].apply(lambda t: TextBlob(str(t)).sentiment.subjectivity)
else:
    posts["tb_polarity"] = np.nan
    posts["tb_subjectivity"] = np.nan

# small keyword list 
pos_kw = ["good","great","win","love","happy","amazing","best"]
neg_kw = ["bad","sad","lose","injury","hate","poor","worst"]

def kw_score(t):
    t = str(t).lower()
    sc = 0
    for w in pos_kw:
        if w in t:
            sc += 1
    for w in neg_kw:
        if w in t:
            sc -= 1
    return sc

posts["kw_score"] = posts[text_col].apply(kw_score)

#
# Many social datasets have likes/retweets/replies columns 
pop_cols = []
for c in ["likes","like_count","retweets","retweet_count","replies","reply_count","favorites","favorite_count"]:
    if c in posts.columns:
        pop_cols.append(c)

if pop_cols:
    print("found popularity columns:", pop_cols)
    # create a simple popularity score (sum of whatever available)
    posts["popularity"] = posts[pop_cols].fillna(0).sum(axis=1)
else:
    posts["popularity"] = 1  # fallback: every post counts as 1 (so average by counts works)

# aggregate per-player (or global) ---------
if player_col is None:
    # global aggregates -> one number
    agg = {
        "vader_compound": "mean",
        "tb_polarity": "mean",
        "tb_subjectivity": "mean",
        "kw_score": "mean",
        "popularity": "sum"
    }
    global_stats = posts.agg(agg).to_dict()
    print("global stats:", {k: round(v,3) if pd.notna(v) else v for k,v in global_stats.items()})

    # add these as columns to FE
    fe["vader_compound_mean"] = global_stats.get("vader_compound", 0)
    fe["tb_polarity_mean"] = global_stats.get("tb_polarity", 0)
    fe["kw_score_mean"] = global_stats.get("kw_score", 0)
    fe["popularity_total"] = global_stats.get("popularity", 0)

else:
    # normalize player names in both df and posts for safer join
    fe_cols = fe.columns.tolist()
    # try to find player name column in FE; prefer 'player_name'
    fe_player_col = None
    for c in ["player_name","name","player","Player"]:
        if c in fe_cols:
            fe_player_col = c
            break
    if fe_player_col is None:
        print("No player_name column found in main features file. Can't join per-player. Exiting.")
        raise SystemExit()
    # normalize names
    posts[player_col] = posts[player_col].astype(str).str.lower().str.strip()
    fe[fe_player_col] = fe[fe_player_col].astype(str).str.lower().str.strip()

    # groupby player -> weighted and unweighted aggregates
    # weighted mean by popularity and plain mean
    agg_funcs = {
        "vader_compound": ["mean","median","std"],
        "tb_polarity": ["mean","median","std"],
        "tb_subjectivity": ["mean"],
        "kw_score": ["mean","median"],
        "popularity": ["sum","count"]
    }

    g = posts.groupby(player_col).agg(agg_funcs)
    # flatten columns
    g.columns = ["_".join(col).strip() for col in g.columns.values]
    g = g.reset_index().rename(columns={player_col: "player_name"})

    # compute weighted vader by popularity (popularity might be 1 for many rows)
    # first get weighted sums
    posts_tmp = posts.copy()
    posts_tmp["vader_times_pop"] = posts_tmp["vader_compound"].fillna(0) * posts_tmp["popularity"].fillna(1)
    pop_sum = posts_tmp.groupby(player_col)["popularity"].sum().reset_index().rename(columns={player_col:"player_name","popularity":"pop_sum"})
    vader_weighted = posts_tmp.groupby(player_col)["vader_times_pop"].sum().reset_index().rename(columns={player_col:"player_name","vader_times_pop":"vader_pop_sum"})
    # merge and compute weighted mean
    weighted = pd.merge(vader_weighted, pop_sum, on="player_name", how="left")
    weighted["vader_pop_mean"] = weighted.apply(lambda r: r["vader_pop_sum"]/r["pop_sum"] if r["pop_sum"]!=0 else 0, axis=1)
    weighted = weighted[["player_name","vader_pop_mean"]]

    # merge aggregates & weighted
    player_sent = pd.merge(g, weighted, on="player_name", how="left")

    # percent positive / negative tweets (VADER thresholds)
    def vader_label(x):
        if pd.isna(x): return "neu"
        if x >= 0.05: return "pos"
        if x <= -0.05: return "neg"
        return "neu"

    posts["vader_label"] = posts["vader_compound"].apply(lambda x: vader_label(x))
    label_counts = posts.groupby(player_col)["vader_label"].value_counts().unstack(fill_value=0).reset_index().rename(columns={player_col:"player_name"})
    # make percent columns
    for lbl in ["pos","neg","neu"]:
        if lbl in label_counts.columns:
            label_counts[f"vader_{lbl}_pct"] = label_counts[lbl] / label_counts[[c for c in ["pos","neg","neu"] if c in label_counts.columns]].sum(axis=1)

    # merge label_counts
    player_sent = pd.merge(player_sent, label_counts[["player_name"] + [c for c in label_counts.columns if c.startswith("vader_")]], on="player_name", how="left")

    # final fillna
    player_sent = player_sent.fillna(0)

    # merge with main features (left join so we keep all players in fe)
    fe = pd.merge(fe, player_sent, left_on=fe_player_col, right_on="player_name", how="left")

    # if any missing for players, fill with global means
    global_means = {
        "vader_compound_mean": posts["vader_compound"].mean(),
        "tb_polarity_mean": posts["tb_polarity"].mean(),
        "kw_score_mean": posts["kw_score"].mean(),
        "popularity_total": posts["popularity"].sum()
    }
    # fill reasonable columns that exist
    for col in fe.columns:
        if col.endswith("_mean") or col.endswith("_median") or col.endswith("_std") or "vader_pop_mean" in col or col.startswith("vader_"):
            fe[col] = fe[col].fillna(0)

# compute correlation between sentiment and market value if market value exists
target_cols = [c for c in fe.columns if "market_value" in c.lower() or "value" in c.lower()]
if target_cols:
    val_col = target_cols[0]
    try:
        corr_val = fe[[val_col, "vader_compound_mean"]].corr().iloc[0,1]
        print("correlation between", val_col, "and vader_compound_mean:", round(corr_val,3))
    except Exception:
        pass

OUT = "../DATA/processed/nlp_sentiment.csv"
fe.to_csv(OUT, index=False)
print("saved ->", OUT)
