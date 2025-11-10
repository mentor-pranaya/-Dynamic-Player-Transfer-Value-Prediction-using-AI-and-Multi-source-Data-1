# - AFE + simple graphs 
# Adds small features, encodes, scales, saves final AFE file and 4 png graphs.

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import matplotlib.pyplot as plt
import seaborn as sns
import os

# ) make sure output folder exists 
out_dir = "../DATA/processed"
os.makedirs(out_dir, exist_ok=True)

# 1) load 
infile = "../DATA/processed/nlp_sentiment.csv"
print("loading:", infile)
df = pd.read_csv(infile, encoding="latin1")
print("loaded shape:", df.shape)

#  quick clean
df.columns = df.columns.str.strip()
df.drop_duplicates(inplace=True)
print("after drop duplicates:", df.shape)

#  fill missing simply
for c in df.select_dtypes(include=[np.number]).columns:
    df[c] = df[c].fillna(df[c].mean())

for c in df.select_dtypes(include=['object']).columns:
    if df[c].isnull().any():
        try:
            df[c] = df[c].fillna(df[c].mode().iloc[0])
        except Exception:
            df[c] = df[c].fillna("unknown")

print("missing values filled")

#  create small derived features
if "Age" in df.columns:
    df["age_group"] = pd.cut(df["Age"],
                             bins=[0, 20, 25, 30, 35, 50, 100],
                             labels=["<20","20-25","25-30","30-35","35-50","50+"]).astype(str)
    print("added age_group")

val_col = None
for candidate in ["market_value_in_eur", "Market Value", "market_value"]:
    if candidate in df.columns:
        val_col = candidate
        break

if val_col is not None:
    df["log_value"] = np.log1p(df[val_col].fillna(0))
    print("added log_value from", val_col)

if "sentiment_score" in df.columns:
    df["sentiment_level"] = pd.cut(df["sentiment_score"],
                                   bins=[-999, -1, 0, 1, 999],
                                   labels=["neg","neutral","pos","high_pos"]).astype(str)
    print("added sentiment_level")

if "attack_index" in df.columns and "sentiment_score" in df.columns:
    df["attack_sent_interaction"] = df["attack_index"].fillna(0) * df["sentiment_score"].fillna(0)
    print("added attack_sent_interaction")

#encode small categorical columns 
to_encode = []
for c in ["position","club","nationality","age_group","sentiment_level"]:
    if c in df.columns:
        to_encode.append(c)

if to_encode:
    print("encoding:", to_encode)
    le = LabelEncoder()
    for c in to_encode:
        df[c] = df[c].astype(str)
        try:
            df[c] = le.fit_transform(df[c])
        except Exception:
            df[c] = pd.factorize(df[c])[0]
else:
    print("no small categorical columns found to encode")

#  select numeric columns and scale 
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()

# drop id-like numeric columns from scaling
drop_like = []
for name in num_cols:
    if name.lower().endswith("id") or name.lower().startswith("id_") or name.lower() in ["player_id","tweet_id"]:
        drop_like.append(name)

if drop_like:
    print("dropping id-like numeric cols from scaling:", drop_like)
    num_cols = [c for c in num_cols if c not in drop_like]

if len(num_cols) > 0:
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols].fillna(0))
    print("scaled numeric columns count:", len(num_cols))
else:
    print("no numeric columns to scale")

#  7) save correlation matrix 
num_only = df.select_dtypes(include=[np.number])
if not num_only.empty:
    corr = num_only.corr()
    corr_file = os.path.join(out_dir, "feature_correlation_matrix_final.csv")
    corr.to_csv(corr_file, index=True)
    print("saved correlation matrix ->", corr_file)
else:
    print("no numeric columns found, skipping correlation matrix")


# 8) save final AFE dataset 
out_file = os.path.join(out_dir, "final_afe_dataset.csv")
df.to_csv(out_file, index=False)
print("saved final AFE file ->", out_file)
print("final shape:", df.shape)

#PLOTS 
# Plot 1: Sentiment distribution (if exists)
try:
    if "sentiment_score" in df.columns:
        plt.figure(figsize=(6,4))
        plt.hist(df["sentiment_score"], bins=20, edgecolor="black")
        plt.title("Sentiment Score Distribution")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Count")
        f1 = os.path.join(out_dir, "plot_sentiment_distribution.png")
        plt.tight_layout()
        plt.savefig(f1)
        plt.close()
        print("saved", f1)
except Exception as e:
    print("plot1 failed:", e)

# Plot 2: Log market value vs sentiment scatter (if both exist)
try:
    if "log_value" in df.columns and "sentiment_score" in df.columns:
        plt.figure(figsize=(6,4))
        plt.scatter(df["sentiment_score"], df["log_value"], alpha=0.5)
        plt.title("Sentiment vs Log Market Value")
        plt.xlabel("Sentiment Score")
        plt.ylabel("Log Market Value")
        f2 = os.path.join(out_dir, "plot_sentiment_vs_value.png")
        plt.tight_layout()
        plt.savefig(f2)
        plt.close()
        print("saved", f2)
except Exception as e:
    print("plot2 failed:", e)

# Plot 3: Average engineered feature means (attack, injury, sentiment, value)
try:
    keys = [k for k in ["attack_index","injury_severity","sentiment_score","log_value"] if k in df.columns]
    if keys:
        means = df[keys].mean().sort_values(ascending=False)
        plt.figure(figsize=(6,4))
        means.plot(kind="bar")
        plt.title("Average of Some Engineered Features")
        plt.ylabel("Scaled mean")
        f3 = os.path.join(out_dir, "plot_feature_means.png")
        plt.tight_layout()
        plt.savefig(f3)
        plt.close()
        print("saved", f3)
except Exception as e:
    print("plot3 failed:", e)

# Plot 4: Correlation heatmap 
try:
    numeric_for_heat = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_for_heat) > 0:
        top = numeric_for_heat[:12]
        plt.figure(figsize=(8,6))
        sns.heatmap(df[top].corr(), cmap="coolwarm", annot=False)
        plt.title("Correlation (top numeric features)")
        f4 = os.path.join(out_dir, "plot_correlation_heatmap.png")
        plt.tight_layout()
        plt.savefig(f4)
        plt.close()
        print("saved", f4)
    else:
        print("no numeric columns for heatmap, skipping")
except Exception as e:
    print("plot4 failed:", e)
