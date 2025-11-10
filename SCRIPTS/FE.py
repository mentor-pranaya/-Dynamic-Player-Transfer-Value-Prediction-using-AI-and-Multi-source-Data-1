# feature engineering 

import pandas as pd
import numpy as np

df = pd.read_csv("../DATA/processed/final_cleaned.csv", encoding="latin1")
print("loaded", df.shape)

# experience
df["experience"] = 0
if "last_season" in df.columns:
    df["experience"] = 2025 - df["last_season"]

# value drop ratio
df["value_drop_ratio"] = 0
if "market_value_in_eur" in df.columns and "highest_market_value_in_eur" in df.columns:
    df["value_drop_ratio"] = (df["market_value_in_eur"] / df["highest_market_value_in_eur"]).fillna(0).clip(0,1)

# attack index
df["attack_index"] = 0
if "goals" in df.columns and "assists" in df.columns:
    df["attack_index"] = df["goals"].fillna(0) + df["assists"].fillna(0)

# injury severity
df["injury_severity"] = 0
if "total_days_injured" in df.columns and "total_games_played" in df.columns:
    df["injury_severity"] = (df["total_days_injured"].fillna(0) /
                             df["total_games_played"].replace({0:np.nan})).fillna(0)

df.to_csv("../DATA/processed/feature_engineered.csv", index=False)
print("saved feature_engineered.csv", df.shape)
