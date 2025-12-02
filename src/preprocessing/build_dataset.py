import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
PROCESSED = Path("data/processed")
PROCESSED.mkdir(parents=True, exist_ok=True)

def build_final_dataset():
    perf = pd.read_csv(INTERIM / "performance_features.csv")
    market = pd.read_csv("data/raw/transfermarkt_values.csv")
    senti = pd.read_csv(INTERIM / "sentiment_features.csv")
    injury = pd.read_csv(INTERIM / "injury_features.csv")

    market.columns = market.columns.str.lower()

    # Merge step by step
    df = perf.merge(market, left_on="player_name", right_on="player_name", how="left")
    df = df.merge(senti, on="player_name", how="left")
    df = df.merge(injury, on="player_name", how="left")

    # Fill missing values
    df["avg_sentiment"] = df["avg_sentiment"].fillna(0)
    df["total_injury_days"] = df["total_injury_days"].fillna(0)

    df.to_csv(PROCESSED / "final_dataset.csv", index=False)
    print("✔ final_dataset.csv saved")


if __name__ == "__main__":
    build_final_dataset()
    print("\nFINAL DATASET READY ✔")
