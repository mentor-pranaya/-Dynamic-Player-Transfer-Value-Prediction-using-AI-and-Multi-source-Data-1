import pandas as pd
from pathlib import Path

INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

def create_performance_features():
    df = pd.read_csv(INTERIM / "performance_clean.csv")

    df["shot_rate"] = df["shots"] / df["events"]
    df["pass_rate"] = df["passes"] / df["events"]
    df["tackle_rate"] = df["tackles"] / df["events"]

    df.to_csv(INTERIM / "performance_features.csv", index=False)
    print("✔ performance_features.csv saved")


def create_sentiment_features():
    df = pd.read_csv(INTERIM / "tweets_clean.csv")

    sentiment = df.groupby("player_keyword")["vader_compound"].mean().reset_index()
    sentiment = sentiment.rename(columns={"player_keyword": "player_name",
                                          "vader_compound": "avg_sentiment"})

    sentiment.to_csv(INTERIM / "sentiment_features.csv", index=False)
    print("✔ sentiment_features.csv saved")


def create_injury_features():
    df = pd.read_csv(INTERIM / "injuries_clean.csv")

    inj = df.groupby("player_name")["days_absent"].sum().reset_index()
    inj = inj.rename(columns={"days_absent": "total_injury_days"})

    inj.to_csv(INTERIM / "injury_features.csv", index=False)
    print("✔ injury_features.csv saved")


if __name__ == "__main__":
    create_performance_features()
    create_sentiment_features()
    create_injury_features()
    print("\nALL FEATURE FILES CREATED ✔")
