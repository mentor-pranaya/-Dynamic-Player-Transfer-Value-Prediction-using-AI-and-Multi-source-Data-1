import pandas as pd
from pathlib import Path

RAW = Path("data/raw")
INTERIM = Path("data/interim")
INTERIM.mkdir(parents=True, exist_ok=True)

def clean_performance():
    df = pd.read_csv(RAW / "performance.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["player_id"] = df["player_id"].astype(str)
    df = df.drop_duplicates()
    df.to_csv(INTERIM / "performance_clean.csv", index=False)
    print("✔ performance_clean.csv saved")

def clean_transfermarkt():
    df = pd.read_csv(RAW / "transfermarkt_values.csv")
    df.columns = df.columns.str.strip().str.lower()
    df = df.drop_duplicates()
    df.to_csv(INTERIM / "transfermarkt_clean.csv", index=False)
    print("✔ transfermarkt_clean.csv saved")

def clean_tweets():
    df = pd.read_csv(RAW / "tweets.csv")
    df.columns = df.columns.str.strip().str.lower()
    df["vader_compound"] = df["vader_compound"].fillna(0)
    df.to_csv(INTERIM / "tweets_clean.csv", index=False)
    print("✔ tweets_clean.csv saved")

def clean_injuries():
    df = pd.read_csv(RAW / "injuries.csv")
    df.columns = df.columns.str.strip().str.lower()

    # Convert date formats
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")

    # Calculate missing days_absent
    df["days_absent"] = df["days_absent"].fillna(
        (df["end_date"] - df["start_date"]).dt.days
    ).fillna(0)

    df.to_csv(INTERIM / "injuries_clean.csv", index=False)
    print("✔ injuries_clean.csv saved")

if __name__ == "__main__":
    clean_performance()
    clean_transfermarkt()
    clean_tweets()
    clean_injuries()
    print("\nALL CLEAN FILES CREATED ✔")
