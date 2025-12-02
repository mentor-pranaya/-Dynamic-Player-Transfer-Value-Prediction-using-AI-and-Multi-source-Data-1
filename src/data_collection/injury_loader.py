"""
injury_loader.py

Load and clean injury datasets. Supports:
- reading a local input CSV (preferred)
- optional lightweight scraping helper for Transfermarkt injury pages (educational)

Usage:
  python src/data_collection/injury_loader.py --input_csv data/raw/injuries.csv --out_csv data/processed/injuries_clean.csv
"""

import os
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import List, Dict

import pandas as pd
import requests
from bs4 import BeautifulSoup
from time import sleep

logger = logging.getLogger("injury_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_local_injury_csv(path: str) -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(f"Injury CSV not found at {path}")
    df = pd.read_csv(path)
    logger.info(f"Loaded {len(df)} injury rows from {path}")
    return df


def clean_injury_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [c.strip().lower() for c in df.columns]

    # Normalize player_name column
    if "player" in df.columns and "player_name" not in df.columns:
        df = df.rename(columns={"player": "player_name"})

    for col in ("start_date", "end_date"):
        if col in df.columns:
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "days_absent" not in df.columns:
        if "start_date" in df.columns and "end_date" in df.columns:
            df["days_absent"] = (df["end_date"] - df["start_date"]).dt.days.fillna(0).astype(int)
        else:
            df["days_absent"] = 0

    # Aggregate total days absent per player
    if "player_name" in df.columns:
        agg = df.groupby("player_name", as_index=False)["days_absent"].sum()
    else:
        agg = pd.DataFrame(columns=["player_name", "days_absent"])
    return agg


def save_clean_injuries(df: pd.DataFrame, out_path: str = "data/processed/injuries_clean.csv"):
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    logger.info(f"Saved cleaned injuries to {out_path}")


def scrape_transfermarkt_injuries(player_injury_url: str, rate_limit_seconds: float = 2.0) -> List[Dict]:
    """
    Lightweight heuristic parser for Transfermarkt injury pages.
    Use only as a helper / research tool; markup can change.
    """
    try:
        r = requests.get(player_injury_url, headers={"User-Agent": "Mozilla/5.0"}, timeout=10)
        r.raise_for_status()
        sleep(rate_limit_seconds)
    except Exception as e:
        logger.error(f"Request failed: {e}")
        return []

    soup = BeautifulSoup(r.text, "html.parser")
    rows = []
    table = soup.find("table")
    if not table:
        logger.warning("No table found on injury page.")
        return rows
    for tr in table.find_all("tr"):
        cols = [td.get_text(" ", strip=True) for td in tr.find_all(["td", "th"])]
        if not cols:
            continue
        # Very coarse heuristics:
        start_date = None
        end_date = None
        try:
            # search for date-like columns
            for c in cols:
                if "." in c and len(c) >= 6:
                    try:
                        dt = datetime.strptime(c.strip(), "%d.%m.%Y")
                        if start_date is None:
                            start_date = dt.date().isoformat()
                        elif end_date is None:
                            end_date = dt.date().isoformat()
                    except Exception:
                        continue
            injury_type = cols[-1] if cols else "unknown"
            rows.append({"player_injury_url": player_injury_url, "injury_type": injury_type, "start_date": start_date, "end_date": end_date, "days_absent": None})
        except Exception:
            continue
    return rows


def cli():
    parser = argparse.ArgumentParser(description="Injury loader and cleaner")
    parser.add_argument("--input_csv", type=str, help="Path to local injury CSV")
    parser.add_argument("--out_csv", type=str, default="data/processed/injuries_clean.csv", help="Output cleaned CSV")
    parser.add_argument("--scrape_url", type=str, help="Optional Transfermarkt injury URL to scrape")
    parser.add_argument("--scrape_only", action="store_true", help="Only scrape the given URL and save")
    args = parser.parse_args()

    if args.scrape_only:
        if not args.scrape_url:
            logger.error("Provide --scrape_url with --scrape_only")
            return
        rows = scrape_transfermarkt_injuries(args.scrape_url)
        df = pd.DataFrame(rows)
        save_clean_injuries(df, args.out_csv)
        return

    if args.input_csv:
        try:
            df = load_local_injury_csv(args.input_csv)
            cleaned = clean_injury_dataframe(df)
            save_clean_injuries(cleaned, args.out_csv)
        except Exception as e:
            logger.error(f"Failed to process input CSV: {e}")
    elif args.scrape_url:
        rows = scrape_transfermarkt_injuries(args.scrape_url)
        df = pd.DataFrame(rows)
        if not df.empty:
            # try compute days_absent
            try:
                df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
                df["end_date"] = pd.to_datetime(df["end_date"], errors="coerce")
                df["days_absent"] = (df["end_date"] - df["start_date"]).dt.days.fillna(0).astype(int)
            except Exception:
                pass
        save_clean_injuries(df, args.out_csv)
    else:
        logger.error("Provide --input_csv or --scrape_url (or use --scrape_only).")


if __name__ == "__main__":
    cli()
