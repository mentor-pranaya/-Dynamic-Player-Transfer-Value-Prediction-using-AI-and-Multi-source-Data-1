"""
statsbomb_loader.py

Simple utilities to download (optional) and/or load StatsBomb Open Data JSON files
and produce a lightweight aggregated player CSV.

Usage:
  # Download minimal files (optional)
  python src/data_collection/statsbomb_loader.py --download --data_dir data/raw/statsbomb

  # Build aggregated CSV from local JSONs
  python src/data_collection/statsbomb_loader.py --data_dir data/raw/statsbomb --out_csv data/processed/statsbomb_player_agg.csv
"""

import os
import json
import logging
import argparse
from pathlib import Path
from time import sleep
from typing import List, Any, Dict, Optional

import pandas as pd
import requests

logger = logging.getLogger("statsbomb_loader")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

STATSBOMB_RAW_GITHUB = "https://raw.githubusercontent.com/statsbomb/open-data/master/data/"


def ensure_dir(path: str):
    Path(path).mkdir(parents=True, exist_ok=True)


def download_json(url: str, out_path: str, retry: int = 2, timeout: int = 20) -> bool:
    try:
        logger.info(f"Downloading {url}")
        for attempt in range(retry + 1):
            try:
                r = requests.get(url, timeout=timeout)
                r.raise_for_status()
                with open(out_path, "w", encoding="utf-8") as fh:
                    fh.write(r.text)
                return True
            except Exception as e:
                logger.warning(f"Attempt {attempt+1} failed: {e}")
                sleep(1)
        return False
    except Exception as e:
        logger.error(f"Download failed: {e}")
        return False


def download_statsbomb_data(dest_dir: str, files: Optional[List[str]] = None):
    ensure_dir(dest_dir)
    if files is None:
        files = ["competitions.json"]  # minimal. extend as needed.
    for rel in files:
        url = STATSBOMB_RAW_GITHUB + rel
        out_path = os.path.join(dest_dir, os.path.basename(rel))
        ok = download_json(url, out_path)
        if not ok:
            logger.warning(f"Failed to download {rel}")


def load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def aggregate_player_stats_from_events(events: List[Dict]) -> Dict[str, Dict]:
    players = {}
    for ev in events:
        try:
            typ = ev.get("type", {}).get("name", "").lower()
            player = ev.get("player")
            if not player:
                continue
            pid = player.get("player_id") or player.get("id") or player.get("wyId")
            if pid is None:
                continue
            pid = str(pid)
            if pid not in players:
                players[pid] = {
                    "player_name": player.get("name") or player.get("player_name") or "",
                    "shots": 0,
                    "passes": 0,
                    "tackles": 0,
                    "events": 0,
                }
            players[pid]["events"] += 1
            if "shot" in typ:
                players[pid]["shots"] += 1
            if "pass" in typ:
                players[pid]["passes"] += 1
            if "tackle" in typ or "block" in typ:
                players[pid]["tackles"] += 1
        except Exception:
            continue
    return players


def build_match_player_summary_from_event_file(event_json_path: str) -> pd.DataFrame:
    try:
        events = load_json(event_json_path)
        if not isinstance(events, list):
            logger.warning(f"{event_json_path} does not look like an events list.")
            return pd.DataFrame()
        players = aggregate_player_stats_from_events(events)
        rows = []
        for pid, info in players.items():
            row = {"player_id": pid}
            row.update(info)
            rows.append(row)
        return pd.DataFrame(rows)
    except Exception as e:
        logger.error(f"Error parsing {event_json_path}: {e}")
        return pd.DataFrame()


def build_dataset_from_local_statsbomb(data_dir: str, out_csv: str):
    p = Path(data_dir)
    if not p.exists():
        logger.error(f"StatsBomb data directory does not exist: {data_dir}")
        return
    event_files = list(p.rglob("*.json"))
    logger.info(f"Found {len(event_files)} JSON files in {data_dir}")
    frames = []
    for ef in event_files:
        df = build_match_player_summary_from_event_file(str(ef))
        if not df.empty:
            df["source_event_file"] = ef.name
            frames.append(df)
    if frames:
        combined = pd.concat(frames, ignore_index=True)
        grouped = (
            combined.groupby(["player_id", "player_name"], as_index=False)
            .agg({"shots": "sum", "passes": "sum", "tackles": "sum", "events": "sum"})
        )
        ensure_dir(os.path.dirname(out_csv) or ".")
        grouped.to_csv(out_csv, index=False)
        logger.info(f"Wrote aggregated player stats to {out_csv}")
    else:
        logger.warning("No event files processed; no CSV written.")


def cli():
    parser = argparse.ArgumentParser(description="StatsBomb loader & light aggregator")
    parser.add_argument("--data_dir", default="data/raw/statsbomb", help="Local folder with StatsBomb JSON files")
    parser.add_argument("--download", action="store_true", help="Download minimal files from GitHub")
    parser.add_argument("--download_files", nargs="*", help="Specific relative files to download")
    parser.add_argument("--out_csv", default="data/processed/statsbomb_player_agg.csv", help="Output aggregated CSV")
    args = parser.parse_args()

    if args.download:
        download_statsbomb_data(args.data_dir, files=args.download_files)
    build_dataset_from_local_statsbomb(args.data_dir, args.out_csv)


if __name__ == "__main__":
    cli()
