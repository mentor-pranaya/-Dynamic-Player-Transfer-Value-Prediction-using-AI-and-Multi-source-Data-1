"""
transfermarkt_scraper.py

Polite, educational Transfermarkt HTML scraper to extract market-value snapshots.
WARNING: scraping websites may violate terms of service — use for learning only.

Usage:
  python src/data_collection/transfermarkt_scraper.py --urls_file data/raw/player_urls.txt --output_csv data/raw/transfermarkt_values.csv
"""

import time
import csv
import logging
import argparse
from typing import List, Optional
from pathlib import Path

import requests
from bs4 import BeautifulSoup

logger = logging.getLogger("transfermarkt_scraper")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


class TransfermarktScraper:
    def __init__(self, rate_limit_seconds: float = 2.5, user_agent: Optional[str] = None, output_csv: Optional[str] = None):
        self.rate_limit_seconds = rate_limit_seconds
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": user_agent or "Mozilla/5.0 (compatible; TransferIQ/1.0; +https://example.com/bot)"
        })
        self.output_csv = output_csv
        if output_csv:
            Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

    def _get(self, url: str, timeout: int = 15) -> Optional[str]:
        try:
            logger.info(f"GET {url}")
            r = self.session.get(url, timeout=timeout)
            r.raise_for_status()
            time.sleep(self.rate_limit_seconds)
            return r.text
        except Exception as e:
            logger.warning(f"Request failed for {url}: {e}")
            return None

    @staticmethod
    def _parse_value_text(value_text: str) -> Optional[float]:
        if not value_text:
            return None
        t = value_text.strip().replace("\n", " ").replace(" ", "")
        if t in ("-", "—", ""):
            return None
        t = t.replace("€", "").replace(",", ".")
        try:
            if t.lower().endswith("m"):
                return float(t[:-1]) * 1_000_000
            if t.lower().endswith("k"):
                return float(t[:-1]) * 1_000
            return float(t)
        except Exception:
            return None

    def parse_player_page(self, html: str, source_url: str) -> dict:
        soup = BeautifulSoup(html, "html.parser")
        # player name heuristics
        name_tag = soup.select_one("div.dataMain h1") or soup.select_one("h1")
        name = name_tag.get_text(strip=True) if name_tag else None

        raw_value = None
        selectors = [
            ".marktwert",
            "div.dataMarktwert .marktwert",
            "div.player-header__market-value",
            ".dataMarktwert .marktwert"
        ]
        for sel in selectors:
            el = soup.select_one(sel)
            if el and el.get_text(strip=True):
                raw_value = el.get_text(strip=True)
                break

        if raw_value is None:
            # fallback scanning small text nodes
            texts = soup.find_all(text=True)
            for i, t in enumerate(texts[:300]):
                text = str(t).strip().lower()
                if "marktwert" in text or "market value" in text:
                    if i + 1 < len(texts):
                        candidate = texts[i + 1].strip()
                        if candidate:
                            raw_value = candidate
                            break

        parsed_value = self._parse_value_text(raw_value) if raw_value else None

        snapshot = None
        try:
            date_el = soup.find(lambda tag: tag.name in ("p", "div", "span") and ("as of" in tag.get_text().lower() or "stand:" in tag.get_text().lower()))
            if date_el:
                snapshot = date_el.get_text(" ", strip=True)
        except Exception:
            snapshot = None

        return {
            "source_url": source_url,
            "player_name": name,
            "raw_value_text": raw_value,
            "market_value_eur": parsed_value,
            "snapshot_info": snapshot
        }

    def scrape_player(self, player_url: str) -> Optional[dict]:
        html = self._get(player_url)
        if not html:
            return None
        return self.parse_player_page(html, player_url)

    def scrape_player_list(self, player_urls: List[str]) -> List[dict]:
        rows = []
        for url in player_urls:
            try:
                data = self.scrape_player(url)
                if data:
                    rows.append(data)
                    if self.output_csv:
                        self._append_to_csv(data)
            except Exception as e:
                logger.exception(f"Error scraping {url}: {e}")
        return rows

    def _append_to_csv(self, row: dict):
        fieldnames = ["source_url", "player_name", "raw_value_text", "market_value_eur", "snapshot_info"]
        out = Path(self.output_csv)
        write_header = not out.exists()
        try:
            with open(out, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                if write_header:
                    writer.writeheader()
                writer.writerow(row)
        except Exception as e:
            logger.warning(f"Failed to write CSV {self.output_csv}: {e}")


def cli():
    parser = argparse.ArgumentParser(description="Transfermarkt scraper (educational/demo use only)")
    parser.add_argument("--urls_file", type=str, default="data/raw/player_urls.txt", help="Text file with one Transfermarkt player URL per line")
    parser.add_argument("--output_csv", type=str, default="data/raw/transfermarkt_values.csv", help="Output CSV")
    parser.add_argument("--rate", type=float, default=2.5, help="Seconds between requests")
    args = parser.parse_args()

    urls = []
    try:
        with open(args.urls_file, "r", encoding="utf-8") as fh:
            urls = [line.strip() for line in fh if line.strip()]
    except FileNotFoundError:
        logger.error(f"{args.urls_file} not found. Create it with one Transfermarkt player URL per line.")
        return

    if urls:
        scraper = TransfermarktScraper(rate_limit_seconds=args.rate, output_csv=args.output_csv)
        scraper.scrape_player_list(urls)
        logger.info("Scraping complete.")
    else:
        logger.info("No URLs supplied.")


if __name__ == "__main__":
    cli()
