"""
twitter_sentiment.py

Collect tweets (snscrape or Tweepy) and compute VADER sentiment scores.
Outputs a CSV with: player_keyword, tweet_date, tweet_text, vader_compound

Usage examples:
  # snscrape mode (no API key required)
  python src/data_collection/twitter_sentiment.py --mode snscrape --query "Lionel Messi since:2024-01-01" --player "Lionel Messi" --max_tweets 500 --output data/raw/messi_tweets.csv

  # tweepy mode (requires TWITTER_BEARER_TOKEN env var)
  python src/data_collection/twitter_sentiment.py --mode tweepy --query "Kylian Mbappe" --player "Kylian Mbappe" --max_tweets 200
"""

import os
import csv
import logging
import argparse
from typing import List, Dict

from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

logger = logging.getLogger("twitter_sentiment")
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

analyzer = SentimentIntensityAnalyzer()


def score_text(text: str) -> float:
    return analyzer.polarity_scores(text)["compound"]


def save_to_csv(rows: List[Dict], outpath: str):
    os.makedirs(os.path.dirname(outpath) or ".", exist_ok=True)
    fieldnames = ["player_keyword", "tweet_date", "tweet_text", "vader_compound"]
    write_header = not os.path.exists(outpath)
    with open(outpath, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for r in rows:
            writer.writerow(r)


def fetch_tweets_snscrape(query: str, max_tweets: int = 500) -> List[Dict]:
    try:
        import snscrape.modules.twitter as sntwitter
    except Exception as e:
        raise RuntimeError("snscrape not installed. Install via: pip install snscrape") from e

    tweets = []
    logger.info("Using snscrape (no API) mode.")
    for i, tweet in enumerate(sntwitter.TwitterSearchScraper(query).get_items()):
        if i >= max_tweets:
            break
        tweets.append({"tweet_date": tweet.date.isoformat(), "tweet_text": tweet.content})
    logger.info(f"Fetched {len(tweets)} tweets via snscrape.")
    return tweets


def fetch_tweets_tweepy(query: str, max_tweets: int = 200) -> List[Dict]:
    try:
        import tweepy
    except ImportError:
        raise RuntimeError("tweepy not installed. Install via: pip install tweepy")

    bearer_token = os.environ.get("TWITTER_BEARER_TOKEN")
    if not bearer_token:
        raise RuntimeError("TWITTER_BEARER_TOKEN environment variable not set for Tweepy mode.")

    client = tweepy.Client(bearer_token=bearer_token, wait_on_rate_limit=True)
    tweets = []
    logger.info("Using Tweepy API mode.")
    for resp in tweepy.Paginator(client.search_recent_tweets, query=query, tweet_fields=["created_at","text"], max_results=100):
        if resp.data is None:
            continue
        for t in resp.data:
            tweets.append({"tweet_date": t.created_at.isoformat(), "tweet_text": t.text})
            if len(tweets) >= max_tweets:
                break
        if len(tweets) >= max_tweets:
            break
    logger.info(f"Fetched {len(tweets)} tweets via tweepy.")
    return tweets


def collect_and_score(player_keyword: str, query: str, mode: str = "snscrape", max_tweets: int = 300, output_csv: str = "data/raw/tweets.csv"):
    if mode not in ("tweepy", "snscrape"):
        raise ValueError("mode must be 'tweepy' or 'snscrape'")

    if mode == "tweepy":
        tweets = fetch_tweets_tweepy(query, max_tweets=max_tweets)
    else:
        tweets = fetch_tweets_snscrape(query, max_tweets=max_tweets)

    rows = []
    for t in tweets:
        txt = t.get("tweet_text", "")
        date = t.get("tweet_date")
        try:
            score = score_text(str(txt))
        except Exception:
            score = 0.0
        rows.append({"player_keyword": player_keyword, "tweet_date": date, "tweet_text": txt, "vader_compound": score})
    save_to_csv(rows, output_csv)
    logger.info(f"Saved {len(rows)} tweets to {output_csv}")
    return rows


def cli():
    parser = argparse.ArgumentParser(description="Collect tweets and compute VADER sentiment")
    parser.add_argument("--mode", choices=["tweepy", "snscrape"], default="snscrape")
    parser.add_argument("--query", required=True, help="Search query (for snscrape include date range if desired)")
    parser.add_argument("--player", required=True, help="Player short identifier to store in CSV")
    parser.add_argument("--max_tweets", type=int, default=300)
    parser.add_argument("--output", default="data/raw/tweets.csv")
    args = parser.parse_args()

    try:
        collect_and_score(player_keyword=args.player, query=args.query, mode=args.mode, max_tweets=args.max_tweets, output_csv=args.output)
    except Exception as e:
        logger.error(f"Error during tweet collection: {e}")


if __name__ == "__main__":
    cli()
