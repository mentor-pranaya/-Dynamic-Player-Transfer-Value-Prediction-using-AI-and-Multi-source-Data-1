import asyncpraw
import asyncio
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import os

# Reddit API setup
client_id = "kv3fpeZpJyBL1P0TGX-U1Q"
client_secret = "51VesG6uGZ_qhd0camnbtNDJqC_HjA"
username = "Weird-Pension-1907"
password = "Bhanu@2004"
user_agent = "reddit-sentiment-bot by u/Weird-Pension-1907"

# players I want to analyze
players = ["Lionel Messi", "Erling Haaland", "Kylian Mbappe"]
output_path = "data/raw/reddit_player_sentiment.csv"
post_limit = 500


os.makedirs(os.path.dirname(output_path), exist_ok=True)

# sentiment analyzer
analyzer = SentimentIntensityAnalyzer()


def find_sentiment(text):
    
    score = analyzer.polarity_scores(text)
    compound = score["compound"]
    if compound >= 0.05:
        label = "positive"
    elif compound <= -0.05:
        label = "negative"
    else:
        label = "neutral"
    return compound, label


async def get_posts(reddit, player):
    print("Collecting posts for:", player)
    collected = []
    try:
        sub = await reddit.subreddit("all")
        async for post in sub.search(player, limit=post_limit, sort="new"):
            text = (post.title + "\n" + post.selftext).strip()
            if text == "":
                continue

            collected.append({
                "player": player,
                "title": post.title,
                "text": text,
                "url": post.url,
                "score": post.score,
                "created_utc": post.created_utc
            })
        print("Done:", player, "-", len(collected), "posts")
    except Exception as e:
        print("Error for", player, ":", e)
    return collected


async def main():
    try:
        async with asyncpraw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            username=username,
            password=password,
            user_agent=user_agent
        ) as reddit:

            all_data = []
            tasks = [get_posts(reddit, p) for p in players]
            results = await asyncio.gather(*tasks)

            for player_posts in results:
                for post in player_posts:
                    score, label = find_sentiment(post["text"])
                    post["sentiment_score"] = score
                    post["sentiment_label"] = label
                    all_data.append(post)

            df = pd.DataFrame(all_data)
            df.to_csv(output_path, index=False)
            print("Data saved to", output_path)

    except Exception as e:
        print("Some error occurred:", e)


if __name__ == "__main__":
    asyncio.run(main())
