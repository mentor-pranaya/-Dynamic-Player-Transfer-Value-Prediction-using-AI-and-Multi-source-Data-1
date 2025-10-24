import requests
import pandas as pd
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
import time

BEARER_TOKEN = "AAAAAAAAAAAAAAAAAAAAAIqA4wEAAAAAZUJ6PNGB%2Beu7pj1g65OnshdZm2U%3Di0cmfk92Ss0LFYXuRuujICEVnjb9j8DKPwkXXpUXNcaTiJyjeS"

def create_headers():
    return {"Authorization": f"Bearer {BEARER_TOKEN}"}

def search_tweets(query, max_results=20):
    url = "https://api.twitter.com/2/tweets/search/recent"
    params = {
        "query": query,
        "max_results": max_results,
        "tweet.fields": "id,text,author_id,created_at,lang",
    }
    while True:
        response = requests.get(url, headers=create_headers(), params=params)
        if response.status_code == 429:  
            print("Rate limit hit. Sleeping for 60 seconds...")
            time.sleep(60)
            continue
        response.raise_for_status()
        return response.json()

def analyze_sentiment(text):
    analyzer = SentimentIntensityAnalyzer()
    score = analyzer.polarity_scores(text)['compound']
    if score > 0.05:
        return score, "positive"
    elif score < -0.05:
        return score, "negative"
    else:
        return score, "neutral"

def main():
    players = ["Kylian Mbappe", "Erling Haaland", "Lionel Messi"]
    all_data = []

    for player in players:
        query = f'"{player}" lang:en'
        print(f"Collecting tweets about {player}...")
        data = search_tweets(query)
        for tweet in data.get('data', []):
            score, label = analyze_sentiment(tweet['text'])
            all_data.append({
                "player": player,
                "tweet_id": tweet["id"],
                "created_at": tweet["created_at"],
                "text": tweet["text"],
                "sentiment_score": score,
                "sentiment_label": label
            })
        time.sleep(5) 

    output_path = "data/raw/twitter_player_mentions_sentiment.csv"
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(all_data)
    df.to_csv(output_path, index=False, encoding='utf-8')
    print(f"Done! File saved in {output_path}")

if __name__ == "__main__":
    main()
