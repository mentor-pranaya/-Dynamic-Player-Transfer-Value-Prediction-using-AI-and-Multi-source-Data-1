# combine Reddit and Twitter player sentiment datasets
import pandas as pd

reddit_file = r"D:\INFOSYS\SCRIPTS\data\raw\reddit_player_sentiment.csv"
twitter_file = r"D:\INFOSYS\SCRIPTS\data\raw\twitter_player_mentions_sentiment.csv"
output_file = r"D:\INFOSYS\SCRIPTS\data\processed\combined_sentiment.csv"


reddit_data = pd.read_csv(reddit_file)
twitter_data = pd.read_csv(twitter_file)

print("Reddit data rows:", len(reddit_data))
print("Twitter data rows:", len(twitter_data))


reddit_data["source"] = "reddit"
twitter_data["source"] = "twitter"


reddit_data.rename(columns={"created_utc": "created_time"}, inplace=True)
twitter_data.rename(columns={"created_at": "created_time"}, inplace=True)


all_data = pd.concat([reddit_data, twitter_data], ignore_index=True)
print("Rows before cleaning:", len(all_data))


all_data.drop_duplicates(subset=["player", "text"], inplace=True)

for col in all_data.columns:
    if all_data[col].dtype == 'object':
        all_data[col].fillna("Unknown", inplace=True)
    else:
        all_data[col].fillna(0, inplace=True)

print("Rows after cleaning:", len(all_data))

all_data.to_csv(output_file, index=False, encoding="utf-8-sig")
print("Combined social media dataset saved ")
