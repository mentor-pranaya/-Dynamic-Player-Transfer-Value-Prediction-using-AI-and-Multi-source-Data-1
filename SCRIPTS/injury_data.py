
# cleaned injury dataset


import pandas as pd

input_file = r"D:\INFOSYS\DATA\Raw_data\injury_data_kggle.csv"
output_file = r"D:\INFOSYS\DATA\processed\injury_data_cleaned.csv"

df = pd.read_csv(input_file)
print("Rows before cleaning:", len(df))

df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')  
    else:
        df[col] = df[col].fillna(0)          

numeric_cols = [
    'start_year','season_days_injured','total_days_injured',
    'season_minutes_played','season_games_played','season_matches_in_squad',
    'total_minutes_played','total_games_played','height_cm','weight_kg',
    'fifa_rating','age','cumulative_minutes_played','cumulative_games_played',
    'minutes_per_game_prev_seasons','avg_days_injured_prev_seasons',
    'avg_games_per_season_prev_seasons','bmi','work_rate_numeric',
    'position_numeric','significant_injury_prev_season','cumulative_days_injured',
    'season_days_injured_prev_season'
]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)

df.to_csv(output_file, index=False)
print("Cleaned injury dataset saved ")
print("Rows after cleaning:", len(df))
