# transfermarket dataset cleaning
import pandas as pd


input_file = r"D:\INFOSYS\DATA\Raw_data\transfermarkt_github.csv"
output_file = r"D:\INFOSYS\DATA\processed\transfermarket_cleaned.csv"


df = pd.read_csv(input_file)
print("Rows before cleaning:", len(df))

df = df.drop_duplicates()

for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = df[col].fillna('Unknown')
    else:
        df[col] = df[col].fillna(0)

df['Market Value'] = pd.to_numeric(df['Market Value'], errors='coerce').fillna(0)
df['Fee'] = pd.to_numeric(df['Fee'], errors='coerce').fillna(0)

df.to_csv(output_file, index=False)
print("Cleaned transfermarket dataset saved ")
print("Rows after cleaning:", len(df))
