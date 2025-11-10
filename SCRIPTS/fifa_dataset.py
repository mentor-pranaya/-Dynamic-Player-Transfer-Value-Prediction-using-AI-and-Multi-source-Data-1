import pandas as pd

# just reading the file normally
fifa = pd.read_csv("../DATA/Raw_data/fifa_dataset.csv", encoding="latin1")

print("before:", fifa.shape)

# removing duplicates
fifa.drop_duplicates(inplace=True)

# checking missing values
print("missing values:", fifa.isnull().sum().sum())

# filling missing data
for col in fifa.columns:
    if fifa[col].dtype == "float64" or fifa[col].dtype == "int64":
        fifa[col].fillna(fifa[col].mean(), inplace=True)
    else:
        fifa[col].fillna(fifa[col].mode()[0], inplace=True)

# cleaning player names if there is a column
if "Player" in fifa.columns:
    fifa["Player"] = fifa["Player"].astype(str)
    fifa["Player"] = fifa["Player"].str.strip().str.lower()

print("after:", fifa.shape)

# saving the cleaned file
fifa.to_csv("../DATA/processed/fifa_cleaned.csv", index=False)
print("done cleaning fifa data")
