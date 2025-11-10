
# adding injury, matches, competitions to already combined.csv
import pandas as pd

# load your already merged file (fifa + transfer + sentiment)
combined = pd.read_csv("../DATA/processed/combined.csv", encoding="latin1")
print("Base combined shape:", combined.shape)

# load the remaining datasets
injury = pd.read_csv("../DATA/processed/injury_data_cleaned.csv", encoding="latin1")
matches = pd.read_csv("../DATA/processed/matches_cleaned.csv", encoding="latin1")
competitions = pd.read_csv("../DATA/processed/competitions_cleaned.csv", encoding="latin1")

# ----------- clean and prepare ------------
# fix player_name column in base
if "player_name" in combined.columns:
    combined["player_name"] = combined["player_name"].astype(str).str.lower().str.strip()
else:
    # check if any similar column
    for c in combined.columns:
        if "player" in c.lower() or "name" in c.lower():
            combined.rename(columns={c:"player_name"}, inplace=True)
            combined["player_name"] = combined["player_name"].astype(str).str.lower().str.strip()
            break

# clean injury nationality/age if exist
if "nationality" in injury.columns:
    injury["nationality"] = injury["nationality"].astype(str).str.lower().str.strip()

# ---------------- Merge 1: try injury ----------------
if "player_name" in combined.columns and "player_name" in injury.columns:
    combined = combined.merge(injury, on="player_name", how="left")
elif "age" in combined.columns and "nationality" in combined.columns:
    # rough match on age + nationality
    if "age" in injury.columns and "nationality" in injury.columns:
        combined = combined.merge(injury, on=["age","nationality"], how="left")
print("After adding injury:", combined.shape)

# ---------------- Merge 2: competitions ----------------
if "League" in combined.columns and "competition_name" in competitions.columns:
    combined = combined.merge(competitions, left_on="League", right_on="competition_name", how="left")
elif "League Country" in combined.columns and "country_name" in competitions.columns:
    combined = combined.merge(competitions, left_on="League Country", right_on="country_name", how="left")
print("After adding competitions:", combined.shape)

# ---------------- Merge 3: matches ----------------
if "League" in combined.columns and "competition" in matches.columns:
    combined = combined.merge(matches, left_on="League", right_on="competition", how="left")
else:
    print("Could not find a perfect match column for matches, skipping partial merge.")
print("After adding matches:", combined.shape)

combined.to_csv("../DATA/processed/final_combined_all.csv", index=False)
print("Saved final_combined_all.csv")
