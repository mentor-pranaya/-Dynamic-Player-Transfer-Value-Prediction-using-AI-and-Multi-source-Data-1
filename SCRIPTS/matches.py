# cleaning the matches.csv file

import os
import json
import pandas as pd

matches_folder = r"D:\INFOSYS\DATA\Raw_data\open-data\data\matches"
output_folder = r"D:\INFOSYS\DATA\processed"
os.makedirs(output_folder, exist_ok=True)

csv_path = os.path.join(output_folder, "matches.csv")
first_file = True
file_counter = 0

json_files = [f for f in os.listdir(matches_folder) if f.endswith(".json")]
if not json_files:
    print("No JSON files in matches folder")
else:
    for file in json_files:
        file_path = os.path.join(matches_folder, file)
        print(f"Processing {file}")
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                data = json.load(f)

            if not data: 
                continue

            df = pd.DataFrame(data)

            if first_file:
                df.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                first_file = False
            else:
                df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

            file_counter += 1
            if file_counter % 100 == 0:
                print(f"{file_counter} files processed")

        except json.JSONDecodeError:
            print(f" Skipping broken JSON: {file}")
        except MemoryError:
            print(f" Skipping too large file: {file}")

print("matches.csv saved ")
