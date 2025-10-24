#converting json files to csv files (matches)

import os
import json
import pandas as pd

base_path = r"D:\INFOSYS\DATA\Raw_data\open-data\data"
output_folder = r"D:\INFOSYS\DATA\processed"
os.makedirs(output_folder, exist_ok=True)

folders = ["matches", "events", "lineups", "three-sixty"]

for folder in folders:
    folder_path = os.path.join(base_path, folder)
    if os.path.exists(folder_path):
        csv_path = os.path.join(output_folder, folder + ".csv")
        first_file = True  

        json_files = [f for f in os.listdir(folder_path) if f.endswith(".json")]
        if not json_files:
            print(f"No JSON files in {folder}")
            continue

        for file in json_files:
            print(f"Processing {file} in {folder}...")
            with open(os.path.join(folder_path, file), "r", encoding="utf-8") as f:
                data = json.load(f)
            df = pd.DataFrame(data)

            if first_file:
                df.to_csv(csv_path, index=False, mode='w', encoding='utf-8-sig')
                first_file = False
            else:
                df.to_csv(csv_path, index=False, mode='a', header=False, encoding='utf-8-sig')

        print(f"{folder}.csv saved ")
    else:
        print(f"Folder not found: {folder}")
