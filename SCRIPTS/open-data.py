
# CLEANING OPEN-DATA CSV FILES 

import pandas as pd
import os

# paths
data_folder = r"D:\INFOSYS\DATA\processed"
output_folder = r"D:\INFOSYS\DATA\processed"
os.makedirs(output_folder, exist_ok=True)

files = ["competitions.csv", "matches.csv", "events.csv", "lineups.csv", "three-sixty.csv"]

for file_name in files:
    input_file = os.path.join(data_folder, file_name)
    output_file = os.path.join(output_folder, f"{file_name.split('.')[0]}_cleaned.csv")
    
    if not os.path.exists(input_file):
        print(f"{file_name} not found, skipping...")
        continue
    
    print(f"\nCleaning {file_name}...")

    try:
        chunk_list = []  
        for chunk in pd.read_csv(input_file, chunksize=500000):  

            chunk = chunk.drop_duplicates()
            
            for col in chunk.columns:
                if chunk[col].dtype == 'object':
                    chunk[col] = chunk[col].fillna('Unknown')
                else:
                    chunk[col] = chunk[col].fillna(0)
            
            numeric_cols = chunk.select_dtypes(include=['int64','float64']).columns
            for col in numeric_cols:
                chunk[col] = pd.to_numeric(chunk[col], errors='coerce').fillna(0)
            
            chunk_list.append(chunk)
        
        df = pd.concat(chunk_list, ignore_index=True)
        df.to_csv(output_file, index=False)
        print(f"{file_name} cleaned ")
        print("Rows after cleaning:", len(df))
    
    except Exception as e:
        print(f"Error cleaning {file_name}: {e}")
