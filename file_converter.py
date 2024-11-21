import os
import pandas as pd

def combine_parquet_to_csv(folder_path, output_csv_file):
    
    all_dataframes = []
    
    # Loop through all files in the directory
    for file_name in os.listdir(folder_path):
        if file_name.endswith(".parquet"):  # Check for .parquet extension
            file_path = os.path.join(folder_path, file_name)
            print(f"Reading {file_path}...")
            df = pd.read_parquet(file_path)  # Read the parquet file
            all_dataframes.append(df)  # Add the DataFrame to the list
    
    # Combine all DataFrames into one
    if all_dataframes:
        combined_df = pd.concat(all_dataframes, ignore_index=True)
        print(f"Saving combined data to {output_csv_file}...")
        combined_df.to_csv(output_csv_file, index=False)
        print("CSV file created successfully.")
    else:
        print("No .parquet files found in the specified folder.")

# Usage
folder_path = input("Enter the folder path containing .parquet files: ")
output_csv_file = input("Enter the path for the output CSV file (e.g., combined_data.csv): ")

combine_parquet_to_csv(folder_path, output_csv_file)
