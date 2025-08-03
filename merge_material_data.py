import pandas as pd
import os

# Define file paths
downloads_folder = os.path.expanduser("~/Downloads")
scripts_folder = os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "material-advisor", "scripts")
data_folder= os.path.join(os.path.expanduser("~"), "OneDrive", "Desktop", "material-advisor", "data")

file_1 = os.path.join(downloads_folder, "Book1.csv")
file_2 = os.path.join(downloads_folder, "1.csv")
target_file = os.path.join(data_folder, "text_prompts.csv")

# Load CSVs
df1 = pd.read_csv(file_1)
df2 = pd.read_csv(file_2)
target_df = pd.read_csv(target_file)

# Function to align and merge
def align_and_merge(source_df, target_columns):
    # Keep only columns that exist in target
    matching_cols = [col for col in source_df.columns if col in target_columns]
    aligned_df = source_df[matching_cols].copy()
    
    # Add any missing columns with NaN
    for col in target_columns:
        if col not in aligned_df.columns:
            aligned_df[col] = pd.NA
            
    # Reorder columns to match target
    aligned_df = aligned_df[target_columns]
    return aligned_df

# Align both new files
merged1 = align_and_merge(df1, target_df.columns)
merged2 = align_and_merge(df2, target_df.columns)

# Combine everything
updated_df = pd.concat([target_df, merged1, merged2], ignore_index=True)

# Save updated file
updated_df.to_csv(target_file, index=False)
print(f" Updated text_prompts.csv saved at:\n{target_file}")
print(f" Total rows after merge: {updated_df.shape[0]}")
