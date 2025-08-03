import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler

# STEP 1: Load the combined material data
df = pd.read_csv("materials_combined.csv")
print("Initial shape:", df.shape)

# STEP 2: Check and drop duplicates
df = df.drop_duplicates()
print("After removing duplicates:", df.shape)

# STEP 3: Check and handle missing values
missing_summary = df.isnull().sum()
print("Missing values:\n", missing_summary)

# Drop rows with critical missing fields
df = df.dropna(subset=["metalness", "roughness", "color_r", "color_g", "color_b"])

# Optional: Fill less important missing values with defaults (if appropriate)
df.fillna({
    "specular_r": 0.0,
    "specular_g": 0.0,
    "specular_b": 0.0,
}, inplace=True)

# STEP 4: Ensure numeric fields have correct data types
numeric_cols = ["metalness", "roughness", "color_r", "color_g", "color_b",
                "specular_r", "specular_g", "specular_b"]

for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Re-check for invalid conversions (optional sanity check)
print("Column types:\n", df[numeric_cols].dtypes)

# STEP 5: Optional Normalisation (for ML models like FAISS)
scaler = MinMaxScaler()
df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# STEP 6: Save cleaned version
df.to_csv("cleaned_materials.csv", index=False)
print(f" Cleaned data saved to cleaned_materials.csv with shape: {df.shape}")
