"""Make sure there's a `data` folder at the same level as the `src` folder before running this script. It should contain all the necessary dataset files for the code to work correctly."""

"""Since the CSV file is large, data cleaning might take some time."""

"""Run `python src/01_data_cleaning.py`"""

import os
import pandas as pd

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "BERPublicsearch.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# 🔹 Load dataset
df = pd.read_csv(
    file_path,
    encoding="ISO-8859-1",
    delimiter="\t",
    on_bad_lines="skip",
    low_memory=False,
    dtype=str
)

# 🔹 Standardize column names (lowercase, remove spaces)
df.columns = df.columns.str.strip().str.lower()

# 🔹 Convert specific numeric columns
numeric_cols = [
    "year_of_construction", "groundfloorarea(sq m)", "co2rating",
    "hsmainsystemefficiency", "mpcdervalue", "hseffadjfactor",
    "supplshfuel", "supplwhfuel", "noofchimneys", "primaryenergylighting",
    "primaryenergyspace", "co2lighting", "co2space", "totalprimaryenergyfact",
    "totalco2emissions"
]

# Convert numeric columns while keeping errors manageable
for col in numeric_cols:
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")

# 🔹 Handle missing values
categorical_cols = ["energyrating", "dwellingtypedescr", "typeofrating"]

# Fill missing categorical values with mode
for col in categorical_cols:
    if col in df.columns:
        mode_value = df[col].mode()[0]
        df[col] = df[col].fillna(mode_value)

# Fill missing numeric values with median
for col in numeric_cols:
    if col in df.columns:
        median_value = df[col].median()
        df[col] = df[col].fillna(median_value)

# 🔹 Drop columns with too many missing values (optional)
missing_threshold = 0.5  # Drop columns with >50% missing values
df = df.dropna(axis=1, thresh=int(missing_threshold * len(df)))

# Save the cleaned dataset
cleaned_file_path = os.path.join(data_folder, "cleaned_data.csv")
df.to_csv(cleaned_file_path, index=False)

print("✅ Data cleaning completed. Saved as 'cleaned_data.csv'.")