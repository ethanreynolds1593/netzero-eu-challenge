"""Make sure there's a `data` folder at the same level as the `src` folder before running this script. It should contain all the necessary dataset files for the code to work correctly."""

"""Since the CSV file is large, data preprocessing might take some time."""

"""Run `python src/03_data_preprocessing.py`"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from scipy import sparse

# Get the directory of the current script
script_dir = os.getcwd()

# Construct the file path dynamically for input dataset
data_folder = os.path.join(script_dir, ".", "data")
file_path = os.path.join(data_folder, "processed_data.csv")

# Ensure the data folder exists
os.makedirs(data_folder, exist_ok=True)

# Load the dataset
data = pd.read_csv(file_path, delimiter=",", low_memory=False, dtype=str)

# 🎯 Define the target variable
target_column = "totalprimaryenergyfact"
X = data.drop(columns=[target_column])  # Features
y = data[target_column]                 # Target

# 🛠️ Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 🚀 Drop columns that are **entirely missing** (fix for UserWarning)
X_train = X_train.dropna(axis=1, how="all")
X_test = X_test.dropna(axis=1, how="all")

# 🔍 Identify feature types
numeric_features = X_train.select_dtypes(include=["int64", "float64"]).columns.tolist()
categorical_features = X_train.select_dtypes(include=["object"]).columns.tolist()

# 🔢 Handle categorical features:
# Define **ordinal** and **one-hot** encoding based on cardinality
low_cardinality_features = [col for col in categorical_features if X_train[col].nunique() < 10]
high_cardinality_features = [col for col in categorical_features if X_train[col].nunique() >= 10]

print(f"High-cardinality categorical features: {len(high_cardinality_features)}")
print(f"Low-cardinality categorical features: {len(low_cardinality_features)}")

# 🏗️ Define Pipelines

# Pipeline for numeric features (Imputation + No Scaling)
numeric_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))  # Fill missing values with median
])

# Pipeline for low-cardinality categorical features (One-Hot Encoding)
onehot_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),  # Fill missing categorical values
    ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))  # ✅ Fixed sparse argument
])

# Pipeline for high-cardinality categorical features (Ordinal Encoding)
ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))
])

# 🏗️ Combine transformations
preprocessor = ColumnTransformer([
    ("num", numeric_pipeline, numeric_features),
    ("onehot", onehot_pipeline, low_cardinality_features),
    ("ordinal", ordinal_pipeline, high_cardinality_features)
])

# 🔄 Fit and transform training data
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# ✅ Convert sparse matrices to CSR format if applicable
if sparse.issparse(X_train_transformed):
    X_train_transformed = X_train_transformed.tocsr()
    X_test_transformed = X_test_transformed.tocsr()
    
# Save the preprocessed (optional)
cleaned_file_path = os.path.join(data_folder, "preprocessed_data.npz")
np.savez(cleaned_file_path, X_train=X_train_transformed, X_test=X_test_transformed, y_train=y_train, y_test=y_test)

print("✅ Preprocessing complete! Data saved as 'preprocessed_data.npz'.")
