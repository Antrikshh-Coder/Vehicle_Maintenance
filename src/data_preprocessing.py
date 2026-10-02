"""
Data Preprocessing Module for Vehicle Maintenance Analysis.
Handles dataset loading, quality checks, data cleaning, train-test splitting,
and feature scaling without data leakage.
"""

import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

def load_data(filepath="data/dataset.csv"):
    """Load dataset from CSV file."""
    if not os.path.exists(filepath):
        # Fallback path if run from subfolder
        filepath = os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv")
    df = pd.read_csv(filepath)
    return df

def clean_data(df):
    """
    Perform data cleaning:
    - Check missing values
    - Check duplicates
    - Validate feature types
    Returns cleaned dataframe and a cleaning report dict.
    """
    cleaned_df = df.copy()
    initial_shape = cleaned_df.shape
    
    # 1. Check & drop duplicates
    duplicates_count = int(cleaned_df.duplicated().sum())
    if duplicates_count > 0:
        cleaned_df = cleaned_df.drop_duplicates().reset_index(drop=True)
        
    # 2. Check missing values
    missing_counts = cleaned_df.isnull().sum().to_dict()
    
    report = {
        "initial_rows": initial_shape[0],
        "final_rows": cleaned_df.shape[0],
        "duplicates_removed": duplicates_count,
        "missing_values": missing_counts
    }
    return cleaned_df, report

def prepare_pipeline(df, target_col="Engine Condition", test_size=0.2, random_state=42):
    """
    Separates features and target, performs 80/20 train-test split,
    fits StandardScaler on training data only (preventing data leakage),
    and saves feature names and scaler.
    """
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    feature_names = list(X.columns)
    
    # 80/20 Train-Test split with stratification
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Feature Scaling: Fit scaler ONLY on X_train to prevent data leakage
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save scaler and feature names
    os.makedirs("models", exist_ok=True)
    joblib.dump(scaler, "models/scaler.joblib")
    joblib.dump(feature_names, "models/feature_names.joblib")
    
    return {
        "X_train_raw": X_train,
        "X_test_raw": X_test,
        "X_train_scaled": X_train_scaled,
        "X_test_scaled": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler
    }

if __name__ == "__main__":
    df = load_data()
    cleaned_df, report = clean_data(df)
    print("Data Cleaning Report:", report)
    pipeline_data = prepare_pipeline(cleaned_df)
    print("Train shape:", pipeline_data["X_train_scaled"].shape)
    print("Test shape:", pipeline_data["X_test_scaled"].shape)
